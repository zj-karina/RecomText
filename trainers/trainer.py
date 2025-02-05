import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import get_losses
from utils.metrics import MetricsCalculator
import os
from transformers import AutoModel
import pandas as pd
import numpy as np
import faiss

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.contrastive_loss_fn, self.recommendation_loss_fn = get_losses()
        self.metrics_calculator = MetricsCalculator()
        self.ks = config.get('evaluation', {}).get('ks', [1, 5, 10])
        
        # Для отслеживания лучшей модели
        self.best_metric = float('-inf')
        self.best_epoch = 0
        self.patience = config.get('training', {}).get('patience', 5)
        self.no_improvement = 0

    def train(self, epochs):
        """Полный цикл обучения с валидацией."""
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Обучение
            train_metrics = self.train_epoch()
            if epoch == 0:
                self._save_checkpoint(epoch)
            print("\nTraining metrics:")
            self._print_metrics(train_metrics)
            
            # Валидация
            val_metrics = self.validate()
            print("\nValidation metrics:")
            self._print_metrics(val_metrics)
            
            # Проверка на улучшение
            current_metric = val_metrics.get('ndcg@10', 0)  # Можно настроить метрику для отслеживания
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.no_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.no_improvement += 1
            
            # Early stopping
            if self.no_improvement >= self.patience:
                print(f"\nEarly stopping triggered. No improvement for {self.patience} epochs.")
                break

    def train_epoch(self):
        """Один эпох обучения."""
        self.model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_recommendation_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            loss, c_loss, r_loss = self.training_step(batch)
            
            total_loss += loss
            total_contrastive_loss += c_loss
            total_recommendation_loss += r_loss
            break

        return {
            'loss': total_loss / len(self.train_loader),
            'contrastive_loss': total_contrastive_loss / len(self.train_loader),
            'recommendation_loss': total_recommendation_loss / len(self.train_loader)
        }

    def validate(self):
        """Валидация модели с использованием актуального FAISS индекса и кастомных метрик."""
        self.model.eval()
        total_loss = 0
        total_contrastive_loss = 0
        total_recommendation_loss = 0
        
        # Загружаем необходимые данные
        textual_history = pd.read_parquet('./data/textual_history.parquet')
        
        # Проверяем наличие FAISS индекса и эмбеддингов
        index_path = self.config['inference'].get('index_path', 'video_index.faiss')
        ids_path = self.config['inference'].get('ids_path', 'video_ids.npy')
        embeddings_path = './data/item_embeddings.npy'
        
        if not os.path.exists(index_path) or not os.path.exists(ids_path):
            print("\nFAISS index not found, creating new index...")
            try:
                from indexer import main as create_index
                create_index(config=self.config)
                print("FAISS index created successfully")
            except Exception as e:
                print(f"Warning: Failed to create FAISS index: {str(e)}")
                return None

        # Загружаем необходимые данные
        index = faiss.read_index(index_path)
        video_ids = np.load(ids_path).tolist()
        df_videos = pd.read_parquet("./data/video_info.parquet")
        df_videos_map = df_videos.set_index('clean_video_id').to_dict(orient='index')
        
        # Загружаем или создаем эмбеддинги товаров
        if os.path.exists(embeddings_path):
            item_embeddings_array = np.load(embeddings_path)
        else:
            print("Warning: item_embeddings.npy not found. Computing embeddings...")
            # TODO: Добавить логику создания эмбеддингов, если необходимо
            return None

        # Загружаем демографические данные
        try:
            user_demographics_df = pd.read_parquet('./data/user_demographics.parquet')
            demographic_features = ['age_group', 'gender', 'location']  # пример характеристик
            has_demographics = True
        except FileNotFoundError:
            print("Warning: Demographics data not found, DAS will not be calculated")
            has_demographics = False

        # Вычисляем центроиды для демографических групп, если данные доступны
        demographic_centroids = {}
        if has_demographics:
            print("Computing demographic centroids...")
            for feature in demographic_features:
                demographic_centroids[feature] = {}
                for group in user_demographics_df[feature].unique():
                    group_users = user_demographics_df[user_demographics_df[feature] == group].index
                    if len(group_users) > 0:
                        # Получаем эмбеддинги для пользователей группы
                        group_embeddings = []
                        for user_id in group_users:
                            # Здесь нужно получить эмбеддинги товаров, которые смотрел пользователь
                            # Это зависит от вашей структуры данных
                            user_items = textual_history[textual_history['user_id'] == user_id]
                            if len(user_items) > 0:
                                item_embs = torch.tensor(
                                    [item_embeddings_array[idx] for idx in user_items['item_id']],
                                    device=self.device
                                )
                                group_embeddings.append(item_embs.mean(dim=0))
                        
                        if group_embeddings:
                            group_centroid = torch.stack(group_embeddings).mean(dim=0)
                            demographic_centroids[feature][group] = F.normalize(group_centroid, p=2, dim=0)

        # Инициализируем калькулятор метрик
        metrics_calculator = MetricsCalculator(sim_threshold=0.7)
        
        # Инициализируем аккумуляторы для метрик
        metrics_accum = {
            "semantic_precision@k": 0.0,
            "cross_category_relevance": 0.0,
            "contextual_ndcg": 0.0
        }
        num_users = 0
        
        # Получаем параметры для метрик
        top_k = self.config['inference'].get('top_k', 10)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                items_text_inputs, user_text_inputs, item_ids, user_ids = [
                    self.to_device(x) for x in batch
                ]
                
                # Выводим историю просмотров для первого батча
                if batch_idx == 0:
                    print("\nValidation Example:")
                    print("\nUser viewing history:")
                    print(textual_history.iloc[batch_idx]['detailed_view'].replace("query: ", ""))

                # Forward pass
                items_embeddings, user_embeddings = self.model(
                    items_text_inputs,
                    user_text_inputs,
                    item_ids,
                    user_ids
                )

                # Нормализация эмбеддингов
                items_embeddings = F.normalize(items_embeddings, p=2, dim=1)
                user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

                # Вычисляем потери
                recommendation_loss = self.compute_recommendation_loss(user_embeddings, items_embeddings)
                contrastive_loss = self.compute_contrastive_loss(items_embeddings, user_embeddings)
                loss = contrastive_loss + self.config['training']['lambda_rec'] * recommendation_loss

                total_loss += loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_recommendation_loss += recommendation_loss.item()

                # Вычисляем метрики для каждого пользователя в батче
                batch_size = user_embeddings.size(0)
                for i in range(batch_size):
                    user_emb = user_embeddings[i].unsqueeze(0)
                    target_item_emb = items_embeddings[i]
                    
                    # Получаем ID и категорию целевого товара
                    target_item_id = str(item_ids[i].item() if item_ids[i].dim() == 0 else item_ids[i][0].item())
                    target_category = df_videos_map.get(target_item_id, {}).get('category', 'Unknown')

                    # Поиск топ-K рекомендаций
                    user_emb_np = user_emb.cpu().numpy().astype('float32')
                    distances, faiss_indices = index.search(user_emb_np, top_k)
                    
                    # Получаем эмбеддинги и категории рекомендованных товаров
                    recommended_embeddings = torch.tensor(
                        [item_embeddings_array[idx] for idx in faiss_indices[0]],
                        device=target_item_emb.device
                    )
                    recommended_embeddings = F.normalize(recommended_embeddings, p=2, dim=1)
                    
                    recommended_categories = []
                    for idx in faiss_indices[0]:
                        vid = video_ids[idx]
                        vid_value = vid[0] if isinstance(vid, (list, tuple, np.ndarray)) else vid
                        video_data = df_videos_map.get(str(vid_value), {})
                        recommended_categories.append(video_data.get('category', 'Unknown'))

                    # Получаем демографические данные пользователя, если доступны
                    user_demo = {}
                    if has_demographics:
                        user_id = user_ids[i].item()
                        if user_id in user_demographics_df.index:
                            for feature in demographic_features:
                                user_demo[feature] = user_demographics_df.loc[user_id, feature]

                    # Вычисляем метрики для текущего пользователя
                    user_metrics = metrics_calculator.compute_metrics(
                        target_item_emb,
                        recommended_embeddings,
                        target_category,
                        recommended_categories,
                        top_k,
                        user_demographics=user_demo if has_demographics else None,
                        demographic_centroids=demographic_centroids if has_demographics else None
                    )

                    # Аккумулируем метрики
                    for metric_name, metric_value in user_metrics.items():
                        metrics_accum[metric_name] += metric_value
                    num_users += 1

                    # Выводим примеры рекомендаций для первого пользователя первого батча
                    if batch_idx == 0 and i == 0:
                        print(f"\nTop-{top_k} recommendations:")
                        for rank, (idx, score) in enumerate(zip(faiss_indices[0], distances[0]), 1):
                            vid = video_ids[idx]
                            vid_value = vid[0] if isinstance(vid, (list, tuple, np.ndarray)) else vid
                            video_data = df_videos_map.get(str(vid_value), {})
                            title = video_data.get('title', 'Unknown title')
                            cat = video_data.get('category', 'Unknown category')
                            print(f"  {rank}. Video ID={vid_value}, Score={score:.4f}, Category={cat}")
                            print(f"     Title: {title}")

        # Вычисляем средние значения метрик
        metrics_dict = {
            'val_loss': total_loss / len(self.val_loader),
            'val_contrastive_loss': total_contrastive_loss / len(self.val_loader),
            'val_recommendation_loss': total_recommendation_loss / len(self.val_loader)
        }
        
        # Добавляем средние значения кастомных метрик
        if num_users > 0:
            for metric_name, metric_sum in metrics_accum.items():
                metrics_dict[metric_name] = metric_sum / num_users

        # Выводим все метрики
        print("\nValidation Metrics:")
        for metric_name, metric_value in metrics_dict.items():
            print(f"{metric_name}: {metric_value:.4f}")

        return metrics_dict

    def _save_checkpoint(self, epoch, metrics=None):
        """Сохраняет чекпоинт и обновляет индекс."""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        if metrics:
            # Сохраняем дополнительные данные
            checkpoint_meta = {
                'epoch': epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'best_metric': self.best_metric,
                'config': self.config  # Сохраняем конфиг для воспроизводимости
            }
            
            meta_path = os.path.join(checkpoint_dir, f'meta_epoch_{epoch}.pt')
            torch.save(checkpoint_meta, meta_path)
        
        # Сохраняем модель
        self.model.save_pretrained(os.path.join(checkpoint_dir, f'model_epoch_{epoch}'))
        print(f"\nSaved checkpoint for epoch {epoch}")
        
        # Обновляем FAISS индекс с путем к текущему чекпоинту
        try:
            print("\nUpdating FAISS index...")
            from indexer import main as update_index
            # Создаем временный конфиг с обновленным путем к модели
            temp_config = self.config.copy()
            temp_config['training']['checkpoint_dir'] = os.path.join(checkpoint_dir, f'model_epoch_{epoch}')
            update_index(config=temp_config)
            print("FAISS index updated successfully")
        except Exception as e:
            print(f"Warning: Failed to update FAISS index: {str(e)}")

    def _print_metrics(self, metrics):
        """Вывод метрик в консоль."""
        # Группируем метрики по типу
        losses = {k: v for k, v in metrics.items() if 'loss' in k}
        # precision = {k: v for k, v in metrics.items() if 'precision' in k}
        # recall = {k: v in metrics.items() if 'recall' in k}
        # ndcg = {k: v for k, v in metrics.items() if 'ndcg' in k}
        # coverage = {k: v for k, v in metrics.items() if 'coverage' in k}
        # diversity = {k: v for k, v in metrics.items() if 'diversity' in k}
        # other = {k: v for k, v in metrics.items() if not any(x in k for x in ['loss', 'precision', 'recall', 'ndcg', 'coverage', 'diversity'])}

        # Выводим метрики по группам
        print("\nLosses:")
        for name, value in losses.items():
            print(f"  {name}: {value:.4f}")
        
        # print("\nPrecision metrics:")
        # for name, value in precision.items():
        #     print(f"  {name}: {value:.4f}")
        
        # print("\nRecall metrics:")
        # for name, value in recall.items():
        #     print(f"  {name}: {value:.4f}")
        
        # print("\nNDCG metrics:")
        # for name, value in ndcg.items():
        #     print(f"  {name}: {value:.4f}")
        
        # print("\nCoverage metrics:")
        # for name, value in coverage.items():
        #     print(f"  {name}: {value:.4f}")
        
        # print("\nDiversity metrics:")
        # for name, value in diversity.items():
        #     print(f"  {name}: {value:.4f}")
        
        # if other:
        #     print("\nOther metrics:")
        #     for name, value in other.items():
        #         print(f"  {name}: {value:.4f}")

    def training_step(self, batch):
        """Один шаг обучения."""
        items_text_inputs, user_text_inputs, item_ids, user_ids = [
            self.to_device(x) for x in batch
        ]

        # Forward pass
        items_embeddings, user_embeddings = self.model(
            items_text_inputs, user_text_inputs, item_ids, user_ids
        )

        # Нормализация эмбеддингов
        items_embeddings = F.normalize(items_embeddings, p=2, dim=1)
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

        # Потери
        recommendation_loss = self.compute_recommendation_loss(
            user_embeddings, items_embeddings
        )
        contrastive_loss = self.compute_contrastive_loss(
            items_embeddings, user_embeddings
        )

        # Общая потеря
        loss = contrastive_loss + self.config['training']['lambda_rec'] * recommendation_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), contrastive_loss.item(), recommendation_loss.item()

    def compute_recommendation_loss(self, user_embeddings, items_embeddings):
        """Вычисление потери рекомендаций."""
        logits = torch.matmul(user_embeddings, items_embeddings.T)
        labels = torch.arange(len(user_embeddings)).to(self.device)
        return self.recommendation_loss_fn(logits, labels)

    def compute_contrastive_loss(self, items_embeddings, user_embeddings):
        """Вычисление контрастивной потери."""
        batch_size = items_embeddings.size(0)
        positive_labels = torch.ones(batch_size, device=self.device)
        
        contrastive_goods_loss = self.contrastive_loss_fn(
            items_embeddings,
            items_embeddings.roll(shifts=1, dims=0),
            positive_labels
        )
        
        contrastive_users_loss = self.contrastive_loss_fn(
            user_embeddings,
            user_embeddings.roll(shifts=1, dims=0),
            positive_labels
        )
        
        return contrastive_goods_loss + contrastive_users_loss

    def to_device(self, x):
        """Перемещение данных на устройство."""
        if isinstance(x, dict):
            return {k: v.to(self.device) for k, v in x.items()}
        return x.to(self.device) 