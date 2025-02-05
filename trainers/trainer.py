import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import get_losses
from utils.metrics import MetricsCalculator
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
        
        self.best_metric = float('-inf')
        self.best_epoch = 0
        self.patience = config.get('training', {}).get('patience', 5)
        self.no_improvement = 0

    def validate(self):
        """Валидация модели с использованием актуального FAISS индекса"""
        self.model.eval()
        total_loss = 0
        total_contrastive_loss = 0
        total_recommendation_loss = 0
        
        # Загрузка необходимых данных
        textual_history = pd.read_parquet('./data/textual_history.parquet')
        df_videos = pd.read_parquet("./data/video_info.parquet")
        df_videos_map = df_videos.set_index('clean_video_id').to_dict(orient='index')

        # Проверка и обновление индекса
        index_path = self.config['inference']['index_path']
        ids_path = self.config['inference']['ids_path']
        embeddings_path = self.config['inference']['embeddings_path']
        
        if not all(os.path.exists(p) for p in [index_path, ids_path, embeddings_path]):
            print("\nIndex files not found, creating new index...")
            try:
                from indexer import main as create_index
                create_index(config=self.config)
            except Exception as e:
                print(f"Error creating index: {str(e)}")
                return None

        # Загрузка индекса и данных
        index = faiss.read_index(index_path)
        video_ids = np.load(ids_path)
        item_embeddings_array = np.load(embeddings_path)
        
        # Демографические данные
        try:
            user_demographics_df = pd.read_parquet('./data/user_demographics.parquet')
            demographic_features = ['age_group', 'gender', 'location']
            demographic_centroids = self._compute_demographic_centroids(
                user_demographics_df, textual_history, item_embeddings_array
            )
        except FileNotFoundError:
            print("Demographics data not found")
            demographic_centroids = None

        # Инициализация метрик
        metrics_calculator = MetricsCalculator(sim_threshold=0.7)
        metrics_accum = {metric: 0.0 for metric in ["semantic_precision@k", "cross_category_relevance", "contextual_ndcg"]}
        num_users = 0
        top_k = self.config['inference'].get('top_k', 10)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                # Обработка батча
                items_text, user_text, item_ids, user_ids = self.to_device(batch)
                
                # Forward pass
                items_emb, user_emb = self.model(items_text, user_text, item_ids, user_ids)
                items_emb = F.normalize(items_emb, p=2, dim=1)
                user_emb = F.normalize(user_emb, p=2, dim=1)

                # Расчет потерь
                rec_loss = self.compute_recommendation_loss(user_emb, items_emb)
                con_loss = self.compute_contrastive_loss(items_emb, user_emb)
                total_loss += (con_loss + self.config['training']['lambda_rec'] * rec_loss).item()
                
                # Поиск рекомендаций и расчет метрик
                for i in range(user_emb.size(0)):
                    user_metrics = self._process_user(
                        user_emb[i], 
                        items_emb[i], 
                        item_ids[i], 
                        user_ids[i], 
                        index, 
                        video_ids, 
                        df_videos_map,
                        item_embeddings_array,
                        metrics_calculator,
                        top_k,
                        user_demographics_df,
                        demographic_centroids
                    )
                    self._update_metrics(metrics_accum, user_metrics)
                    num_users += 1

        return self._compile_metrics(total_loss, total_contrastive_loss, total_recommendation_loss, metrics_accum, num_users)

    def _compute_demographic_centroids(self, user_demographics_df, textual_history, item_embeddings):
        """Вычисление центроидов демографических групп"""
        centroids = {}
        for feature in ['age_group', 'gender', 'location']:
            centroids[feature] = {}
            for group in user_demographics_df[feature].unique():
                group_users = user_demographics_df[user_demographics_df[feature] == group].index
                embeddings = []
                for user_id in group_users:
                    user_items = textual_history[textual_history['user_id'] == user_id]
                    if not user_items.empty:
                        item_indices = user_items['item_id'].values
                        embeddings.append(torch.mean(torch.tensor(
                            item_embeddings[item_indices], 
                            device=self.device
                        ), dim=0))
                if embeddings:
                    centroids[feature][group] = F.normalize(torch.mean(torch.stack(embeddings), dim=0), p=2, dim=0)
        return centroids

    def _process_user(self, user_emb, target_emb, item_id, user_id, index, video_ids, df_videos_map, item_embeddings, metrics_calculator, top_k, user_demographics, centroids):
        """Обработка одного пользователя для расчета метрик"""
        # Поиск рекомендаций
        user_emb_np = user_emb.cpu().numpy().astype('float32')
        distances, indices = index.search(user_emb_np.reshape(1, -1), top_k)
        
        # Получение рекомендаций
        rec_embeddings = torch.tensor(item_embeddings[indices[0]], device=self.device)
        rec_embeddings = F.normalize(rec_embeddings, p=2, dim=1)
        
        # Метаданные рекомендаций
        rec_categories = []
        for idx in indices[0]:
            video_id = str(video_ids[idx])
            rec_categories.append(df_videos_map.get(video_id, {}).get('category', 'Unknown'))

        # Демографические данные
        user_demo = {}
        if user_demographics is not None and user_id.item() in user_demographics.index:
            for feature in ['age_group', 'gender', 'location']:
                user_demo[feature] = user_demographics.loc[user_id.item(), feature]

        # Целевой товар
        target_id = str(item_id.item())
        target_category = df_videos_map.get(target_id, {}).get('category', 'Unknown')

        return metrics_calculator.compute_metrics(
            target_emb,
            rec_embeddings,
            target_category,
            rec_categories,
            top_k,
            user_demographics=user_demo,
            demographic_centroids=centroids
        )

    def _update_metrics(self, metrics_accum, user_metrics):
        """Обновление аккумуляторов метрик"""
        for metric in metrics_accum:
            metrics_accum[metric] += user_metrics.get(metric, 0)

    def _compile_metrics(self, total_loss, contrastive_loss, recommendation_loss, metrics_accum, num_users):
        """Компиляция финальных метрик"""
        metrics_dict = {
            'val_loss': total_loss / len(self.val_loader),
            'val_contrastive_loss': contrastive_loss / len(self.val_loader),
            'val_recommendation_loss': recommendation_loss / len(self.val_loader)
        }
        
        if num_users > 0:
            for metric in metrics_accum:
                metrics_dict[metric] = metrics_accum[metric] / num_users
                
        print("\nValidation Metrics:")
        for name, value in metrics_dict.items():
            print(f"{name}: {value:.4f}")
            
        return metrics_dict

    def _save_checkpoint(self, epoch, metrics=None):
        """Улучшенное сохранение чекпоинта"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Сохранение модели
        model_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}')
        self.model.save_pretrained(model_path)
        
        # Обновление конфига для индексатора
        index_config = self.config.copy()
        index_config['inference']['model_path'] = model_path
        
        try:
            from indexer import main as update_index
            update_index(config=index_config)
            print("FAISS index updated successfully")
        except Exception as e:
            print(f"Error updating index: {str(e)}")

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