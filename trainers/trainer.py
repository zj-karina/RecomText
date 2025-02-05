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
        """Валидация модели с использованием актуального FAISS индекса."""
        self.model.eval()
        total_loss = 0
        total_contrastive_loss = 0
        total_recommendation_loss = 0
        
        # Загружаем необходимые данные
        textual_history = pd.read_parquet('./data/textual_history.parquet')
        
        # Проверяем наличие FAISS индекса
        index_path = self.config['inference'].get('index_path', 'video_index.faiss')
        ids_path = self.config['inference'].get('ids_path', 'video_ids.npy')
        
        if not os.path.exists(index_path) or not os.path.exists(ids_path):
            print("\nFAISS index not found, creating new index...")
            try:
                from indexer import main as create_index
                create_index()
                print("FAISS index created successfully")
            except Exception as e:
                print(f"Warning: Failed to create FAISS index: {str(e)}")
                detailed_output = False
            
        # Загружаем индекс и видео информацию
        if os.path.exists(index_path) and os.path.exists(ids_path):
            index = faiss.read_index(index_path)
            video_ids = np.load(ids_path).tolist()
            df_videos = pd.read_parquet("./data/video_info.parquet")
            df_videos_map = df_videos.set_index('clean_video_id').to_dict(orient='index')
            detailed_output = True
        else:
            detailed_output = False

        metrics_dict = {}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                items_text_inputs, user_text_inputs, item_ids, user_ids = [
                    self.to_device(x) for x in batch
                ]
                
                # Выводим историю просмотров для первого батча
                if batch_idx == 0 and detailed_output:
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

                # Потери
                recommendation_loss = self.compute_recommendation_loss(
                    user_embeddings, items_embeddings
                )
                contrastive_loss = self.compute_contrastive_loss(
                    items_embeddings, user_embeddings
                )
                loss = contrastive_loss + self.config['training']['lambda_rec'] * recommendation_loss

                # Аккумулируем потери
                total_loss += loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_recommendation_loss += recommendation_loss.item()
                
                # Для первого батча показываем рекомендации
                if batch_idx == 0 and detailed_output:
                    k = self.config['inference'].get('top_k', 10)
                    user_emb = user_embeddings[0].unsqueeze(0)
                    user_emb_np = user_emb.cpu().numpy().astype('float32')
                    
                    # Поиск в FAISS
                    distances, faiss_indices = index.search(user_emb_np, k)
                    
                    # Получаем ID видео и скоры
                    retrieved_ids = [video_ids[idx] for idx in faiss_indices[0]]
                    retrieved_scores = distances[0]

                    print(f"\nTop-{k} recommendations:")
                    for rank, (vid, score) in enumerate(zip(retrieved_ids, retrieved_scores), start=1):
                        vid_value = vid[0]
                        video_data = df_videos_map.get(str(vid_value), {})
                        title = video_data.get('title', 'Unknown title')
                        cat = video_data.get('category', 'Unknown category')
                        
                        print(f"  {rank}. Video ID={vid_value}, Score={score:.4f}, Category={cat}")
                        print(f"     Title: {title}")

                num_batches += 1

        # Вычисляем средние значения
        metrics_dict['val_loss'] = total_loss / num_batches
        metrics_dict['val_contrastive_loss'] = total_contrastive_loss / num_batches
        metrics_dict['val_recommendation_loss'] = total_recommendation_loss / num_batches

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