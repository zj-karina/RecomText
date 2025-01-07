import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import get_losses
from utils.metrics import MetricsCalculator

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

        return {
            'loss': total_loss / len(self.train_loader),
            'contrastive_loss': total_contrastive_loss / len(self.train_loader),
            'recommendation_loss': total_recommendation_loss / len(self.train_loader)
        }

    def validate(self):
        """Валидация модели."""
        self.model.eval()
        val_loss = 0
        val_contrastive_loss = 0
        val_recommendation_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                items_text_inputs, user_text_inputs, item_ids, user_ids, categories = [
                    self.to_device(x) for x in batch
                ]

                # Forward pass
                items_embeddings, user_embeddings = self.model(
                    items_text_inputs, user_text_inputs, item_ids, user_ids
                )

                # Нормализация эмбеддингов
                items_embeddings = F.normalize(items_embeddings, p=2, dim=1)
                user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

                batch_size = user_embeddings.size(0)
                
                # Вычисляем схожесть каждого пользователя со всеми items
                predictions = []
                for user_emb in user_embeddings:
                    # Расширяем размерность для пользователя
                    user_emb = user_emb.unsqueeze(0)  # [1, embed_dim]
                    
                    # Вычисляем схожесть с каждым item
                    user_predictions = F.cosine_similarity(
                        user_emb.unsqueeze(0),  # [1, 1, embed_dim]
                        items_embeddings.unsqueeze(0),  # [1, n_items, embed_dim]
                        dim=2
                    )  # [1, n_items]
                    predictions.append(user_predictions.squeeze(0))
                
                # Объединяем предсказания для всех пользователей
                predictions = torch.stack(predictions)  # [batch_size, n_items]
                
                # Создаем ground truth
                ground_truth = torch.zeros(batch_size, items_embeddings.size(0), device=self.device)
                ground_truth[torch.arange(batch_size), torch.arange(batch_size)] = 1

                # Вычисляем потери
                recommendation_loss = self.compute_recommendation_loss(
                    user_embeddings, items_embeddings
                )
                contrastive_loss = self.compute_contrastive_loss(
                    items_embeddings, user_embeddings
                )

                val_contrastive_loss += contrastive_loss.item()
                val_recommendation_loss += recommendation_loss.item()
                val_loss += (contrastive_loss + self.config['training']['lambda_rec'] * recommendation_loss).item()

                # Вычисляем метрики
                metrics = self.metrics_calculator.compute_metrics(
                    predictions=predictions,
                    ground_truth=ground_truth,
                    categories=categories,
                    item_embeddings=items_embeddings,
                    ks=self.ks
                )

        # Усредняем потери
        num_batches = len(self.val_loader)
        metrics.update({
            'val_loss': val_loss / num_batches,
            'val_contrastive_loss': val_contrastive_loss / num_batches,
            'val_recommendation_loss': val_recommendation_loss / num_batches
        })

        return metrics

    def _save_checkpoint(self, epoch, metrics):
        """Сохранение чекпоинта модели."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
        print(f"\nSaved checkpoint for epoch {epoch}")

    def _print_metrics(self, metrics):
        """Вывод метрик в консоль."""
        # Группируем метрики по типу
        losses = {k: v for k, v in metrics.items() if 'loss' in k}
        precision = {k: v for k, v in metrics.items() if 'precision' in k}
        recall = {k: v for k, v in metrics.items() if 'recall' in k}
        ndcg = {k: v for k, v in metrics.items() if 'ndcg' in k}
        coverage = {k: v for k, v in metrics.items() if 'coverage' in k}
        diversity = {k: v for k, v in metrics.items() if 'diversity' in k}
        other = {k: v for k, v in metrics.items() if not any(x in k for x in ['loss', 'precision', 'recall', 'ndcg', 'coverage', 'diversity'])}

        # Выводим метрики по группам
        print("\nLosses:")
        for name, value in losses.items():
            print(f"  {name}: {value:.4f}")
        
        print("\nPrecision metrics:")
        for name, value in precision.items():
            print(f"  {name}: {value:.4f}")
        
        print("\nRecall metrics:")
        for name, value in recall.items():
            print(f"  {name}: {value:.4f}")
        
        print("\nNDCG metrics:")
        for name, value in ndcg.items():
            print(f"  {name}: {value:.4f}")
        
        print("\nCoverage metrics:")
        for name, value in coverage.items():
            print(f"  {name}: {value:.4f}")
        
        print("\nDiversity metrics:")
        for name, value in diversity.items():
            print(f"  {name}: {value:.4f}")
        
        if other:
            print("\nOther metrics:")
            for name, value in other.items():
                print(f"  {name}: {value:.4f}")

    def training_step(self, batch):
        """Один шаг обучения."""
        items_text_inputs, user_text_inputs, item_ids, user_ids, categories = [
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