import torch
import numpy as np
from typing import Dict, List, Union, Tuple
from scipy.spatial.distance import pdist, squareform

class MetricsCalculator:
    """Калькулятор метрик для рекомендательной системы с учетом категорий."""
    
    @staticmethod
    def precision_at_k(predictions: torch.Tensor, 
                      ground_truth: torch.Tensor, 
                      categories: torch.Tensor,
                      k: int) -> float:
        """
        Вычисляет Precision@K с учетом категорий.
        
        Args:
            predictions: Тензор предсказанных рейтингов (batch_size x n_items)
            ground_truth: Тензор истинных значений (batch_size x n_items)
            categories: Тензор категорий товаров (n_items)
            k: Количество топ-элементов
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        # Получаем категории для топ-k предсказаний
        pred_categories = categories[top_k_indices]
        
        # Получаем категории истинных товаров
        true_categories = categories[ground_truth.bool()]
        
        # Считаем совпадения по категориям
        matches = (pred_categories.unsqueeze(2) == true_categories.unsqueeze(1)).any(dim=2)
        
        return matches.float().mean(dim=1).mean().item()

    @staticmethod
    def coverage(predictions: torch.Tensor, 
                categories: torch.Tensor,
                k: int) -> float:
        """
        Вычисляет покрытие категорий в топ-k рекомендациях.
        
        Args:
            predictions: Тензор предсказанных рейтингов
            categories: Тензор категорий товаров
            k: Количество топ-элементов
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        recommended_categories = categories[top_k_indices]
        unique_categories = torch.unique(recommended_categories)
        total_categories = torch.unique(categories)
        
        return len(unique_categories) / len(total_categories)

    @staticmethod
    def intra_list_diversity(predictions: torch.Tensor,
                            item_embeddings: torch.Tensor,
                            k: int) -> float:
        """
        Вычисляет внутреннее разнообразие в списках рекомендаций.
        
        Args:
            predictions: Тензор предсказанных рейтингов
            item_embeddings: Эмбеддинги товаров
            k: Количество топ-элементов
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        diversities = []
        for indices in top_k_indices:
            embeddings = item_embeddings[indices]
            similarities = torch.matmul(embeddings, embeddings.T)
            # Убираем диагональные элементы
            mask = torch.ones_like(similarities) - torch.eye(k, device=similarities.device)
            diversity = (similarities * mask).sum() / (k * (k - 1))
            diversities.append(1 - diversity.item())  # Конвертируем сходство в разнообразие
            
        return np.mean(diversities)

    def compute_metrics(
        self, 
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        categories: torch.Tensor,
        item_embeddings: torch.Tensor,
        ks: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Вычисляет все метрики для заданных k.
        
        Args:
            predictions: Тензор предсказанных рейтингов
            ground_truth: Тензор истинных значений
            categories: Тензор категорий товаров
            item_embeddings: Эмбеддинги товаров
            ks: Список значений k для вычисления метрик @K
        """
        metrics = {}
        
        for k in ks:
            metrics[f'precision@{k}'] = self.precision_at_k(
                predictions, ground_truth, categories, k
            )
            metrics[f'recall@{k}'] = self.recall_at_k(
                predictions, ground_truth, categories, k
            )
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(
                predictions, ground_truth, categories, k
            )
            metrics[f'coverage@{k}'] = self.coverage(
                predictions, categories, k
            )
            metrics[f'diversity@{k}'] = self.intra_list_diversity(
                predictions, item_embeddings, k
            )
        
        metrics['mrr'] = self.mean_reciprocal_rank(
            predictions, ground_truth, categories
        )
        
        return metrics 