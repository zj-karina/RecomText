import torch
import numpy as np
from typing import Dict, List, Union, Tuple

class MetricsCalculator:
    """Калькулятор метрик для рекомендательной системы с учетом категорий."""
    
    @staticmethod
    def precision_at_k(predictions: torch.Tensor, 
                      ground_truth: torch.Tensor, 
                      categories: torch.Tensor,
                      k: int) -> float:
        """
        Вычисляет Precision@K с учетом категорий.
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        pred_categories = categories[top_k_indices]
        true_indices = torch.arange(batch_size, device=predictions.device)
        true_categories = categories[true_indices].unsqueeze(1)
        
        matches = (pred_categories == true_categories)
        precision = matches.float().mean(dim=1)
        
        return precision.mean().item()

    @staticmethod
    def recall_at_k(predictions: torch.Tensor, 
                    ground_truth: torch.Tensor, 
                    categories: torch.Tensor,
                    k: int) -> float:
        """
        Вычисляет Recall@K с учетом категорий.
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        pred_categories = categories[top_k_indices]
        true_indices = torch.arange(batch_size, device=predictions.device)
        true_categories = categories[true_indices].unsqueeze(1)
        
        matches = (pred_categories == true_categories)
        recall = (matches.sum(dim=1) > 0).float()
        
        return recall.mean().item()

    @staticmethod
    def ndcg_at_k(predictions: torch.Tensor, 
                  ground_truth: torch.Tensor, 
                  categories: torch.Tensor,
                  k: int) -> float:
        """
        Вычисляет NDCG@K с учетом категорий.
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        pred_categories = categories[top_k_indices]
        true_indices = torch.arange(batch_size, device=predictions.device)
        true_categories = categories[true_indices].unsqueeze(1)
        
        matches = (pred_categories == true_categories).float()
        
        # Вычисляем DCG
        discounts = torch.log2(torch.arange(k, device=predictions.device) + 2.0)
        dcg = (matches / discounts.unsqueeze(0)).sum(dim=1)
        
        # Вычисляем IDCG (идеальный DCG)
        idcg = (1 / discounts[0]).unsqueeze(0).expand(batch_size)
        
        ndcg = dcg / idcg
        
        return ndcg.mean().item()

    @staticmethod
    def coverage(predictions: torch.Tensor, 
                categories: torch.Tensor,
                k: int) -> float:
        """
        Вычисляет покрытие категорий в топ-k рекомендациях.
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        recommended_categories = categories[top_k_indices]
        unique_recommended = torch.unique(recommended_categories)
        unique_total = torch.unique(categories)
        
        return len(unique_recommended) / len(unique_total)

    @staticmethod
    def diversity(predictions: torch.Tensor,
                 item_embeddings: torch.Tensor,
                 k: int) -> float:
        """
        Вычисляет разнообразие рекомендаций на основе эмбеддингов.
        """
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        diversities = []
        for indices in top_k_indices:
            embeddings = item_embeddings[indices]
            similarities = torch.matmul(embeddings, embeddings.T)
            mask = torch.ones_like(similarities) - torch.eye(k, device=similarities.device)
            diversity = 1 - (similarities * mask).sum() / (k * (k - 1))
            diversities.append(diversity.item())
            
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
            metrics[f'diversity@{k}'] = self.diversity(
                predictions, item_embeddings, k
            )
        
        return metrics 