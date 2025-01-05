import torch
import numpy as np
from typing import Dict, List, Union, Tuple

class MetricsCalculator:
    """Калькулятор метрик для рекомендательной системы."""
    
    @staticmethod
    def precision_at_k(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int) -> float:
        """
        Вычисляет Precision@K.
        
        Args:
            predictions: Тензор предсказанных рейтингов или скоров (batch_size x n_items)
            ground_truth: Тензор истинных значений (batch_size x n_items)
            k: Количество топ-элементов для рассмотрения
        
        Returns:
            float: Значение метрики Precision@K
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        relevant_items_in_k = torch.gather(ground_truth, 1, top_k_indices).sum(dim=1)
        return (relevant_items_in_k / k).mean().item()

    @staticmethod
    def recall_at_k(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int) -> float:
        """
        Вычисляет Recall@K.
        
        Args:
            predictions: Тензор предсказанных рейтингов
            ground_truth: Тензор истинных значений
            k: Количество топ-элементов
        
        Returns:
            float: Значение метрики Recall@K
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        relevant_items_in_k = torch.gather(ground_truth, 1, top_k_indices).sum(dim=1)
        total_relevant = ground_truth.sum(dim=1)
        return (relevant_items_in_k / total_relevant.clamp(min=1)).mean().item()

    @staticmethod
    def ndcg_at_k(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int) -> float:
        """
        Вычисляет Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            predictions: Тензор предсказанных рейтингов
            ground_truth: Тензор истинных значений
            k: Количество топ-элементов
        
        Returns:
            float: Значение метрики NDCG@K
        """
        # Получаем топ-k предсказаний
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        # Получаем релевантность для топ-k элементов
        actual_relevance = torch.gather(ground_truth, 1, top_k_indices)
        
        # Вычисляем идеальный DCG
        ideal_relevance, _ = torch.sort(ground_truth, dim=1, descending=True)
        ideal_relevance = ideal_relevance[:, :k]
        
        # Вычисляем коэффициенты дисконтирования
        position_discount = 1 / torch.log2(torch.arange(k, device=predictions.device) + 2)
        
        # Вычисляем DCG и IDCG
        dcg = (actual_relevance * position_discount).sum(dim=1)
        idcg = (ideal_relevance * position_discount).sum(dim=1)
        
        return (dcg / idcg.clamp(min=1e-8)).mean().item()

    @staticmethod
    def mean_reciprocal_rank(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Вычисляет Mean Reciprocal Rank (MRR).
        
        Args:
            predictions: Тензор предсказанных рейтингов
            ground_truth: Тензор истинных значений
        
        Returns:
            float: Значение метрики MRR
        """
        # Получаем ранги для всех элементов
        _, indices = torch.sort(predictions, dim=1, descending=True)
        ranks = torch.zeros_like(indices)
        ranks.scatter_(1, indices, torch.arange(predictions.size(1), device=predictions.device).expand_as(indices) + 1)
        
        # Находим ранг первого релевантного элемента
        relevant_ranks = ranks[ground_truth.bool()]
        first_relevant_ranks = relevant_ranks.view(-1, relevant_ranks.size(-1))[:, 0]
        
        return (1.0 / first_relevant_ranks).mean().item()

    def compute_metrics(
        self, 
        predictions: torch.Tensor, 
        ground_truth: torch.Tensor, 
        ks: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Вычисляет все метрики для заданных k.
        
        Args:
            predictions: Тензор предсказанных рейтингов
            ground_truth: Тензор истинных значений
            ks: Список значений k для вычисления метрик @K
        
        Returns:
            Dict[str, float]: Словарь с результатами всех метрик
        """
        metrics = {}
        
        for k in ks:
            metrics[f'precision@{k}'] = self.precision_at_k(predictions, ground_truth, k)
            metrics[f'recall@{k}'] = self.recall_at_k(predictions, ground_truth, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(predictions, ground_truth, k)
        
        metrics['mrr'] = self.mean_reciprocal_rank(predictions, ground_truth)
        
        return metrics

def compute_metrics(user_embeddings: torch.Tensor, 
                   item_embeddings: torch.Tensor, 
                   ground_truth: torch.Tensor,
                   ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Вычисляет метрики для эмбеддингов пользователей и товаров.
    
    Args:
        user_embeddings: Нормализованные эмбеддинги пользователей
        item_embeddings: Нормализованные эмбеддинги товаров
        ground_truth: Тензор истинных значений
        ks: Список значений k для вычисления метрик @K
    
    Returns:
        Dict[str, float]: Словарь с результатами метрик
    """
    # Вычисляем предсказания как косинусную схожесть между эмбеддингами
    predictions = torch.matmul(user_embeddings, item_embeddings.T)
    
    # Инициализируем калькулятор метрик
    calculator = MetricsCalculator()
    
    # Вычисляем все метрики
    return calculator.compute_metrics(predictions, ground_truth, ks)