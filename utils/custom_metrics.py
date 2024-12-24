import torch
import numpy as np
from typing import Dict, List, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationMetrics:
    def __init__(self, categories_map: Dict[str, str]):
        """
        Args:
            categories_map: Словарь соответствия video_id -> category
        """
        self.categories_map = categories_map
        
    def _get_category(self, video_id: str) -> str:
        """Получает категорию видео по его ID."""
        return self.categories_map.get(video_id, "unknown")

    def calculate_precision_recall_at_k(self, 
                                      predictions: List[str], 
                                      ground_truth: List[str], 
                                      k: int) -> Tuple[float, float]:
        """
        Вычисляет Precision@K и Recall@K на основе категорий.
        
        Args:
            predictions: Список ID предсказанных видео
            ground_truth: Список ID фактически просмотренных видео
            k: Количество топовых рекомендаций для оценки
        """
        if not ground_truth:
            return 0.0, 0.0
            
        # Получаем категории для рекомендаций и истинных просмотров
        pred_categories = set(self._get_category(vid) for vid in predictions[:k])
        true_categories = set(self._get_category(vid) for vid in ground_truth)
        
        # Находим пересечение категорий
        relevant_categories = pred_categories.intersection(true_categories)
        
        precision = len(relevant_categories) / k if k > 0 else 0
        recall = len(relevant_categories) / len(true_categories) if true_categories else 0
        
        return precision, recall

    def calculate_ndcg_at_k(self, 
                           predictions: List[str], 
                           ground_truth: List[str], 
                           k: int) -> float:
        """
        Вычисляет NDCG@K с учетом категорий.
        """
        if not ground_truth:
            return 0.0
            
        true_categories = set(self._get_category(vid) for vid in ground_truth)
        
        # Формируем релевантности для DCG
        relevance = []
        for i, pred_id in enumerate(predictions[:k]):
            pred_category = self._get_category(pred_id)
            # Если категория совпадает с любой из истинных, считаем элемент релевантным
            rel = 1.0 if pred_category in true_categories else 0.0
            relevance.append(rel)
            
        # Вычисляем DCG
        dcg = 0.0
        for i, rel in enumerate(relevance, 1):
            dcg += rel / np.log2(i + 1)
            
        # Вычисляем IDCG (идеальный DCG)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance, 1):
            idcg += rel / np.log2(i + 1)
            
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg

    def calculate_coverage(self, 
                         all_predictions: List[List[str]], 
                         all_categories: Set[str]) -> float:
        """
        Вычисляет покрытие категорий в рекомендациях.
        
        Args:
            all_predictions: Список списков ID рекомендованных видео для всех пользователей
            all_categories: Множество всех возможных к��тегорий
        """
        recommended_categories = set()
        for predictions in all_predictions:
            for vid in predictions:
                recommended_categories.add(self._get_category(vid))
                
        coverage = len(recommended_categories) / len(all_categories)
        return coverage

    def calculate_mrr(self, 
                     predictions: List[str], 
                     ground_truth: List[str]) -> float:
        """
        Вычисляет MRR на основе первого совпадения категорий.
        """
        if not ground_truth:
            return 0.0
            
        true_categories = set(self._get_category(vid) for vid in ground_truth)
        
        for rank, pred_id in enumerate(predictions, 1):
            if self._get_category(pred_id) in true_categories:
                return 1.0 / rank
                
        return 0.0

    def calculate_intra_list_diversity(self, 
                                     predictions: List[str], 
                                     embeddings_dict: Dict[str, np.ndarray]) -> float:
        """
        Вычисляет разнообразие внутри списка рекомендаций.
        
        Args:
            predictions: Список ID рекомендованных видео
            embeddings_dict: Словарь эмбеддингов для каждого video_id
        """
        if len(predictions) < 2:
            return 0.0
            
        # Получаем эмбеддинги для рекомендаций
        embeddings = np.array([embeddings_dict[vid] for vid in predictions])
        
        # Вычисляем попарные косинусные сходства
        similarities = cosine_similarity(embeddings)
        
        # Считаем среднее сходство (исключая диагональ)
        n = len(predictions)
        total_similarity = (similarities.sum() - n) / (n * (n - 1))
        
        # Возвращаем разнообразие (1 - сходство)
        return 1.0 - total_similarity

def evaluate_recommendations(metrics: RecommendationMetrics,
                           model_predictions: List[List[str]],
                           ground_truth: List[List[str]],
                           embeddings_dict: Dict[str, np.ndarray],
                           all_categories: Set[str],
                           k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Вычисляет все метрики для набора рекомендаций.
    """
    results = {}
    n_users = len(model_predictions)
    
    for k in k_values:
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        
        for preds, truth in zip(model_predictions, ground_truth):
            precision, recall = metrics.calculate_precision_recall_at_k(preds, truth, k)
            ndcg = metrics.calculate_ndcg_at_k(preds, truth, k)
            
            precision_sum += precision
            recall_sum += recall
            ndcg_sum += ndcg
            
        results[f'precision@{k}'] = precision_sum / n_users
        results[f'recall@{k}'] = recall_sum / n_users
        results[f'ndcg@{k}'] = ndcg_sum / n_users
    
    # Вычисляем MRR
    mrr_sum = 0.0
    for preds, truth in zip(model_predictions, ground_truth):
        mrr_sum += metrics.calculate_mrr(preds, truth)
    results['mrr'] = mrr_sum / n_users
    
    # Вычисляем Coverage
    results['coverage'] = metrics.calculate_coverage(model_predictions, all_categories)
    
    # Вычисляем среднее Intra-List Diversity
    diversity_sum = 0.0
    for preds in model_predictions:
        diversity_sum += metrics.calculate_intra_list_diversity(preds, embeddings_dict)
    results['diversity'] = diversity_sum / n_users
    
    return results 