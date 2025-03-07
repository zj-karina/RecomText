import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Tuple, Set

class MetricsCalculator:
    """Калькулятор специализированных метрик для рекомендательной системы."""
    
    def __init__(self, sim_threshold_precision: float = 0.79, sim_threshold_ndcg: float = 0.8):
        """
        Args:
            sim_threshold: порог для "успешной" семантической близости
        """
        self.sim_threshold_precision = sim_threshold_precision
        self.sim_threshold_ndcg = sim_threshold_ndcg

    def semantic_precision_at_k(self, 
                            target_embedding: torch.Tensor,
                            recommended_embeddings: torch.Tensor,
                            k: int) -> float:
        """
        Вычисляет Semantic Precision@K.
        """
        # Вычисление косинусного сходства
        similarities = F.cosine_similarity(
            target_embedding.unsqueeze(0),
            recommended_embeddings,
            dim=1
        )

        # Считаем, сколько попало выше порога
        successes = (similarities >= self.sim_threshold_precision).sum().item()
        precision_at_k = successes / min(k, recommended_embeddings.shape[0])

        return precision_at_k


    def cross_category_relevance(self,
                               sp_at_k: float,
                               target_category: str,
                               recommended_categories: List[str]) -> float:
        """
        Вычисляет Cross-Category Relevance на основе SP@K и категориальной избыточности.
        """
        if not recommended_categories:
            return sp_at_k
        same_category_count = sum(1 for cat in recommended_categories if cat == target_category)
        redundancy = same_category_count / len(recommended_categories)
        return 0.7 * sp_at_k + 0.3 * (1 - redundancy)

    def contextual_ndcg(self,
                       target_embedding: torch.Tensor,
                       recommended_embeddings: torch.Tensor,
                       target_category: str,
                       recommended_categories: List[str]) -> float:
        """
        Вычисляет Contextual NDCG с учетом семантической близости и категорий.
        """
        similarities = F.cosine_similarity(
            target_embedding.unsqueeze(0),
            recommended_embeddings,
            dim=1
        )
        
        relevances = []
        for sim, rec_category in zip(similarities, recommended_categories):
            sim_val = sim.item()
            if rec_category == target_category and sim_val >= self.sim_threshold_ndcg:
                rel = 3
            elif rec_category != target_category and sim_val >= self.sim_threshold_ndcg:
                rel = 2
            elif rec_category == target_category and sim_val < self.sim_threshold_ndcg:
                rel = 1
            else:
                rel = 0
            relevances.append(rel)

        dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances, 1))
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_relevances, 1))
        
        return dcg / idcg if idcg > 0 else 0

    def demographic_alignment_score(self,
                                  user_demographics: Dict[str, str],
                                  recommended_embeddings: torch.Tensor,
                                  demographic_centroids: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Вычисляет Demographic Alignment Score для рекомендаций.
        
        Args:
            user_demographics: словарь с демографическими характеристиками пользователя
            recommended_embeddings: эмбеддинги рекомендованных товаров
            demographic_centroids: словарь центроидов для каждой демографической группы
        
        Returns:
            Dict с DAS scores для каждой демографической характеристики
        """
        das_scores = {}

        if user_demographics is None or demographic_centroids is None:
            return das_scores

        for demo_feature, user_group in user_demographics.items():
            if demo_feature in demographic_centroids and user_group in demographic_centroids[demo_feature]:
                group_centroid = demographic_centroids[demo_feature][user_group]
                
                # Вычисляем среднее косинусное сходство между рекомендациями и центроидом группы
                similarities = F.cosine_similarity(
                    recommended_embeddings,
                    group_centroid.unsqueeze(0),
                    dim=1
                )
                das_scores[f"das_{demo_feature}"] = similarities.mean().item()
        return das_scores

    def precision_at_k(self, 
                     recommended_ids: List[str], 
                     relevant_ids: Set[str], 
                     k: int) -> float:
        """
        Args:
            recommended_ids: список ID рекомендованных элементов
            relevant_ids: множество ID релевантных элементов
            k: количество рекомендаций для оценки
        """
        if not recommended_ids or k <= 0:
            return 0.0
            
        # Обрезаем рекомендации до k элементов
        recommended_at_k = recommended_ids[:k]
        
        # Считаем количество релевантных элементов в топ-k
        hits = sum(1 for item_id in recommended_at_k if item_id in relevant_ids)
        
        return hits / min(k, len(recommended_at_k))

    def recall_at_k(self, 
                  recommended_ids: List[str], 
                  relevant_ids: Set[str], 
                  k: int) -> float:
        """
        Args:
            recommended_ids: список ID рекомендованных элементов
            relevant_ids: множество ID релевантных элементов
            k: количество рекомендаций для оценки
        """
        if not recommended_ids or not relevant_ids or k <= 0:
            return 0.0
            
        recommended_at_k = recommended_ids[:k]
        hits = sum(1 for item_id in recommended_at_k if item_id in relevant_ids)
        
        return hits / len(relevant_ids) if len(relevant_ids) > 0 else 0.0

    def ndcg_at_k(self, 
                recommended_ids: List[str], 
                relevant_ids: Set[str], 
                relevance_scores: Dict[str, float] = None,
                k: int = 10) -> float:
        """
        Args:
            recommended_ids: список ID рекомендованных элементов
            relevant_ids: множество ID релевантных элементов
            relevance_scores: словарь с оценками релевантности для каждого ID (опционально)
            k: количество рекомендаций для оценки
        """
        if not recommended_ids or k <= 0:
            return 0.0
            
        recommended_at_k = recommended_ids[:k]
        
        if relevance_scores:
            relevances = [
                relevance_scores.get(item_id, 0.0) if item_id in relevant_ids else 0.0
                for item_id in recommended_at_k
            ]
        else:
            relevances = [1.0 if item_id in relevant_ids else 0.0 for item_id in recommended_at_k]
            
        dcg = sum(
            rel / math.log2(i + 2) 
            for i, rel in enumerate(relevances)
        )
            
        if relevance_scores:
            ideal_relevances = sorted(
                [relevance_scores.get(item_id, 0.0) for item_id in relevant_ids if item_id in relevance_scores],
                reverse=True
            )
        else:
            ideal_relevances = [1.0] * len(relevant_ids)
            
        ideal_relevances = ideal_relevances[:k]
            
        idcg = sum(
            rel / math.log2(i + 2) 
            for i, rel in enumerate(ideal_relevances)
        )
            
        return dcg / idcg if idcg > 0 else 0.0

    def mrr_at_k(self, 
               recommended_ids: List[str], 
               relevant_ids: Set[str], 
               k: int) -> float:
        """
        Args:
            recommended_ids: список ID рекомендованных элементов
            relevant_ids: множество ID релевантных элементов
            k: количество рекомендаций для оценки
        """
        if not recommended_ids or not relevant_ids or k <= 0:
            return 0.0
            
        recommended_at_k = recommended_ids[:k]
        
        for i, item_id in enumerate(recommended_at_k):
            if item_id in relevant_ids:
                return 1.0 / (i + 1)  # RR = 1 / rank
                
        return 0.0

    def compute_classic_metrics(self,
                              recommended_ids: List[str],
                              relevant_ids: Set[str],
                              ks: List[int] = [5, 10],
                              relevance_scores: Dict[str, float] = None) -> Dict[str, float]:
        """
        Вычисляет все классические метрики рекомендательных систем для заданных K.
        
        Args:
            recommended_ids: список ID рекомендованных элементов
            relevant_ids: множество ID релевантных элементов
            ks: список значений K для вычисления метрик
            relevance_scores: словарь с оценками релевантности для каждого ID (опционально)
            
        Returns:
            Dict с вычисленными метриками
        """
        metrics = {}
        
        for k in ks:
            metrics[f"precision@k"] = self.precision_at_k(recommended_ids, relevant_ids, k)
            metrics[f"recall@k"] = self.recall_at_k(recommended_ids, relevant_ids, k)
            metrics[f"ndcg@k"] = self.ndcg_at_k(recommended_ids, relevant_ids, relevance_scores, k)
            metrics[f"mrr@k"] = self.mrr_at_k(recommended_ids, relevant_ids, k)
            
        return metrics

    def compute_metrics(self,
                       target_embedding: torch.Tensor,
                       recommended_embeddings: torch.Tensor,
                       target_category: str,
                       recommended_categories: List[str],
                       k: int,
                       user_demographics: Dict[str, str] = None,
                       demographic_centroids: Dict[str, Dict[str, torch.Tensor]] = None,
                       recommended_ids: List[str] = None,
                       relevant_ids: Set[str] = None,
                       relevance_scores: Dict[str, float] = None) -> Dict[str, float]:
        """
        Вычисляет все метрики для одного пользователя.
        
        Args:
            target_embedding: целевой эмбеддинг
            recommended_embeddings: эмбеддинги рекомендованных элементов
            target_category: целевая категория
            recommended_categories: категории рекомендованных элементов
            k: количество рекомендаций для оценки
            user_demographics: словарь с демографическими характеристиками пользователя
            demographic_centroids: словарь центроидов для каждой демографической группы
            recommended_ids: список ID рекомендованных элементов
            relevant_ids: множество ID релевантных элементов
            relevance_scores: словарь с оценками релевантности для каждого ID
            
        Returns:
            Dict с вычисленными метриками
        """
        metrics = {}
        metrics = {
            "semantic_precision@k": self.semantic_precision_at_k(
                target_embedding,
                recommended_embeddings,
                k
            ),
            "cross_category_relevance": self.cross_category_relevance(
                self.semantic_precision_at_k(target_embedding, recommended_embeddings, k),
                target_category,
                recommended_categories
            ),
            "contextual_ndcg": self.contextual_ndcg(
                target_embedding,
                recommended_embeddings,
                target_category,
                recommended_categories
            )
        }
        
        # Добавляем DAS если доступны демографические данные
        if user_demographics and demographic_centroids:
            das_scores = self.demographic_alignment_score(
                user_demographics,
                recommended_embeddings,
                demographic_centroids
            )
            metrics.update(das_scores)
            
        classic_metrics = self.compute_classic_metrics(
            recommended_ids,
            relevant_ids,
            ks=[k],
            relevance_scores=relevance_scores
        )
        metrics.update(classic_metrics)
            
        return metrics