import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Tuple

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

    def compute_metrics(self,
                       target_embedding: torch.Tensor,
                       recommended_embeddings: torch.Tensor,
                       target_category: str,
                       recommended_categories: List[str],
                       k: int,
                       user_demographics: Dict[str, str] = None,
                       demographic_centroids: Dict[str, Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Вычисляет все метрики для одного пользователя.
        """
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
            
        return metrics

    def compute_ranking_metrics(self, target_ids, rec_items, top_k):
        """
        Вычисляет метрики ранжирования: Hit Rate, MRR, NDCG, Precision и Recall.
        
        :param target_ids: ID целевого элемента или список ID (Ground Truth)
        :param rec_items: Список рекомендованных элементов (list или np.array)
        :param top_k: Количество топ-K рекомендаций
        :return: Словарь с метриками
        """

        # Обеспечиваем, что target_ids - это множество (на случай списка)
        if isinstance(target_ids, int):  
            target_ids = {target_ids}  # Один товар в множество
        else:
            target_ids = set(target_ids)  # Несколько товаров

        rec_items = rec_items[:top_k]  # Берем только топ-K рекомендаций

        # Hit Rate (HR@K) – хотя бы один таргетный товар в рекомендациях
        hit = int(len(target_ids.intersection(rec_items)) > 0)

        # MRR@K – Reciprocal Rank первого найденного релевантного товара
        mrr = 0.0
        for i, item in enumerate(rec_items):
            if item in target_ids:
                mrr = 1.0 / (i + 1)
                break  # Берем первый релевантный товар

        # DCG@K – Discounted Cumulative Gain
        dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(rec_items) if item in target_ids])

        # IDCG@K – Ideal DCG (на случай, если все target_items были наверху)
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(target_ids), top_k))])
        ndcg = dcg / idcg if idcg > 0 else 0.0  # NDCG@K

        # Precision@K – Доля релевантных товаров среди топ-K
        precision = len(target_ids.intersection(rec_items)) / top_k

        # Recall@K – Доля покрытых таргетных товаров
        recall = len(target_ids.intersection(rec_items)) / len(target_ids)

        return {
            "HitRate@K": hit,
            "MRR@K": mrr,
            "NDCG@K": ndcg,
            "Precision@K": precision,
            "Recall@K": recall
        }

