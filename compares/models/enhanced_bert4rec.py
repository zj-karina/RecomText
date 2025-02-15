import torch
import torch.nn as nn
from recbole.model.sequential_recommender import BERT4Rec
from typing import Dict, Optional

class EnhancedBERT4Rec(BERT4Rec):
    """Расширенная версия BERT4Rec с поддержкой дополнительных признаков"""
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # Получаем конфигурацию признаков
        self.numerical_features = config['data'].get('numerical_features', [])
        self.categorical_features = config['data'].get('token_features', [])
        
        # Размерности для разных типов признаков
        self.hidden_size = config['hidden_size']
        self.num_numerical = len(self.numerical_features)
        
        # Создаем слои для обработки признаков
        if self.num_numerical > 0:
            self.numerical_projection = nn.Sequential(
                nn.Linear(self.num_numerical, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(config['hidden_dropout_prob'])
            )
            
        if self.categorical_features:
            self.categorical_embeddings = nn.ModuleDict({
                feature: nn.Embedding(dataset.num_features[feature], self.hidden_size)
                for feature in self.categorical_features
            })
            
        # Слой для объединения всех признаков
        self.feature_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def forward(self, interaction):
        """
        Args:
            interaction: Словарь с полями взаимодействий
        """
        # Получаем последовательность элементов
        item_seq = interaction[self.ITEM_SEQ]  # Используем константу ITEM_SEQ вместо прямого обращения
        
        # Получаем базовые эмбеддинги последовательности от родительского класса
        seq_output = super().forward(interaction)
        
        # Обрабатываем числовые признаки, используя маппинг полей
        if self.num_numerical > 0:
            numerical_values = []
            for feature in self.numerical_features:
                # Используем маппинг для преобразования имен полей
                mapped_feature = self._get_mapped_field(feature)
                if mapped_feature in interaction:
                    numerical_values.append(interaction[mapped_feature])
            
            if numerical_values:  # Проверяем, что есть хотя бы один признак
                numerical_features = torch.stack(numerical_values, dim=-1)
                numerical_emb = self.numerical_projection(numerical_features)
                seq_output = self.feature_fusion(
                    torch.cat([seq_output, numerical_emb], dim=-1)
                )
        
        return seq_output

    def _get_mapped_field(self, feature):
        """Преобразует имя признака в соответствующее поле RecBole"""
        field_mapping = {
            'total_watchtime': 'rating',
            'timestamp': 'timestamp',
            'rutube_video_id': 'item_id',
            'viewer_uid': 'user_id',
            'plays': 'rating'
            # Добавьте другие маппинги при необходимости
        }
        return field_mapping.get(feature, feature) 