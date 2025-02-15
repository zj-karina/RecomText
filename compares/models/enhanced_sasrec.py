import torch
import torch.nn as nn
from recbole.model.sequential_recommender import SASRec
from typing import Dict, Optional

class EnhancedSASRec(SASRec):
    """Расширенная версия SASRec с поддержкой дополнительных признаков"""
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # Получаем конфигурацию признаков
        self.numerical_features = config['data'].get('numerical_features', [])
        self.categorical_features = config['data'].get('token_features', [])
        
        # Размерности для разных типов признаков
        self.hidden_size = config['hidden_size']
        self.num_numerical = len(self.numerical_features)
        
        # Создаем слои для обработки числовых признаков
        if self.num_numerical > 0:
            self.numerical_projection = nn.Sequential(
                nn.Linear(self.num_numerical, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(config['hidden_dropout_prob'])
            )
            
        # Создаем эмбеддинги для категориальных признаков
        if self.categorical_features:
            self.categorical_embeddings = nn.ModuleDict({
                feature: nn.Embedding(dataset.num_features[feature], self.hidden_size)
                for feature in self.categorical_features
            })
            
        # Слой для объединения всех признаков
        self.feature_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def forward(self, item_seq, item_seq_len, **kwargs):
        """
        Аргументы:
            item_seq: базовая последовательность (индексы), ожидаемая форма [batch_size, seq_len]
            item_seq_len: длины последовательностей
            **kwargs: дополнительные признаки (например, числовые или категориальные), передаваемые как именованные аргументы
        """
        # Получаем базовые эмбеддинги последовательности от родительского класса
        seq_output = super().forward(item_seq, item_seq_len)

        # Обработка числовых признаков, если они переданы в kwargs
        if self.num_numerical > 0 and all(feature in kwargs for feature in self.numerical_features):
            numerical_features = torch.stack(
                [kwargs[feature] for feature in self.numerical_features], dim=-1
            )
            print(numerical_features)
            numerical_emb = self.numerical_projection(numerical_features)
            seq_output = self.feature_fusion(torch.cat([seq_output, numerical_emb], dim=-1))
            
        # Обработка категориальных признаков, если они переданы в kwargs
        if self.categorical_features and all(feature in kwargs for feature in self.categorical_features):
            categorical_emb = torch.zeros_like(seq_output)
            for feature in self.categorical_features:
                feature_emb = self.categorical_embeddings[feature](kwargs[feature])
                categorical_emb = categorical_emb + feature_emb
            seq_output = self.feature_fusion(torch.cat([seq_output, categorical_emb], dim=-1))
            
        return seq_output