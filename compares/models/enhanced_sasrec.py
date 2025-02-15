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
        
    def forward(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # Получаем базовый выход от SASRec
        seq_output = super().forward(interaction)
        
        # Добавляем обработку числовых признаков
        if hasattr(self, 'numerical_features'):
            numerical_tensors = []
            for field in self.numerical_features:
                field_list = f'{field}_list'
                if field_list in interaction:
                    numerical_tensors.append(interaction[field_list])
            
            if numerical_tensors:
                numerical_features = torch.cat(numerical_tensors, dim=-1)
                numerical_emb = self.numerical_projection(numerical_features)
                seq_output = self.feature_fusion(
                    torch.cat([seq_output, numerical_emb], dim=-1)
                )
        
        return seq_output