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
        print(interaction)
        print(self.ITEM_SEQ)
        item_seq = interaction[self.ITEM_SEQ]
        item_seq = item_seq[:, 0, :]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # Получаем базовые эмбеддинги последовательности
        seq_output = super().forward(item_seq)
        
        # Обрабатываем числовые признаки
        if self.num_numerical > 0:
            numerical_features = torch.stack([
                interaction[feature] for feature in self.numerical_features
            ], dim=-1)
            numerical_emb = self.numerical_projection(numerical_features)
            seq_output = self.feature_fusion(
                torch.cat([seq_output, numerical_emb], dim=-1)
            )
            
        # Обрабатываем категориальные признаки
        if self.categorical_features:
            categorical_emb = torch.zeros_like(seq_output)
            for feature in self.categorical_features:
                feature_emb = self.categorical_embeddings[feature](interaction[feature])
                categorical_emb = categorical_emb + feature_emb
            seq_output = self.feature_fusion(
                torch.cat([seq_output, categorical_emb], dim=-1)
            )
            
        return seq_output 