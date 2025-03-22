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
        
        # Создаем слои для обработки числовых признаков
        if self.num_numerical > 0:
            self.numerical_projection = nn.Sequential(
                nn.Linear(self.num_numerical, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(config['hidden_dropout_prob'])
            )
            
        # Создаем эмбеддинги для категориальных признаков (если они используются)
        if self.categorical_features:
            self.categorical_embeddings = nn.ModuleDict({
                feature: nn.Embedding(dataset.num_features[feature], self.hidden_size)
                for feature in self.categorical_features
            })
            
        # Слой для объединения всех признаков
        self.feature_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def forward(self, item_seq, **kwargs):
        """
        Args:
            item_seq: базовая последовательность (тензор)
            **kwargs: дополнительные признаки (например, числовые или категориальные), передаваемые как именованные аргументы
        """
        # Получаем базовые эмбеддинги последовательности от родительского класса
        seq_output = super().forward(item_seq)
        
        # Обрабатываем числовые признаки, если они переданы через kwargs
        if self.num_numerical > 0 and all(feature in kwargs for feature in self.numerical_features):
            numerical_features = torch.stack(
                [kwargs[feature] for feature in self.numerical_features], dim=-1
            )
            numerical_emb = self.numerical_projection(numerical_features)
            seq_output = self.feature_fusion(torch.cat([seq_output, numerical_emb], dim=-1))
            
        # Обработка категориальных признаков, если они переданы через kwargs
        if self.categorical_features and all(feature in kwargs for feature in self.categorical_features):
            categorical_emb = torch.zeros_like(seq_output)
            for feature in self.categorical_features:
                feature_emb = self.categorical_embeddings[feature](kwargs[feature])
                categorical_emb = categorical_emb + feature_emb
            seq_output = self.feature_fusion(torch.cat([seq_output, categorical_emb], dim=-1))
        
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

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]  # [batch, seq_len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        
        seq_output = self.forward(item_seq)  # [batch, seq_len, hidden_size]
        
        final_hidden = self.gather_indexes(seq_output, item_seq_len)  # [batch, hidden_size]
        
        target_item = interaction[self.ITEM_ID]
        # Применяем squeeze(1) только если tensor имеет более одной размерности
        if target_item.dim() > 1:
            target_item = target_item.squeeze(1)  # [batch]
        
        logits = torch.matmul(final_hidden, self.item_embedding.weight[:self.n_items].transpose(0, 1))  # [batch, n_items]
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, target_item)
        return loss