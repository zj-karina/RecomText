import torch
import torch.nn as nn
from recbole.model.sequential_recommender import SASRec
from typing import Dict, Optional

class EnhancedSASRec(SASRec):
    """Расширенная версия SASRec с поддержкой дополнительных признаков"""
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # Получаем размерности для текстовых эмбеддингов
        self.text_fields = config['data']['TEXT_FIELDS']
        self.text_embedding_dim = {}
        for field_info in config['data']['field_preparation']['inter']:
            if field_info['field'].endswith('_emb'):
                self.text_embedding_dim[field_info['field']] = field_info['dim']
        
        # Получаем информацию о пользовательских признаках
        self.user_features = config['data']['USER_FEATURES']
        self.user_feature_dims = {
            field: dataset.field2token_num[field]
            for field in self.user_features
        }
        
        # Создаем эмбеддинги для пользовательских признаков
        self.user_feature_embeddings = nn.ModuleDict({
            field: nn.Embedding(dim, self.hidden_size)
            for field, dim in self.user_feature_dims.items()
        })
        
        # Проекции для текстовых эмбеддингов
        self.text_projections = nn.ModuleDict({
            f"{field}_emb": nn.Linear(dim, self.hidden_size)
            for field in self.text_fields
            for emb_field, dim in self.text_embedding_dim.items()
            if emb_field == f"{field}_emb"
        })
        
        # Слой для объединения всех признаков
        total_dims = self.hidden_size * (1 + len(self.text_fields) + len(self.user_features))
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dims, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, interaction):
        # Получаем базовый выход от SASRec
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = super().forward(interaction)
        
        # Обрабатываем текстовые эмбеддинги
        text_embeddings = []
        for field in self.text_fields:
            emb_field = f"{field}_emb"
            if emb_field in self.text_projections:
                # Собираем все компоненты эмбеддинга
                emb_dim = self.text_embedding_dim[emb_field]
                field_vectors = []
                for i in range(emb_dim):
                    field_name = f"{emb_field}_{i}_list"
                    if field_name in interaction:
                        field_vectors.append(interaction[field_name])
                if field_vectors:
                    # Объединяем компоненты и проецируем
                    field_embedding = torch.stack(field_vectors, dim=-1)
                    projected_embedding = self.text_projections[emb_field](field_embedding)
                    text_embeddings.append(projected_embedding)
        
        # Обрабатываем пользовательские признаки
        user_embeddings = []
        for field in self.user_features:
            if field in interaction:
                user_feature = interaction[field]
                user_embedding = self.user_feature_embeddings[field](user_feature)
                # Расширяем до размера последовательности
                user_embedding = user_embedding.unsqueeze(1).expand(-1, seq_output.size(1), -1)
                user_embeddings.append(user_embedding)
        
        # Объединяем все признаки
        all_embeddings = [seq_output]
        if text_embeddings:
            all_embeddings.extend(text_embeddings)
        if user_embeddings:
            all_embeddings.extend(user_embeddings)
        
        combined_embedding = torch.cat(all_embeddings, dim=-1)
        final_output = self.feature_fusion(combined_embedding)
        
        return final_output