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

    def forward(self, item_seq, interaction=None):
        """
        Args:
            item_seq: базовая последовательность (тензор)
            interaction: полный объект interaction с дополнительными признаками
        """
        # Получаем базовые эмбеддинги последовательности
        seq_output = super().forward(item_seq)  # [batch_size, seq_len, hidden_size]
        
        # Если нет дополнительных признаков, возвращаем базовый выход
        if interaction is None:
            return seq_output
            
        print(f"Interaction keys: {interaction.keys()}")
        print(f"Sequence output shape: {seq_output.shape}")
        
        all_features = [seq_output]
        
        # Обрабатываем эмбеддинги заголовков
        if 'detailed_view_embedding' in interaction:
            title_emb = interaction['detailed_view_embedding']  # [batch_size, embedding_size]
            print(f"Title embedding shape: {title_emb.shape}")
            
            if title_emb.shape[-1] != self.embedding_size:
                raise ValueError(f"Expected embedding size {self.embedding_size}, got {title_emb.shape[-1]}")
                
            if not isinstance(title_emb, torch.Tensor):
                title_emb = torch.tensor(title_emb, dtype=torch.float32, device=seq_output.device)
            
            # Проецируем эмбеддинги заголовков
            title_proj = self.title_projection(title_emb)  # [batch_size, hidden_size]
            title_proj = self.title_norm(title_proj)
            
            # Расширяем до размера последовательности
            title_proj = title_proj.unsqueeze(1).expand(-1, seq_output.size(1), -1)  # [batch_size, seq_len, hidden_size]
            all_features.append(title_proj)
            print(f"Title projection shape: {title_proj.shape}")
        
        # Обрабатываем числовые признаки
        if self.numerical_features:
            numerical_tensors = []
            for field in self.numerical_features:
                if field in interaction and not field.endswith('_embedding'):
                    num_feature = interaction[field].unsqueeze(-1)  # [batch_size, 1]
                    if not isinstance(num_feature, torch.Tensor):
                        num_feature = torch.tensor(num_feature, dtype=torch.float32, device=seq_output.device)
                    projected_num = self.numerical_projections[field](num_feature)  # [batch_size, hidden_size]
                    numerical_tensors.append(projected_num)
            
            if numerical_tensors:
                numerical_output = torch.stack(numerical_tensors, dim=1)  # [batch_size, num_features, hidden_size]
                numerical_output = self.numerical_norm(numerical_output.mean(dim=1))  # [batch_size, hidden_size]
                numerical_output = numerical_output.unsqueeze(1).expand(-1, seq_output.size(1), -1)  # [batch_size, seq_len, hidden_size]
                all_features.append(numerical_output)
                print(f"Numerical output shape: {numerical_output.shape}")
        
        # Обрабатываем категориальные признаки
        if self.categorical_features:
            categorical_tensors = []
            for field in self.categorical_features:
                if field in interaction:
                    cat_feature = interaction[field]
                    if not isinstance(cat_feature, torch.Tensor):
                        cat_feature = torch.tensor(cat_feature, dtype=torch.long, device=seq_output.device)
                    cat_embedding = self.categorical_embeddings[field](cat_feature)  # [batch_size, hidden_size]
                    categorical_tensors.append(cat_embedding)
            
            if categorical_tensors:
                categorical_output = torch.stack(categorical_tensors, dim=1)  # [batch_size, num_features, hidden_size]
                categorical_output = self.categorical_norm(categorical_output.mean(dim=1))  # [batch_size, hidden_size]
                categorical_output = categorical_output.unsqueeze(1).expand(-1, seq_output.size(1), -1)  # [batch_size, seq_len, hidden_size]
                all_features.append(categorical_output)
                print(f"Categorical output shape: {categorical_output.shape}")
        
        # Объединяем все признаки
        combined = torch.cat(all_features, dim=-1)  # [batch_size, seq_len, num_features * hidden_size]
        print(f"Combined features shape: {combined.shape}")
        combined = self.dropout(combined)
        output = self.feature_fusion(combined)  # [batch_size, seq_len, hidden_size]
        print(f"Final output shape: {output.shape}")
        
        return output