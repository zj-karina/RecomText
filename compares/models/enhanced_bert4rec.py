import torch
import torch.nn as nn
from recbole.model.sequential_recommender import BERT4Rec

class EnhancedBERT4Rec(BERT4Rec):
    """Расширенная версия BERT4Rec с поддержкой эмбеддингов заголовков, категориальных и числовых признаков"""
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # Получаем конфигурацию признаков
        self.numerical_features = config['data'].get('numerical_features', [])
        self.categorical_features = config['data'].get('token_features', [])
        
        # Размерность эмбеддингов BERT
        self.embedding_size = config.get('embedding_size', 384)
        
        # Проекция для эмбеддингов заголовков
        self.title_projection = nn.Linear(self.embedding_size, self.hidden_size)
        
        # Создаем проекции для числовых признаков
        self.numerical_projections = nn.ModuleDict({
            field: nn.Linear(1, self.hidden_size)  # Каждый числовой признак проецируется в hidden_size
            for field in self.numerical_features
            if not field.endswith('_embedding')
        })
        
        # Создаем эмбеддинги для категориальных признаков
        if self.categorical_features:
            self.categorical_embeddings = nn.ModuleDict({
                feature: nn.Embedding(dataset.num_features[feature], self.hidden_size)
                for feature in self.categorical_features
            })
        
        # Слои нормализации
        self.title_norm = nn.LayerNorm(self.hidden_size)
        self.numerical_norm = nn.LayerNorm(self.hidden_size)
        self.categorical_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob', 0.1))
        
        # Финальный слой объединения (базовая последовательность + все признаки)
        num_features = (
            1 +  # базовая последовательность
            bool(self.numerical_features) +  # числовые признаки
            bool(self.categorical_features)  # категориальные признаки
        )
        self.feature_fusion = nn.Linear(self.hidden_size * num_features, self.hidden_size)
        
    def forward(self, item_seq, interaction=None):
        """
        Args:
            item_seq: базовая последовательность (тензор)
            interaction: полный объект interaction с дополнительными признаками
        """
        # Получаем базовые эмбеддинги последовательности
        seq_output = super().forward(item_seq)  # [batch_size, seq_len, hidden_size]
        all_features = [seq_output]
        
        if interaction is not None:
            # Обрабатываем эмбеддинги заголовков
            if 'detailed_view_embedding' in interaction:
                title_emb = interaction['detailed_view_embedding']  # [batch_size, embedding_size]
                if not isinstance(title_emb, torch.Tensor):
                    title_emb = torch.tensor(title_emb, dtype=torch.float32, device=seq_output.device)
                
                # Проецируем эмбеддинги заголовков
                title_proj = self.title_projection(title_emb)  # [batch_size, hidden_size]
                title_proj = self.title_norm(title_proj)
                
                # Расширяем до размера последовательности
                title_proj = title_proj.unsqueeze(1).expand(-1, seq_output.size(1), -1)  # [batch_size, seq_len, hidden_size]
                all_features.append(title_proj)
            
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
        
        # Объединяем все признаки
        combined = torch.cat(all_features, dim=-1)  # [batch_size, seq_len, num_features * hidden_size]
        combined = self.dropout(combined)
        output = self.feature_fusion(combined)  # [batch_size, seq_len, hidden_size]
        
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Восстанавливаем тестовые данные
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        
        # Получаем выход модели, передавая полный interaction
        seq_output = self.forward(item_seq, interaction)
        
        # Получаем последний hidden state
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        
        test_item_emb = self.item_embedding.weight[:self.n_items]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        # Восстанавливаем тестовые данные
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        
        # Получаем выход модели, передавая полный interaction
        seq_output = self.forward(item_seq, interaction)
        
        # Получаем последний hidden state
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # Восстанавливаем тестовые данные
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        
        # Получаем выход модели, передавая полный interaction
        seq_output = self.forward(item_seq, interaction)
        
        # Получаем последний hidden state
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        
        test_items_emb = self.item_embedding.weight[:self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
