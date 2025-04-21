import torch
import torch.nn as nn
from recbole.model.sequential_recommender import BERT4Rec
import numpy as np
import os


class EnhancedBERT4Rec(BERT4Rec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.numerical_features = config['data'].get('numerical_features', [])
        self.categorical_features = config['data'].get('token_features', [])
        self.embedding_sequence_fields = config['data'].get('embedding_sequence_fields', [])
        self.hidden_size = config['hidden_size']

        # Числовые признаки: линейные проекции
        self.numerical_projections = nn.ModuleDict({
            field: nn.Linear(1, self.hidden_size)
            for field in self.numerical_features
        })

        # Категориальные эмбеддинги
        self.categorical_embeddings = nn.ModuleDict({
            field: nn.Embedding(len(dataset.field2id_token[field]), self.hidden_size)
            for field in self.categorical_features
            if field in dataset.field2id_token
        })

        # Эмбеддинг матрицы из внешних npy
        self.embedding_list_weights = {}
        for field in self.embedding_sequence_fields:
            emb_field_name = field.replace('_idx', '')
            emb_path = os.path.join(config['data_path'], f"{emb_field_name}.npy")
            self.embedding_list_weights[field] = torch.from_numpy(np.load(emb_path)).float()

        # Нормализация и дропаут
        self.norm_numerical = nn.LayerNorm(self.hidden_size)
        self.norm_categorical = nn.LayerNorm(self.hidden_size)
        self.embedding_sequence_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        num_sources = 1
        if self.numerical_features:
            num_sources += 1
        if self.categorical_features:
            num_sources += 1
        if self.embedding_sequence_fields:
            num_sources += 1

        self.feature_fusion = nn.Linear(self.hidden_size * num_sources, self.hidden_size)
        self.output_project = nn.Linear(self.hidden_size * num_sources, self.hidden_size)


    def forward(self, item_seq, interaction=None):
        seq_output = super().forward(item_seq)  # [B, L, H]
    
        all_features = []
    
        if interaction is not None:
            device = seq_output.device
    
            # Числовые признаки
            if self.numerical_features:
                numerical_stack = []
                for field in self.numerical_features:
                    if field in interaction:
                        val = interaction[field]
                        if not isinstance(val, torch.Tensor):
                            val = torch.tensor(val, dtype=torch.float32, device=device)
                        if val.dim() == 1:
                            val = val.unsqueeze(-1)
                        proj = self.numerical_projections[field](val)
                        numerical_stack.append(proj)
                if numerical_stack:
                    num_avg = torch.stack(numerical_stack, dim=0).mean(dim=0)
                    num_normed = self.norm_numerical(num_avg)
                    all_features.append(num_normed)
    
            # Категориальные признаки
            if self.categorical_features:
                categorical_stack = []
                for field in self.categorical_features:
                    if field in interaction:
                        val = interaction[field]
                        if not isinstance(val, torch.Tensor):
                            val = torch.tensor(val, dtype=torch.long, device=device)
                        emb = self.categorical_embeddings[field](val)
                        categorical_stack.append(emb)
                if categorical_stack:
                    cat_avg = torch.stack(categorical_stack, dim=0).mean(dim=0)
                    cat_normed = self.norm_categorical(cat_avg)
                    all_features.append(cat_normed)
    
            for idx_field in self.embedding_sequence_fields:
                if idx_field not in interaction:
                    continue
                indices = interaction[idx_field]
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices, dtype=torch.long, device=seq_output.device)
                if indices.dim() == 1:
                    indices = indices.view(seq_output.shape[0], -1)
                embed_matrix = self.embedding_list_weights[idx_field].to(seq_output.device)
                embedded = embed_matrix[indices]  # [B, L, H]
                mask = (indices != 0).float().unsqueeze(-1)  # [B, L, 1]
                masked_embeds = embedded * mask
                sum_embeds = masked_embeds.sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1e-6)
                pooled = sum_embeds / count  # [B, H]
                pooled = pooled.view(-1, self.hidden_size)
                pooled = self.embedding_sequence_norm(pooled)
                all_features.append(pooled)
    
        # Конкатенируем доп. признаки
        side_features = torch.cat(all_features, dim=-1) if all_features else None # [B, D_extra]
        return seq_output, side_features  # [B, L, H], [B, D_extra]
    

    def calculate_loss(self, interaction):
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        pos_items = interaction[self.POS_ITEMS]
        masked_index = interaction[self.MASK_INDEX]
    
        seq_output, side_features = self.forward(masked_item_seq, interaction)  # [B, L, H]
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))        
        pred_index_map = pred_index_map.unsqueeze(-1)
        
        seq_output = seq_output.unsqueeze(2)
        seq_output = torch.bmm(pred_index_map.view(-1, masked_index.size(1), masked_item_seq.size(-1)), seq_output.view(-1, masked_item_seq.size(-1), seq_output.size(-1)))
        
        seq_output = seq_output.view(masked_index.size(0), masked_index.size(1), -1)  # [B, mask_len, H]
        
        if side_features is not None:
            side_features = side_features.unsqueeze(1).expand(-1, masked_index.size(1), -1)
            seq_output = torch.cat([seq_output, side_features], dim=-1)  # [B, mask_len, H + D_extra]
        
        # BPR Loss
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEMS]
            pos_items_emb = self.item_embedding(pos_items)  # [B, M, H]
            neg_items_emb = self.item_embedding(neg_items)  # [B, M, H]
            # если размеры эмбеддинга изменились, project их в общий hidden:
            if seq_output.size(-1) != pos_items_emb.size(-1):
                seq_output = self.output_project(seq_output)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            targets = (masked_index > 0).float()
            loss = -torch.sum(torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets) / torch.sum(targets)
            return loss
        
        # Cross Entropy
        elif self.loss_type == "CE":
            test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num, H]
            if seq_output.size(-1) != test_item_emb.size(-1):
                seq_output = self.output_project(seq_output)  # [B, M, H]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, M, item_num]
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            targets = (masked_index > 0).float().view(-1)  # [B*mask_len]
            loss = torch.sum(
                loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets
            ) / torch.sum(targets)
            return loss

    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
    
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        
        seq_output, side_features = self.forward(item_seq, interaction)  # [B, L, H], [B, D_extra]
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B, H]
    
        if side_features is not None:
            seq_output = torch.cat([seq_output, side_features], dim=-1)  # [B, H + D_extra]
    
        test_item_emb = self.item_embedding(test_item)  # [B, H]
        
        if seq_output.size(-1) != test_item_emb.size(-1):
            seq_output = self.output_project(seq_output)  # [B, H]
    
        return (seq_output * test_item_emb).sum(dim=-1)  # [B]
    

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
    
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        
        seq_output, side_features = self.forward(item_seq, interaction)  # [B, L, H], [B, D_extra]
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B, H]
    
        if side_features is not None:
            seq_output = torch.cat([seq_output, side_features], dim=-1)  # [B, H + D_extra]
    
        test_items_emb = self.item_embedding.weight[:self.n_items]  # [item_num, H]
        
        if seq_output.size(-1) != test_items_emb.size(-1):
            seq_output = self.output_project(seq_output)  # [B, H]
    
        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
