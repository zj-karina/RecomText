import os
import torch
import torch.nn as nn
from recbole.model.sequential_recommender import SASRec
import numpy as np


class EnhancedSASRec(SASRec):
    """SASRec с добавлением числовых, категориальных признаков и embedding list."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.numerical_features = config['data'].get('numerical_features', [])
        self.categorical_features = config['data'].get('token_features', [])
        self.embedding_sequence_fields = config['data'].get('embedding_sequence_fields', [])

        self.embedding_list_weights = {}
        for field in self.embedding_sequence_fields:
            emb_field_name = field.replace('_idx', '')
            emb_path = os.path.join(config['data_path'], f"{emb_field_name}.npy")
            self.embedding_list_weights[field] = torch.from_numpy(np.load(emb_path)).float()

        # Размер скрытого пространства
        self.hidden_size = config["hidden_size"]

        # Проекции для числовых признаков (1 → hidden_size)
        self.numerical_projections = nn.ModuleDict({
            field: nn.Linear(1, self.hidden_size)
            for field in self.numerical_features
            if not field.endswith('_embedding')
        })

        # Эмбеддинги для категориальных признаков
        self.categorical_embeddings = nn.ModuleDict({
            feature: nn.Embedding(len(dataset.field2id_token[feature]), self.hidden_size)
            for feature in self.categorical_features
            if feature in dataset.field2id_token
        })

        self.norm_numerical = nn.LayerNorm(self.hidden_size)
        self.norm_categorical = nn.LayerNorm(self.hidden_size)
        self.embedding_sequence_norm = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        # Кол-во типов признаков, участвующих в concat
        num_feature_sources = 1  # output SASRec
        if self.numerical_features:
            num_feature_sources += 1
        if self.categorical_features:
            num_feature_sources += 1
        if self.embedding_sequence_fields:
            num_feature_sources += 1

        self.feature_fusion = nn.Linear(self.hidden_size * num_feature_sources, self.hidden_size)

    def forward(self, item_seq, item_seq_len, interaction=None):
        # Основной output SASRec (B, H)
        seq_output = super().forward(item_seq, item_seq_len)
        all_features = [seq_output]

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

            # Эмбеддинг-последовательности (embedding_list)
            for idx_field in self.embedding_sequence_fields:
                if idx_field not in interaction:
                    continue

                indices = interaction[idx_field]  # [B, L]
                print(f"{idx_field} indices: min={indices.min().item()}, max={indices.max().item()}, shape={indices.shape}")
                
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices, dtype=torch.long, device=seq_output.device)

                if indices.dim() == 1:
                    indices = indices.view(seq_output.shape[0], -1)

                embed_matrix = self.embedding_list_weights[idx_field].to(seq_output.device)
                assert indices.max().item() < embed_matrix.shape[0], f"Index out of bounds for {idx_field}"

                embedded = embed_matrix[indices]  # [B, L, H]

                # MASKED MEAN AGGREGATION
                mask = (indices != 0).float().unsqueeze(-1)      # [B, L, 1]
                masked_embeds = embedded * mask                  # [B, L, H]
                sum_embeds = masked_embeds.sum(dim=1)            # [B, H]
                count = mask.sum(dim=1).clamp(min=1e-6)          # [B, 1]
                pooled = sum_embeds / count                      # [B, H]
                pooled = self.embedding_sequence_norm(pooled)    # norm it
                all_features.append(pooled)

        concat = torch.cat(all_features, dim=-1)  # [B, D * num_parts]
        
        # Проверка перед передачей в слой feature_fusion
        fused = self.dropout(self.feature_fusion(concat))  # [B, D]
        return fused

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, interaction)
        pos_items = interaction[self.POS_ITEM_ID]

        if pos_items.max() >= self.item_embedding.num_embeddings:
            raise ValueError("Invalid item index in pos_items")

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_emb, dim=-1)
            return self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, interaction)
        test_item_emb = self.item_embedding(test_item)
        return torch.sum(seq_output * test_item_emb, dim=-1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, interaction)
        test_items_emb = self.item_embedding.weight
        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))
