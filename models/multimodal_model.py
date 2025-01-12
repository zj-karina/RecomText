import torch
import torch.nn as nn
from transformers import AutoModel

class MultimodalRecommendationModel(nn.Module):
    """A multimodal model for text and recommendation tasks."""
    def __init__(self, text_model_name, user_vocab_size, items_vocab_size, 
                 id_embed_dim=32, text_embed_dim=768):
        super(MultimodalRecommendationModel, self).__init__()
        
        # Text-based model
        self.text_model = AutoModel.from_pretrained(text_model_name)

        # Embeddings for user IDs and items IDs
        self.user_id_embeddings = nn.Embedding(user_vocab_size, id_embed_dim)
        self.items_id_embeddings = nn.Embedding(items_vocab_size, id_embed_dim)

        # Fusion layers
        self.user_fusion = nn.Linear(text_embed_dim + id_embed_dim, text_embed_dim)
        self.items_fusion = nn.Linear(text_embed_dim + id_embed_dim, text_embed_dim)

    def forward(self, items_text_inputs, user_text_inputs, item_ids, user_id):
        # Text embeddings
        items_text_embeddings = self.text_model(**items_text_inputs).last_hidden_state.mean(dim=1)
        user_text_embeddings = self.text_model(**user_text_inputs).last_hidden_state.mean(dim=1)
    
        # ID embeddings
        items_id_embeddings = self.items_id_embeddings(item_ids).mean(dim=1)
        user_id_embedding = self.user_id_embeddings(user_id)
    
        # Fusion
        items_embeddings = self.items_fusion(
            torch.cat([items_text_embeddings, items_id_embeddings], dim=-1)
        )
        user_embeddings = self.user_fusion(
            torch.cat([user_text_embeddings, user_id_embedding], dim=-1)
        )
    
        return items_embeddings, user_embeddings