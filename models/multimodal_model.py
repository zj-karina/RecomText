import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel

class MultimodalRecommendationModel(nn.Module):
    """A multimodal model for text and recommendation tasks."""
    def __init__(self, text_model_name, user_vocab_size, items_vocab_size, 
                 id_embed_dim=32, text_embed_dim=768):
        super(MultimodalRecommendationModel, self).__init__()
        
        # Сохраняем параметры конструктора
        self.text_model_name = text_model_name
        self.user_vocab_size = user_vocab_size
        self.items_vocab_size = items_vocab_size
        self.id_embed_dim = id_embed_dim
        self.text_embed_dim = text_embed_dim

        # Text-based (HF) model
        self.text_model = AutoModel.from_pretrained(text_model_name)

        # Embeddings for user IDs and items IDs
        self.user_id_embeddings = nn.Embedding(user_vocab_size, id_embed_dim)
        self.items_id_embeddings = nn.Embedding(items_vocab_size, id_embed_dim)

        # Fusion layers
        self.user_fusion = nn.Linear(text_embed_dim + id_embed_dim, text_embed_dim)
        self.items_fusion = nn.Linear(text_embed_dim + id_embed_dim, text_embed_dim)

        # Добавляем dropout для регуляризации
        self.dropout = nn.Dropout(0.3)  # Сильный dropout для борьбы с переобучением

    def forward(self, items_text_inputs, user_text_inputs, item_ids, user_id):
        # Text embeddings
        items_text_embeddings = self.text_model(**items_text_inputs).last_hidden_state.mean(dim=1)
        user_text_embeddings  = self.text_model(**user_text_inputs).last_hidden_state.mean(dim=1)
    
        # Применяем dropout к текстовым эмбеддингам
        items_text_embeddings = self.dropout(items_text_embeddings)
        user_text_embeddings = self.dropout(user_text_embeddings)
    
        # ID embeddings
        items_id_embeddings = self.items_id_embeddings(item_ids).mean(dim=1)
        user_id_embedding   = self.user_id_embeddings(user_id)
    
        # Fusion с большим весом для ID эмбеддингов
        items_embeddings = self.items_fusion(
            torch.cat([items_text_embeddings * 0.7, items_id_embeddings * 1.3], dim=-1)  # Увеличиваем вес ID
        )
        user_embeddings  = self.user_fusion(
            torch.cat([user_text_embeddings * 0.7, user_id_embedding * 1.3], dim=-1)  # Увеличиваем вес ID
        )
    
        # Финальный dropout
        items_embeddings = self.dropout(items_embeddings)
        user_embeddings = self.dropout(user_embeddings)
    
        return items_embeddings, user_embeddings

    def save_pretrained(self, save_directory: str):
        """
        Сохраняем часть text_model в стиле HuggingFace
        и наши "кастомные" слои + конфиг отдельно.
        """
        os.makedirs(save_directory, exist_ok=True)

        # (1) Сохранение text_model (HF-способ)
        self.text_model.save_pretrained(save_directory)

        # (2) Сохранение конфигурации кастомной части
        config = {
            "text_model_name": self.text_model_name,
            "user_vocab_size": self.user_vocab_size,
            "items_vocab_size": self.items_vocab_size,
            "id_embed_dim": self.id_embed_dim,
            "text_embed_dim": self.text_embed_dim,
            "model_type": "MultimodalRecommendationModel"
        }
        config_path = os.path.join(save_directory, "multimodal_config.json")
        import json
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # (3) Сохранение "кастомных" весов отдельно
        full_state_dict = self.state_dict()
        text_model_keys = set(k for k in self.text_model.state_dict().keys())

        custom_weights = {}
        for k, v in full_state_dict.items():
            # Если ключ принадлежит AutoModel, пропускаем (он уже в pytorch_model.bin)
            if (k in text_model_keys) or k.startswith("text_model."):
                continue
            custom_weights[k] = v

        torch.save(custom_weights, os.path.join(save_directory, "pytorch_multimodal.bin"))
        
        print(f"\nModel saved to {save_directory}")
        print(f"  HF weights: pytorch_model.bin (inside {save_directory})")
        print(f"  Custom weights: pytorch_multimodal.bin")
        print(f"  Config: {config_path}")

    @classmethod
    def from_pretrained(cls, save_directory: str, device="cpu"):
        """
        Восстанавливаем модель из директории,
        где были сохранены HF-весы (text_model), конфиг (multimodal_config.json)
        и кастомные веса (pytorch_multimodal.bin).
        """
        # (1) Считываем конфиг
        config_path = os.path.join(save_directory, "multimodal_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        text_model_name  = config_data["text_model_name"]
        user_vocab_size  = config_data["user_vocab_size"]
        items_vocab_size = config_data["items_vocab_size"]
        id_embed_dim     = config_data["id_embed_dim"]
        text_embed_dim   = config_data["text_embed_dim"]

        # (2) Создаём экземпляр класса (пока без весов), 
        #     но text_model внутри будет прочитана из HF-весов
        model = cls(
            text_model_name=text_model_name,
            user_vocab_size=user_vocab_size,
            items_vocab_size=items_vocab_size,
            id_embed_dim=id_embed_dim,
            text_embed_dim=text_embed_dim
        )
        
        # (3) Загрузим state_dict для text_model (пока HF уже это сделала)
        #     На самом деле, AutoModel.from_pretrained(...) в конструкторе 
        #     уже подтянуло pytorch_model.bin/config.json.
        #     Но если вы хотите жёстко перезагрузить:
        #        model.text_model = AutoModel.from_pretrained(save_directory)
        #     Так или иначе, уже это сделано во время __init__.

        # (4) Загрузка весов кастомной части (Embedding, Fusion)
        custom_weights_path = os.path.join(save_directory, "pytorch_multimodal.bin")
        if not os.path.isfile(custom_weights_path):
            raise FileNotFoundError(f"Custom weights not found: {custom_weights_path}")

        custom_state_dict = torch.load(custom_weights_path, map_location=device)

        # Объединяем с нынешним state_dict
        # Внимание: нужно только подгрузить ключи, имеющиеся в custom_state_dict:
        model.load_state_dict(custom_state_dict, strict=False)

        # Перемещаем на device
        model.to(device)
        model.eval()

        print(f"\nLoaded MultimodalRecommendationModel from {save_directory}")
        return model