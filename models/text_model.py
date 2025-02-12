import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModel

class TextOnlyRecommendationModel(nn.Module):
    """
    A text-only recommendation model that processes text inputs for users and items.
    """
    def __init__(self, text_model_name: str):
        super(TextOnlyRecommendationModel, self).__init__()
        
        # Сохраняем параметры конструктора
        self.text_model_name = text_model_name

        # Text-based (HF) model
        self.text_model = AutoModel.from_pretrained(text_model_name)

    def forward(self, 
               items_text_inputs: Dict[str, torch.Tensor], 
               user_text_inputs: Dict[str, torch.Tensor], 
               item_ids: Optional[torch.Tensor] = None, 
               user_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Text embeddings
        items_embeddings = self.text_model(**items_text_inputs).last_hidden_state.mean(dim=1)
        user_embeddings = self.text_model(**user_text_inputs).last_hidden_state.mean(dim=1)
        
        return items_embeddings, user_embeddings

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)

        # (1) Сохранение text_model (HF-способ)
        self.text_model.save_pretrained(save_directory)

        # (2) Сохранение конфигурации кастомной части
        config = {
            "text_model_name": self.text_model_name,
            "model_type": "TextOnlyRecommendationModel"
        }
        config_path = os.path.join(save_directory, "textonly_config.json")
        import json
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\nModel saved to {save_directory}")
        print(f"  HF weights: pytorch_model.bin (inside {save_directory})")
        print(f"  Config: {config_path}")

    @classmethod
    def from_pretrained(cls, save_directory: str, device="cpu"):
        """
        Восстанавливаем модель из директории,
        где были сохранены HF-весы (text_model), конфиг (multimodal_config.json)
        и кастомные веса (pytorch_multimodal.bin).
        """
        # (1) Считываем конфиг
        config_path = os.path.join(save_directory, "textonly_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        text_model_name  = config_data["text_model_name"]

        # (2) Создаём экземпляр класса (пока без весов), 
        #     но text_model внутри будет прочитана из HF-весов
        model = cls(
            text_model_name=text_model_name,
        )

        # Перемещаем на device
        model.to(device)
        model.eval()

        print(f"\nLoaded TextOnlyRecommendationModel from {save_directory}")
        return model