import os
import faiss
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from data.dataset import BuildTrainDataset  
from models.multimodal_model import MultimodalRecommendationModel 

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def to_device(data, device):
    """
    Переносит данные на указанное устройство (ваша вспомогательная функция).
    """
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif hasattr(data, 'to'):
        return data.to(device)
    return data

def main():
    # 1) Загружаем конфиг
    config = load_config()
    model_path = config['inference']['model_path']
    batch_size = config['data']['batch_size']
    max_length = config['data']['max_length']
    
    top_k = config['inference']['top_k']
    num_examples = config['inference']['num_examples']  # сколько пользователей показать
    
    index_path = config['inference'].get('index_path', 'video_index.faiss')
    ids_path = config['inference'].get('ids_path', 'video_ids.npy')
    
    # 2) Загружаем модель (кастомный класс)
    text_model_name = config['model']['text_model_name']
    user_vocab_size = 10000
    items_vocab_size = 50000
    id_embed_dim = 32
    text_embed_dim = 768

    model = MultimodalRecommendationModel(
        text_model_name=text_model_name,
        user_vocab_size=user_vocab_size,
        items_vocab_size=items_vocab_size,
        id_embed_dim=id_embed_dim,
        text_embed_dim=text_embed_dim
    )
    
    # Загрузка state_dict
    if os.path.isdir(model_path):
        print(f"Warning: {model_path} is directory, adapt as needed.")
    else:
        print(f"Loading state_dict from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3) Загружаем FAISS-индекс
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Faiss index not found at {index_path}")
    index = faiss.read_index(index_path)
    
    # Загружаем список товаров (video_ids)
    video_ids = np.load(ids_path).tolist()
    print(f"Loaded FAISS index with {index.ntotal} vectors. video_ids size={len(video_ids)}")
    
    # Если захотим быстро находить title/category, можно подгрузить df и словарь:
    df_videos = pd.read_parquet("./data/all_videos.parquet")
    df_videos_map = df_videos.set_index('rutube_video_id').to_dict(orient='index')
    # Теперь df_videos_map[item_id] даст словарь с полями [title, category, category_id, ...]

    # 4) Инициализируем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # 5) Готовим валидационный датасет (как в вашем коде)
    textual_history = pd.read_parquet('./data/textual_history.parquet')
    id_history = pd.read_parquet('./data/id_history.parquet')
    user_descriptions = pd.read_parquet('./data/user_descriptions.parquet')

    val_dataset = BuildTrainDataset(
        textual_history=textual_history,
        user_descriptions=user_descriptions,
        id_history=id_history,
        tokenizer=tokenizer,
        max_length=max_length,
        split='val',
        val_size=config['training'].get('validation_size', 0.1),
        random_state=config['training'].get('random_seed', 42)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 6) Пробегаемся по val_loader, но рекомендации достаём из FAISS
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            items_text_inputs, user_text_inputs, item_ids, user_ids, categories = [
                to_device(x, device) for x in batch
            ]
            
            # Считаем user_embeddings
            # (item_embeddings нам НЕ нужны, они уже в FAISS)
            # Заглушка для items_text_inputs
            dummy_items_inputs = {
                "input_ids": torch.zeros_like(items_text_inputs["input_ids"]),
                "attention_mask": torch.zeros_like(items_text_inputs["attention_mask"])
            }
            dummy_item_ids = torch.zeros_like(item_ids)
            
            # Прогоняем модель
            _, user_embeddings = model(
                items_text_inputs=dummy_items_inputs,
                user_text_inputs=user_text_inputs,
                item_ids=dummy_item_ids,
                user_id=user_ids
            )
            
            # Нормируем
            user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

            # Вычисляем рекомендации для первых num_examples пользователей в батче
            for u in range(min(num_examples, user_embeddings.size(0))):
                user_emb = user_embeddings[u].unsqueeze(0)  # shape [1, emb_dim]
                user_emb_np = user_emb.cpu().numpy().astype('float32')
                
                # Поиск в FAISS
                distances, faiss_indices = index.search(user_emb_np, top_k)
                # distances, faiss_indices -> shape [1, top_k]
                
                retrieved_ids = [video_ids[idx] for idx in faiss_indices[0]]
                retrieved_scores = distances[0]
                
                # Выводим
                print(f"\nUser {u+1} (batch {batch_idx}):")
                print(f" User ID: {user_ids[u].item()}")
                
                # Декодируем текст пользователя, если нужно
                user_text_decoded = tokenizer.decode(
                    user_text_inputs["input_ids"][u],
                    skip_special_tokens=True
                )
                print(f" User text: {user_text_decoded}")

                # Покажем top-K
                print(f" Top-{top_k} recommendations from FAISS:")
                for rank, (vid, score) in enumerate(zip(retrieved_ids, retrieved_scores), start=1):
                    # Можно достать title/category из df_videos_map
                    video_data = df_videos_map.get(vid, {})
                    title = video_data.get('title', 'Unknown title')
                    cat = video_data.get('category', 'Unknown category')
                    
                    print(f"  {rank}. Video ID={vid}, Score={score:.4f}, Category={cat}")
                    print(f"     Title: {title}")

            # Для демонстрации прерываемся после первого батча
            break

    print("Inference done.")


if __name__ == "__main__":
    main()