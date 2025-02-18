import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from data.dataset import BuildTrainDataset, get_dataloader
from torch.utils.data import DataLoader
import os
import yaml
import pandas as pd
import faiss
import json
import numpy as np

from models.multimodal_model import MultimodalRecommendationModel
from models.text_model import TextOnlyRecommendationModel

def to_device(data, device):
    """
    Переносит данные на указанное устройство.
    Поддерживает словари, списки, кортежи и тензоры.
    """
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif hasattr(data, 'to'):  # Проверяем наличие метода to
        return data.to(device)
    return data

def evaluate_predictions(model, val_loader, textual_history, device, k=10, num_examples=5, tokenizer=None):
    """
    Evaluate model predictions on validation set and show examples.
    """
    # Добавляем получение путей из конфига
    config = load_config()
    index_path = config['inference'].get('index_path', 'video_index.faiss')
    ids_path = config['inference'].get('ids_path', 'video_ids.npy')

    # Проверяем наличие необходимых параметров
    if not tokenizer:
        raise ValueError("Tokenizer must be provided")

    # Загружаем FAISS индекс
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Faiss index not found at {index_path}")
    index = faiss.read_index(index_path)
    
    # Загружаем список товаров (video_ids)
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"Video IDs not found at {ids_path}")
    video_ids = np.load(ids_path).tolist()
    print(f"Loaded FAISS index with {index.ntotal} vectors. video_ids size={len(video_ids)}")
    
    # Загружаем информацию о видео
    df_videos = pd.read_parquet("./data/video_info.parquet")
    # Используем clean_video_id вместо rutube_video_id для соответствия с индексером
    df_videos_map = df_videos.set_index('clean_video_id').to_dict(orient='index')

    # Пробегаемся по val_loader
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            items_text_inputs, user_text_inputs, item_ids, user_ids = [
                to_device(x, device) for x in batch
            ]

            # Считаем user_embeddings
            _, user_embeddings = model(
                items_text_inputs,
                user_text_inputs,
                item_ids,
                user_ids
            )
            
            # Нормируем
            user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    
            # Вычисляем рекомендации для первых num_examples пользователей в батче
            for u in range(min(num_examples, user_embeddings.size(0))):
                user_emb = user_embeddings[u].unsqueeze(0)
                user_emb_np = user_emb.cpu().numpy().astype('float32')
                
                # Поиск в FAISS
                distances, faiss_indices = index.search(user_emb_np, k)
                
                # Получаем ID видео и скоры
                retrieved_ids = [video_ids[idx] for idx in faiss_indices[0]]
                retrieved_scores = distances[0]
    
                # Декодируем текст пользователя
                user_text_decoded = tokenizer.decode(
                    user_text_inputs["input_ids"][u],
                    skip_special_tokens=True
                )
                print(f'\nUser text: {user_text_decoded.replace("passage: ", "")}')
    
                # Получаем детальную историю просмотров
                print("\nUser viewing history:")
                print(textual_history.iloc[u]['detailed_view'].replace("query: ", ""))
    
                # Показываем top-K
                print(f"Top-{k} recommendations from FAISS:")
                for rank, (vid, score) in enumerate(zip(retrieved_ids, retrieved_scores), start=1):
                    vid_value = vid[0]
                    video_data = df_videos_map.get(str(vid_value), {})  # Преобразуем в строку для соответствия
                    title = video_data.get('title', 'Unknown title')
                    cat = video_data.get('category', 'Unknown category')
                    
                    print(f"  {rank}. Video ID={vid_value}, Score={score:.4f}, Category={cat}")
                    print(f"     Title: {title}")

            # Для демонстрации прерываемся после первого батча
            break

    print("Inference done.")

def load_config(config_path="configs/config.yaml"):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Загрузка конфигурации
    config = load_config()
    
    # Получение параметров из конфига
    model_path = config['inference']['model_path']
    batch_size = config['data']['batch_size']
    num_examples = config['inference']['num_examples']
    top_k = config['inference']['top_k']
    max_length = config['data']['max_length']
    checkpoint_dir = config['training']['checkpoint_dir']

    index_path = config['inference'].get('index_path', 'video_index.faiss')
    ids_path = config['inference'].get('ids_path', 'video_ids.npy')

    # Prepare data
    textual_history = pd.read_parquet('./data/textual_history.parquet')
    id_history = pd.read_parquet('./data/id_history.parquet')
    user_descriptions = pd.read_parquet('./data/user_descriptions.parquet')

    # Загружаем маппинг
    with open('./data/mappings/item_id_map.json', 'r', encoding='utf-8') as f:
        item_id_map = json.load(f)
    print(f"Loaded item_id_map with {len(item_id_map)} items")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['model'].get('use_fusion', True):
        model = MultimodalRecommendationModel.from_pretrained(checkpoint_dir)
    else:
        model = TextOnlyRecommendationModel.from_pretrained(checkpoint_dir)

    model.to(device)
    
    # Создание валидационного датасета и лоадера
    val_dataset = BuildTrainDataset(
        textual_history=textual_history,
        user_descriptions=user_descriptions,
        id_history=id_history,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        split='val',
        val_size=config['training'].get('validation_size', 0.1),
        random_state=config['training'].get('random_seed', 42),
        item_id_map=item_id_map  # Передаем тот же маппинг
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Оценка предсказаний с передачей токенизатора
    evaluate_predictions(
        model, 
        val_loader, 
        textual_history,
        device, 
        k=top_k, 
        num_examples=num_examples,
        tokenizer=tokenizer  # Передаем токенизатор
    ) 