import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from data.dataset import BuildTrainDataset, get_dataloader
from torch.utils.data import DataLoader
import os
import yaml
import pandas as pd

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

def evaluate_predictions(model, val_loader, device, k=10, num_examples=5, tokenizer=None):
    """
    Evaluate model predictions on validation set and show examples.
    """
    model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            items_text_inputs, user_text_inputs, item_ids, user_ids, categories = [
                to_device(x, device) for x in batch
            ]

            # Forward pass
            items_embeddings, user_embeddings = model(
                items_text_inputs, user_text_inputs, item_ids, user_ids
            )

            # Нормализация эмбеддингов
            items_embeddings = F.normalize(items_embeddings, p=2, dim=1)
            user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

            # Вычисляем схожесть для каждого пользователя
            predictions = []
            for user_emb in user_embeddings[:num_examples]:
                user_emb = user_emb.unsqueeze(0)
                user_predictions = F.cosine_similarity(
                    user_emb.unsqueeze(0),
                    items_embeddings.unsqueeze(0),
                    dim=2
                )
                predictions.append(user_predictions.squeeze(0))
            
            predictions = torch.stack(predictions)
            
            # Получаем топ-k предсказаний для каждого пользователя
            top_k_scores, top_k_indices = torch.topk(predictions, k=min(k, items_embeddings.size(0)))
            
            # Выводим результаты
            print("\nPrediction Examples:")
            for user_idx in range(min(num_examples, len(user_ids))):
                print(f"\nUser {user_idx + 1}:")
                print(f"User ID: {user_ids[user_idx].item()}")
                
                # Декодируем текст пользователя
                if tokenizer:
                    user_text = tokenizer.decode(
                        user_text_inputs['input_ids'][user_idx],
                        skip_special_tokens=True
                    )
                    print(f"User Text Input: {user_text}")
                
                print("\nTop recommendations:")
                for rank, (score, idx) in enumerate(zip(top_k_scores[user_idx], top_k_indices[user_idx]), 1):
                    item_id = item_ids[idx.item()].item()  # Получаем скалярное значение
                    
                    # Декодируем текст товара
                    if tokenizer:
                        item_text = tokenizer.decode(
                            items_text_inputs['input_ids'][idx],
                            skip_special_tokens=True
                        )
                        print(f"{rank}. Item ID: {item_id}")
                        print(f"   Score: {score:.4f}")
                        print(f"   Category: {categories[idx].item()}")
                        print(f"   Text: {item_text}\n")
                    else:
                        print(f"{rank}. Item ID: {item_id}")
                        print(f"   Score: {score:.4f}")
                        print(f"   Category: {categories[idx].item()}\n")
            
            break  # Выходим после первого батча

def load_model(model_path, device):
    """
    Load model from HuggingFace format checkpoint.
    
    Args:
        model_path: Path to the saved model directory
        device: Torch device
    """
    # Загружаем модель
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Загружаем метаданные если нужно
    meta_path = os.path.join(os.path.dirname(model_path), 
                            f'meta_{os.path.basename(model_path)}.pt')
    meta_data = None
    if os.path.exists(meta_path):
        meta_data = torch.load(meta_path)
    
    return model, meta_data

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

    # Prepare data
    textual_history = pd.read_parquet('./data/textual_history.parquet')
    id_history = pd.read_parquet('./data/id_history.parquet')
    user_descriptions = pd.read_parquet('./data/user_descriptions.parquet')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загрузка модели
    model, meta_data = load_model(model_path, device)
    
    # Если нужно, можно использовать метаданные
    if meta_data:
        print(f"Model was trained for {meta_data['epoch']} epochs")
        print(f"Best metric: {meta_data['best_metric']}")
    
    # Создание валидационного датасета и лоадера
    val_dataset = BuildTrainDataset(
        textual_history=textual_history,
        user_descriptions=user_descriptions,
        id_history=id_history,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        split='val',
        val_size=config['training'].get('validation_size', 0.1),
        random_state=config['training'].get('random_seed', 42)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Оценка предсказаний с передачей токенизатора
    evaluate_predictions(
        model, 
        val_loader, 
        device, 
        k=top_k, 
        num_examples=num_examples,
        tokenizer=tokenizer  # Передаем токенизатор
    ) 