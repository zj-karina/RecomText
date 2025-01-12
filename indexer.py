import os
import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.multimodal_model import MultimodalRecommendationModel

class VideoInfoDataset(Dataset):
    """
    Датасет для индексирования товаров (rutube_video_id + title).
    """
    def __init__(self, df, tokenizer, max_length=128):
        """
        df: DataFrame с колонками [rutube_video_id, title, ...]
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_id = int(row['rutube_video_id'])
        title = str(row['title'])

        tokens = self.tokenizer(
            title,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Из batch=1 в обычный вид [seq_len]
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        
        return tokens, item_id

def collate_fn(batch):
    """
    Объединяем список (tokens, item_id) в батч.
    """
    tokens_list, item_ids_list = zip(*batch)

    input_ids = torch.stack([x['input_ids'] for x in tokens_list], dim=0)
    attention_mask = torch.stack([x['attention_mask'] for x in tokens_list], dim=0)

    item_ids_tensor = torch.tensor(item_ids_list, dtype=torch.long)

    tokens_batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    return tokens_batch, item_ids_tensor


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1) Загружаем конфиг
    config = load_config()
    model_path = config['inference']['model_path']
    text_model_name = config['model']['text_model_name']
    batch_size = config['data']['batch_size']
    max_length = config['data']['max_length']
    index_path = config['inference'].get('index_path', 'video_index.faiss')
    ids_path = config['inference'].get('ids_path', 'video_ids.npy')
    
    # 2) Загружаем всю таблицу с товарами
    df_videos = pd.read_parquet("./data/video_info.parquet")
    print(f"Loaded {len(df_videos)} items for indexing.")
    
    # 3) Загружаем модель и tokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4) Создаем датасет для индексации
    dataset = VideoInfoDataset(df_videos, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 5) Инициализируем модель с размерами из датасета
    model = MultimodalRecommendationModel(
        text_model_name=text_model_name,
        user_vocab_size=len(dataset.user_id_map),
        items_vocab_size=len(dataset.item_id_map),
        id_embed_dim=config['model']['id_embed_dim'],
        text_embed_dim=config['model']['text_embed_dim']
    )

    # Загрузка обученных весов
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        # Загружаем модель в формате HuggingFace
        model.text_model = AutoModel.from_pretrained(model_path)
        
        # Загружаем метаданные
        meta_path = os.path.join(os.path.dirname(model_path), 
                                f'meta_{os.path.basename(model_path)}.pt')
        if os.path.exists(meta_path):
            checkpoint_meta = torch.load(meta_path, map_location="cpu")
            print(f"Loaded metadata from epoch {checkpoint_meta['epoch']}")
            print(f"Best metric: {checkpoint_meta['best_metric']}")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    model.to(device)
    model.eval()

    # 6) Создаём FAISS index
    embed_dim = model.text_model.config.hidden_size
    index = faiss.IndexFlatIP(embed_dim)
    
    all_embeddings = []
    all_ids = []

    # 7) Прогоняем товары через модель
    with torch.no_grad():
        for tokens_batch, item_ids_batch in tqdm(loader, desc="Indexing items"):
            tokens_batch = {k: v.to(device) for k, v in tokens_batch.items()}
            item_ids_batch = item_ids_batch.to(device)

            # Заглушки для user-части
            dummy_user_inputs = {
                "input_ids": torch.zeros_like(tokens_batch["input_ids"]),
                "attention_mask": torch.zeros_like(tokens_batch["attention_mask"])
            }
            dummy_user_ids = torch.zeros_like(item_ids_batch)

            # Прямой проход (только item_embeddings нам нужен)
            items_embeddings, _ = model(
                items_text_inputs=tokens_batch,
                user_text_inputs=dummy_user_inputs,
                item_ids=item_ids_batch,
                user_ids=dummy_user_ids
            )

            # Нормируем для косинус-похожести
            items_embeddings = F.normalize(items_embeddings, p=2, dim=1)

            # На CPU
            items_embeddings_np = items_embeddings.cpu().numpy().astype('float32')
            item_ids_np = item_ids_batch.cpu().numpy()

            all_embeddings.append(items_embeddings_np)
            all_ids.extend(item_ids_np)

    # Объединяем
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print("Adding to Faiss index...", all_embeddings.shape)
    index.add(all_embeddings)

    # 8) Сохраняем индекс + IDs
    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(all_ids, dtype=np.int64))
    print(f"Saved FAISS index to {index_path}, IDs to {ids_path}")
    print("Done.")


if __name__ == "__main__":
    main()