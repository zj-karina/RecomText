import os
import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Импортируем вашу модель
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
    model_path = config['inference']['model_path']           # где лежит обученная модель (HF формат)
    text_model_name = config['model']['text_model_name']   
    batch_size = config['data']['batch_size']                # размер батча
    max_length = config['data']['max_length']                # макс. длина токенов
    index_path = config['inference'].get('index_path', 'video_index.faiss')
    ids_path = config['inference'].get('ids_path', 'video_ids.npy')
    
    # 2) Загружаем всю таблицу с товарами
    df_videos = pd.read_parquet("./data/video_info.parquet")
    print(f"Loaded {len(df_videos)} items for indexing.")
    
    # 3) Загружаем модель и tokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Инициализируем вашу кастомную модель
    # (Обратите внимание: вы, возможно, сохраняли её не стандартным AutoModel,
    #  а через torch.save(state_dict). Тогда смотрите, как правильно загрузить веса.)
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

    # Загрузка обученных весов (пример):
    if os.path.isdir(model_path):
        # Если вы сохраняли через AutoModel.save_pretrained(...) — это один вариант
        print(f"Warning: {model_path} seems to be a HF directory; adapt load accordingly.")
    else:
        # Если вы сохраняли через torch.save(state_dict, ...)
        print(f"Loading state_dict from {model_path} ...")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4) Делаем DataLoader по всем товарам
    dataset = VideoInfoDataset(df_videos, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 5) Создаём FAISS index
    # Возьмём размерность text_model.config.hidden_size или у вас может быть text_embed_dim
    embed_dim = model.text_model.config.hidden_size  
    index = faiss.IndexFlatIP(embed_dim)  # dot product (Inner Product)
    
    all_embeddings = []
    all_ids = []

    # 6) Прогоняем товары через модель, получаем item_embeddings
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
                user_id=dummy_user_ids
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

    # 7) Сохраняем индекс + IDs
    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(all_ids, dtype=np.int64))
    print(f"Saved FAISS index to {index_path}, IDs to {ids_path}")
    print("Done.")


if __name__ == "__main__":
    main()