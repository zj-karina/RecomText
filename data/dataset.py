import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

class BuildTrainDataset(Dataset):
    """Dataset for training with unpackable outputs and user-description fallback."""
    def __init__(self, textual_history, user_descriptions, id_history, tokenizer, max_length=128, split='train', val_size=0.1, random_state=42):
        self.textual_history = textual_history
        self.user_descriptions = user_descriptions
        self.id_history = id_history
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Split data
        if split in ['train', 'val']:
            indices = np.arange(len(id_history))
            np.random.seed(random_state)
            np.random.shuffle(indices)
            split_idx = int(len(indices) * (1 - val_size))
            
            if split == 'train':
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]
        else:
            self.indices = np.arange(len(id_history))

        # Create mappings
        self.user_id_map = {uid: idx for idx, uid in enumerate(id_history['viewer_uid'].unique())}
        self.item_id_map = {iid: idx for idx, iid in enumerate(id_history['clean_video_id'].explode().unique())}
        
        # Create reverse mappings
        self.reverse_user_id_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_id_map = {idx: iid for iid, idx in self.item_id_map.items()}

        self.categories = textual_history['category'].values

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Получаем ID видео
        item_id = self.item_ids[idx]
        
        # Получаем информацию о видео из таблицы video_info
        video_info = self.video_info[self.video_info['rutube_video_id'] == item_id].iloc[0]
        item_text = video_info['title']
        category_idx = video_info['category_id']  # Берем готовый category_id из таблицы
        
        # Получаем текст пользователя
        user_id = self.user_ids[idx]
        user_text = self.user_descriptions[
            self.user_descriptions['viewer_uid'] == user_id
        ]['user_description'].iloc[0]
        
        # Tokenize texts
        item_encoding = self.tokenizer(
            item_text, padding='max_length', truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )
        user_encoding = self.tokenizer(
            user_text, padding='max_length', truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )
        
        item_text_inputs = {key: val.squeeze(0) for key, val in item_encoding.items()}
        user_text_inputs = {key: val.squeeze(0) for key, val in user_encoding.items()}
        
        return item_text_inputs, user_text_inputs, item_id, user_id, category_idx

def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn
    )

def custom_collate_fn(batch):
    item_text_inputs, user_text_inputs, item_ids, user_ids, categories = zip(*batch)
    
    item_text_inputs = {
        key: pad_sequence([x[key] for x in item_text_inputs], batch_first=True) 
        for key in item_text_inputs[0]
    }
    user_text_inputs = {
        key: pad_sequence([x[key] for x in user_text_inputs], batch_first=True) 
        for key in user_text_inputs[0]
    }
    
    item_ids = pad_sequence([x for x in item_ids], batch_first=True, padding_value=0)
    user_ids = torch.stack(user_ids)
    
    categories = torch.stack([cat for cat in categories])  # теперь можно использовать stack, так как все элементы - тензоры
    
    return item_text_inputs, user_text_inputs, item_ids, user_ids, categories