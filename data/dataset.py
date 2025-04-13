import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

class BuildTrainDataset(Dataset):
    """Dataset for training with unpackable outputs and user-description fallback."""
    def __init__(self, textual_history, user_descriptions, id_history, tokenizer, max_length=128, split='train', val_size=0.1, random_state=42, item_id_map=None):
        self.textual_history = textual_history
        self.user_descriptions = user_descriptions
        self.id_history = id_history
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Получаем уникальных пользователей
        unique_users = id_history['viewer_uid'].unique()
        np.random.seed(random_state)
        np.random.shuffle(unique_users)
        split_idx = int(len(unique_users) * (1 - val_size))
        
        # Разделяем пользователей на train и val
        if split == 'train':
            train_users = unique_users[:split_idx]
            self.users = train_users
            print(f"Training on {len(train_users)} users")
        else:  # val
            val_users = unique_users[split_idx:]
            self.users = val_users
            print(f"Validating on {len(val_users)} users")
        
        # Создаем индексы пользователей для итерации
        self.user_indices = np.arange(len(self.users))
        
        # Create mappings
        self.user_id_map = {uid: idx for idx, uid in enumerate(id_history['viewer_uid'].unique())}
        if item_id_map is not None:
            self.item_id_map = item_id_map
        else:
            self.item_id_map = {iid: idx for idx, iid in enumerate(id_history['clean_video_id'].explode().unique())}
        
        # Create reverse mappings
        self.reverse_user_id_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_id_map = {idx: iid for iid, idx in self.item_id_map.items()}

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        # Получаем ID пользователя из списка пользователей для этого сплита
        user_idx = self.user_indices[idx]
        viewer_uid = self.users[user_idx]
        
        # Находим данные для этого пользователя
        user_history_idx = self.id_history[self.id_history['viewer_uid'] == viewer_uid].index
        
        if len(user_history_idx) == 0:
            # Если данных нет, возвращаем заглушку
            print(f"Warning: No history found for user {viewer_uid}")
            return self._get_dummy_item()
        
        # Берем первую запись для этого пользователя
        history_idx = user_history_idx[0]
        
        # Получаем текстовую историю и ID видео
        item_text = self.textual_history.iloc[history_idx]['detailed_view']
        item_ids = self.id_history.iloc[history_idx]['clean_video_id']
        
        # Аугментация данных для тренировочного сета
        if len(item_text) > 0 and isinstance(item_text, str):
            # Случайное удаление слов (с вероятностью 30%)
            if np.random.random() < 0.3:
                words = item_text.split()
                keep_prob = 0.8  # вероятность сохранения слова
                kept_words = [word for word in words if np.random.random() < keep_prob]
                item_text = ' '.join(kept_words) if kept_words else item_text
                
            # Случайное перемешивание частей текста (с вероятностью 20%)
            if np.random.random() < 0.2:
                sentences = item_text.split('.')
                if len(sentences) > 1:
                    np.random.shuffle(sentences)
                    item_text = '.'.join(sentences)
        
        # Convert absolute indices to sequential
        try:
            item_ids = [self.item_id_map[str(x)] for x in item_ids]
        except KeyError as e:
            print(f"KeyError for item_id: {e}")
            # Если ID нет в маппинге, используем заглушку
            item_ids = [0]
            
        mapped_user_id = self.user_id_map[viewer_uid]
        
        # Match viewer_uid with user_descriptions
        user_row = self.user_descriptions[self.user_descriptions['viewer_uid'] == viewer_uid]
        user_text = user_row.iloc[0]['user_description'] if not user_row.empty else ""
        
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
    
        return (
            item_text_inputs,
            user_text_inputs,
            torch.tensor(item_ids, dtype=torch.int64),
            torch.tensor(mapped_user_id, dtype=torch.int64),
        )
    
    def _get_dummy_item(self):
        """Создает заглушку для случаев, когда данные отсутствуют"""
        dummy_text = "нет данных"
        dummy_encoding = self.tokenizer(
            dummy_text, padding='max_length', truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )
        dummy_text_inputs = {key: val.squeeze(0) for key, val in dummy_encoding.items()}
        
        return (
            dummy_text_inputs,
            dummy_text_inputs,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor(0, dtype=torch.int64),
        )

def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn
    )

def custom_collate_fn(batch):
    item_text_inputs, user_text_inputs, item_ids, user_ids = zip(*batch)
    
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
    
    return item_text_inputs, user_text_inputs, item_ids, user_ids