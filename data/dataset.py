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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        # Get viewer_uid and related data
        viewer_uid = self.id_history.iloc[idx]['viewer_uid']
        item_text = self.textual_history.iloc[idx]['detailed_view']
        item_ids = self.id_history.iloc[idx]['clean_video_id']
        
        # Convert item_ids to numeric format
        if isinstance(item_ids, (list, np.ndarray)):
            item_ids = [int(x) for x in item_ids]
        else:
            item_ids = [int(item_ids)]

        # Convert absolute indices to sequential
        item_ids = [self.item_id_map[str(x)] for x in item_ids]
        mapped_user_id = self.user_id_map[viewer_uid]

        # Match viewer_uid with user_descriptions
        user_row = self.user_descriptions[self.user_descriptions['viewer_uid'] == viewer_uid]
        user_text = user_row.iloc[0]['user_description'] if not user_row.empty else ""
        
        # Combine item texts
        item_text = ' '.join(item_text)
        
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
            torch.tensor(mapped_user_id, dtype=torch.int64)
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