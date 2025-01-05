import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class BuildTrainDataset(Dataset):
    """Dataset for training with unpackable outputs and user-description fallback."""
    def __init__(self, textual_history, user_descriptions, id_history, tokenizer, max_length=128, split='train', val_size=0.1, random_state=42):
        # Split data
        train_idx, val_idx = train_test_split(
            range(len(id_history)), 
            test_size=val_size, 
            random_state=random_state
        )
        
        # Select appropriate indices based on split
        indices = train_idx if split == 'train' else val_idx
        
        self.textual_history = textual_history.iloc[indices].reset_index(drop=True)
        self.id_history = id_history.iloc[indices].reset_index(drop=True)
        self.user_descriptions = user_descriptions
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create mappings for index conversion
        self.user_id_map = {uid: idx for idx, uid in enumerate(id_history['viewer_uid'].unique())}
        self.item_id_map = {iid: idx for idx, iid in enumerate(id_history['clean_video_id'].explode().unique())}
        
        # Create reverse mappings
        self.reverse_user_id_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_id_map = {idx: iid for iid, idx in self.item_id_map.items()}

        # Add special index for unknown values
        self.unknown_item_idx = len(self.item_id_map)
        self.unknown_user_idx = len(self.user_id_map)
        
        # Update reverse mappings
        self.reverse_item_id_map[self.unknown_item_idx] = -1
        self.reverse_user_id_map[self.unknown_user_idx] = -1

    def __len__(self):
        return len(self.id_history)

    def __getitem__(self, idx):
        # Get viewer_uid and related data
        viewer_uid = self.id_history.iloc[idx]['viewer_uid']
        item_text = self.textual_history.iloc[idx]['detailed_view']
        item_ids = self.id_history.iloc[idx]['clean_video_id']
    
        # Convert item_ids to numeric format
        if isinstance(item_ids, (list, np.ndarray)):
            item_ids = [int(x) for x in item_ids]
        else:
            item_ids = [int(item_ids)]

        # Safe index conversion with unknown value handling
        if isinstance(item_ids, (list, np.ndarray)):
            item_ids = [self.item_id_map.get(int(x), self.unknown_item_idx) for x in item_ids]
        else:
            item_ids = [self.item_id_map.get(int(item_ids), self.unknown_item_idx)]
            
        # Safe user_id conversion
        mapped_user_id = self.user_id_map.get(viewer_uid, self.unknown_user_idx)

        # Match viewer_uid with user_descriptions
        user_row = self.user_descriptions[self.user_descriptions['viewer_uid'] == viewer_uid]
        if not user_row.empty:
            user_text = user_row.iloc[0]['user_description']
        else:
            user_text = ""
    
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

        item_ids = torch.tensor(item_ids, dtype=torch.int64)
        user_id = torch.tensor(mapped_user_id, dtype=torch.int64)
    
        return item_text_inputs, user_text_inputs, item_ids, user_id 