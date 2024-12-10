import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized tensors in a batch."""
    item_text_inputs, user_text_inputs, item_ids, user_ids = zip(*batch)
    
    # Padding for text data
    item_text_inputs = {
        key: pad_sequence([x[key] for x in item_text_inputs], batch_first=True) 
        for key in item_text_inputs[0]
    }
    user_text_inputs = {
        key: pad_sequence([x[key] for x in user_text_inputs], batch_first=True) 
        for key in user_text_inputs[0]
    }
    
    # Padding for item_ids
    item_ids = pad_sequence(
        [torch.tensor(x, dtype=torch.int64) for x in item_ids],
        batch_first=True,
        padding_value=0
    )
    
    # Stack user_ids
    user_ids = torch.stack(user_ids)
    
    return item_text_inputs, user_text_inputs, item_ids, user_ids

def create_dataloader(dataset, batch_size=32, shuffle=True):
    """Create DataLoader with custom collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    ) 