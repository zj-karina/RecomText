import yaml
import torch
import pandas as pd
import json
import os
from transformers import AutoTokenizer
from data.dataset import BuildTrainDataset, get_dataloader
from models.multimodal_model import MultimodalRecommendationModel
from trainers.trainer import Trainer

def train_model(config_path='configs/config.yaml'):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Prepare data
    textual_history = pd.read_parquet('./data/textual_history.parquet')
    id_history = pd.read_parquet('./data/id_history.parquet')
    user_descriptions = pd.read_parquet('./data/user_descriptions.parquet')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])
    
    # Create training dataset
    train_dataset = BuildTrainDataset(
        textual_history=textual_history,
        user_descriptions=user_descriptions,
        id_history=id_history,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        split='train',
        val_size=config['training'].get('validation_size', 0.1),
        random_state=config['training'].get('random_seed', 42)
    )
    
    # Сохраняем маппинги
    mappings_dir = os.path.join(config['training']['checkpoint_dir'], 'mappings')
    os.makedirs(mappings_dir, exist_ok=True)
    
    with open(os.path.join(mappings_dir, 'item_id_map.json'), 'w', encoding='utf-8') as f:
        json.dump(train_dataset.item_id_map, f, ensure_ascii=False, indent=2)
    
    print(f"Saved item_id_map with {len(train_dataset.item_id_map)} items")
    
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
    
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=config['data']['batch_size']
    )
    val_loader = get_dataloader(
        val_dataset, 
        batch_size=config['data']['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    model = MultimodalRecommendationModel(
        text_model_name=config['model']['text_model_name'],
        user_vocab_size=len(train_dataset.user_id_map),
        items_vocab_size=len(train_dataset.item_id_map),
        id_embed_dim=config['model']['id_embed_dim'],
        text_embed_dim=config['model']['text_embed_dim']
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate'])
    )
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, optimizer, config)
    
    # Training loop
    metrics = trainer.train(config['training']['epochs'])

if __name__ == "__main__":
    train_model()