import yaml
import torch
import pandas as pd
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
    
    # Create datasets
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
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        metrics = trainer.train_epoch()
        
        print(f"Average Loss: {metrics['loss']:.4f}")
        print(f"Contrastive Loss: {metrics['contrastive_loss']:.4f}")
        print(f"Recommendation Loss: {metrics['recommendation_loss']:.4f}")

if __name__ == "__main__":
    train_model()