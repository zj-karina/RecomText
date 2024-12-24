import torch
import torch.nn as nn
from torch import optim
import yaml
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Set

from data.dataset import BuildTrainDataset
from data.dataloader import create_dataloader
from models.multimodal import MultimodalRecommendationModel
from utils.mappings import save_mappings
from utils.custom_metrics import RecommendationMetrics, evaluate_recommendations

def train_model(config_path='config/config.yaml'):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Prepare data
    textual_history = pd.read_parquet('textual_history.parquet')
    id_history = pd.read_parquet('id_history.parquet')
    user_descriptions = pd.read_parquet('user_descriptions.parquet')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])
    
    # Create dataset and dataloader
    dataset = BuildTrainDataset(
        textual_history, 
        user_descriptions, 
        id_history, 
        tokenizer, 
        max_length=config['data']['max_length']
    )
    dataloader = create_dataloader(
        dataset, 
        batch_size=config['training']['batch_size']
    )
    
    # Save mappings
    save_mappings(dataset)
    
    # Initialize model
    user_vocab_size = len(dataset.user_id_map)
    items_vocab_size = len(dataset.item_id_map)
    
    model = MultimodalRecommendationModel(
        text_model_name=config['model']['text_model_name'],
        user_vocab_size=user_vocab_size,
        items_vocab_size=items_vocab_size,
        id_embed_dim=config['model']['id_embed_dim'],
        text_embed_dim=config['model']['text_embed_dim']
    )
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss functions
    contrastive_loss_fn = nn.CosineEmbeddingLoss()
    recommendation_loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate']
    )
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}"):
            item_text_inputs, user_text_inputs, item_ids, user_ids = batch
            
            # Move to device
            item_text_inputs = {k: v.to(device) for k, v in item_text_inputs.items()}
            user_text_inputs = {k: v.to(device) for k, v in user_text_inputs.items()}
            item_ids = item_ids.to(device)
            user_ids = user_ids.to(device)

            # Forward pass
            items_embeddings, user_embeddings = model(
                item_text_inputs, 
                user_text_inputs, 
                item_ids, 
                user_ids
            )

            # Losses
            logits = torch.matmul(user_embeddings, items_embeddings.T)
            labels = torch.arange(len(user_embeddings)).to(device)
            recommendation_loss = recommendation_loss_fn(logits, labels)

            positive_labels = torch.ones(items_embeddings.size(0)).to(device)
            negative_labels = -torch.ones(items_embeddings.size(0)).to(device)
            
            contrastive_items_loss = contrastive_loss_fn(
                items_embeddings, 
                items_embeddings, 
                positive_labels
            )
            contrastive_users_loss = contrastive_loss_fn(
                user_embeddings, 
                user_embeddings, 
                negative_labels
            )
            contrastive_loss = contrastive_items_loss + contrastive_users_loss

            # Total loss
            loss = contrastive_loss + config['training']['lambda_rec'] * recommendation_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")

def evaluate_model(model: torch.nn.Module, 
                  dataloader: torch.utils.data.DataLoader,
                  categories_map: Dict[str, str],
                  embeddings_dict: Dict[str, np.ndarray],
                  all_categories: Set[str],
                  device: torch.device) -> Tuple[Dict[str, float], float]:
    """
    Оценивает модель с использованием кастомных метрик
    """
    model.eval()
    metrics = RecommendationMetrics(categories_map)
    
    all_predictions = []
    all_ground_truth = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            item_text_inputs, user_text_inputs, item_ids, user_ids = batch
            
            # Перемещаем данные на устройство
            item_text_inputs = {k: v.to(device) for k, v in item_text_inputs.items()}
            user_text_inputs = {k: v.to(device) for k, v in user_text_inputs.items()}
            item_ids = item_ids.to(device)
            user_ids = user_ids.to(device)
            
            # Forward pass
            items_embeddings, user_embeddings = model(
                item_text_inputs, 
                user_text_inputs, 
                item_ids, 
                user_ids
            )
            
            # Получаем предсказания
            similarities = torch.matmul(user_embeddings, items_embeddings.T)
            _, top_indices = torch.topk(similarities, k=20, dim=1)
            
            # Преобразуем индексы в video_ids
            batch_predictions = []
            batch_ground_truth = []
            
            for i, indices in enumerate(top_indices):
                pred_videos = [dataset.reverse_item_id_map[idx.item()] for idx in indices]
                true_videos = [dataset.reverse_item_id_map[idx.item()] for idx in item_ids[i]]
                
                batch_predictions.append(pred_videos)
                batch_ground_truth.append(true_videos)
            
            all_predictions.extend(batch_predictions)
            all_ground_truth.extend(batch_ground_truth)
            
            num_batches += 1
    
    # Вычисляем метрики
    evaluation_results = evaluate_recommendations(
        metrics=metrics,
        model_predictions=all_predictions,
        ground_truth=all_ground_truth,
        embeddings_dict=embeddings_dict,
        all_categories=all_categories,
        k_values=[5, 10, 20]
    )
    
    return evaluation_results, total_loss / num_batches

if __name__ == "__main__":
    train_model() 