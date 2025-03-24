import yaml
import torch
import pandas as pd
import json
import os
import numpy as np
import copy
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from transformers import AutoTokenizer
from data.dataset import BuildTrainDataset, get_dataloader
from models.multimodal_model import MultimodalRecommendationModel
from trainers.trainer import Trainer
from datetime import datetime

def objective(params, config_path='configs/config.yaml'):
    """
    Hyperopt objective function for hyperparameter optimization.
    """
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    
    config = copy.deepcopy(base_config)
    
    # Set parameters from hyperopt search
    config['training']['learning_rate'] = params['learning_rate']
    config['training']['lambda_rec'] = params['lambda_rec']
    config['metrics']['sim_threshold_precision'] = params['sim_threshold_precision']
    config['metrics']['sim_threshold_ndcg'] = params['sim_threshold_ndcg']
    config['inference']['top_k'] = int(params['top_k'])
    
    # Set data fraction to 0.1 (10%) for faster optimization
    config['training']['data_fraction'] = 0.1
    
    # Only run for 1 epoch during optimization
    config['training']['epochs'] = 1
    
    # Create unique directory for this trial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = f"hyperopt_trial_{timestamp}"
    config['training']['checkpoint_dir'] = os.path.join("hyperopt_checkpoints", trial_dir)
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Access global dataloader variables
    global train_loader, val_loader, train_dataset
    
    # Initialize model with the dataset vocab sizes
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
    
    # Create trainer and run training
    trainer = Trainer(model, train_loader, val_loader, optimizer, config)
    trainer.train(config['training']['epochs'])
    
    # Get validation metrics
    val_metrics = trainer.validate()
    contextual_ndcg = val_metrics.get('contextual_ndcg', 0.0)
    
    # Store all metrics for this trial
    result = {
        'learning_rate': params['learning_rate'],
        'lambda_rec': params['lambda_rec'],
        'sim_threshold_precision': params['sim_threshold_precision'],
        'sim_threshold_ndcg': params['sim_threshold_ndcg'],
        'top_k': int(params['top_k']),
        'contextual_ndcg': contextual_ndcg,
        'semantic_precision@k': val_metrics.get('semantic_precision@k', 0.0),
        'cross_category_relevance': val_metrics.get('cross_category_relevance', 0.0),
        'precision@k': val_metrics.get('precision@k', 0.0),
        'recall@k': val_metrics.get('recall@k', 0.0),
        'ndcg@k': val_metrics.get('ndcg@k', 0.0),
        'mrr@k': val_metrics.get('mrr@k', 0.0)
    }
    
    print(f"\nTrial completed with contextual_ndcg = {contextual_ndcg:.4f}")
    print(f"Parameters: {params}")
    
    # Add to global results storage
    global trials_results
    trials_results.append(result)
    
    # Hyperopt minimizes, so return negative NDCG
    return {'loss': -contextual_ndcg, 'status': STATUS_OK}

def run_hyperopt_optimization(n_trials=20, config_path='configs/config.yaml'):
    """
    Run hyperopt hyperparameter optimization.
    """
    # Load base configuration
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    
    # Define hyperparameter search space
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-4)),
        'lambda_rec': hp.uniform('lambda_rec', 0.1, 0.5),
        'sim_threshold_precision': hp.uniform('sim_threshold_precision', 0.7, 0.95),
        'sim_threshold_ndcg': hp.uniform('sim_threshold_ndcg', 0.7, 0.95),
        'top_k': hp.quniform('top_k', 5, 20, 5)
    }
    
    # Initialize global variables
    global train_loader, val_loader, train_dataset, trials_results
    trials_results = []
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hyperopt_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading datasets and preparing dataloaders...")
    
    # Load all data
    textual_history = pd.read_parquet('./data/textual_history.parquet')
    id_history = pd.read_parquet('./data/id_history.parquet')
    user_descriptions = pd.read_parquet('./data/user_descriptions.parquet')
    
    # Original data size
    print(f"Original data size: {len(textual_history)} records")
    
    # Set random seed for reproducibility
    random_state = base_config['training'].get('random_seed', 42)
    np.random.seed(random_state)
    
    # Sample 10% of unique users
    unique_users = textual_history['viewer_uid'].unique()
    sampled_users = np.random.choice(
        unique_users, 
        size=max(1, int(len(unique_users) * 0.1)),
        replace=False
    )
    
    # Filter data for sampled users only
    textual_history_sample = textual_history[textual_history['viewer_uid'].isin(sampled_users)]
    id_history_sample = id_history[id_history['viewer_uid'].isin(sampled_users)]
    user_descriptions_sample = user_descriptions[user_descriptions['viewer_uid'].isin(sampled_users)]
    
    print(f"Sampled data size: {len(textual_history_sample)} records (approximately 10%)")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_config['model']['text_model_name'])
    
    # Load item ID mapping
    with open('./data/mappings/item_id_map.json', 'r', encoding='utf-8') as f:
        item_id_map = json.load(f)
    
    # Create train dataset
    train_dataset = BuildTrainDataset(
        textual_history=textual_history_sample,
        user_descriptions=user_descriptions_sample,
        id_history=id_history_sample,
        tokenizer=tokenizer,
        max_length=base_config['data']['max_length'],
        split='train',
        val_size=base_config['training'].get('validation_size', 0.1),
        random_state=random_state,
        item_id_map=item_id_map
    )
    
    # Create validation dataset
    val_dataset = BuildTrainDataset(
        textual_history=textual_history_sample,
        user_descriptions=user_descriptions_sample,
        id_history=id_history_sample,
        tokenizer=tokenizer,
        max_length=base_config['data']['max_length'],
        split='val',
        val_size=base_config['training'].get('validation_size', 0.1),
        random_state=random_state,
        item_id_map=item_id_map
    )
    
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=base_config['data']['batch_size']
    )
    val_loader = get_dataloader(
        val_dataset, 
        batch_size=base_config['data']['batch_size'],
        shuffle=False
    )
    
    # Initialize trials object
    trials = Trials()
    
    # Run optimization
    print(f"Starting hyperopt optimization with {n_trials} trials using 10% of data...")
    best = fmin(
        fn=lambda params: objective(params, config_path),
        space=space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials
    )
    
    # Process best parameters
    best_params = {k: v for k, v in best.items()}
    if 'top_k' in best_params:
        best_params['top_k'] = int(best_params['top_k'])
    
    # Print best parameters
    print("\nBest parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # Get best metric value
    best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
    best_contextual_ndcg = -trials.trials[best_trial_idx]['result']['loss']
    print(f"Best contextual_ndcg: {best_contextual_ndcg:.4f}")
    
    # Save all trial results
    results_df = pd.DataFrame(trials_results)
    results_path = os.path.join(results_dir, 'hyperopt_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"All trial results saved to {results_path}")
    
    # Create optimized config with best parameters
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update config with best parameters
    config['training']['learning_rate'] = best_params['learning_rate']
    config['training']['lambda_rec'] = best_params['lambda_rec']
    config['metrics']['sim_threshold_precision'] = best_params['sim_threshold_precision']
    config['metrics']['sim_threshold_ndcg'] = best_params['sim_threshold_ndcg']
    config['inference']['top_k'] = best_params['top_k']
    
    # Save optimized config
    optimized_config_path = os.path.join(results_dir, 'optimized_config.yaml')
    with open(optimized_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Also save to configs directory
    with open('configs/optimized_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nOptimized configuration saved to {optimized_config_path}")
    print(f"and to configs/optimized_config.yaml")
    
    return best_params, trials

if __name__ == "__main__":
    run_hyperopt_optimization(n_trials=5)