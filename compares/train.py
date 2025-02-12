import pandas as pd
import numpy as np
import os
import logging
import yaml
import torch
from datetime import datetime
from typing import Optional, Dict, List
from recbole.quick_start import run_recbole
from recbole.utils import init_seed
from data.preprocessing.feature_preprocessor import FeaturePreprocessor, get_full_features_config
from data.preprocessing.rutube_preprocessor import RutubePreprocessor
from data.preprocessing.lastfm_preprocessor import LastFMPreprocessor
from utils.logger import setup_logging

DATASET_PREPROCESSORS = {
    'rutube': RutubePreprocessor,
    'lastfm': LastFMPreprocessor
}

def generate_config(
    features: Dict,
    model_params: Dict,
    output_dir: str,
    experiment_name: str
) -> str:
    """Generate and save RecBole configuration"""
    # Загружаем базовый конфиг
    base_config_path = os.path.join(
        os.path.dirname(__file__),
        'configs',
        'base_config.yaml'
    )
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Обновляем load_col из features
    config['data']['load_col'] = {
        'inter': features['interaction_features'],
        'item': features.get('item_features', []),
        'user': features.get('user_features', [])
    }
    
    # Обновляем numerical_features
    config['data']['numerical_features'] = features['numerical_features']
    
    # Добавляем параметры модели
    config.update(model_params)
    
    # Добавляем пути к данным и чекпоинтам
    config['data_path'] = output_dir
    config['checkpoint_dir'] = f'./ckpts/saved_{experiment_name}'
    
    # Сохраняем итоговый конфиг
    os.makedirs(f"{output_dir}/{experiment_name}", exist_ok=True)
    config_path = f'{output_dir}/{experiment_name}/{experiment_name}.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def run_experiment(
    input_file: str,
    experiment_name: str,
    model_params: Dict,
    feature_config: str,
    dataset_type: str,
    output_dir: str = 'dataset'
):
    """Основная функция для запуска обучения и оценки."""
    logger = setup_logging()

    try:
        # Загружаем данные
        df = pd.read_csv(input_file)
        
        # Получаем препроцессор для конкретного датасета
        dataset_preprocessor = DATASET_PREPROCESSORS.get(dataset_type)
        if dataset_preprocessor is None:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Инициализируем препроцессор датасета
        preprocessor = dataset_preprocessor()
        
        if feature_config in ['text_and_id', 'full_features']:
            # Загружаем конфигурацию признаков
            with open(f'configs/feature_configs/{feature_config}.yaml', 'r') as f:
                feature_config_dict = yaml.safe_load(f)
            
            # Предобработка данных с учетом специфики датасета
            df = preprocessor.preprocess(df, feature_config_dict)
            
            # Инициализируем препроцессор для фичей
            feature_preprocessor = FeaturePreprocessor()
            
            # Обрабатываем признаки в зависимости от типа датасета
            if dataset_type == 'rutube':
                # Обработка признаков для Rutube
                items_df = df[['item_id', 'title', 'description']].drop_duplicates()
                feature_preprocessor.process_features(
                    items_df,
                    output_dir,
                    experiment_name,
                    'item'
                )
                
                if feature_config == 'full_features':
                    users_df = df[['user_id', 'age', 'sex', 'region']].drop_duplicates()
                    feature_preprocessor.process_features(
                        users_df,
                        output_dir,
                        experiment_name,
                        'user'
                    )
                
                interactions_df = df[['user_id', 'item_id', 'timestamp', 'watch_time', 
                                    'ua_device_type', 'ua_os']].copy()
                
            elif dataset_type == 'lastfm':
                # Обработка признаков для LastFM
                items_df = df[['artist_id', 'artist_name', 'tags']].drop_duplicates()
                items_df = items_df.rename(columns={
                    'artist_id': 'item_id',
                    'artist_name': 'title',
                    'tags': 'description'
                })
                feature_preprocessor.process_features(
                    items_df,
                    output_dir,
                    experiment_name,
                    'item'
                )
                
                if feature_config == 'full_features':
                    users_df = df[['user_id', 'age', 'gender', 'country']].drop_duplicates()
                    users_df = users_df.rename(columns={
                        'gender': 'sex',
                        'country': 'region'
                    })
                    feature_preprocessor.process_features(
                        users_df,
                        output_dir,
                        experiment_name,
                        'user'
                    )
                
                interactions_df = df[['user_id', 'artist_id', 'timestamp', 'play_count']].copy()
                interactions_df = interactions_df.rename(columns={
                    'artist_id': 'item_id',
                    'play_count': 'watch_time'
                })
            
            feature_preprocessor.process_features(
                interactions_df,
                output_dir,
                experiment_name,
                'inter'
            )
            
            features = get_full_features_config()
            
        else:
            # Используем базовую конфигурацию только с ID
            with open(f'configs/feature_configs/{feature_config}.yaml', 'r') as f:
                features = yaml.safe_load(f)['features']
                
            # Базовая предобработка данных через специфичный препроцессор
            df = preprocessor.preprocess(df, features)
            
            # Сохраняем взаимодействия
            if features.get('interaction_features'):
                inter_df = df[features['interaction_features']].copy()
                os.makedirs(f"{output_dir}/{experiment_name}", exist_ok=True)
                inter_df.to_csv(
                    f'{output_dir}/{experiment_name}/{experiment_name}.inter',
                    sep='\t',
                    index=False
                )

        # Генерируем конфиг и запускаем обучение
        config_path = generate_config(
            features=features,
            model_params=model_params,
            output_dir=output_dir,
            experiment_name=experiment_name
        )
        
        init_seed(42, True)
        result = run_recbole(
            model=model_params['model'],
            dataset=experiment_name,
            config_file_list=[config_path],
            config_dict={
                'data_path': output_dir,
                'checkpoint_dir': f'./ckpts/saved_{experiment_name}',
                'save_dataset': True
            }
        )
        
        logger.info(f"Training completed. Model saved in ./ckpts/saved_{experiment_name}")
        return result
        
    except Exception as e:
        logger.error(f"Error in experiment pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train recommendation model')
    parser.add_argument('--model', type=str, required=True, choices=['SASRec', 'BERT4Rec'],
                      help='Model to train')
    parser.add_argument('--feature_config', type=str, required=True,
                      choices=['id_only', 'text_and_id', 'full_features'],
                      help='Feature configuration to use')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input data file')
    parser.add_argument('--dataset_type', type=str, required=True,
                      choices=['rutube', 'lastfm'],
                      help='Type of dataset')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                      help='Output directory')
    parser.add_argument('--experiment_name', type=str, required=True,
                      help='Name of the experiment')
    
    args = parser.parse_args()
    
    # Загружаем параметры модели
    model_config_path = f'configs/model_configs/{args.model.lower()}.yaml'
    with open(model_config_path, 'r') as f:
        model_params = yaml.safe_load(f)
    
    # Запускаем эксперимент
    result = run_experiment(
        input_file=args.input_file,
        experiment_name=args.experiment_name,
        model_params=model_params,
        feature_config=args.feature_config,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir
    ) 