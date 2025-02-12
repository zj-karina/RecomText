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
    experiment_name: str,
    dataset_type: str
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
    
    # Получаем конфигурацию для конкретного датасета
    dataset_features = features[dataset_type]
    
    # Обновляем load_col из features
    config['data']['load_col'] = {
        'inter': dataset_features['features']['interaction_features'],
        'item': dataset_features['features'].get('item_features', []),
        'user': dataset_features['features'].get('user_features', [])
    }
    
    # Обновляем numerical_features
    config['data']['numerical_features'] = dataset_features['features']['numerical_features']
    
    # Добавляем маппинг полей
    config['data'].update(dataset_features['field_mapping'])
    
    # Проверяем наличие текстовых полей
    if 'TEXT_FIELDS' in dataset_features['field_mapping']:
        text_fields = dataset_features['field_mapping']['TEXT_FIELDS']
        # Добавляем эмбеддинги в numerical_features
        for field in text_fields:
            emb_features = [f'{field}_emb_{i}' for i in range(384)]  # Размерность BERT
            config['data']['numerical_features'].extend(emb_features)
    
    # Проверяем наличие категориальных признаков
    if 'categorical_features' in dataset_features['features']:
        cat_fields = dataset_features['features']['categorical_features']
        config['data']['token_features'] = cat_fields
    
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
        
        if dataset_type == 'rutube':
            df['rutube_video_id'] = df['rutube_video_id'].apply(lambda x: x.strip('video_'))
        
        # Загружаем конфигурацию признаков
        with open(f'configs/feature_configs/{feature_config}.yaml', 'r') as f:
            feature_config_dict = yaml.safe_load(f)
        
        # Инициализируем препроцессор датасета
        preprocessor = dataset_preprocessor()
        
        # Предобработка данных
        df = preprocessor.preprocess(df, feature_config_dict[dataset_type])
        
        # Сохраняем взаимодействия с явным указанием типов
        if dataset_type == 'rutube':
            inter_df = df[['viewer_uid', 'rutube_video_id', 'timestamp', 'total_watchtime']].copy()
            inter_df = inter_df.rename(columns={
                'viewer_uid': 'user_id',
                'rutube_video_id': 'item_id',
                'total_watchtime': 'rating'
            })
        else:  # lastfm
            inter_df = df[['user_id', 'artist_id', 'timestamp', 'plays']].copy()
            inter_df = inter_df.rename(columns={
                'artist_id': 'item_id',
                'plays': 'rating'
            })
        
        # Убеждаемся, что timestamp присутствует и отсортирован
        if 'timestamp' not in inter_df.columns:
            raise ValueError("timestamp field is required for sequential recommendation")
            
        # Сортируем по времени
        inter_df = inter_df.sort_values('timestamp')
        
        # Создаем директорию для эксперимента
        os.makedirs(f"{output_dir}/{experiment_name}", exist_ok=True)
        
        # Записываем файл с заголовками, содержащими типы
        with open(f'{output_dir}/{experiment_name}/{experiment_name}.inter', 'w', encoding='utf-8') as f:
            # Определяем типы для каждого поля
            header_types = [
                'user_id:token',
                'item_id:token',
                'rating:float',
                'timestamp:float'  # Убеждаемся, что timestamp включен
            ]
            # Записываем заголовок и данные
            f.write('\t'.join(header_types) + '\n')
            inter_df.to_csv(f, sep='\t', index=False, header=False)

        # Обновляем конфигурацию
        config_dict = {
            'data_path': output_dir,
            'checkpoint_dir': f'./ckpts/saved_{experiment_name}',
            'save_dataset': True,
            'load_col': {
                'inter': ['user_id', 'item_id', 'rating', 'timestamp']  # Явно указываем все необходимые поля
            },
            'eval_args': {
                'split': {'RS': [0.8, 0.1, 0.1]},
                'order': 'TO',
                'group_by': 'user',
                'mode': 'full'
            },
            'MAX_ITEM_LIST_LENGTH': 50,
            'ITEM_LIST_LENGTH_FIELD': 'item_length',
            'LIST_SUFFIX': '_list',
            'max_seq_length': 50
        }

        # Генерируем конфиг и запускаем обучение
        config_path = generate_config(
            features=feature_config_dict,
            model_params=model_params,
            output_dir=output_dir,
            experiment_name=experiment_name,
            dataset_type=dataset_type
        )
        
        init_seed(42, True)
        result = run_recbole(
            model=model_params['model'],
            dataset=experiment_name,
            config_file_list=[config_path],
            config_dict=config_dict
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