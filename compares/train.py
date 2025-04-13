import os

import pandas as pd
import numpy as np
import os
import logging
import glob
import math
import yaml
import torch
import copy
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, List
from recbole.quick_start import run_recbole
from recbole.utils import init_seed
from data.preprocessing.feature_preprocessor import FeaturePreprocessor
from data.preprocessing.rutube_preprocessor import RutubePreprocessor
from data.preprocessing.lastfm_preprocessor import LastFMPreprocessor
from utils.logger import setup_logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils import init_seed
from recbole.utils.case_study import full_sort_scores
from typing import Optional, Dict, List
from sklearn.metrics.pairwise import cosine_similarity
import torch
import math
import faiss
import numpy as np
from tqdm import tqdm
from recbole.utils.case_study import full_sort_scores
from sklearn.preprocessing import normalize
import warnings
from collections.abc import Mapping
import copy

warnings.filterwarnings("ignore", category=UserWarning)

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
    
    # Проверяем и логируем GPU
    gpu_id = config.get('gpu_id', 0)
    use_gpu = config.get('use_gpu', False)
    config['device'] = f'cuda:{gpu_id}' if use_gpu else 'cpu'
    
    # Получаем конфигурацию для конкретного датасета
    dataset_features = features[dataset_type]

    # Обновляем numerical_features
    print(f"DATASET FEATURES: {dataset_features}")
    if 'numerical_features' in dataset_features['features']:
        config['data']['numerical_features'] = dataset_features['features']['numerical_features']

    if 'embedding_sequence_fields' in dataset_features['features']:
        config['data']['embedding_sequence_fields'] = dataset_features['features']['embedding_sequence_fields']
    
    # Добавляем маппинг полей
    config['data'].update(dataset_features['field_mapping'])
    
    # Проверяем наличие категориальных признаков
    if 'categorical_features' in dataset_features['features']:
        cat_fields = dataset_features['features']['categorical_features']
        config['data']['token_features'] = cat_fields
    
    # Добавляем параметры модели
    config.update(model_params)

    # Добавляем пути к данным и чекпоинтам
    config['data_path'] = output_dir
    config['experiment_name'] = experiment_name
    config['checkpoint_dir'] = f'./ckpts/saved_{experiment_name}'

    # Сохраняем итоговый конфиг
    os.makedirs(f"{output_dir}/{experiment_name}", exist_ok=True)
    config_path = f'{output_dir}/{experiment_name}/{experiment_name}.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config, config_path 


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
        base_config_path = os.path.join(os.path.dirname(__file__), 'configs', 'base_config.yaml')
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        use_gpu = base_config.get('use_gpu', False)
        gpu_id = base_config.get('gpu_id', 0)
        
        logger.info(f"Using GPU ID from config: {gpu_id}")
        torch.cuda.set_device(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Загружаем данные
        df = pd.read_csv(input_file)
        df = df.sample(100_000)
        # Получаем препроцессор для конкретного датасета
        dataset_preprocessor = DATASET_PREPROCESSORS.get(dataset_type)
        if dataset_preprocessor is None:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        if dataset_type == 'rutube':
            df['rutube_video_id'] = df['rutube_video_id'].apply(lambda x: x.strip('video_'))

        # Загружаем конфигурацию признаков
        with open(f'configs/feature_configs/{feature_config}.yaml', 'r') as f:
            feature_config_dict = yaml.safe_load(f)
        
        # Load textual data before preprocessing
        if dataset_type == 'rutube':
            month_dict = {
                1: 'января',
                2: 'февраля',
                3: 'марта',
                4: 'апреля',
                5: 'мая',
                6: 'июня',
                7: 'июля',
                8: 'августа',
                9: 'сентября',
                10: 'октября',
                11: 'ноября',
                12: 'декабря'
            }
            
            # Форматируем дату
            df['formatted_date'] = df['day'].astype(str) + ' ' + df['month'].map(month_dict)
        
            # Определяем тип клиента
            df['client_type'] = df['ua_client_type'].apply(
                lambda x: 'браузере' if x == 'browser' else 'приложении' if x == 'mobile app' else x
            )
            
            # Создаем описание просмотра
            def create_view_description(row):
                parts = []
        
                if pd.notna(row['title']):
                    parts.append('Название видео: ' + str(row['title']))
                    
                if pd.notna(row['category']):
                    parts.append('категории ' + str(row['category']))
                    
                if pd.notna(row['client_type']):
                    parts.append(f"просмотрено в {row['client_type']}")
                    
                if pd.notna(row['ua_os']):
                    parts.append(f"ОС {row['ua_os']}")
                    
                if pd.notna(row['formatted_date']):
                    parts.append(str(row['formatted_date']))
                    
                # Сохраняем категорию отдельно
                category = row.get('category', 'unknown')
                    
                return ' '.join(parts) if parts else None, category
            
            # Добавляем подробности о просмотре и категорию
            df[['detailed_view', 'category']] = df.apply(create_view_description, axis=1, result_type='expand')

        output_path = f"{output_dir}/{experiment_name}/"
        os.makedirs(output_path, exist_ok=True)
        device = f'cuda:{gpu_id}' if use_gpu else 'cpu'
        print(output_dir)
        print(experiment_name)
        preprocessor = dataset_preprocessor(device=device, output_dir=output_dir,
            experiment_name=experiment_name)

        logger.info(f"Start preprocessing, available features: {df.columns.tolist()}")
        
        # Предобработка данных
        df = preprocessor.preprocess(
            df=df,
            feature_config=feature_config_dict[dataset_type],
            model_type=model_params['model'].lower(),
        )
        logger.info(f"After preprocessing, available features: {df.columns.tolist()}")

        # Ищем все embedding_list-поля
        embedding_list_cols = [col for col in df.columns if col.endswith('_embedding')]
        emb_list_idx_fields = [col for col in df.columns if col.endswith('_idx')]
        emb_list_idx_fields_header = [f'{col}:token' for col in emb_list_idx_fields]
        
        if dataset_type == 'rutube':
            inter_cols = ['viewer_uid', 'rutube_video_id', 'timestamp', 'total_watchtime']
            # на самом деле не все эмбеддинги будут относится с inter
            # они могут быть связаны только к item или user и это надо придумать как делать автоматически
            inter_cols.extend(emb_list_idx_fields)
            print(f"HERE1 = {emb_list_idx_fields}")
            
            inter_df = df[inter_cols].copy()
            inter_df = inter_df.rename(columns={
                'viewer_uid': 'user_id',
                'rutube_video_id': 'item_id',
                'total_watchtime': 'rating'
            })
            inter_cols = list(inter_df.columns)
        
            df_videos = pd.read_parquet("~/RecomText/data/video_info.parquet")
            df_videos_map = df_videos.set_index('clean_video_id').to_dict(orient='index')

            item_df = df[['rutube_video_id', 'category']].copy().rename(columns={'rutube_video_id': 'item_id'})
            item_header = ['item_id:token', 'category:token']

            with open(f'{output_path}{experiment_name}.item', 'w', encoding='utf-8') as f:
                f.write('\t'.join(item_header) + '\n')
                item_df.to_csv(f, sep='\t', index=False, header=False)
    
            logger.info(f"Saved item file: {output_path}{experiment_name}.item with headers: {item_header}")

            user_df = df[['viewer_uid', 'sex', 'region']].copy().rename(columns={'viewer_uid': 'user_id'})
            user_header = ['user_id:token', 'sex:token', 'region:token']
    
            with open(f'{output_path}{experiment_name}.user', 'w', encoding='utf-8') as f:
                f.write('\t'.join(user_header) + '\n')
                user_df.to_csv(f, sep='\t', index=False, header=False)
    
            logger.info(f"Saved item file: {output_path}{experiment_name}.user with headers: {user_header}")
                
        else:  # lastfm dataset
            inter_df = df[['user_id', 'artist_id', 'timestamp', 'plays']].copy()
            inter_df = inter_df.rename(columns={
                'artist_id': 'item_id',
                'plays': 'rating'
            })
        
        # Убеждаемся, что timestamp присутствует и отсортирован
        if 'timestamp' not in inter_df.columns:
            raise ValueError("timestamp field is required for sequential recommendation")
        
        inter_df = inter_df.sort_values('timestamp')
        inter_header = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'] # для записи в .inter
        # на самом деле не все эмбеддинги будут относится с inter
        # они могут быть связаны только к item или user и это надо придумать как делать автоматически
        inter_header.extend(emb_list_idx_fields_header)
                
        with open(f'{output_path}{experiment_name}.inter', 'w', encoding='utf-8') as f:
            f.write('\t'.join(inter_header) + '\n')
            inter_df.to_csv(f, sep='\t', index=False, header=False)
        
        logger.info(f"Saved interaction file: {output_path}{experiment_name}.inter with headers: {inter_header}")

        # Обновляем конфигурацию
        base_config_dict = {
            'data_path': output_dir,
            'checkpoint_dir': f'./ckpts/saved_{experiment_name}',
            'save_dataset': True,
            'load_col': {
                'inter': inter_cols,
                'item': ['item_id', 'category'],
                'user': ['user_id', 'sex', 'region'],
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
            'max_seq_length': 50,
            'cuda_id': gpu_id,
            'pytorch_gpu_id': gpu_id,
            'gpu_id': gpu_id,
            'use_gpu': use_gpu,
            'device': device,

            
            # Add text feature configuration
            'feature_fusion': True,
            'embedding_size': 384,
            'numerical_projection_dropout': 0.1,
            
            # Add data configuration for embeddings
            'data': {
                # 'numerical_features': inter_embedding_cols + item_embedding_cols + user_embedding_cols,
                'token_features': feature_config_dict[dataset_type]['features'].get('categorical_features', []),
                'embedding_sequence_fields': emb_list_idx_fields,
                # 'text_fields': feature_config_dict[dataset_type]['features'].get('text_fields', [])
            }
        }

        logger.info("Start generate_config()...")
        # Генерируем конфиг и запускаем обучение
        config, config_path = generate_config(
            features=feature_config_dict,
            model_params=model_params,
            output_dir=output_dir,
            experiment_name=experiment_name,
            dataset_type=dataset_type
        )
        
        # Обновляем только базовые параметры, сохраняя конфигурацию признаков        
        def deep_update(d: dict, u: dict) -> dict:
            """
            Обновляет словарь `d` значениями из `u` рекурсивно.
            Если ключ уже существует и оба значения — словари, то делает глубокое обновление.
            Иначе — перезаписывает значение.
            """
            d = copy.deepcopy(d)
            for k, v in u.items():
                if isinstance(v, Mapping) and isinstance(d.get(k), Mapping):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        config, config_path = generate_config(
            features=feature_config_dict,
            model_params=model_params,
            output_dir=output_dir,
            experiment_name=experiment_name,
            dataset_type=dataset_type
        )
        
        # Обновляем конфигурацию только один раз
        config = deep_update(config, base_config_dict)
        
        logger.info("Configuration updated and saved")
        
        logger.info(f"Final config: {config}")
        logger.info(f"Using GPU ID: {config['gpu_id']}")
        logger.info(f"Numerical features: {config['data']['numerical_features']}")
        logger.info(f"Token features: {config['data'].get('token_features', [])}")
        logger.info(f"Text fields: {config['data'].get('text_fields', [])}")
        logger.info(f"Interaction features: {inter_df.columns.tolist()}")

        logger.info("Start run_recbole()...")
        
        init_seed(42, True)
        print(f"MODEL {model_params['model']}")
        print(f"config['data_path'] = {config['data_path']}")

        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        print(f"""{os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}""")


        def run_custom_model(config, config_path, experiment_name):
            from recbole.config import Config
            from recbole.data import create_dataset, data_preparation
            from recbole.trainer import Trainer
            from recbole.utils import init_seed
        
            # 1. Создаём конфиг с оригинальной моделью
            config = Config(
                model=config['model'],  # 'BERT4Rec' или 'SASRec'
                dataset=experiment_name,
                config_file_list=[config_path],
                config_dict=config
            )
        
            # 2. Seed
            init_seed(config['seed'], config['reproducibility'])
        
            # 3. Создание оригинального датасета на основе имени модели
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
        
            # 4. Подмена на кастомную модель из конфигурации
            model_name = config['model']
            if model_name == 'BERT4Rec':
                from models.enhanced_bert4rec import EnhancedBERT4Rec
                model_class = EnhancedBERT4Rec
            elif model_name == 'SASRec':
                from models.enhanced_sasrec import EnhancedSASRec
                model_class = EnhancedSASRec
            else:
                raise ValueError(f"Model {model_name} is not recognized!")
        
            # 5. Создание модели
            model = model_class(config, train_data.dataset).to(config['device'])
        
            # 6. Тренер
            trainer = Trainer(config, model)
            best_valid_score, best_valid_result = trainer.fit(train_data,
                                                              valid_data,
                                                              saved=True,
                                                              show_progress=config["show_progress"])
        
            # 7. Тест
            test_result = trainer.evaluate(test_data)
        
            return {
                'best_valid_score': best_valid_score,
                'best_valid_result': best_valid_result,
                'test_result': test_result
            }
            
        # Если используем кастомную модель
        if feature_config != 'id_only':            
            print(f"NAME = {config['model']}")
            print(f"=-=-==-=-=-=--=")
            
            print(f"config['data_path'] = {config['data_path']}")
            # Используем наш прямой запуск
            print(f"CUSTOMS MODEL")
            result = run_custom_model(
                config=config,
                config_path=config_path,
                experiment_name=experiment_name
            )
        else:
            result = run_recbole(
                model=model_params["model"],
                dataset=experiment_name,
                config_file_list=[config_path],
                config_dict=config
            )

        logger.info(f"RecBole config: {config}")
        logger.info(f"Dataset columns: {inter_df.columns.tolist()}")

        # Запускаем кастомные метрики
        # custom_metrics = evaluate_with_custom_metrics(preprocessor, config, dataset_type, df_videos_map)
        # logger.info(f"Custom Metrics: {custom_metrics}")

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