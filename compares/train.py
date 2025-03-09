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
from data.preprocessing.feature_preprocessor import FeaturePreprocessor, get_full_features_config
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

    return config, config_path


def cosine_similarity_faiss(vecs1, vecs2):
    """
    Вычисляет косинусное сходство с использованием Faiss на GPU.
    """
    vecs1 = vecs1.to(torch.float32).contiguous()
    vecs2 = vecs2.to(torch.float32).contiguous()

    # Проверка формы vecs2
    if len(vecs2.shape) > 2:
        print(f"Reshaping vecs2 from {vecs2.shape} to ({vecs2.shape[0] * vecs2.shape[1]}, {vecs2.shape[2]})")
        vecs2 = vecs2.view(-1, vecs2.shape[-1])  # Преобразуем в двумерный массив

    # Проверка формы vecs1
    if len(vecs1.shape) > 2:
        print(f"Reshaping vecs1 from {vecs1.shape} to ({vecs1.shape[0] * vecs1.shape[1]}, {vecs1.shape[2]})")
        vecs1 = vecs1.view(-1, vecs1.shape[-1])  # Преобразуем в двумерный массив
    elif len(vecs1.shape) == 2:
        # Если vecs1 имеет форму [n_users, embedding_dim], расширяем её
        vecs1 = vecs1.unsqueeze(1)  # [n_users, 1, embedding_dim]
        vecs1 = vecs1.expand(-1, vecs2.shape[0] // vecs1.shape[0], -1)  # [n_users, 10, embedding_dim]
        vecs1 = vecs1.reshape(-1, vecs1.shape[-1])  # [n_users * 10, embedding_dim]

    # Создание индекса Faiss
    index = faiss.IndexFlatIP(vecs2.shape[1])  # Используем внутреннее произведение
    res = faiss.StandardGpuResources()  # Используем GPU
    index = faiss.index_cpu_to_gpu(res, 0, index)  # Переносим индекс на GPU

    # Добавление векторов в индекс
    index.add(vecs2.detach().cpu().numpy())

    # Поиск сходства
    k_search = vecs2.shape[0]
    D, _ = index.search(vecs1.detach().cpu().numpy(), k_search)
    # D, _ = index.search(vecs1.detach().cpu().numpy(), vecs2.shape[0])
    return torch.tensor(D, device=vecs1.device)

def evaluate_with_custom_metrics(preprocessor, config, dataset_type, category_info, k=10):
    """
    Evaluate metrics on GPU using a single batch loop to avoid memory issues.
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Загрузка модели и данных
    model_path = get_latest_checkpoint(config['checkpoint_dir'])

    (_, model, dataset, train_data, valid_data, test_data) = load_data_and_model(model_path)
    model.to(device).eval()

    # Получение эмбеддингов айтемов и данных для тестирования
    item_embeddings = model.item_embedding.weight.to(device)
    all_users = test_data.dataset.inter_feat['user_id'].to(device)
    all_items = test_data.dataset.inter_feat['item_id'].to(device)
    test_data.dataset.inter_feat['user_id'] = test_data.dataset.inter_feat['user_id'].to(device)
    test_data.dataset.inter_feat['item_id'] = test_data.dataset.inter_feat['item_id'].to(device)
    
    # Уникальные пользователи для обработки
    unique_users, user_indices = torch.unique(all_users, return_inverse=True)
    # Настройка батчей
    batch_size = config.get('eval_batch_size', 2048)
    results = {'SP@K': 0, 'CCR': 0, 'NDCG': 0}
    count = 0

    sim_threshold_precision = config.get('sim_threshold_precision', 0.89)
    sim_threshold_ndcg = config.get('sim_threshold_ndcg', 0.83)

    print(f"Total users: {len(all_users)}")
    original_inter_feat = test_data.dataset.inter_feat
    # Обработка батчей
    for i in tqdm(range(0, len(all_users), batch_size), total=len(all_users) // batch_size, desc="Processing batches"):
        batch_users = all_users[i : i + batch_size].to(device)
        batch_items = all_items[i : i + batch_size].to(device)
        
        # Получение предсказаний модели
        batch_user_set = set(batch_users.cpu().numpy())  # Уникальные пользователи из батча
        batch_item_set = set(batch_items.cpu().numpy())  # Уникальные айтемы из батча

        # Фильтруем dataset по user_id и item_id, которые есть в батче
        mask_users = torch.isin(test_data.dataset.inter_feat['user_id'], batch_users)
        mask_items = torch.isin(test_data.dataset.inter_feat['item_id'], batch_items)
        mask = mask_users & mask_items

        filtered_inter_feat = test_data.dataset.inter_feat[mask]
        
        # Подменяем inter_feat на отфильтрованный
        test_data.dataset.inter_feat = filtered_inter_feat

        with torch.no_grad():
            batch_scores = full_sort_scores(batch_users, model, test_data, device=device)

        # Перенос предсказаний на CPU для дальнейшей обработки
        batch_scores_cpu = batch_scores.detach().cpu()
        print(f"Batch scores shape: {batch_scores_cpu.shape}")

        # Векторизованное вычисление метрик
        sp, ccr, ndcg = compute_metrics_for_batch(
            preprocessor, dataset_type, batch_scores_cpu, batch_items, item_embeddings, category_info, device, sim_threshold_precision, sim_threshold_ndcg, k=k
        )
        
        # Обновление результатов
        results['SP@K'] += sp
        results['CCR'] += ccr
        results['NDCG'] += ndcg
        count += batch_scores_cpu.shape[0]

        test_data.dataset.inter_feat = original_inter_feat

        # Очистка памяти
        del batch_scores, batch_scores_cpu
        torch.cuda.empty_cache()
    
    # Нормализация результатов
    for key in results:
        results[key] /= count if count > 0 else 1

    return results

def compute_metrics_for_batch(preprocessor, dataset_type, batch_scores, batch_items, item_embeddings, category_info, device, sim_threshold_precision, sim_threshold_ndcg, k=10):
    """
    Векторизованное вычисление метрик для всего батча.
    """
    # Получение топ-k рекомендаций
    pred_items = batch_scores.argsort(dim=1, descending=True)[:, :k]
    ground_truth_items = batch_items.unsqueeze(1).cpu().numpy()

    # Векторизованное вычисление метрик
    sp = semantic_precision_at_k_batch(pred_items, ground_truth_items, item_embeddings, device, sim_threshold_precision)
    ccr, ndcg = 0, 0
    if dataset_type == 'rutube':
        ccr = cross_category_relevance_batch(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device, k=k)
        ndcg = contextual_ndcg_batch(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device, sim_threshold_ndcg)

    return sp, ccr, ndcg

def semantic_precision_at_k_batch(pred_items, ground_truth_items, item_embeddings, device, sim_threshold_precision):
    """
    Векторизованная семантическая точность.
    """
    pred_items = pred_items.to(device)
    ground_truth_items = torch.tensor(ground_truth_items, device=device)

    # Векторизованное вычисление сходства
    gt_vectors = item_embeddings[ground_truth_items]
    rec_vectors = item_embeddings[pred_items]
    similarity_matrix = cosine_similarity_faiss(gt_vectors, rec_vectors)

    # Подсчет успешных рекомендаций
    successful_recs = (similarity_matrix >= sim_threshold_precision).sum(dim=1)
    return successful_recs.float().mean().item()

def cross_category_relevance_batch(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device, k=10):
    """
    Векторизованная кросс-категорийная релевантность.
    """
    pred_items = pred_items.to(device)
    ground_truth_items = torch.tensor(ground_truth_items, device=device)

    # Векторизованное вычисление сходства
    gt_vectors = item_embeddings[ground_truth_items]
    rec_vectors = item_embeddings[pred_items]
    similarity_matrix = cosine_similarity_faiss(gt_vectors, rec_vectors)

    # Подсчет релевантности
    same_category_count = 0
    cross_category_success = 0

    for i, gt_item in enumerate(ground_truth_items):
        orig_gt_item = preprocessor.item_encoder.inverse_transform([gt_item.cpu().item()])[0]
        gt_category = category_info.get(str(orig_gt_item), {}).get('category_id')

        for j, rec_item in enumerate(pred_items[i]):
            orig_rec_item = preprocessor.item_encoder.inverse_transform([rec_item.cpu().item()])[0]
            rec_category = category_info.get(str(orig_rec_item), {}).get('category_id')

            similarity = similarity_matrix[i, j].item()
            if rec_category == gt_category:
                same_category_count += 1
            elif similarity >= 0.8:
                cross_category_success += 1

    # Вычисление итоговой метрики
    sp_k = semantic_precision_at_k_batch(pred_items, ground_truth_items, item_embeddings, device, k)
    category_diversity = 1 - (same_category_count / (k * len(ground_truth_items)))
    return 0.7 * sp_k + 0.3 * category_diversity

def contextual_ndcg_batch(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device, sim_threshold_ndcg):
    relevances = []

    # Приводим ground_truth_items к тензору для корректной работы
    ground_truth_items = torch.tensor(ground_truth_items, device=device)
    pred_items = pred_items.to(device)

    # Векторизованное вычисление сходства
    gt_vectors = item_embeddings[ground_truth_items]
    rec_vectors = item_embeddings[pred_items]
    similarity_matrix = cosine_similarity_faiss(gt_vectors, rec_vectors)

    # Вычисление релевантности
    for i, gt_item in enumerate(ground_truth_items):
        # Теперь gt_item — это тензор, и можно вызвать .cpu().item()
        orig_gt_item = preprocessor.item_encoder.inverse_transform([gt_item.cpu().item()])[0]
        gt_category = category_info.get(str(orig_gt_item), {}).get('category_id')

        for j, rec_item in enumerate(pred_items[i]):
            orig_rec_item = preprocessor.item_encoder.inverse_transform([rec_item.cpu().item()])[0]
            rec_category = category_info.get(str(orig_rec_item), {}).get('category_id')

            similarity = similarity_matrix[i, j].item()
            if rec_category == gt_category and similarity >= sim_threshold_ndcg:
                relevances.append(3)
            elif rec_category != gt_category and similarity >= sim_threshold_ndcg:
                relevances.append(2)
            elif rec_category == gt_category and similarity < sim_threshold_ndcg:
                relevances.append(1)
            else:
                relevances.append(0)

    # Вычисление DCG и IDCG
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances, 1))
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_relevances, 1))
    
    return dcg / idcg if idcg > 0 else 0

def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """Находит самый последний (по времени модификации) чекпоинт в указанной директории."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime) # Выбираем самый последний по дате изменения
    return latest_checkpoint


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
        df_videos_map = None
        if dataset_type == 'rutube':
            inter_df = df[['viewer_uid', 'rutube_video_id', 'timestamp', 'total_watchtime']].copy()
            inter_df = inter_df.rename(columns={
                'viewer_uid': 'user_id',
                'rutube_video_id': 'item_id',
                'total_watchtime': 'rating'
            })
            df_videos = pd.read_parquet("~/RecomText/data/video_info.parquet")
            df_videos_map = df_videos.set_index('clean_video_id').to_dict(orient='index')

            textual_history = pd.read_parquet('~/RecomText/data/textual_history.parquet')
            inter_df['title'] = textual_history['detailed_view']
            # id_history = pd.read_parquet('./data/id_history.parquet')
            # user_descriptions = pd.read_parquet('./data/user_descriptions.parquet')

        else:  # lastfm
            inter_df = df[['user_id', 'artist_id', 'timestamp', 'plays']].copy()
            # inter_df = df[['user_id', 'artist_id', 'plays']].copy()
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
                'item_title:token_seq', # Текстовое поле (название товара)
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
                'inter': ['user_id', 'item_id', 'title', 'rating', 'timestamp']  # Явно указываем все необходимые поля
            },
           'eval_args': {
                'split': {'RS': [0.8, 0.1, 0.1]},
                'order': 'TO',
                'group_by': 'user',
                'mode': 'full'
            },
            'TEXT_FIELD': 'title',  # Указываем, что это текстовая фича
            'MAX_ITEM_LIST_LENGTH': 50,
            'ITEM_LIST_LENGTH_FIELD': 'item_length',
            'LIST_SUFFIX': '_list',
            'max_seq_length': 50
        }

        # Генерируем конфиг и запускаем обучение
        config, config_path = generate_config(
            features=feature_config_dict,
            model_params=model_params,
            output_dir=output_dir,
            experiment_name=experiment_name,
            dataset_type=dataset_type
        )
        print(f"CONFIG DICT = {config_dict}")
        init_seed(42, True)
        result = run_recbole(
            model=model_params['model'],
            dataset=experiment_name,
            config_file_list=[config_path],
            config_dict=config_dict
        )
        logger.info(f"RecBole config: {config}")
        logger.info(f"Dataset columns: {inter_df.columns.tolist()}")

        # Запускаем кастомные метрики
        custom_metrics = evaluate_with_custom_metrics(preprocessor, config, dataset_type, df_videos_map)
        logger.info(f"Custom Metrics: {custom_metrics}")

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