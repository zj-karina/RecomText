import pandas as pd
import numpy as np
import os
import logging
import glob
import math
import yaml
import torch
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

# def extract_embeddings(model, config_path):
#     """Выгружает обученные эмбеддинги пользователей и товаров из модели RecBole."""
#     model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
#     model.eval()

#     user_embeddings = model.user_embedding.weight.detach().cpu().numpy()
#     item_embeddings = model.item_embedding.weight.detach().cpu().numpy()

#     np.save("user_embeddings.npy", user_embeddings)
#     np.save("item_embeddings.npy", item_embeddings)

#     return user_embeddings, item_embeddings


# def contextual_ndcg(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, k=10):
#     """
#     Вычисляет Contextual NDCG с учетом семантической близости и категорий.
#     """
#     sim_threshold_ndcg = 0.8
#     relevances = []
#     for gt_item in ground_truth_items:
#         orig_gt_item = preprocessor.item_encoder.inverse_transform([gt_item])[0]
#         gt_category = category_info.get(str(orig_gt_item), None)['category_id']
#         gt_vector = item_embeddings[gt_item]
#         for rec_item in pred_items[:k]:
#             orig_rec_item = preprocessor.item_encoder.inverse_transform([rec_item])[0]
#             rec_category = category_info.get(str(orig_rec_item), None)['category_id']
#             rec_vector = item_embeddings[rec_item]
#             similarity = cosine_similarity([gt_vector], [rec_vector])[0][0]
#             if rec_category == gt_category and similarity >= sim_threshold_ndcg:
#                 rel = 3
#             elif rec_category != gt_category and similarity >= sim_threshold_ndcg:
#                 rel = 2
#             elif rec_category == gt_category and similarity < sim_threshold_ndcg:
#                 rel = 1
#             else:
#                 rel = 0
#             relevances.append(rel)
            

#     dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances, 1))
#     ideal_relevances = sorted(relevances, reverse=True)
#     idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_relevances, 1))
    
#     return dcg / idcg if idcg > 0 else 0

# def semantic_precision_at_k(pred_items, ground_truth_items, item_embeddings, k=10, threshold=0.7):
#     """Вычисляет SP@K - семантическую точность рекомендаций."""
#     successful_recs = 0
#     for gt_item in ground_truth_items:
#         gt_vector = item_embeddings[gt_item]
#         for rec_item in pred_items[:k]:
#             rec_vector = item_embeddings[rec_item]
#             similarity = cosine_similarity([gt_vector], [rec_vector])[0][0]
#             if similarity >= threshold:
#                 successful_recs += 1
#     return successful_recs / (k * len(ground_truth_items))


# def cross_category_relevance(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, k=10):
#     """Оценивает качество рекомендаций с учетом категорий."""
#     same_category_count = 0
#     cross_category_success = 0
#     for gt_item in ground_truth_items:
#         orig_gt_item = preprocessor.item_encoder.inverse_transform([gt_item])[0]
#         gt_category = category_info.get(str(orig_gt_item), None)['category_id']
#         gt_vector = item_embeddings[gt_item]
#         for rec_item in pred_items[:k]:
#             orig_rec_item = preprocessor.item_encoder.inverse_transform([rec_item])[0]
#             rec_category = category_info.get(str(orig_rec_item), None)['category_id']
#             rec_vector = item_embeddings[rec_item]
#             similarity = cosine_similarity([gt_vector], [rec_vector])[0][0]
#             if rec_category == gt_category:
#                 same_category_count += 1
#             elif similarity >= 0.7:
#                 cross_category_success += 1
#     sp_k = semantic_precision_at_k(pred_items, ground_truth_items, item_embeddings, k)
#     category_diversity = 1 - (same_category_count / k)
#     return 0.7 * sp_k + 0.3 * category_diversity

# def evaluate_with_custom_metrics(preprocessor, config, category_info, k=10):
#     """Запускает кастомные метрики"""
#     model_path = get_latest_checkpoint(config['checkpoint_dir'])
#     (_, model, dataset, train_data, valid_data, test_data) = load_data_and_model(model_path)
#     model.eval()
    
#     item_embeddings = model.item_embedding.weight.detach().cpu().numpy()
#     # item_embeddings = model.item_embedding.weight.detach().to(device)  # Переносим эмбеддинги на GPU

#     all_users = test_data.dataset.inter_feat['user_id'].numpy()
#     all_items = test_data.dataset.inter_feat['item_id'].numpy()
#     unique_users = np.unique(all_users)
    
#     batch_size = config['eval_batch_size']  # Можно подобрать
#     scores_list = []

#     for i in tqdm(range(0, len(unique_users), batch_size), desc="Computing scores"):
#         batch_users = unique_users[i : i + batch_size]

#         with torch.no_grad():
#             batch_scores = full_sort_scores(batch_users, model, test_data, device=torch.device('cuda:0'))

#         scores_list.append(batch_scores)

#     scores_matrix = torch.cat(scores_list, dim=0).detach().cpu().numpy()

#     # Метрики
#     results = {'SP@K': 0, 'CCR': 0, 'NDCG': 0}
#     num_users = len(unique_users)

#     for idx, user_id in tqdm(enumerate(unique_users), total=num_users, desc="Evaluating users"):
#         user_indices = np.where(all_users == user_id)[0]
#         ground_truth_items = all_items[user_indices]

#         # Индексируем правильно: берём строку `idx` (а не `user_indices`)
#         pred_items = scores_matrix[idx].argsort()[-k:]

#         results['SP@K'] += semantic_precision_at_k(pred_items, ground_truth_items, item_embeddings)
#         results['CCR'] += cross_category_relevance(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info)
#         results['NDCG'] += contextual_ndcg(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info)

#     for key in results:
#         results[key] /= num_users

#     return results

import torch
import math
import faiss
import numpy as np
from tqdm import tqdm
from recbole.utils.case_study import full_sort_scores
from sklearn.preprocessing import normalize

def evaluate_with_custom_metrics(preprocessor, config, category_info, k=10):
    """Запускает кастомные метрики, используя GPU"""
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = get_latest_checkpoint(config['checkpoint_dir'])
    (_, model, dataset, train_data, valid_data, test_data) = load_data_and_model(model_path)
    model.to(device).eval()

    item_embeddings = model.item_embedding.weight.to(device)

    all_users = test_data.dataset.inter_feat['user_id'].to(device)
    all_items = test_data.dataset.inter_feat['item_id'].to(device)
    unique_users = torch.unique(all_users)

    batch_size = config.get('eval_batch_size', 512)
    scores_list = []

    test_data.dataset.inter_feat['user_id'] = test_data.dataset.inter_feat['user_id'].to(device)
    test_data.dataset.inter_feat['item_id'] = test_data.dataset.inter_feat['item_id'].to(device)


    for i in tqdm(range(0, len(all_users), batch_size), desc="Computing scores"):
        batch_users = all_users[i : i + batch_size].to(device)  # Добавили .to(device)
        print(f"batch_users = {batch_users.device}")
        print(f"model = {model.device}")
        with torch.no_grad():
            batch_scores = full_sort_scores(batch_users, model, test_data, device=device)
        print(f"batch_scores = {batch_scores.device}")
        scores_list.append(batch_scores)

    scores_list_cpu = [batch.cpu() for batch in scores_list]  # Переносим по частям
    scores_matrix = torch.cat(scores_list_cpu, dim=0).numpy()
    del scores_list, scores_list_cpu  # Освобождаем память
    torch.cuda.empty_cache()  # Чистим кэш CUDA


    # scores_matrix = torch.cat(scores_list, dim=0).cpu().numpy()  # Переносим на CPU перед NumPy

    results = {'SP@K': 0, 'CCR': 0, 'NDCG': 0}
    num_users = len(unique_users)

    for idx, user_id in tqdm(enumerate(unique_users), total=num_users, desc="Evaluating users"):
        user_indices = torch.where(all_users == user_id)[0].detach().cpu().numpy()  # Переносим на CPU перед NumPy
        ground_truth_items = all_items[user_indices].detach().cpu().numpy()  # Аналогично

        pred_items = scores_matrix[idx].argsort()[-k:]

        results['SP@K'] += semantic_precision_at_k(pred_items, ground_truth_items, item_embeddings, device)
        results['CCR'] += cross_category_relevance(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device)
        results['NDCG'] += contextual_ndcg(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device)

    for key in results:
        results[key] /= num_users

    return results

    # all_users = test_data.dataset.inter_feat['user_id'].to_numpy()
    # all_items = test_data.dataset.inter_feat['item_id'].to_numpy()
    # unique_users = np.unique(all_users)

    # batch_size = config.get('eval_batch_size', 512)
    # scores_list = []

    # for i in tqdm(range(0, len(all_users), batch_size), desc="Computing scores"):
    #     batch_users = all_users[i : i + batch_size]
    #     batch_users_tensor = torch.tensor(batch_users, device=device)

    #     with torch.no_grad():
    #         batch_scores = full_sort_scores(batch_users_tensor, model, test_data, device=device)
    #     scores_list.append(batch_scores)

    # scores_matrix = torch.cat(scores_list, dim=0).cpu().numpy()

    # results = {'SP@K': 0, 'CCR': 0, 'NDCG': 0}
    # num_users = len(unique_users)

    # for idx, user_id in tqdm(enumerate(unique_users), total=num_users, desc="Evaluating users"):
    #     user_indices = np.where(all_users == user_id)[0]
    #     ground_truth_items = all_items[user_indices]
    #     pred_items = scores_matrix[idx].argsort()[-k:]

    #     results['SP@K'] += semantic_precision_at_k(pred_items, ground_truth_items, item_embeddings, device)
    #     results['CCR'] += cross_category_relevance(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device)
    #     results['NDCG'] += contextual_ndcg(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device)

    # for key in results:
    #     results[key] /= num_users

    # return results

def contextual_ndcg(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device, k=10):
    """Contextual NDCG с использованием GPU"""
    sim_threshold_ndcg = 0.8
    relevances = []

    item_embeddings = item_embeddings.to(device)
    ground_truth_items = torch.tensor(ground_truth_items, device=device)
    pred_items = torch.tensor(pred_items[:k], device=device)

    # Векторизованный подсчет сходства
    gt_vectors = item_embeddings[ground_truth_items]
    rec_vectors = item_embeddings[pred_items]
    similarity_matrix = cosine_similarity_faiss(gt_vectors, rec_vectors)

    for i, gt_item in enumerate(ground_truth_items):
        orig_gt_item = preprocessor.item_encoder.inverse_transform([gt_item.cpu().item()])[0]
        gt_category = category_info.get(str(orig_gt_item), {}).get('category_id')

        for j, rec_item in enumerate(pred_items):
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

    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances, 1))
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_relevances, 1))
    
    return dcg / idcg if idcg > 0 else 0

def semantic_precision_at_k(pred_items, ground_truth_items, item_embeddings, device, k=10, threshold=0.7):
    """Семантическая точность с Faiss на GPU"""
    pred_items = torch.tensor(pred_items[:k], device=device)
    ground_truth_items = torch.tensor(ground_truth_items, device=device)

    gt_vectors = item_embeddings[ground_truth_items]
    rec_vectors = item_embeddings[pred_items]

    similarity_matrix = cosine_similarity_faiss(gt_vectors, rec_vectors)

    successful_recs = (similarity_matrix >= threshold).sum().item()
    return successful_recs / (k * len(ground_truth_items))

def cross_category_relevance(preprocessor, pred_items, ground_truth_items, item_embeddings, category_info, device, k=10):
    """Кросс-категорийная релевантность"""
    same_category_count = 0
    cross_category_success = 0

    pred_items = torch.tensor(pred_items[:k], device=device)
    ground_truth_items = torch.tensor(ground_truth_items, device=device)

    gt_vectors = item_embeddings[ground_truth_items]
    rec_vectors = item_embeddings[pred_items]
    similarity_matrix = cosine_similarity_faiss(gt_vectors, rec_vectors)

    for i, gt_item in enumerate(ground_truth_items):
        orig_gt_item = preprocessor.item_encoder.inverse_transform([gt_item.cpu().item()])[0]
        gt_category = category_info.get(str(orig_gt_item), {}).get('category_id')

        for j, rec_item in enumerate(pred_items):
            orig_rec_item = preprocessor.item_encoder.inverse_transform([rec_item.cpu().item()])[0]
            rec_category = category_info.get(str(orig_rec_item), {}).get('category_id')

            similarity = similarity_matrix[i, j].item()
            if rec_category == gt_category:
                same_category_count += 1
            elif similarity >= 0.8:
                cross_category_success += 1

    sp_k = semantic_precision_at_k(pred_items, ground_truth_items, item_embeddings, device, k)
    category_diversity = 1 - (same_category_count / k)
    return 0.7 * sp_k + 0.3 * category_diversity

# def cosine_similarity_faiss(vecs1, vecs2):
#     """Вычисляет косинусное сходство с использованием Faiss"""
#     vecs1 = normalize(vecs1.cpu().numpy(), axis=1)
#     vecs2 = normalize(vecs2.cpu().numpy(), axis=1)

#     index = faiss.IndexFlatIP(vecs2.shape[1])
#     index.add(vecs2)
    
#     D, _ = index.search(vecs1, vecs2.shape[0])
#     return torch.tensor(D, device=vecs1.device)
def cosine_similarity_faiss(vecs1, vecs2):
    """Вычисляет косинусное сходство с использованием Faiss на GPU"""
    vecs1 = vecs1.to(torch.float32).contiguous()  # Приводим к нужному формату
    vecs2 = vecs2.to(torch.float32).contiguous()

    index = faiss.IndexFlatIP(vecs2.shape[1])  
    res = faiss.StandardGpuResources()  # Используем GPU
    index = faiss.index_cpu_to_gpu(res, 0, index)  

    index.add(vecs2.detach().cpu().numpy())  # Faiss требует numpy
    D, _ = index.search(vecs1.detach().cpu().numpy(), vecs2.shape[0])
    return torch.tensor(D, device=vecs1.device)

def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """Находит самый последний (по времени модификации) чекпоинт в указанной директории."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)  # Выбираем самый последний по дате изменения
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
        if dataset_type == 'rutube':
            inter_df = df[['viewer_uid', 'rutube_video_id', 'timestamp', 'total_watchtime']].copy()
            inter_df = inter_df.rename(columns={
                'viewer_uid': 'user_id',
                'rutube_video_id': 'item_id',
                'total_watchtime': 'rating'
            })
            df_videos = pd.read_parquet("../data/video_info.parquet")
            df_videos_map = df_videos.set_index('clean_video_id').to_dict(orient='index')
        else:  # lastfm
            # inter_df = df[['user_id', 'artist_id', 'timestamp', 'plays']].copy()
            inter_df = df[['user_id', 'artist_id', 'plays']].copy()
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
        config, config_path = generate_config(
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
        # Запускаем кастомные метрики
        # custom_metrics = evaluate_with_custom_metrics(preprocessor, config, df_videos_map)
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