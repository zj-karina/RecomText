from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import torch

class FeaturePreprocessor:
    def __init__(
        self,
        text_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim: int = 16,
        device: str = None  # Добавляем параметр device
    ):
        # Определяем устройство: если не указано явно, используем CUDA при наличии
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.text_model = SentenceTransformer(text_model_name)
        self.text_model.to(device)  # Перемещаем модель на нужное устройство
        
        self.embedding_dim = embedding_dim
        self.text_embedding_size = self.text_model.get_sentence_embedding_dimension()
        self.label_encoders = {}
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Using device: {device} for text embeddings generation")
        
    def _process_text_features(self, df: pd.DataFrame, text_fields: List[str]) -> pd.DataFrame:
        """Обработка текстовых признаков"""
        df_processed = df.copy()
        
        for field in text_fields:
            if field in df.columns:
                self.logger.info(f"Processing text field: {field}")
                texts = df[field].fillna('').tolist()
                
                # Используем device при генерации эмбеддингов
                embeddings = self.text_model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=True,
                    device=self.device  # Явно указываем устройство
                )
                
                # Создаем колонки для эмбеддингов
                emb_columns = [f'{field}_emb_{i}' for i in range(self.text_embedding_size)]
                df_processed = df_processed.join(
                    pd.DataFrame(
                        embeddings,
                        columns=emb_columns,
                        index=df_processed.index
                    )
                )
                # Удаляем исходное текстовое поле
                df_processed = df_processed.drop(columns=[field])
                
        return df_processed
    
    def _process_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_fields: List[str],
        is_train: bool = True
    ) -> pd.DataFrame:
        """Обработка категориальных признаков"""
        df_processed = df.copy()
        
        for field in categorical_fields:
            if field in df.columns:
                df_processed[field] = df_processed[field].fillna('unknown')
                
                if is_train or field not in self.label_encoders:
                    self.label_encoders[field] = LabelEncoder()
                    df_processed[field] = self.label_encoders[field].fit_transform(df_processed[field])
                else:
                    known_categories = set(self.label_encoders[field].classes_)
                    df_processed[field] = df_processed[field].map(
                        lambda x: x if x in known_categories else 'unknown'
                    )
                    df_processed[field] = self.label_encoders[field].transform(df_processed[field])
                
        return df_processed

    def _process_numerical_features(
        self,
        df: pd.DataFrame,
        numerical_fields: List[str],
        is_train: bool = True
    ) -> pd.DataFrame:
        """Обработка числовых признаков"""
        df_processed = df.copy()
        
        for field in numerical_fields:
            if field in df.columns:
                if is_train:
                    fill_value = df_processed[field].mean()
                    self.scalers[field] = StandardScaler()
                    df_processed[field] = self.scalers[field].fit_transform(
                        df_processed[field].fillna(fill_value).values.reshape(-1, 1)
                    )
                else:
                    fill_value = self.scalers[field].mean_
                    df_processed[field] = self.scalers[field].transform(
                        df_processed[field].fillna(fill_value).values.reshape(-1, 1)
                    )
                
        return df_processed

    def process_features(
        self,
        df: pd.DataFrame,
        feature_config: Dict,
        output_dir: str,
        experiment_name: str,
        dataset_type: str,
        is_train: bool = True
    ) -> pd.DataFrame:
        """
        Обработка всех признаков
        
        Args:
            df: DataFrame с признаками
            feature_config: Конфигурация признаков
            output_dir: Директория для сохранения
            experiment_name: Название эксперимента
            dataset_type: Тип датасета ('rutube' или 'lastfm')
            is_train: Флаг обучающей выборки
        """
        os.makedirs(f"{output_dir}/{experiment_name}", exist_ok=True)
        
        # Получаем конфигурацию для конкретного датасета
        dataset_config = (feature_config[dataset_type] 
                         if dataset_type in feature_config 
                         else feature_config)
        
        # Получаем списки признаков разных типов
        text_fields = dataset_config['field_mapping'].get('TEXT_FIELDS', [])
        categorical_fields = dataset_config['features'].get('categorical_features', [])
        numerical_fields = dataset_config['features'].get('numerical_features', [])
        
        # Обрабатываем признаки
        if text_fields:
            self.logger.info(f"Processing text fields: {text_fields}")
            df = self._process_text_features(df, text_fields)
        
        if categorical_fields:
            self.logger.info(f"Processing categorical fields: {categorical_fields}")
            df = self._process_categorical_features(df, categorical_fields, is_train)
            
        if numerical_fields:
            self.logger.info(f"Processing numerical fields: {numerical_fields}")
            df = self._process_numerical_features(df, numerical_fields, is_train)
        
        return df

def get_full_features_config(dataset_type: str, feature_config: Dict) -> Dict:
    """
    Получение полной конфигурации признаков для конкретного датасета
    
    Args:
        dataset_type: Тип датасета ('rutube' или 'lastfm')
        feature_config: Загруженная конфигурация признаков
    """
    dataset_config = feature_config[dataset_type]
    
    # Базовые признаки из конфигурации
    base_features = dataset_config['features']['interaction_features']
    
    # Получаем размерности эмбеддингов
    text_fields = dataset_config['field_mapping'].get('TEXT_FIELDS', [])
    categorical_fields = dataset_config['features'].get('categorical_features', [])
    
    # Создаем списки эмбеддингов
    text_embeddings = []
    for field in text_fields:
        text_embeddings.extend([f'{field}_emb_{i}' for i in range(384)])  # Стандартный размер BERT
        
    categorical_embeddings = []
    for field in categorical_fields:
        categorical_embeddings.extend([f'{field}_emb_{i}' for i in range(16)])  # Размер из конфига
    
    return {
        'user_features': dataset_config['features']['user_features'],
        'interaction_features': base_features,
        'item_features': dataset_config['features']['item_features'],
        'numerical_features': (
            dataset_config['features']['numerical_features'] +
            text_embeddings +
            categorical_embeddings
        )
    } 