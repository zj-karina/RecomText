import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from data.preprocessing.feature_preprocessor import FeaturePreprocessor

class RutubePreprocessor:
    def __init__(self, device=None):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.device = device
        
    def _process_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка временных признаков"""
        if 'timestamp' not in df.columns and all(col in df.columns for col in ['year', 'month', 'day', 'hour', 'minute', 'second']):
            # Создаем timestamp из компонентов
            df['timestamp'] = pd.to_datetime(
                df[['year', 'month', 'day', 'hour', 'minute', 'second']].assign(microsecond=0)
            ).astype('int64') // 10**9
            
            # Добавляем циклические признаки для часа
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Добавляем признак выходного дня
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
        return df
    
    def _process_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка социально-демографических признаков"""
        df = df.copy()
        
        # Обработка пола
        if 'sex' in df.columns:
            df['sex'] = df['sex'].fillna('unknown')
            df['sex'] = df['sex'].map({
                'M': 'male',
                'F': 'female',
                'm': 'male',
                'f': 'female'
            }).fillna('unknown')
        
        # Обработка возраста
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            median_age = df['age'].median()
            df['age'] = df['age'].fillna(median_age)
            # Ограничиваем возраст разумными пределами
            df.loc[df['age'] < 13, 'age'] = 13
            df.loc[df['age'] > 90, 'age'] = 90
            # Нормализация
            df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
        
        # Обработка региона
        if 'region' in df.columns:
            df['region'] = df['region'].fillna('unknown')
        
        return df
    
    def preprocess(self, 
                  df: pd.DataFrame, 
                  feature_config: Dict,
                  min_interactions: int = 5) -> pd.DataFrame:
        """
        Предобработка данных Rutube
        
        Args:
            df: Исходный датафрейм
            feature_config: Конфигурация используемых признаков
            min_interactions: Минимальное количество взаимодействий
        """
        df = df.copy()

        # Фильтруем пользователей и видео с малым количеством взаимодействий
        user_counts = df['viewer_uid'].value_counts()
        item_counts = df['rutube_video_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        df = df[df['viewer_uid'].isin(valid_users) & df['rutube_video_id'].isin(valid_items)]
        # Обработка ID
        df['viewer_uid'] = self.user_encoder.fit_transform(df['viewer_uid'])
        df['rutube_video_id'] = self.item_encoder.fit_transform(df['rutube_video_id'])
        # Обработка временных признаков
        df = self._process_temporal_features(df)
        
        # Обработка социально-демографических признаков
        df = self._process_demographic_features(df)
        
        # Получаем все необходимые признаки из конфига
        all_features = set()
        for feature_type in ['interaction_features', 'user_features', 'item_features']:
            features = feature_config['features'].get(feature_type, [])
            # Добавляем только те признаки, которые есть в датафрейме
            all_features.update([f for f in features if f in df.columns])
        
        # Обработка текстовых и других признаков через FeaturePreprocessor
        feature_processor = FeaturePreprocessor(device=self.device)
        df = feature_processor.process_features(
            df=df,
            feature_config=feature_config,
            output_dir='dataset',
            experiment_name='temp',
            dataset_type='rutube'
        )
        
        # Обработка категориальных признаков
        categorical_features = feature_config['features'].get('categorical_features', [])
        for feature in categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna('unknown')
        
        # Обработка числовых признаков
        numerical_features = feature_config['features'].get('numerical_features', [])
        for feature in numerical_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].mean())
        
        # Добавляем эмбеддинги текстовых полей
        text_fields = feature_config['field_mapping'].get('TEXT_FIELDS', [])
        for field in text_fields:
            if field in df.columns:
                emb_field = f'{field}_embedding'
                emb_list_field = f'{field}_embedding_list'
                if emb_field in df.columns:
                    all_features.add(emb_field)
                if emb_list_field in df.columns:
                    all_features.add(emb_list_field)
        
        # Обновляем список признаков после всех преобразований
        available_features = [f for f in all_features if f in df.columns]
        print(f"available_features = {available_features}")
        return df[available_features] 