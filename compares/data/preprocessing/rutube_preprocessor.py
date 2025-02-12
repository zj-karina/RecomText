import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class RutubePreprocessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
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
        
        # Получаем все необходимые признаки из конфига
        all_features = set()
        for feature_type in ['interaction_features', 'user_features', 'item_features']:
            all_features.update(feature_config['features'].get(feature_type, []))
            
        # Добавляем дополнительные признаки, если они нужны
        if 'hour_sin' in numerical_features or 'hour_cos' in numerical_features:
            all_features.update(['hour_sin', 'hour_cos'])
        if 'is_weekend' in categorical_features:
            all_features.add('is_weekend')
            
        # Оставляем только нужные колонки
        return df[list(all_features)] 