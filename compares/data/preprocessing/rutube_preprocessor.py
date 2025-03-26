import os
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from data.preprocessing.feature_preprocessor import FeaturePreprocessor


class RutubePreprocessor:
    def __init__(self, device=None):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.feature_processor = FeaturePreprocessor(device=device)
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

    def preprocess(self, df: pd.DataFrame, feature_config: Dict, min_interactions: int = 5, is_train: bool = True, model_type: str = None) -> pd.DataFrame:
        df = df.copy()
        
        # Фильтрация пользователей и видео
        print(f"SHAPE = {df.shape}")
        valid_users = df['viewer_uid'].value_counts()[lambda x: x >= min_interactions].index
        valid_items = df['rutube_video_id'].value_counts()[lambda x: x >= min_interactions].index
        df = df[df['viewer_uid'].isin(valid_users) & df['rutube_video_id'].isin(valid_items)]
        print(f"SHAPE = {df.shape}")
        # Кодирование ID
        df['viewer_uid'] = self.user_encoder.fit_transform(df['viewer_uid'])
        df['rutube_video_id'] = self.item_encoder.fit_transform(df['rutube_video_id'])
        
        # Обработка временных признаков
        df = self._process_temporal_features(df)
        
        # Обработка социально-демографических признаков
        df = self._process_demographic_features(df)
        
        # Обработка всех признаков через FeaturePreprocessor
        df = self.feature_processor.process_features(df, feature_config, is_train, model_type)
        
        return df
