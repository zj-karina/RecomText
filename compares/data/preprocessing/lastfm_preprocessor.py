import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class LastFMPreprocessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.artist_encoder = LabelEncoder()
        
    def _process_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка временных признаков"""
        df = df.copy()
        
        if 'signup' in df.columns:
            # Преобразуем строковую дату в timestamp с учетом формата "MMM d, YYYY"
            df['timestamp'] = pd.to_datetime(df['signup'], format='%b %d, %Y').astype('int64') // 10**9
        
        return df
    
    def _process_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка демографических признаков"""
        # Обработка пола
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'m': 'male', 'f': 'female'}).fillna('unknown')
            
        # Обработка возраста
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['age'] = df['age'].fillna(df['age'].mean())
            
        return df
        
    def preprocess(self, 
                  df: pd.DataFrame, 
                  feature_config: Dict,
                  min_interactions: int = 5) -> pd.DataFrame:
        """
        Предобработка данных LastFM
        
        Args:
            df: Исходный датафрейм
            feature_config: Конфигурация используемых признаков
            min_interactions: Минимальное количество взаимодействий
        """
        df = df.copy()
        
        # Фильтруем пользователей и артистов с малым количеством взаимодействий
        user_counts = df['user_id'].value_counts()
        artist_counts = df['artist_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_artists = artist_counts[artist_counts >= min_interactions].index
        
        df = df[df['user_id'].isin(valid_users) & df['artist_id'].isin(valid_artists)]
        
        # Обработка ID
        df['user_id'] = self.user_encoder.fit_transform(df['user_id'])
        df['artist_id'] = self.artist_encoder.fit_transform(df['artist_id'])
        
        # Обработка временных признаков (создаем timestamp из signup)
        df = self._process_temporal_features(df)
        
        # Обработка демографических признаков
        if any(field in df.columns for field in ['gender', 'age', 'country']):
            df = self._process_demographic_features(df)
        
        # Обработка текстовых признаков через FeaturePreprocessor
        text_fields = feature_config['field_mapping'].get('TEXT_FIELDS', [])
        if text_fields:
            feature_processor = FeaturePreprocessor()
            df = feature_processor.process_features(
                df=df,
                feature_config=feature_config,
                output_dir='dataset',  # временная директория
                experiment_name='temp',
                dataset_type='lastfm'
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
        
        # Получаем все необходимые признаки из конфига
        all_features = set()
        for feature_type in ['interaction_features', 'user_features', 'item_features']:
            all_features.update(feature_config['features'].get(feature_type, []))
            
        # Оставляем только нужные колонки
        return df[list(all_features)] 