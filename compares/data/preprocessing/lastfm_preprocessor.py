import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from data.preprocessing.feature_preprocessor import FeaturePreprocessor

class LastFMPreprocessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.artist_encoder = LabelEncoder()
    
    def _process_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка демографических признаков"""
        # Обработка пола
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'m': 'male', 'f': 'female'}).fillna('unknown')
            
        # Обработка возраста
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['age'] = df['age'].fillna(df['age'].mean())
            df.loc[df['age'] < 13, 'age'] = 13
            df.loc[df['age'] > 90, 'age'] = 90
            df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
            
        if 'country' in df.columns:
            df['country'] = df['country'].fillna('unknown')
        
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
        
        # Обработка временных признаков
        if 'signup' in df.columns:
            df['timestamp'] = pd.to_datetime(df['signup']).astype('int64') // 10**9
        
        # Обработка социально-демографических признаков
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'m': 'male', 'f': 'female'}).fillna('unknown')
        
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['age'] = df['age'].fillna(df['age'].mean())
            df.loc[df['age'] < 13, 'age'] = 13
            df.loc[df['age'] > 90, 'age'] = 90
            df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
        
        if 'country' in df.columns:
            df['country'] = df['country'].fillna('unknown')
        
        # Получаем все необходимые признаки из конфига
        all_features = set()
        for feature_type in ['interaction_features', 'user_features', 'item_features']:
            features = feature_config['features'].get(feature_type, [])
            # Добавляем только те признаки, которые есть в датафрейме
            all_features.update([f for f in features if f in df.columns])
        
        # Обработка текстовых и других признаков через FeaturePreprocessor
        feature_processor = FeaturePreprocessor()
        df = feature_processor.process_features(
            df=df,
            feature_config=feature_config,
            output_dir='dataset',
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
        
        # Добавляем эмбеддинги текстовых полей
        text_fields = feature_config['field_mapping'].get('TEXT_FIELDS', [])
        for field in text_fields:
            emb_features = [f'{field}_emb_{i}' for i in range(384)]
            all_features.update(emb_features)
        
        # Обновляем список признаков после всех преобразований
        available_features = [f for f in all_features if f in df.columns]
        
        return df[available_features] 