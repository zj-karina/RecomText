import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import LabelEncoder

class RutubePreprocessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
    def preprocess(self, 
                  df: pd.DataFrame, 
                  feature_config: Dict[str, List[str]],
                  min_interactions: int = 5) -> pd.DataFrame:
        """
        Предобработка данных Rutube
        
        Args:
            df: Исходный датафрейм
            feature_config: Конфигурация используемых признаков
            min_interactions: Минимальное количество взаимодействий для пользователя/видео
        """
        # Копируем датафрейм
        df = df.copy()
        
        # Фильтруем пользователей и видео с малым количеством взаимодействий
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        
        # Кодируем ID
        df['user_id'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_id'] = self.item_encoder.fit_transform(df['item_id'])
        
        # Создаем timestamp из компонентов даты если нужно
        if 'timestamp' in feature_config['features'].get('interaction_features', []):
            if 'timestamp' not in df.columns and all(col in df.columns for col in ['year', 'month', 'day', 'hour', 'minute', 'second']):
                df['timestamp'] = pd.to_datetime(
                    df[['year', 'month', 'day', 'hour', 'minute', 'second']].assign(microsecond=0)
                ).astype('int64') // 10**9
                
        # Оставляем только нужные признаки
        all_features = []
        for feature_list in feature_config['features'].values():
            all_features.extend(feature_list)
        
        return df[list(set(all_features))] 