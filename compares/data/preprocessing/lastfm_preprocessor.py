import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import LabelEncoder

class LastFMPreprocessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.artist_encoder = LabelEncoder()
        
    def preprocess(self, 
                  df: pd.DataFrame, 
                  feature_config: Dict[str, List[str]],
                  min_interactions: int = 5) -> pd.DataFrame:
        """
        Предобработка данных LastFM
        
        Args:
            df: Исходный датафрейм
            feature_config: Конфигурация используемых признаков
            min_interactions: Минимальное количество взаимодействий для пользователя/артиста
        """
        # Копируем датафрейм
        df = df.copy()
        
        # Фильтруем пользователей и артистов с малым количеством взаимодействий
        user_counts = df['user_id'].value_counts()
        artist_counts = df['artist_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_artists = artist_counts[artist_counts >= min_interactions].index
        
        df = df[df['user_id'].isin(valid_users) & df['artist_id'].isin(valid_artists)]
        
        # Кодируем ID
        df['user_id'] = self.user_encoder.fit_transform(df['user_id'])
        df['artist_id'] = self.artist_encoder.fit_transform(df['artist_id'])
        
        # Преобразуем дату в timestamp
        if 'signup' in df.columns:
            df['timestamp'] = pd.to_datetime(df['signup']).astype('int64') // 10**9
            
        # Оставляем только нужные признаки
        all_features = []
        for feature_list in feature_config['features'].values():
            all_features.extend(feature_list)
        
        return df[list(set(all_features))] 