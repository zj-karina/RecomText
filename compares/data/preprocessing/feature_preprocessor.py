from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os
from sklearn.preprocessing import LabelEncoder

class FeaturePreprocessor:
    def __init__(
        self,
        text_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        text_fields: List[str] = ['title', 'description'],
        categorical_fields: List[str] = ['sex', 'region', 'ua_device_type', 'ua_os'],
        embedding_dim: int = 16  # размерность для категориальных признаков
    ):
        self.text_model = SentenceTransformer(text_model_name)
        self.text_fields = text_fields
        self.categorical_fields = categorical_fields
        self.embedding_dim = embedding_dim
        self.text_embedding_size = self.text_model.get_sentence_embedding_dimension()
        self.label_encoders = {field: LabelEncoder() for field in categorical_fields}
        
    def _process_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка текстовых признаков"""
        df_processed = df.copy()
        
        for field in self.text_fields:
            if field in df.columns:
                embeddings = df[field].apply(
                    lambda x: self.text_model.encode(str(x)) if pd.notna(x) 
                    else np.zeros(self.text_embedding_size)
                )
                
                emb_columns = [f'{field}_emb_{i}' for i in range(self.text_embedding_size)]
                df_processed = df_processed.join(
                    pd.DataFrame(
                        embeddings.tolist(),
                        columns=emb_columns,
                        index=df_processed.index
                    )
                )
                df_processed = df_processed.drop(columns=[field])
                
        return df_processed
    
    def _process_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка категориальных признаков"""
        df_processed = df.copy()
        
        for field in self.categorical_fields:
            if field in df.columns:
                # Заполняем пропуски специальным значением
                df_processed[field] = df_processed[field].fillna('UNKNOWN')
                
                # Кодируем категории
                df_processed[field] = self.label_encoders[field].fit_transform(df_processed[field])
                
                # Создаем простые эмбеддинги (one-hot или случайные)
                n_categories = len(self.label_encoders[field].classes_)
                
                # Используем случайные эмбеддинги для больших словарей
                if n_categories > self.embedding_dim:
                    embeddings = np.random.normal(0, 1/self.embedding_dim, 
                                               (n_categories, self.embedding_dim))
                else:
                    # Для маленьких словарей используем one-hot
                    embeddings = np.eye(n_categories)
                    if n_categories < self.embedding_dim:
                        # Дополняем нулями до нужной размерности
                        padding = np.zeros((n_categories, self.embedding_dim - n_categories))
                        embeddings = np.hstack([embeddings, padding])
                
                # Получаем эмбеддинги для каждого значения
                category_embeddings = embeddings[df_processed[field]]
                
                # Создаем колонки для эмбеддингов
                emb_columns = [f'{field}_emb_{i}' for i in range(self.embedding_dim)]
                df_processed = df_processed.join(
                    pd.DataFrame(
                        category_embeddings,
                        columns=emb_columns,
                        index=df_processed.index
                    )
                )
                
        return df_processed

    def process_features(
        self,
        df: pd.DataFrame,
        output_dir: str,
        experiment_name: str,
        file_type: str = 'item'  # или 'user' для пользовательских признаков
    ) -> pd.DataFrame:
        """
        Обработка всех признаков
        
        Args:
            df: DataFrame с признаками
            output_dir: Директория для сохранения
            experiment_name: Название эксперимента
            file_type: Тип файла ('item' или 'user')
        """
        os.makedirs(f"{output_dir}/{experiment_name}", exist_ok=True)
        
        # Обрабатываем текстовые признаки если это item-файл
        if file_type == 'item' and any(field in df.columns for field in self.text_fields):
            df = self._process_text_features(df)
            
        # Обрабатываем категориальные признаки
        df = self._process_categorical_features(df)
        
        # Сохраняем результат
        df.to_csv(
            f"{output_dir}/{experiment_name}/{experiment_name}.{file_type}",
            sep="\t",
            index=False
        )
        
        return df

def get_full_features_config(
    text_embedding_size: int = 384,
    cat_embedding_size: int = 16
) -> Dict:
    """
    Получение полной конфигурации признаков
    """
    # Базовые признаки
    base_features = ['user_id', 'item_id', 'timestamp', 'watch_time']
    
    # Числовые признаки пользователя
    user_numerical = ['age']
    
    # Категориальные признаки и их эмбеддинги
    categorical_fields = ['sex', 'region', 'ua_device_type', 'ua_os']
    categorical_embeddings = []
    for field in categorical_fields:
        categorical_embeddings.extend([f'{field}_emb_{i}' for i in range(cat_embedding_size)])
    
    # Текстовые эмбеддинги
    text_embeddings = []
    for field in ['title', 'description']:
        text_embeddings.extend([f'{field}_emb_{i}' for i in range(text_embedding_size)])
    
    return {
        'user_features': ['user_id', 'age'] + \
                        [f for f in categorical_embeddings if f.startswith(('sex_', 'region_'))],
        'interaction_features': base_features + \
                              [f for f in categorical_embeddings if f.startswith(('ua_device_', 'ua_os_'))],
        'item_features': ['item_id'] + text_embeddings,
        'numerical_features': ['timestamp', 'watch_time', 'age'] + \
                            text_embeddings + categorical_embeddings
    } 