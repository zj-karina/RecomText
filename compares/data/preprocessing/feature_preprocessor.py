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
        print(f"device = {device}")
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.text_model = SentenceTransformer(text_model_name)
        self.text_model.to(device)
        
        self.embedding_dim = embedding_dim
        self.text_embedding_size = self.text_model.get_sentence_embedding_dimension()
        self.label_encoders = {}
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Using device: {device} for text embeddings generation")
        
    def _process_text_features(self, df: pd.DataFrame, text_fields: List[str], max_seq_length: int = 50, model_type: str = 'sasrec') -> pd.DataFrame:
        """
        Обработка текстовых признаков
        
        Args:
            df: DataFrame с данными
            text_fields: Список текстовых полей для обработки
            max_seq_length: Максимальная длина последовательности (по умолчанию 50)
            model_type: Тип модели ('sasrec' или 'bert4rec')
        """
        df_processed = df.copy()
        
        for field in text_fields:
            if field in df.columns:
                self.logger.info(f"_process_text_features: field: {field}")
                texts = df[field].fillna('').tolist()
                
                embeddings = self.text_model.encode(texts, show_progress_bar=False)
                embeddings = torch.tensor(embeddings, device=self.device)
                
                df_processed[f'{field}_embedding'] = embeddings.cpu().numpy()
                
                # Создаем последовательности только для BERT4Rec
                if model_type == 'bert4rec':
                    if 'viewer_uid' not in df.columns:
                        self.logger.warning("Field viewer_uid not found in DataFrame. Skipping sequence creation.")
                        continue
                    
                    # Для каждого пользователя создаем последовательность эмбеддингов
                    user_sequences = {}
                    for user_id, emb in zip(df['viewer_uid'], embeddings):
                        if user_id not in user_sequences:
                            user_sequences[user_id] = []
                        user_sequences[user_id].append(emb.tolist())
                    
                    # Создаем словарь с готовыми последовательностями для каждого пользователя
                    processed_sequences = {}
                    for user_id in df['viewer_uid'].unique():
                        seq = user_sequences[user_id]
                        if len(seq) < max_seq_length:
                            # Создаем padding как список нулей
                            padding = [[0.0] * embeddings.shape[1]] * (max_seq_length - len(seq))
                            seq.extend(padding)
                        else:
                            seq = seq[:max_seq_length]
                        processed_sequences[user_id] = seq
                    
                    sequence_embeddings = []
                    for user_id in df['viewer_uid']:
                        sequence_embeddings.append(processed_sequences[user_id])
                    
                    df_processed[f'{field}_embedding_list'] = sequence_embeddings
                
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
                print(f"_process_numerical_features: field = {field}")
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
        is_train: bool = True,
        model_type: str = 'sasrec'
    ) -> pd.DataFrame:
        """Обработка всех признаков"""
        os.makedirs(f"{output_dir}/{experiment_name}", exist_ok=True)
        
        dataset_config = (feature_config[dataset_type] 
                         if dataset_type in feature_config 
                         else feature_config)
        
        # Получаем списки признаков разных типов
        text_fields = dataset_config['features'].get('text_fields', [])
        categorical_fields = dataset_config['features'].get('categorical_features', [])
        numerical_fields = dataset_config['features'].get('numerical_features', [])
        
        self.logger.info(f"Processing features:")
        self.logger.info(f"Text fields: {text_fields}")
        self.logger.info(f"Categorical fields: {categorical_fields}")
        self.logger.info(f"Numerical fields: {numerical_fields}")
        
        # Получаем максимальную длину последовательности из конфигурации
        max_seq_length = dataset_config.get('MAX_ITEM_LIST_LENGTH', 50)
        
        # Обрабатываем признаки
        if text_fields:
            self.logger.info(f"Processing text fields: {text_fields}")
            df = self._process_text_features(df, text_fields, max_seq_length=max_seq_length, model_type=model_type)
        
        if categorical_fields:
            self.logger.info(f"Processing categorical fields: {categorical_fields}")
            df = self._process_categorical_features(df, categorical_fields, is_train)
            
        if numerical_fields:
            self.logger.info(f"Processing numerical fields: {numerical_fields}")
            df = self._process_numerical_features(df, numerical_fields, is_train)
        
        for field in categorical_fields:
            if field in df.columns:
                df[field] = df[field].astype('int64')
                self.logger.info(f"Converted {field} to int64")
            
        for field in numerical_fields:
            if field in df.columns:
                df[field] = df[field].astype('float32')
                self.logger.info(f"Converted {field} to float32")
        
        return df

def get_full_features_config(dataset_config: Dict) -> Dict:
    """Создает полную конфигурацию признаков для RecBole"""
    numerical_features = dataset_config['features']['numerical_features']
    categorical_features = dataset_config['features']['categorical_features']
    text_fields = dataset_config['features'].get('text_fields', [])
    
    text_embeddings = []
    categorical_embeddings = []
    
    for field in text_fields:
        text_embeddings.extend([
            f'{field}_embedding',
            f'{field}_embedding_list'
        ])
    
    for field in categorical_features:
        categorical_embeddings.append(f'{field}_embedding')
    
    full_config = {
        'numerical_features': (
            numerical_features +
            text_embeddings +
            categorical_embeddings
        ),
        'categorical_features': categorical_features,
        'text_fields': text_fields,
        'embedding_size': dataset_config.get('embedding_size', 384),
        'numerical_projection_dropout': dataset_config.get('numerical_projection_dropout', 0.1)
    }
    
    return full_config 