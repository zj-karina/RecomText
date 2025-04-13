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
        embedding_dim: int = 384,
        device: str = None,
        output_dir='',
        experiment_name=''
    ):
        self.device = device or ('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.text_model = SentenceTransformer(text_model_name).to(self.device)
        self.embedding_dim = embedding_dim
        self.text_embedding_size = self.text_model.get_sentence_embedding_dimension()
        self.label_encoders = {}
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device} for text embeddings generation")
        self.output_dir = output_dir
        self.experiment_name = experiment_name

    def _process_text_features(
        self, df, text_fields, max_seq_length=50, model_type='sasrec', field_mapping=None
    ):
        df_processed = df.copy()
        item_id_field = field_mapping.get("ITEM_ID_FIELD", "item_id") if field_mapping else "item_id"  # предположим, что у нас есть поле для item_id
    
        for field in text_fields:
            print(f"dsdsdsdsdsdds Processing text field: {field}")
            if field in df.columns:
                texts = df[field].fillna('').tolist()
                embeddings = torch.tensor(
                    self.text_model.encode(texts, show_progress_bar=True),
                    device=self.device
                )
                embedding_dim = embeddings.shape[1]
                df_processed[f'{field}_embedding'] = embeddings.cpu().numpy().tolist()
    
                # --- Группируем по айтемам ---
                item_sequences = (
                    df_processed.groupby(item_id_field)[f'{field}_embedding']
                    .apply(list)
                    .to_dict()
                )
    
                # --- Подготовка к индексированию ---
                embedding_to_index = {}
                all_embeddings = []
                indexed_sequences = {}
    
                for item_id, seq in item_sequences.items():
                    # Паддинг
                    seq = seq[:max_seq_length]  # обрезаем до max_seq_length
                    pad_len = max_seq_length - len(seq)
                    seq += [[0.0] * embedding_dim] * pad_len  # добавляем паддинг, если нужно
    
                    indexed_seq = []
                    for emb in seq:
                        emb = np.array(emb)
                        # Делаем hashable для индексации
                        emb_key = tuple(emb / (np.linalg.norm(emb) + 1e-12))  # нормализуем эмбеддинг
                        if emb_key not in embedding_to_index:
                            embedding_to_index[emb_key] = len(all_embeddings)
                            all_embeddings.append(emb)
                        indexed_seq.append(embedding_to_index[emb_key])
    
                    indexed_sequences[item_id] = indexed_seq
    
                # --- Сохраняем индексы в df ---
                df_processed[f'{field}_embedding_idx'] = df_processed[item_id_field].map(indexed_sequences)
                print(f"COLUMNS = {df_processed.columns}")
                # Проверка индексов на валидность
                max_idx = len(all_embeddings)
                all_idx = df_processed[f'{field}_embedding_idx'].explode()
                invalid = all_idx[all_idx >= max_idx]
                if not invalid.empty:
                    print(f"[!] Found {len(invalid)} invalid indices in {field}_embedding_idx. Max index allowed: {max_idx}")

                # --- Сохраняем .npy файл весов ---
                weights = np.vstack(all_embeddings).astype(np.float32)
                experiment_dir = os.path.join(self.output_dir, self.experiment_name)
                os.makedirs(experiment_dir, exist_ok=True)
                np.save(os.path.join(experiment_dir, f"{field}_embedding.npy"), weights)
    
                print(f"Saved embedding matrix: {field}_embedding.npy")
    
        return df_processed



    def _process_categorical_features(self, df: pd.DataFrame, categorical_fields: List[str], is_train: bool = True) -> pd.DataFrame:
        df_processed = df.copy()
        for field in categorical_fields:
            if field in df.columns:
                df_processed[field] = df_processed[field].fillna('unknown')
                if is_train or field not in self.label_encoders:
                    self.label_encoders[field] = LabelEncoder()
                    df_processed[field] = self.label_encoders[field].fit_transform(df_processed[field])
                else:
                    df_processed[field] = df_processed[field].map(lambda x: x if x in self.label_encoders[field].classes_ else 'unknown')
                    df_processed[field] = self.label_encoders[field].transform(df_processed[field])
        return df_processed

    def _process_numerical_features(self, df: pd.DataFrame, numerical_fields: List[str], is_train: bool = True) -> pd.DataFrame:
        df_processed = df.copy()
        for field in numerical_fields:
            if field in df.columns:
                if is_train:
                    self.scalers[field] = StandardScaler()
                    df_processed[field] = self.scalers[field].fit_transform(df_processed[field].fillna(df_processed[field].mean()).values.reshape(-1, 1))
                else:
                    df_processed[field] = self.scalers[field].transform(df_processed[field].fillna(self.scalers[field].mean_).values.reshape(-1, 1))
        return df_processed

    def process_features(self, df: pd.DataFrame, feature_config: Dict, is_train: bool = True, model_type: str = 'sasrec') -> pd.DataFrame:
        print(f"feature_config['features'] = {feature_config['features']}")
        text_fields = feature_config['features'].get('text_features', [])
        categorical_fields = feature_config['features'].get('categorical_features', [])
        numerical_fields = feature_config['features'].get('numerical_features', [])
        max_seq_length = feature_config.get('MAX_ITEM_LIST_LENGTH', 50)
        print(f"text_fields from ['features'] = {text_fields}")

        field_mapping = feature_config.get("field_mapping", {})

        if text_fields:
            df = self._process_text_features(df, text_fields, max_seq_length, model_type, field_mapping)
        if categorical_fields:
            df = self._process_categorical_features(df, categorical_fields, is_train)
        if numerical_fields:
            df = self._process_numerical_features(df, numerical_fields, is_train)
        
        return df

def get_full_features_config(dataset_config: Dict, embedding_dim=384) -> Dict:
    text_fields = dataset_config['features'].get('text_features', [])
    categorical_features = dataset_config['features'].get('categorical_features', [])
    numerical_features = dataset_config['features'].get('numerical_features', [])
    text_embeddings = [f'{field}_embedding' for field in text_fields] + [f'{field}_embedding_list' for field in text_fields]
    
    return {
        'numerical_features': numerical_features + text_embeddings,
        'categorical_features': categorical_features,
        # 'text_fields': text_fields,
        'embedding_size': dataset_config.get('embedding_size', embedding_dim),
        'numerical_projection_dropout': dataset_config.get('numerical_projection_dropout', 0.1)
    }