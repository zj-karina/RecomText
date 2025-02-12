from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os

class TextPreprocessor:
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        text_fields: List[str] = ['title', 'description']
    ):
        """
        Инициализация препроцессора для текстовых данных
        
        Args:
            model_name: Название модели для эмбеддингов
            text_fields: Список текстовых полей для обработки
        """
        self.model = SentenceTransformer(model_name)
        self.text_fields = text_fields
        self.embedding_size = self.model.get_sentence_embedding_dimension()
        
    def process_text_features(
        self,
        df: pd.DataFrame,
        output_dir: str,
        experiment_name: str
    ) -> None:
        """
        Обработка текстовых признаков и сохранение их в формате RecBole
        
        Args:
            df: DataFrame с текстовыми полями
            output_dir: Директория для сохранения результатов
            experiment_name: Название эксперимента
        """
        # Создаем директорию если её нет
        os.makedirs(f"{output_dir}/{experiment_name}", exist_ok=True)
        
        # Создаем копию DataFrame
        df_processed = df.copy()
        
        # Обрабатываем каждое текстовое поле
        for field in self.text_fields:
            if field in df.columns:
                # Получаем эмбеддинги
                embeddings = df[field].apply(
                    lambda x: self.model.encode(str(x)) if pd.notna(x) 
                    else np.zeros(self.embedding_size)
                )
                
                # Создаем колонки для эмбеддингов
                emb_columns = [f'{field}_emb_{i}' for i in range(self.embedding_size)]
                df_processed = df_processed.join(
                    pd.DataFrame(
                        embeddings.tolist(),
                        columns=emb_columns,
                        index=df_processed.index
                    )
                )
                
                # Удаляем исходное текстовое поле
                df_processed = df_processed.drop(columns=[field])
        
        # Сохраняем результат
        df_processed.to_csv(
            f"{output_dir}/{experiment_name}/{experiment_name}.item",
            sep="\t",
            index=False
        )
        
        return df_processed

def get_text_features_config(embedding_size: int = 384) -> Dict:
    """
    Получение конфигурации признаков с текстовыми эмбеддингами
    """
    return {
        'user_features': ['user_id', 'age', 'sex', 'region'],
        'interaction_features': ['user_id', 'item_id', 'timestamp', 'watch_time'],
        'item_features': ['item_id'] + \
                        [f'title_emb_{i}' for i in range(embedding_size)] + \
                        [f'desc_emb_{i}' for i in range(embedding_size)],
        'numerical_features': ['timestamp', 'watch_time', 'duration', 'age'] + \
                            [f'title_emb_{i}' for i in range(embedding_size)] + \
                            [f'desc_emb_{i}' for i in range(embedding_size)]
    } 