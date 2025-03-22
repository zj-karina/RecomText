from recbole.data.dataset import SequentialDataset
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EnhancedSequentialDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model.to(config['device'])
        self.embedding_dim = 384  # BERT embedding dimension
        self.model_type = config.get('model_type', 'sasrec')  # Добавляем тип модели
        
    def _get_field_from_config(self):
        """Расширяем конфигурацию полей для работы с эмбеддингами"""
        super()._get_field_from_config()
        
        # Получаем текстовые поля из правильного места в конфигурации
        text_fields = self.config['features'].get('text_fields', [])
        
        # Регистрируем поля для эмбеддингов как единые тензоры
        for text_field in text_fields:
            emb_field = f'{text_field}_embedding'
            self.field2type[emb_field] = 'float_seq'
            self.field2source[emb_field] = 'item'
            
            # Добавляем поле для последовательности эмбеддингов только для BERT4Rec
            if self.model_type == 'bert4rec':
                list_field = f'{emb_field}_list'
                self.field2type[list_field] = 'float_seq'
                self.field2source[list_field] = 'interaction'
            
    def _prepare_data_augmentation(self):
        """Подготавливаем данные, включая генерацию эмбеддингов"""
        super()._prepare_data_augmentation()
        
        text_fields = self.config['features'].get('text_fields', [])
        logger.info(f"Processing text fields: {text_fields}")
        logger.info(f"Model type: {self.model_type}")
        
        # Обрабатываем текстовые поля
        for text_field in text_fields:
            if text_field in self.inter_feat:
                logger.info(f"Processing text field: {text_field}")
                
                # Получаем уникальные тексты для всех айтемов
                unique_texts = self.inter_feat[text_field].unique()
                logger.info(f"Found {len(unique_texts)} unique texts")
                
                # Генерируем эмбеддинги батчами
                batch_size = 128
                all_embeddings = []
                
                for i in range(0, len(unique_texts), batch_size):
                    batch_texts = unique_texts[i:i + batch_size]
                    batch_embeddings = self.text_model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                    all_embeddings.append(batch_embeddings.cpu().numpy())
                
                # Объединяем все эмбеддинги
                all_embeddings = np.vstack(all_embeddings)
                logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
                
                # Проверяем размерность эмбеддингов
                if all_embeddings.shape[1] != self.embedding_dim:
                    raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {all_embeddings.shape[1]}")
                
                # Создаем маппинг текст -> эмбеддинг
                text_to_embedding = {text: emb for text, emb in zip(unique_texts, all_embeddings)}
                
                # Создаем тензор эмбеддингов для всех взаимодействий
                sequence_embeddings = np.array([
                    text_to_embedding[text] for text in self.inter_feat[text_field]
                ])
                logger.info(f"Sequence embeddings shape: {sequence_embeddings.shape}")
                
                # Сохраняем эмбеддинги
                emb_field = f'{text_field}_embedding'
                self.inter_feat[emb_field] = torch.FloatTensor(sequence_embeddings)
                logger.info(f"Saved embeddings to field: {emb_field}")
                
                # Создаем последовательности эмбеддингов только для BERT4Rec
                if self.model_type == 'bert4rec':
                    logger.info("Creating sequence embeddings for BERT4Rec")
                    list_field = f'{emb_field}_list'
                    
                    # Проверяем размерность последовательности
                    if sequence_embeddings.shape[0] % self.max_item_list_len != 0:
                        raise ValueError(f"Number of embeddings ({sequence_embeddings.shape[0]}) must be divisible by max_item_list_len ({self.max_item_list_len})")
                    
                    # Создаем последовательности
                    sequence_tensor = torch.FloatTensor(
                        sequence_embeddings.reshape(-1, self.max_item_list_len, self.embedding_dim)
                    )
                    self.inter_feat[list_field] = sequence_tensor
                    logger.info(f"Created sequence tensor of shape: {sequence_tensor.shape}")
                    logger.info(f"Saved sequence embeddings to field: {list_field}")
                
    def _restore_saved_dataset(self, saved_dataset):
        """Восстанавливаем сохраненный датасет"""
        super()._restore_saved_dataset(saved_dataset)
        
        text_fields = self.config['features'].get('text_fields', [])
        
        # Восстанавливаем эмбеддинги
        for text_field in text_fields:
            emb_field = f'{text_field}_embedding'
            list_field = f'{emb_field}_list'
            
            if emb_field in saved_dataset:
                self.inter_feat[emb_field] = saved_dataset[emb_field]
            if self.model_type == 'bert4rec' and list_field in saved_dataset:
                self.inter_feat[list_field] = saved_dataset[list_field] 