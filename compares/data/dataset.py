from recbole.data.dataset import SequentialDataset

class EnhancedSequentialDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        
    def _get_field_from_config(self):
        super()._get_field_from_config()
        
        # Обрабатываем текстовые поля и их эмбеддинги
        for text_field in self.config['data']['TEXT_FIELDS']:
            # Добавляем базовое текстовое поле
            if text_field not in self.field2type:
                self.field2type[text_field] = 'token'
                self.field2source[text_field] = 'item'
            
            # Добавляем поля эмбеддингов
            emb_field = f'{text_field}_emb'
            for field_info in self.config['data']['field_preparation']['inter']:
                if field_info['field'] == emb_field:
                    dim = field_info['dim']
                    for i in range(dim):
                        emb_component = f'{emb_field}_{i}'
                        if emb_component not in self.field2type:
                            self.field2type[emb_component] = 'float'
                            self.field2source[emb_component] = 'item'
                        
        # Обрабатываем пользовательские категориальные признаки
        for field in self.config['data']['USER_FEATURES']:
            if field not in self.field2type:
                self.field2type[field] = 'token'
                self.field2source[field] = 'user'
    
    def _prepare_data_augmentation(self):
        super()._prepare_data_augmentation()
        
        # Подготавливаем последовательности для текстовых полей и их эмбеддингов
        for text_field in self.config['data']['TEXT_FIELDS']:
            # Создаем последовательности для базовых текстовых полей
            if text_field in self.inter_feat:
                list_field = f'{text_field}_list'
                self.inter_feat[list_field] = self.inter_feat[text_field].values.reshape(-1, self.max_item_list_len)
            
            # Создаем последовательности для эмбеддингов
            emb_field = f'{text_field}_emb'
            for field_info in self.config['data']['field_preparation']['inter']:
                if field_info['field'] == emb_field:
                    dim = field_info['dim']
                    for i in range(dim):
                        emb_component = f'{emb_field}_{i}'
                        list_field = f'{emb_component}_list'
                        if emb_component in self.inter_feat:
                            self.inter_feat[list_field] = self.inter_feat[emb_component].values.reshape(-1, self.max_item_list_len)
        
        # Подготавливаем пользовательские признаки
        # Они не требуют преобразования в последовательности, так как относятся к пользователю в целом
        for field in self.config['data']['USER_FEATURES']:
            if field in self.inter_feat:
                self.inter_feat[field] = self.inter_feat[field].astype('int64') 