import os
import logging
import yaml
from typing import Dict, Optional
from recbole.quick_start import run_recbole
from recbole.utils import init_seed
from ..utils.logger import setup_logging

class BaseTrainer:
    def __init__(
        self,
        model_name: str,
        feature_config: str,
        dataset_name: str,
        base_config: str = None,
        output_dir: str = "./output",
        logger: Optional[logging.Logger] = None
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.logger = logger or setup_logging()
        
        # Определяем путь к базовому конфигу
        if base_config is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_config = os.path.join(base_dir, "configs/base_config.yaml")
        
        # Загрузка конфигураций
        self.base_config = self._load_yaml(base_config)
        self.feature_config = self._load_yaml(feature_config)
        
        # Создание директорий
        os.makedirs(output_dir, exist_ok=True)
        
    def _load_yaml(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
            
    def train(self, data_path: str, model_params: Dict) -> None:
        try:
            # Объединяем все конфигурации
            config = {
                **self.base_config,
                **self.feature_config,
                **model_params,
                'data_path': data_path,
                'checkpoint_dir': os.path.join(self.output_dir, 'checkpoints', self.dataset_name),
            }
            
            # Сохраняем итоговую конфигурацию
            config_path = os.path.join(self.output_dir, 'configs', f'{self.dataset_name}.yaml')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            init_seed(config['training']['seed'], True)
            
            # Запускаем обучение
            result = run_recbole(
                model=self.model_name,
                dataset=self.dataset_name,
                config_file_list=[config_path],
                config_dict=config
            )
            
            self.logger.info(f"Training completed. Model saved in {config['checkpoint_dir']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
            raise 