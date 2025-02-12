import wandb
from typing import Dict, Any, Optional
import os
import json
from datetime import datetime

class ExperimentTracker:
    """
    Класс для отслеживания экспериментов с поддержкой WandB и локального логирования
    """
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any],
        output_dir: str = "./experiments",
        use_wandb: bool = True
    ):
        self.experiment_name = experiment_name
        self.output_dir = os.path.join(output_dir, experiment_name)
        self.use_wandb = use_wandb
        
        # Создаем директорию для эксперимента
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Инициализируем WandB если нужно
        if use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config
            )
            
        # Сохраняем конфигурацию локально
        self._save_config(config)
        
    def _save_config(self, config: Dict[str, Any]):
        """Сохранение конфигурации в JSON"""
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Логирование метрик"""
        # Добавляем timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Логируем в WandB
        if self.use_wandb:
            wandb.log(metrics, step=step)
            
        # Сохраняем локально
        metrics_path = os.path.join(self.output_dir, "metrics.jsonl")
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
            
    def save_artifact(self, name: str, artifact_path: str):
        """Сохранение артефакта (например, модели)"""
        if self.use_wandb:
            artifact = wandb.Artifact(name, type='model')
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
            
    def finish(self):
        """Завершение эксперимента"""
        if self.use_wandb:
            wandb.finish() 