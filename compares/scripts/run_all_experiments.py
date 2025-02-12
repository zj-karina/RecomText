import argparse
import yaml
import os
from itertools import product
from typing import List, Dict

def load_configs(config_dir: str) -> Dict[str, List[str]]:
    """Загрузка всех доступных конфигураций"""
    configs = {
        'models': [],
        'features': [],
        'datasets': ['rutube', 'lastfm']
    }
    
    # Загружаем модели
    model_dir = os.path.join(config_dir, 'model_configs')
    configs['models'] = [
        f.split('.')[0] for f in os.listdir(model_dir)
        if f.endswith('.yaml')
    ]
    
    # Загружаем конфигурации признаков
    feature_dir = os.path.join(config_dir, 'feature_configs')
    configs['features'] = [
        f.split('.')[0] for f in os.listdir(feature_dir)
        if f.endswith('.yaml')
    ]
    
    return configs

def run_experiment(model: str, feature_config: str, dataset: str, output_dir: str):
    """Запуск одного эксперимента"""
    command = (
        f"python train.py "
        f"--model {model} "
        f"--feature_config {feature_config} "
        f"--dataset {dataset} "
        f"--output_dir {output_dir}/{dataset}/{model}/{feature_config}"
    )
    print(f"Running: {command}")
    os.system(command)

def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--config_dir', type=str, default='configs',
                      help='Directory with configurations')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Output directory for experiments')
    parser.add_argument('--models', nargs='+',
                      help='Specific models to run (optional)')
    parser.add_argument('--features', nargs='+',
                      help='Specific feature configs to run (optional)')
    parser.add_argument('--datasets', nargs='+',
                      help='Specific datasets to run (optional)')
    
    args = parser.parse_args()
    
    # Загружаем все доступные конфигурации
    configs = load_configs(args.config_dir)
    
    # Фильтруем по аргументам командной строки
    if args.models:
        configs['models'] = [m for m in configs['models'] if m in args.models]
    if args.features:
        configs['features'] = [f for f in configs['features'] if f in args.features]
    if args.datasets:
        configs['datasets'] = [d for d in configs['datasets'] if d in args.datasets]
    
    # Запускаем все комбинации экспериментов
    for model, feature_config, dataset in product(
        configs['models'],
        configs['features'],
        configs['datasets']
    ):
        run_experiment(model, feature_config, dataset, args.output_dir)

if __name__ == '__main__':
    main() 