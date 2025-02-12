import argparse
import pandas as pd
import os
import json
from typing import Dict, List
import plotly.graph_objects as go
from ..utils.visualization import plot_metrics_comparison, plot_learning_curves

def load_experiment_results(experiment_dir: str) -> Dict:
    """Загрузка результатов эксперимента"""
    metrics_path = os.path.join(experiment_dir, "metrics.jsonl")
    config_path = os.path.join(experiment_dir, "config.json")
    
    # Загружаем метрики
    metrics = []
    with open(metrics_path, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    
    # Загружаем конфигурацию
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    return {
        'metrics': pd.DataFrame(metrics),
        'config': config
    }

def analyze_experiments(base_dir: str, output_dir: str):
    """Анализ всех экспериментов"""
    results = []
    
    # Собираем результаты всех экспериментов
    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        for model in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model)
            if not os.path.isdir(model_path):
                continue
                
            for feature_config in os.listdir(model_path):
                experiment_path = os.path.join(model_path, feature_config)
                if not os.path.isdir(experiment_path):
                    continue
                    
                try:
                    exp_results = load_experiment_results(experiment_path)
                    results.append({
                        'dataset': dataset,
                        'model': model,
                        'feature_config': feature_config,
                        **exp_results
                    })
                except Exception as e:
                    print(f"Error loading {experiment_path}: {e}")
    
    # Создаем отчеты
    os.makedirs(output_dir, exist_ok=True)
    
    # Сравнение метрик между моделями
    for dataset in set(r['dataset'] for r in results):
        dataset_results = [r for r in results if r['dataset'] == dataset]
        
        # Сравнение по последним метрикам
        final_metrics = {
            f"{r['model']}_{r['feature_config']}": r['metrics'].iloc[-1]
            for r in dataset_results
        }
        
        fig = plot_metrics_comparison(
            list(final_metrics.values()),
            ['NDCG@10', 'MRR@10', 'Recall@10'],
            list(final_metrics.keys()),
            f"Models Comparison - {dataset}"
        )
        fig.write_html(os.path.join(output_dir, f"{dataset}_comparison.html"))
        
        # Кривые обучения
        for result in dataset_results:
            fig = plot_learning_curves(
                result['metrics'],
                ['train_loss', 'valid_NDCG@10'],
                f"Learning Curves - {result['model']} ({result['feature_config']}) - {dataset}"
            )
            fig.write_html(os.path.join(
                output_dir,
                f"{dataset}_{result['model']}_{result['feature_config']}_learning_curves.html"
            ))
    
    # Создаем сводную таблицу
    summary = pd.DataFrame([
        {
            'dataset': r['dataset'],
            'model': r['model'],
            'feature_config': r['feature_config'],
            **r['metrics'].iloc[-1]
        }
        for r in results
    ])
    
    summary.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory with experiment results')
    parser.add_argument('--output_dir', type=str, default='analysis',
                      help='Output directory for analysis')
    
    args = parser.parse_args()
    analyze_experiments(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main() 