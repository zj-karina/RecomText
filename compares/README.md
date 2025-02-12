# Сравнительные эксперименты с моделями рекомендаций

Этот модуль содержит код для сравнения различных моделей рекомендаций на датасетах Rutube и LastFM.

## Поддерживаемые модели
- SASRec
- BERT4Rec

## Конфигурации признаков
- id_only: только ID пользователей и элементов
- text_and_id: ID + текстовые признаки
- full_features: все доступные признаки, включая социально-демографические

## Запуск экспериментов 
```bash
python train.py --model <model_name> --feature_config <config_name> --dataset <dataset_name>
```

## Параметры
- `--model`: название модели (SASRec или BERT4Rec)
- `--feature_config`: конфигурация признаков (id_only, text_and_id, full_features)
- `--dataset`: название датасета (Rutube или LastFM)

Пример запуска:
```bash
python train.py --model SASRec --feature_config id_only --dataset Rutube
```