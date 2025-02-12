#!/bin/bash

# Массивы с параметрами
MODELS=("SASRec" "BERT4Rec")
FEATURE_CONFIGS=("id_only" "text_and_id" "full_features")
DATASETS=("rutube" "lastfm")
INPUT_FILES=("./data/rutube.csv" "./data/lastfm.csv")

# Создаем директорию для результатов
mkdir -p ./results

# Функция для запуска эксперимента и сохранения метрик
run_experiment() {
    local model=$1
    local feature_config=$2
    local input_file=$3
    local dataset_type=$4
    local experiment_name="${model,,}_${dataset_type}_${feature_config}"
    
    echo "Starting experiment: $experiment_name"
    echo "$(date): Starting $experiment_name" >> ./results/experiment_log.txt
    
    # Запускаем эксперимент
    python train.py \
        --model $model \
        --feature_config $feature_config \
        --input_file $input_file \
        --dataset_type $dataset_type \
        --experiment_name $experiment_name \
        --output_dir ./dataset 2>&1 | tee ./results/${experiment_name}_log.txt
    
    # Извлекаем лучшую метрику (MRR@10) из лога
    best_metric=$(grep "best valid MRR@10" ./results/${experiment_name}_log.txt | tail -n 1 | awk '{print $NF}')
    
    # Сохраняем результат
    echo "${experiment_name}: ${best_metric}" >> ./results/best_metrics.txt
    
    echo "Finished experiment: $experiment_name with best metric: ${best_metric}"
    echo "$(date): Finished $experiment_name" >> ./results/experiment_log.txt
    echo "----------------------------------------"
}

# Очищаем файлы с результатами
> ./results/best_metrics.txt
> ./results/experiment_log.txt

# Запускаем все комбинации
for ((i=0; i<${#DATASETS[@]}; i++)); do
    for model in "${MODELS[@]}"; do
        for feature_config in "${FEATURE_CONFIGS[@]}"; do
            run_experiment "$model" "$feature_config" "${INPUT_FILES[i]}" "${DATASETS[i]}"
        done
    done
done

# Сортируем результаты по метрике
echo "Final Results (sorted by metric):" >> ./results/best_metrics.txt
sort -t':' -k2 -nr ./results/best_metrics.txt >> ./results/best_metrics_sorted.txt

echo "All experiments completed! Check ./results/best_metrics_sorted.txt for final results." 