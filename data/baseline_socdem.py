import pandas as pd
import numpy as np
from typing import Dict, List

def load_data() -> tuple:
    """
    Loads all required datasets
    """
    data = pd.read_csv('./data/train_events.csv')
    video = pd.read_csv('./data/video_info_v2.csv')
    targets = pd.read_csv('./data/train_targets.csv')
    all_events = pd.read_csv('./data/all_events.csv')
    
    return data, video, targets, all_events

def create_user_history_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates sorted viewing history for users
    """
    # Sort by timestamp
    df = df.sort_values('event_timestamp')
    
    # Remove 'video_' prefix from rutube_video_id
    df['clean_video_id'] = df['rutube_video_id'].str.replace('video_', '')
    
    # Group by user and collect list of viewed videos
    user_history = df.groupby('viewer_uid')['clean_video_id'].agg(list).reset_index()
    
    # Keep only users with more than one view
    user_history = user_history[user_history['clean_video_id'].map(len) > 1]
    
    return user_history.reset_index(drop=True)

def create_detailed_user_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates detailed textual viewing history for users as a single string in `detailed_view`.
    """
    # Словарь для перевода месяцев
    month_dict = {
        'January': 'января',
        'February': 'февраля',
        'March': 'марта',
        'April': 'апреля',
        'May': 'мая',
        'June': 'июня',
        'July': 'июля',
        'August': 'августа',
        'September': 'сентября',
        'October': 'октября',
        'November': 'ноября',
        'December': 'декабря'
    }
    
    # Форматируем дату
    df['formatted_date'] = pd.to_datetime(df['event_timestamp']).dt.strftime('%d %B')
    df['formatted_date'] = df['formatted_date'].apply(
        lambda x: f"{x.split()[0]} {month_dict.get(x.split()[1], x.split()[1])}"
    )

    # Определяем тип клиента
    df['client_type'] = df['ua_client_type'].apply(
        lambda x: 'браузере' if x == 'browser' else 'приложении' if x == 'mobile app' else x
    )
    
    # Создаем описание просмотра
    def create_view_description(row):
        parts = []

        if pd.notna(row['title']):
            parts.append('Название видео: ' + str(row['title']))
            
        if pd.notna(row['category']):
            parts.append('категории ' + str(row['category']))
            
        if pd.notna(row['client_type']):
            parts.append(f"просмотрено в {row['client_type']}")
            
        if pd.notna(row['ua_os']):
            parts.append(f"ОС {row['ua_os']}")
            
        if pd.notna(row['formatted_date']):
            parts.append(str(row['formatted_date']))
            
        # Сохраняем категорию отдельно
        category = row.get('category', 'unknown')
            
        return ' '.join(parts) if parts else None, category
    
    # Добавляем подробности о просмотре и категорию
    df[['detailed_view', 'category']] = df.apply(create_view_description, axis=1, result_type='expand')
    
    # Группируем по пользователям
    user_history = df.groupby('viewer_uid').agg({
        'detailed_view': lambda x: 'query: ' + ' ; '.join(filter(None, x)),
        'category': list  # Сохраняем список категорий
    }).reset_index()
    
    return user_history

def create_user_description(targets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates textual description of users based on their characteristics
    """
    # Convert seconds to hours
    
    targets_df['hours_watched'] = targets_df['total_watchtime'] / 3600
    
    # Find maximum watch time for each user
    max_watch_time = targets_df.groupby('viewer_uid')['hours_watched'].transform('max')
    
    # Filter rows with maximum watch time
    filtered_df = targets_df[targets_df['hours_watched'] == max_watch_time]
    
    def create_description(group):
        row = group.iloc[0]
        gender = "мужчина" if row['sex'] == 'male' else "женщина"
        regions = ' и '.join(group['region'].unique())
        
        description = (
            f"passage: {gender}, {row['age']} лет, "
            f"живет в {regions}, "
            f"смотрит на сайте {row['hours_watched']:.1f} часов"
        )
        return description
    
    user_descriptions = (filtered_df.groupby('viewer_uid')
                        .apply(create_description)
                        .reset_index()
                        .rename(columns={0: 'user_description'}))
    
    return user_descriptions

def main():
    # Load data
    data, video, targets, all_events = load_data()
    
    # Combine all events
    result = pd.concat([data, all_events], axis=0).drop_duplicates()
    result = result.reset_index(drop=True)
    
    # Merge with video information
    result_with_video_info = pd.merge(result, video, on='rutube_video_id', how='left')

    # Выбираем только колонки viewer_uid и region из result
    regions = result[['viewer_uid', 'region']].drop_duplicates()
    
    # Добавляем region к targets
    targets = pd.merge(targets, regions, on='viewer_uid', how='left')

    # Выбираем только колонки viewer_uid и region из result
    regions = result[['viewer_uid', 'total_watchtime']].drop_duplicates()
    
    # Добавляем region к targets
    targets = pd.merge(targets, regions, on='viewer_uid', how='left')
            
    Create viewing histories
    user_history_df = create_user_history_sorted(result)
    detailed_history_df = create_detailed_user_history(result_with_video_info)
    user_descriptions = create_user_description(targets)

    print(detailed_history_df)
    
    # Save results
    detailed_history_df.to_parquet('./data/textual_history.parquet')
    user_history_df.to_parquet('./data/id_history.parquet')
    user_descriptions.to_parquet('./data/user_descriptions.parquet')

if __name__ == "__main__":
    main() 