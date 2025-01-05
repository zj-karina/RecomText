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
    Creates detailed textual viewing history for users
    """
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
    
    # Format date
    df['formatted_date'] = pd.to_datetime(df['event_timestamp']).dt.strftime('%d %B')
    df['formatted_date'] = df['formatted_date'].apply(
        lambda x: f"{x.split()[0]} {month_dict.get(x.split()[1], x.split()[1])}"
    )

    # Define client type
    df['client_type'] = df['ua_client_type'].apply(
        lambda x: 'браузере' if x == 'browser' else 'приложении' if x == 'mobile app' else x
    )
    
    def create_view_description(row):
        parts = []

        parts.append('query: ')
        
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
            
        return ' '.join(parts) if parts else None
    
    df['detailed_view'] = df.apply(create_view_description, axis=1)
    
    # Group by user
    user_history = df.groupby('viewer_uid')['detailed_view'].agg(list).reset_index()
    
    # Keep only users with more than one view
    user_history = user_history[user_history['detailed_view'].map(len) > 1]
    
    return user_history.reset_index(drop=True)

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
    
    # Create viewing histories
    user_history_df = create_user_history_sorted(result)
    detailed_history_df = create_detailed_user_history(result_with_video_info)
    user_descriptions = create_user_description(targets)
    
    # Save results
    detailed_history_df.to_parquet('textual_history.parquet')
    user_history_df.to_parquet('id_history.parquet')
    user_descriptions.to_parquet('user_descriptions.parquet')

if __name__ == "__main__":
    main() 