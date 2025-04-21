import pandas as pd
import numpy as np
from typing import Dict, List
import os
import json

def load_data() -> pd.DataFrame:
    """
    Загружает данные о репозиториях из repo_metadata.json
    """
    # Проверяем существование файла
    json_path = './data/repo_metadata.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Файл {json_path} не найден")
    
    # Загружаем данные из JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        repos_data = json.load(f)
    
    # Преобразуем в DataFrame
    data = pd.DataFrame(repos_data)
    
    # Обрабатываем вложенные структуры
    if 'languages' in data.columns:
        data['languages'] = data['languages'].apply(lambda x: [{"name": k, "size": v} for k, v in x.items()] if isinstance(x, dict) else x)
    
    if 'topics' in data.columns:
        data['topics'] = data['topics'].apply(lambda x: [{"name": t, "stars": 0} for t in x] if isinstance(x, list) else x)
    
    # Заполняем пропуски
    for col in ['description', 'primaryLanguage', 'license', 'codeOfConduct']:
        if col in data.columns:
            data[col] = data[col].fillna(None)
    
    # Преобразуем даты
    for col in ['createdAt', 'pushedAt']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])
    
    # Преобразуем числовые колонки
    numeric_cols = ['stars', 'forks', 'watchers', 'languageCount', 'topicCount', 
                   'diskUsageKb', 'pullRequests', 'issues', 'defaultBranchCommitCount', 
                   'assignableUserCount']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    # Преобразуем булевы колонки
    bool_cols = ['isFork', 'isArchived', 'forkingAllowed']
    for col in bool_cols:
        if col in data.columns:
            data[col] = data[col].fillna(False).astype(bool)
    
    print(f"Загружено {len(data)} репозиториев")
    return data

def create_repo_history_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает историю взаимодействия с репозиториями
    """
    # Создаем уникальный идентификатор для каждого репозитория
    df['repo_id'] = df['nameWithOwner'].apply(lambda x: x.replace('/', '_'))
    
    # Сортируем по дате последнего обновления
    df = df.sort_values('pushedAt')
    
    # Группируем по владельцу и собираем список репозиториев
    user_history = df.groupby('owner')['repo_id'].agg(list).reset_index()
    
    # Оставляем только пользователей с более чем одним репозиторием
    user_history = user_history[user_history['repo_id'].map(len) > 1]
    
    return user_history.reset_index(drop=True)

def create_detailed_repo_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает детальное текстовое описание репозиториев
    """
    def create_repo_description(row):
        parts = []
        
        if pd.notna(row['description']):
            parts.append(f"Описание: {row['description']}")
            
        if pd.notna(row['primaryLanguage']):
            parts.append(f"Основной язык: {row['primaryLanguage']}")
            
        if row['languages']:
            lang_str = ', '.join([f"{lang['name']} ({lang['size']} байт)" for lang in row['languages']])
            parts.append(f"Используемые языки: {lang_str}")
            
        if row['topics']:
            topics_str = ', '.join([topic['name'] for topic in row['topics']])
            parts.append(f"Темы: {topics_str}")
            
        if pd.notna(row['createdAt']):
            parts.append(f"Создан: {row['createdAt']}")
            
        if pd.notna(row['pushedAt']):
            parts.append(f"Последнее обновление: {row['pushedAt']}")
            
        return ' ; '.join(parts)
    
    # Создаем детальное описание для каждого репозитория
    df['detailed_view'] = df.apply(create_repo_description, axis=1)
    
    # Группируем по владельцу
    user_history = df.groupby('owner').agg({
        'detailed_view': lambda x: 'query: ' + ' ; '.join(x),
        'primaryLanguage': list  # Сохраняем список основных языков
    }).reset_index()
    
    return user_history

def create_user_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает текстовое описание пользователей на основе их активности
    """
    def create_description(group):
        row = group.iloc[0]
        
        # Собираем статистику по репозиториям пользователя
        total_stars = group['stars'].sum()
        total_forks = group['forks'].sum()
        total_watchers = group['watchers'].sum()
        total_commits = group['defaultBranchCommitCount'].sum()
        
        # Собираем информацию о языках
        languages = {}
        for langs in group['languages']:
            for lang in langs:
                name = lang['name']
                size = lang['size']
                if name in languages:
                    languages[name] += size
                else:
                    languages[name] = size
        
        # Сортируем языки по использованию
        top_languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:3]
        top_langs_str = ', '.join([f"{lang} ({size} байт)" for lang, size in top_languages])
        
        description = (
            f"passage: Владелец {len(group)} репозиториев, "
            f"всего {total_stars} звезд, {total_forks} форков, "
            f"{total_watchers} наблюдателей, {total_commits} коммитов. "
            f"Основные языки: {top_langs_str}"
        )
        return description
    
    user_descriptions = (df.groupby('owner')
                        .apply(create_description)
                        .reset_index()
                        .rename(columns={0: 'user_description'}))
    
    return user_descriptions

def create_repo_info_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает таблицу с информацией о репозиториях для инференса
    """
    # Создаем уникальный маппинг языков в числа
    all_languages = set()
    for langs in df['languages']:
        for lang in langs:
            all_languages.add(lang['name'])
    language_to_id = {lang: idx for idx, lang in enumerate(all_languages)}
    
    # Подготавливаем таблицу
    repo_info = df[['nameWithOwner', 'description', 'primaryLanguage', 'languages', 'topics']].copy()
    
    # Создаем маппинг ID репозиториев в числовые индексы
    repo_info['repo_id'] = repo_info['nameWithOwner'].apply(lambda x: x.replace('/', '_'))
    item_id_map = {repo_id: idx for idx, repo_id in enumerate(repo_info['repo_id'].unique())}
    
    # Добавляем числовой ID языка
    repo_info['language_id'] = repo_info['primaryLanguage'].map(language_to_id)
    
    # Заполняем пропуски
    repo_info['description'] = repo_info['description'].fillna('')
    repo_info['primaryLanguage'] = repo_info['primaryLanguage'].fillna('unknown')
    repo_info['language_id'] = repo_info['language_id'].fillna(-1)
    
    # Сохраняем маппинги
    mappings_dir = './data/mappings'
    os.makedirs(mappings_dir, exist_ok=True)
    
    # Сохраняем маппинг репозиториев
    with open(os.path.join(mappings_dir, 'item_id_map.json'), 'w', encoding='utf-8') as f:
        json.dump(item_id_map, f, ensure_ascii=False, indent=2)
    print(f"Saved item_id_map with {len(item_id_map)} items")
    
    # Сохраняем маппинг языков
    language_mapping = pd.DataFrame({
        'language': list(language_to_id.keys()),
        'language_id': list(language_to_id.values())
    })
    language_mapping.to_parquet(os.path.join(mappings_dir, 'language_mapping.parquet'))
    
    return repo_info

def main():
    # Загружаем данные
    data = load_data()
    
    # Создаем историю репозиториев
    user_history_df = create_repo_history_sorted(data)
    detailed_history_df = create_detailed_repo_history(data)
    user_descriptions = create_user_description(data)

    # Сохраняем результаты
    detailed_history_df.to_parquet('./data/textual_history.parquet')
    user_history_df.to_parquet('./data/id_history.parquet')
    user_descriptions.to_parquet('./data/user_descriptions.parquet')

    # Создаем таблицу для инференса
    repo_info = create_repo_info_table(data)
    repo_info.to_parquet('./data/repo_info.parquet')

if __name__ == "__main__":
    main()