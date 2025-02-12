import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def temporal_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    time_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделение данных по времени на train, validation и test
    """
    # Сортируем по времени
    df = df.sort_values(time_col)
    
    # Определяем границы для split
    n_samples = len(df)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Разделяем данные
    train = df.iloc[:val_idx]
    val = df.iloc[val_idx:test_idx]
    test = df.iloc[test_idx:]
    
    return train, val, test

def random_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Случайное разделение данных на train, validation и test
    """
    # Сначала отделяем test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Затем отделяем validation от train
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=random_state
    )
    
    return train, val, test 