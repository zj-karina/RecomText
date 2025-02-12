import logging
from typing import Optional

def setup_logging(
    log_file: Optional[str] = 'training.log',
    level: int = logging.INFO
) -> logging.Logger:
    """
    Настройка логирования
    
    Args:
        log_file: Путь к файлу лога. Если None, логи только в консоль
        level: Уровень логирования
    """
    # Создаем логгер
    logger = logging.getLogger('recom_text_compares')
    logger.setLevel(level)
    
    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Добавляем вывод в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Добавляем вывод в файл если указан
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 