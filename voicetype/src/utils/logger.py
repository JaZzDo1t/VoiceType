"""
VoiceType - Logger Setup
Настройка loguru для логирования приложения.
"""
from pathlib import Path
from loguru import logger
import sys

from src.utils.constants import APP_NAME, LOGS_DIR, LOG_RETENTION_DAYS


def setup_logger(log_dir: Path = None) -> None:
    """
    Настроить loguru.

    Args:
        log_dir: Директория для логов. По умолчанию LOGS_DIR из constants.
    """
    if log_dir is None:
        log_dir = LOGS_DIR

    # Создаём директорию если не существует
    log_dir.mkdir(parents=True, exist_ok=True)

    # Удаляем дефолтный handler
    logger.remove()

    # Добавляем вывод в stderr для отладки (только если доступен)
    # В PyInstaller windowed mode sys.stderr может быть None
    if sys.stderr is not None:
        logger.add(
            sys.stderr,
            format="<level>{time:HH:mm:ss}</level> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level="DEBUG",
            colorize=True
        )

    # Добавляем файловый handler
    # Ограничение: ~200 записей (~20KB), без бэкапов - просто перезаписывать
    logger.add(
        log_dir / f"{APP_NAME.lower()}.log",
        rotation="20 KB",  # Ротация при ~200 строках
        retention=0,  # Не хранить старые файлы, просто перезаписывать
        format="{time:HH:mm:ss} [{level}] {message}",
        level="DEBUG",
        encoding="utf-8"
    )

    logger.info(f"{APP_NAME} logger initialized")


def get_logger():
    """Получить настроенный logger."""
    return logger


# Экспортируем logger для удобства импорта
__all__ = ["setup_logger", "get_logger", "logger"]
