"""
VoiceType - Models Manager
Управление моделями распознавания речи Whisper.
"""
import threading
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from src.utils.constants import WHISPER_MODEL_SIZES, WHISPER_DEFAULT_MODEL


class ModelsManager:
    """
    Управление моделями Whisper.
    Отвечает за проверку наличия и получение информации о моделях.
    """

    # Информация о моделях Whisper (размер в MB)
    # Только base, small - убраны tiny, medium и large-v3
    WHISPER_MODEL_INFO = {
        "base": {"size_mb": 145, "params": "74M", "quality": "medium"},
        "small": {"size_mb": 488, "params": "244M", "quality": "good"},
    }

    def __init__(self):
        """Инициализация менеджера моделей."""
        logger.debug("ModelsManager initialized")

    # ========== Whisper методы ==========

    def get_whisper_cache_dir(self) -> Path:
        """
        Получить директорию кеша моделей Whisper.

        faster-whisper использует Hugging Face Hub для скачивания,
        модели кешируются в ~/.cache/huggingface/hub/

        Returns:
            Path к директории кеша
        """
        # Hugging Face Hub кеш
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        return hf_cache

    def is_whisper_model_cached(self, model_size: str) -> bool:
        """
        Проверить, скачана ли модель Whisper.

        Args:
            model_size: Размер модели (base, small)

        Returns:
            True если модель уже скачана в кеш
        """
        if model_size not in WHISPER_MODEL_SIZES:
            logger.warning(f"Unknown Whisper model size: {model_size}")
            return False

        cache_dir = self.get_whisper_cache_dir()
        if not cache_dir.exists():
            return False

        # faster-whisper использует модели из Systran/faster-whisper-*
        # Паттерн кеширования: models--Systran--faster-whisper-{size}
        model_cache_name = f"models--Systran--faster-whisper-{model_size}"
        model_path = cache_dir / model_cache_name

        if model_path.exists():
            # Проверяем что там есть файлы модели
            snapshots = model_path / "snapshots"
            if snapshots.exists() and any(snapshots.iterdir()):
                return True

        return False

    def get_whisper_model_info(self, model_size: str) -> Optional[Dict]:
        """
        Получить информацию о модели Whisper.

        Args:
            model_size: Размер модели

        Returns:
            Словарь с информацией или None
        """
        if model_size not in self.WHISPER_MODEL_INFO:
            return None

        info = self.WHISPER_MODEL_INFO[model_size].copy()
        info["name"] = model_size
        info["cached"] = self.is_whisper_model_cached(model_size)
        return info

    def get_available_whisper_models(self) -> List[Dict]:
        """
        Получить список всех моделей Whisper с информацией.

        Returns:
            Список словарей с информацией о моделях
        """
        models = []
        for size in WHISPER_MODEL_SIZES:
            info = self.get_whisper_model_info(size)
            if info:
                models.append(info)
        return models

    def get_whisper_info(self) -> Dict:
        """
        Получить общую информацию о Whisper.

        Returns:
            Словарь с информацией о Whisper
        """
        cached_models = [
            size for size in WHISPER_MODEL_SIZES
            if self.is_whisper_model_cached(size)
        ]

        return {
            "name": "faster-whisper",
            "description": "OpenAI Whisper with CTranslate2 (4x faster)",
            "available_sizes": WHISPER_MODEL_SIZES,
            "cached_models": cached_models,
            "default_model": WHISPER_DEFAULT_MODEL,
            "cache_dir": str(self.get_whisper_cache_dir()),
            "features": ["punctuation", "vad", "multilingual"],
            "languages": ["ru", "en", "auto"],
            "available": self._is_faster_whisper_available(),
        }

    def _is_faster_whisper_available(self) -> bool:
        """Проверить, установлен ли faster-whisper."""
        try:
            import faster_whisper
            return True
        except ImportError:
            return False


# Глобальный экземпляр (thread-safe singleton)
_models_manager: Optional[ModelsManager] = None
_models_lock = threading.Lock()


def get_models_manager() -> ModelsManager:
    """Получить глобальный экземпляр ModelsManager (thread-safe singleton)."""
    global _models_manager
    if _models_manager is None:
        with _models_lock:
            if _models_manager is None:
                _models_manager = ModelsManager()
    return _models_manager
