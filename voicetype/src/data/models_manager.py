"""
VoiceType - Models Manager
Управление моделями распознавания речи (Vosk) и пунктуации (Silero TE).
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from src.utils.constants import SUPPORTED_LANGUAGES, MODEL_SIZES


class ModelsManager:
    """
    Управление моделями Vosk и Silero.
    Отвечает за поиск, проверку наличия и получение путей к моделям.
    """

    # Названия папок с моделями Vosk
    VOSK_MODEL_NAMES = {
        "ru": {
            "small": "vosk-model-small-ru-0.22",
            "large": "vosk-model-ru-0.42"
        },
        "en": {
            "small": "vosk-model-small-en-us-0.15",
            "large": "vosk-model-en-us-0.22"
        }
    }

    # Альтернативные названия (для совместимости)
    VOSK_MODEL_ALIASES = {
        "vosk-model-small-ru": "vosk-model-small-ru-0.22",
        "vosk-model-ru": "vosk-model-ru-0.42",
        "vosk-model-small-en": "vosk-model-small-en-us-0.15",
        "vosk-model-en": "vosk-model-en-us-0.22",
    }

    # Названия файлов моделей Silero TE
    SILERO_MODEL_FILES = {
        "multi": "v2_4lang_q.pt",  # Многоязычная модель
        "ru": "v4_ru.pt",
        "en": "v4_en.pt",
    }

    def __init__(self, models_dir: Path = None):
        """
        Args:
            models_dir: Директория с моделями. По умолчанию ищется рядом с exe/скриптом.
        """
        self.models_dir = models_dir or self._get_default_models_dir()
        logger.debug(f"Models directory: {self.models_dir}")

    def _get_default_models_dir(self) -> Path:
        """Получить директорию с моделями по умолчанию."""
        if getattr(sys, 'frozen', False):
            # Запущено как exe (PyInstaller)
            # PyInstaller кладёт данные в _internal/
            base_dir = Path(sys.executable).parent / "_internal"
        else:
            # Запущено как скрипт
            base_dir = Path(__file__).parent.parent.parent

        return base_dir / "models"

    def get_vosk_model_path(self, language: str, size: str) -> Optional[Path]:
        """
        Получить путь к модели Vosk.

        Args:
            language: "ru" или "en"
            size: "small" или "large"

        Returns:
            Path к папке с моделью или None если не найдена
        """
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language: {language}")
            language = "ru"

        if size not in MODEL_SIZES:
            logger.warning(f"Unsupported model size: {size}")
            size = "small"

        # Основное название модели
        model_name = self.VOSK_MODEL_NAMES.get(language, {}).get(size)

        if model_name:
            model_path = self.models_dir / model_name
            if model_path.exists():
                return model_path

        # Ищем по альтернативным названиям
        for alias, canonical in self.VOSK_MODEL_ALIASES.items():
            if language in alias and (size == "small") == ("small" in alias):
                alias_path = self.models_dir / alias
                if alias_path.exists():
                    return alias_path

        # Ищем любую подходящую модель
        if self.models_dir.exists():
            for folder in self.models_dir.iterdir():
                if folder.is_dir() and "vosk" in folder.name.lower():
                    if language in folder.name.lower():
                        if (size == "small") == ("small" in folder.name.lower()):
                            return folder

        logger.warning(f"Vosk model not found: {language}/{size}")
        return None

    def get_available_vosk_models(self) -> List[Dict]:
        """
        Получить список доступных моделей Vosk.

        Returns:
            Список словарей с информацией о моделях:
            [{"language": "ru", "size": "small", "path": Path, "name": str}, ...]
        """
        models = []

        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return models

        for language in SUPPORTED_LANGUAGES:
            for size in MODEL_SIZES:
                path = self.get_vosk_model_path(language, size)
                if path:
                    models.append({
                        "language": language,
                        "size": size,
                        "path": path,
                        "name": path.name,
                        "size_mb": self._get_folder_size_mb(path)
                    })

        return models

    def _get_folder_size_mb(self, folder: Path) -> float:
        """Получить размер папки в МБ."""
        try:
            total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
            return round(total / (1024 * 1024), 1)
        except Exception:
            return 0.0

    def is_vosk_model_available(self, language: str, size: str) -> bool:
        """
        Проверить, доступна ли модель Vosk.

        Args:
            language: "ru" или "en"
            size: "small" или "large"

        Returns:
            True если модель доступна
        """
        return self.get_vosk_model_path(language, size) is not None

    def get_silero_te_path(self) -> Optional[Path]:
        """
        Получить путь к локальной модели Silero TE.

        Returns:
            Path к папке с моделью или None если не найдена
        """
        silero_dir = self.models_dir / "silero-te"
        if silero_dir.exists():
            # Проверяем есть ли .pt файлы
            pt_files = list(silero_dir.glob("*.pt"))
            if pt_files:
                return silero_dir
        return None

    def get_silero_te_info(self) -> Dict:
        """
        Получить информацию о модели Silero TE.
        Поддерживает как локальную модель, так и загрузку из torch hub.

        Returns:
            Словарь с информацией
        """
        local_path = self.get_silero_te_path()
        local_available = local_path is not None

        # Получаем список локальных моделей
        local_models = []
        if local_path:
            local_models = [f.name for f in local_path.glob("*.pt")]

        return {
            "name": "Silero Text Enhancement",
            "source": "local" if local_available else "torch.hub (snakers4/silero-models)",
            "local_path": str(local_path) if local_path else None,
            "local_models": local_models,
            "local_available": local_available,
            "languages": ["ru", "en"],
            "torch_available": self._is_torch_available(),
            "available": local_available or self._is_torch_available()
        }

    def is_silero_te_local(self) -> bool:
        """
        Проверить, есть ли локальная модель Silero TE.

        Returns:
            True если локальная модель доступна
        """
        return self.get_silero_te_path() is not None

    def _is_torch_available(self) -> bool:
        """Проверить, доступен ли PyTorch."""
        # В frozen build проверяем маркер от runtime hook
        # НЕ делаем import torch - это вызывает двойную инициализацию!
        if getattr(sys, 'frozen', False):
            return getattr(sys, '_torch_rthook_success', False)
        # В обычном режиме проверяем sys.modules или пробуем импорт
        if 'torch' in sys.modules:
            return True
        try:
            import torch
            return True
        except ImportError:
            return False

    def get_models_summary(self) -> Dict:
        """
        Получить сводку по всем моделям.

        Returns:
            {
                "vosk": [list of available models],
                "silero_te": {info},
                "models_dir": str
            }
        """
        return {
            "vosk": self.get_available_vosk_models(),
            "silero_te": self.get_silero_te_info(),
            "models_dir": str(self.models_dir),
            "models_dir_exists": self.models_dir.exists()
        }

    def ensure_models_dir(self) -> bool:
        """
        Создать директорию для моделей если не существует.

        Returns:
            True если директория существует или создана
        """
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create models directory: {e}")
            return False


# Глобальный экземпляр
_models_manager: Optional[ModelsManager] = None


def get_models_manager() -> ModelsManager:
    """Получить глобальный экземпляр ModelsManager."""
    global _models_manager
    if _models_manager is None:
        _models_manager = ModelsManager()
    return _models_manager
