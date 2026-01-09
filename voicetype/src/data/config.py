"""
VoiceType - Configuration Manager
Управление конфигурацией приложения через YAML.
"""
import copy
import threading
from pathlib import Path
from typing import Any, Optional
import yaml
from loguru import logger

from src.utils.constants import (
    CONFIG_FILE, APP_DATA_DIR,
    DEFAULT_HOTKEY_START, DEFAULT_HOTKEY_STOP,
    THEME_DARK, OUTPUT_MODE_KEYBOARD
)


# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "audio": {
        "microphone_id": "default",
        "language": "ru",
        "model": "small",
        "vad_sensitivity": 0.5
    },
    "recognition": {
        "punctuation_enabled": True
    },
    "output": {
        "mode": OUTPUT_MODE_KEYBOARD
    },
    "hotkeys": {
        "start_recording": DEFAULT_HOTKEY_START,
        "stop_recording": DEFAULT_HOTKEY_STOP
    },
    "system": {
        "autostart": False,
        "theme": THEME_DARK
    },
    "internal": {
        "first_run": True,
        "window_geometry": None
    }
}


class Config:
    """
    Управление конфигурацией приложения.
    Поддерживает точечную нотацию для доступа к вложенным ключам.
    Автосохранение при изменениях.
    """

    def __init__(self, config_path: Path = None):
        """
        Args:
            config_path: Путь к файлу конфигурации. По умолчанию CONFIG_FILE.
        """
        self.config_path = config_path or CONFIG_FILE
        self._config: dict = {}
        self._loaded = False

    def load(self) -> None:
        """Загрузить конфиг из YAML файла."""
        # Создаём директорию если не существует
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        self._config = self._merge_with_defaults(loaded)
                    else:
                        self._config = copy.deepcopy(DEFAULT_CONFIG)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self._config = copy.deepcopy(DEFAULT_CONFIG)
        else:
            logger.info("Config file not found, using defaults")
            self._config = copy.deepcopy(DEFAULT_CONFIG)
            self.save()

        self._loaded = True

    def _merge_with_defaults(self, loaded: dict) -> dict:
        """Объединить загруженный конфиг с дефолтными значениями."""
        result = copy.deepcopy(DEFAULT_CONFIG)

        def deep_merge(base: dict, override: dict) -> dict:
            merged = base.copy()
            for key, value in override.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged

        return deep_merge(result, loaded)

    def save(self) -> None:
        """Сохранить конфиг в YAML файл."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            logger.debug(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Получить значение по ключу.
        Поддерживает точечную нотацию: 'audio.language'

        Args:
            key: Ключ (можно использовать точку для вложенных)
            default: Значение по умолчанию
        """
        if not self._loaded:
            self.load()

        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Установить значение и автосохранить.
        Поддерживает точечную нотацию: 'audio.language'

        Args:
            key: Ключ
            value: Новое значение
        """
        if not self._loaded:
            self.load()

        keys = key.split(".")
        config = self._config

        # Навигация до предпоследнего ключа
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Установка значения
        config[keys[-1]] = value
        logger.debug(f"Config updated: {key} = {value}")

        # Автосохранение
        self.save()

    def reset_to_defaults(self) -> None:
        """Сбросить к настройкам по умолчанию."""
        self._config = copy.deepcopy(DEFAULT_CONFIG)
        self.save()
        logger.info("Configuration reset to defaults")

    def get_all(self) -> dict:
        """Получить всю конфигурацию."""
        if not self._loaded:
            self.load()
        return self._config.copy()


# Глобальный экземпляр конфига (thread-safe singleton)
_config_instance: Optional[Config] = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """Получить глобальный экземпляр конфига (thread-safe singleton)."""
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = Config()
    return _config_instance


def _reset_config_instance() -> None:
    """
    Сбросить глобальный singleton экземпляр конфига.
    Используется только для тестов.
    """
    global _config_instance
    _config_instance = None
