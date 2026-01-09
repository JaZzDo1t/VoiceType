"""
VoiceType - Config Tests
Тесты для модуля конфигурации.
"""
import pytest
import tempfile
from pathlib import Path

from src.data.config import Config, DEFAULT_CONFIG


class TestConfig:
    """Тесты класса Config."""

    @pytest.fixture
    def temp_config(self):
        """Создать временный конфиг."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        config = Config(config_path)
        yield config

        # Cleanup
        if config_path.exists():
            config_path.unlink()

    def test_load_creates_default(self, temp_config):
        """При отсутствии файла создаётся конфиг по умолчанию."""
        temp_config.load()

        assert temp_config._loaded
        assert temp_config._config is not None

    def test_get_simple_key(self, temp_config):
        """Получение простого ключа."""
        temp_config.load()

        # Проверяем дефолтные значения
        assert temp_config.get("audio.language") == "ru"
        assert temp_config.get("audio.model") == "small"

    def test_get_with_default(self, temp_config):
        """Получение несуществующего ключа с дефолтом."""
        temp_config.load()

        result = temp_config.get("nonexistent.key", "default_value")
        assert result == "default_value"

    def test_set_and_save(self, temp_config):
        """Установка значения и автосохранение."""
        temp_config.load()

        temp_config.set("audio.language", "en")

        assert temp_config.get("audio.language") == "en"

        # Проверяем что сохранилось
        new_config = Config(temp_config.config_path)
        new_config.load()
        assert new_config.get("audio.language") == "en"

    def test_dotted_notation(self, temp_config):
        """Точечная нотация для вложенных ключей."""
        temp_config.load()

        # Вложенный ключ
        temp_config.set("system.theme", "light")
        assert temp_config.get("system.theme") == "light"

    def test_reset_to_defaults(self, temp_config):
        """Сброс к значениям по умолчанию."""
        temp_config.load()

        # Меняем значение
        temp_config.set("audio.language", "en")
        assert temp_config.get("audio.language") == "en"

        # Сбрасываем
        temp_config.reset_to_defaults()

        assert temp_config.get("audio.language") == "ru"

    def test_get_all(self, temp_config):
        """Получение всей конфигурации."""
        temp_config.load()

        all_config = temp_config.get_all()

        assert isinstance(all_config, dict)
        assert "audio" in all_config
        assert "system" in all_config
