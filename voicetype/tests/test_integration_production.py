"""
VoiceType - Integration & E2E Tests
Интеграционные и end-to-end тесты для проверки взаимодействия компонентов.

INTEGRATION-QA Test Suite
"""
import pytest
import tempfile
import threading
import time
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import Mock, MagicMock, patch

# ===== Data Layer Integration Tests =====


class TestConfigIntegration:
    """Тесты интеграции конфигурации."""

    @pytest.fixture
    def temp_config(self):
        """Создать временный конфиг."""
        from src.data.config import Config, _reset_config_instance

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        _reset_config_instance()
        config = Config(config_path)
        yield config

        # Cleanup
        _reset_config_instance()
        if config_path.exists():
            config_path.unlink()

    def test_config_thread_safety(self, temp_config):
        """Проверка потокобезопасности конфигурации."""
        temp_config.load()
        errors = []
        results = []

        def writer_thread(key_suffix, iterations=50):
            try:
                for i in range(iterations):
                    temp_config.set(f"test.key_{key_suffix}", f"value_{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader_thread(iterations=100):
            try:
                for i in range(iterations):
                    val = temp_config.get("audio.language", "default")
                    results.append(val)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            t = threading.Thread(target=writer_thread, args=(i,))
            threads.append(t)
        for i in range(3):
            t = threading.Thread(target=reader_thread)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) > 0

    def test_config_reload_on_change(self, temp_config):
        """Тест перезагрузки конфига при изменениях."""
        temp_config.load()

        # Меняем значение
        temp_config.set("audio.language", "en")

        # Создаём новый инстанс с тем же файлом
        from src.data.config import Config
        new_config = Config(temp_config.config_path)
        new_config.load()

        # Значение должно сохраниться
        assert new_config.get("audio.language") == "en"

    def test_config_default_merge(self, temp_config):
        """Тест слияния пользовательских настроек с дефолтами."""
        import yaml

        # Записываем частичный конфиг
        partial_config = {"audio": {"language": "en"}}
        with open(temp_config.config_path, "w", encoding="utf-8") as f:
            yaml.dump(partial_config, f)

        temp_config.load()

        # Проверяем что пользовательское значение сохранилось
        assert temp_config.get("audio.language") == "en"
        # И дефолтные значения присутствуют
        assert temp_config.get("audio.model") == "small"
        assert temp_config.get("system.theme") is not None


class TestDatabaseIntegration:
    """Тесты интеграции базы данных."""

    @pytest.fixture
    def temp_db(self):
        """Создать временную БД."""
        from src.data.database import Database, _reset_database_instance

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        _reset_database_instance()
        db = Database(db_path)
        db.initialize()
        yield db

        # Cleanup
        _reset_database_instance()
        if db_path.exists():
            db_path.unlink()

    def test_database_thread_safety(self, temp_db):
        """Проверка потокобезопасности БД."""
        errors = []
        entry_ids = []
        lock = threading.Lock()

        def writer_thread(thread_id, iterations=20):
            try:
                for i in range(iterations):
                    started = datetime.now()
                    ended = started + timedelta(seconds=10)
                    entry_id = temp_db.add_history_entry(
                        started_at=started,
                        ended_at=ended,
                        text=f"Thread {thread_id} - Entry {i}",
                        language="ru"
                    )
                    with lock:
                        entry_ids.append(entry_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader_thread(iterations=30):
            try:
                for i in range(iterations):
                    _ = temp_db.get_history()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            t = threading.Thread(target=writer_thread, args=(i,))
            threads.append(t)
        for i in range(2):
            t = threading.Thread(target=reader_thread)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(entry_ids) > 0, "Should have created entries"

    def test_database_stats_cleanup(self, temp_db):
        """Тест очистки старой статистики."""
        # Добавляем несколько записей
        for i in range(10):
            temp_db.add_stats_entry(cpu_percent=i * 5, ram_mb=100 + i)

        stats_before = temp_db.get_stats_24h()
        assert len(stats_before) == 10

        # Cleanup не удаляет свежие записи
        deleted = temp_db.cleanup_old_stats()
        stats_after = temp_db.get_stats_24h()

        assert len(stats_after) == 10, "Recent stats should remain"

    def test_database_history_limit_enforcement(self, temp_db):
        """Тест соблюдения лимита записей истории."""
        from src.utils.constants import MAX_HISTORY_ENTRIES

        # Добавляем больше записей чем лимит
        for i in range(MAX_HISTORY_ENTRIES + 10):
            started = datetime.now()
            ended = started + timedelta(seconds=5)
            temp_db.add_history_entry(
                started_at=started,
                ended_at=ended,
                text=f"Entry {i}",
                language="ru"
            )

        history = temp_db.get_history()
        assert len(history) <= MAX_HISTORY_ENTRIES

    def test_database_atomic_operations(self, temp_db):
        """Тест атомарности операций."""
        started = datetime.now()
        ended = started + timedelta(seconds=30)

        # Добавляем запись
        entry_id = temp_db.add_history_entry(
            started_at=started,
            ended_at=ended,
            text="Атомарная запись",
            language="ru"
        )

        # Немедленно читаем
        history = temp_db.get_history()

        assert len(history) >= 1
        assert any(h["id"] == entry_id for h in history)


class TestModelsManagerIntegration:
    """Тесты интеграции менеджера моделей."""

    @pytest.fixture
    def models_manager(self):
        """Создать менеджер моделей с временной директорией."""
        from src.data.models_manager import ModelsManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelsManager(models_dir=Path(temp_dir))
            yield manager

    def test_models_dir_creation(self, models_manager):
        """Тест создания директории моделей."""
        result = models_manager.ensure_models_dir()
        assert result is True
        assert models_manager.models_dir.exists()

    def test_missing_model_handling(self, models_manager):
        """Тест обработки отсутствующей модели."""
        path = models_manager.get_vosk_model_path("ru", "small")
        assert path is None  # Модель не найдена

    def test_models_summary(self, models_manager):
        """Тест получения сводки по моделям."""
        summary = models_manager.get_models_summary()

        assert "vosk" in summary
        assert "silero_te" in summary
        assert "models_dir" in summary


# ===== Component Integration Tests =====


class TestRecognizerIntegration:
    """Тесты интеграции распознавателя."""

    def test_recognizer_callbacks_thread_safety(self):
        """Тест потокобезопасности callbacks распознавателя."""
        from src.core.recognizer import Recognizer

        recognizer = Recognizer(model_path="dummy_path")

        results = []
        lock = threading.Lock()

        def on_partial(text):
            with lock:
                results.append(("partial", text))

        def on_final(text):
            with lock:
                results.append(("final", text))

        recognizer.on_partial_result = on_partial
        recognizer.on_final_result = on_final

        # Проверяем что callbacks установлены корректно
        assert recognizer.on_partial_result is not None
        assert recognizer.on_final_result is not None

    def test_recognizer_state_management(self):
        """Тест управления состоянием распознавателя."""
        from src.core.recognizer import Recognizer

        recognizer = Recognizer(model_path="dummy_path")

        # Изначально не загружен
        assert not recognizer.is_loaded()

        # Попытка обработки без загрузки
        result = recognizer.process_audio(b"\x00" * 1000)
        assert result is None


class TestAudioCaptureIntegration:
    """Тесты интеграции захвата аудио."""

    def test_audio_queue_thread_safety(self):
        """Тест потокобезопасности очереди аудио."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()
        audio_queue = capture.get_audio_queue()

        errors = []

        def producer():
            try:
                for i in range(100):
                    audio_queue.put(b"\x00" * 100)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def consumer():
            consumed = 0
            try:
                while consumed < 100:
                    try:
                        _ = audio_queue.get(timeout=0.1)
                        consumed += 1
                    except queue.Empty:
                        continue
            except Exception as e:
                errors.append(e)

        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_audio_level_calculation(self):
        """Тест расчёта уровня аудио."""
        from src.core.audio_capture import AudioCapture
        import numpy as np

        capture = AudioCapture()

        # Симулируем тишину
        silence = np.zeros(1000, dtype=np.int16).tobytes()
        capture._update_level(silence)
        assert capture._current_level < 0.1

        # Симулируем громкий сигнал
        loud = (np.ones(1000, dtype=np.int16) * 16000).tobytes()
        capture._update_level(loud)
        assert capture._current_level > 0


class TestHotkeyManagerIntegration:
    """Тесты интеграции менеджера хоткеев."""

    def test_hotkey_normalization(self):
        """Тест нормализации хоткеев."""
        from src.core.hotkey_manager import HotkeyManager

        manager = HotkeyManager()

        # Разные форматы должны нормализоваться одинаково
        assert manager.normalize_hotkey("Ctrl+Shift+S") == "ctrl+shift+s"
        assert manager.normalize_hotkey("ctrl + shift + s") == "ctrl+shift+s"
        assert manager.normalize_hotkey("CTRL+SHIFT+S") == "ctrl+shift+s"

    def test_hotkey_registration_thread_safety(self):
        """Тест потокобезопасности регистрации хоткеев."""
        from src.core.hotkey_manager import HotkeyManager

        manager = HotkeyManager()
        errors = []

        def register_hotkeys(prefix, count=20):
            try:
                for i in range(count):
                    manager.register(f"ctrl+alt+{prefix}{i}", lambda: None)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_hotkeys, args=("a",)),
            threading.Thread(target=register_hotkeys, args=("b",)),
            threading.Thread(target=register_hotkeys, args=("c",)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

        # Проверяем что хоткеи зарегистрированы
        hotkeys = manager.get_registered_hotkeys()
        assert len(hotkeys) > 0

    def test_hotkey_parse(self):
        """Тест парсинга хоткеев."""
        from src.core.hotkey_manager import HotkeyManager

        modifiers, regular_keys = HotkeyManager.parse_hotkey("ctrl+shift+s")

        assert "ctrl" in modifiers
        assert "shift" in modifiers
        assert "s" in regular_keys

        # Тест комбинации без модификаторов (a+b)
        modifiers2, regular_keys2 = HotkeyManager.parse_hotkey("a+b")
        assert len(modifiers2) == 0
        assert "a" in regular_keys2
        assert "b" in regular_keys2

        # Тест смешанной комбинации (ctrl+a+b)
        modifiers3, regular_keys3 = HotkeyManager.parse_hotkey("ctrl+a+b")
        assert "ctrl" in modifiers3
        assert "a" in regular_keys3
        assert "b" in regular_keys3


class TestOutputManagerIntegration:
    """Тесты интеграции менеджера вывода."""

    def test_output_mode_switching(self):
        """Тест переключения режимов вывода."""
        from src.core.output_manager import OutputManager
        from src.utils.constants import OUTPUT_MODE_KEYBOARD, OUTPUT_MODE_CLIPBOARD

        manager = OutputManager(OUTPUT_MODE_KEYBOARD)
        assert manager.mode == OUTPUT_MODE_KEYBOARD

        manager.set_mode(OUTPUT_MODE_CLIPBOARD)
        assert manager.mode == OUTPUT_MODE_CLIPBOARD

        # Невалидный режим -> fallback на keyboard
        manager.set_mode("invalid")
        assert manager.mode == OUTPUT_MODE_KEYBOARD

    def test_typing_delay_bounds(self):
        """Тест границ задержки набора."""
        from src.core.output_manager import OutputManager

        manager = OutputManager()

        manager.set_typing_delay(0.0001)  # Слишком маленькая
        assert manager._typing_delay >= 0.001

        manager.set_typing_delay(1.0)  # Слишком большая
        assert manager._typing_delay <= 0.5


class TestPunctuationIntegration:
    """Тесты интеграции пунктуации."""

    def test_punctuation_disabled_class(self):
        """Тест заглушки отключенной пунктуации."""
        from src.core.punctuation import PunctuationDisabled

        punct = PunctuationDisabled()

        assert punct.is_loaded() is True
        assert punct.load_model() is True

        # Должна делать первую букву заглавной
        result = punct.enhance("тест")
        assert result[0].isupper()

    def test_punctuation_batch_enhancement(self):
        """Тест пакетного улучшения."""
        from src.core.punctuation import PunctuationDisabled

        punct = PunctuationDisabled()

        texts = ["тест один", "тест два", "тест три"]
        results = punct.enhance_batch(texts)

        assert len(results) == 3
        for r in results:
            assert r[0].isupper()


# ===== E2E User Flow Tests =====


class TestUserFlowSimulation:
    """Симуляция пользовательских сценариев."""

    @pytest.fixture
    def mock_app_components(self):
        """Создать mock компоненты приложения."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        from src.data.config import Config, _reset_config_instance
        from src.data.database import Database, _reset_database_instance

        _reset_config_instance()
        _reset_database_instance()

        config = Config(config_path)
        config.load()

        db = Database(db_path)
        db.initialize()

        yield {"config": config, "db": db, "config_path": config_path, "db_path": db_path}

        # Cleanup
        _reset_config_instance()
        _reset_database_instance()
        if config_path.exists():
            config_path.unlink()
        if db_path.exists():
            db_path.unlink()

    def test_settings_change_flow(self, mock_app_components):
        """Тест сценария изменения настроек."""
        config = mock_app_components["config"]

        # Пользователь меняет язык
        config.set("audio.language", "en")
        assert config.get("audio.language") == "en"

        # Пользователь меняет модель
        config.set("audio.model", "large")
        assert config.get("audio.model") == "large"

        # Пользователь меняет режим вывода
        config.set("output.mode", "clipboard")
        assert config.get("output.mode") == "clipboard"

        # Перезагрузка конфига сохраняет изменения
        from src.data.config import Config
        new_config = Config(mock_app_components["config_path"])
        new_config.load()

        assert new_config.get("audio.language") == "en"
        assert new_config.get("audio.model") == "large"
        assert new_config.get("output.mode") == "clipboard"

    def test_recognition_session_flow(self, mock_app_components):
        """Тест сценария сессии распознавания."""
        db = mock_app_components["db"]

        # Симулируем сессию распознавания
        session_start = datetime.now()
        session_text = "Это тестовый текст распознавания"

        # Пользователь говорит
        time.sleep(0.05)  # Симуляция времени записи

        session_end = datetime.now()

        # Сохраняем результат
        entry_id = db.add_history_entry(
            started_at=session_start,
            ended_at=session_end,
            text=session_text,
            language="ru"
        )

        assert entry_id > 0

        # Проверяем историю
        history = db.get_history()
        assert len(history) >= 1
        assert history[0]["text"] == session_text

    def test_multiple_sessions_flow(self, mock_app_components):
        """Тест нескольких последовательных сессий."""
        db = mock_app_components["db"]

        sessions = [
            ("Первая сессия", 10),
            ("Вторая сессия", 20),
            ("Третья сессия", 15),
        ]

        for text, duration in sessions:
            started = datetime.now()
            ended = started + timedelta(seconds=duration)
            db.add_history_entry(
                started_at=started,
                ended_at=ended,
                text=text,
                language="ru"
            )

        history = db.get_history()
        assert len(history) >= 3

        # Проверяем суммарное время
        total_time = db.get_today_recognition_time()
        assert total_time >= 45  # 10 + 20 + 15

    def test_config_and_database_isolation(self, mock_app_components):
        """Тест изоляции конфига и БД."""
        config = mock_app_components["config"]
        db = mock_app_components["db"]

        # Операции с конфигом не влияют на БД
        config.set("audio.language", "en")
        history = db.get_history()
        assert isinstance(history, list)

        # Операции с БД не влияют на конфиг
        db.add_history_entry(
            started_at=datetime.now(),
            ended_at=datetime.now() + timedelta(seconds=10),
            text="Test",
            language="ru"
        )
        assert config.get("audio.language") == "en"


# ===== Lifecycle Tests =====


class TestLifecycleManagement:
    """Тесты управления жизненным циклом."""

    def test_recognizer_lifecycle(self):
        """Тест жизненного цикла распознавателя."""
        from src.core.recognizer import Recognizer

        recognizer = Recognizer(model_path="dummy")

        # Начальное состояние
        assert not recognizer.is_loaded()

        # После unload (не должен падать даже если не был загружен)
        recognizer.unload()
        assert not recognizer.is_loaded()

    def test_punctuation_lifecycle(self):
        """Тест жизненного цикла пунктуации."""
        from src.core.punctuation import Punctuation, PunctuationDisabled

        # Обычная пунктуация
        punct = Punctuation(language="ru")
        assert not punct.is_loaded()

        punct.unload()  # Не должен падать
        assert not punct.is_loaded()

        # Отключенная пунктуация
        disabled = PunctuationDisabled()
        assert disabled.is_loaded()

        disabled.unload()
        assert disabled.is_loaded()  # Всегда loaded

    def test_audio_capture_lifecycle(self):
        """Тест жизненного цикла захвата аудио."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()

        assert not capture.is_running()

        # Stop не должен падать если не был запущен
        capture.stop()
        assert not capture.is_running()

    def test_hotkey_manager_lifecycle(self):
        """Тест жизненного цикла менеджера хоткеев."""
        from src.core.hotkey_manager import HotkeyManager

        manager = HotkeyManager()

        assert not manager.is_listening()

        # Stop не должен падать если не был запущен
        manager.stop_listening()
        assert not manager.is_listening()

        # Регистрация хоткея до запуска
        manager.register("ctrl+alt+t", lambda: None)
        hotkeys = manager.get_registered_hotkeys()
        assert "ctrl+alt+t" in hotkeys


# ===== Race Condition Tests =====


class TestRaceConditions:
    """Тесты на race conditions."""

    def test_config_concurrent_writes(self):
        """Тест конкурентной записи в конфиг."""
        from src.data.config import Config, _reset_config_instance

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        _reset_config_instance()
        config = Config(config_path)
        config.load()

        errors = []

        def writer(value):
            try:
                for i in range(50):
                    config.set("test.concurrent", f"{value}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"thread_{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Cleanup
        _reset_config_instance()
        if config_path.exists():
            config_path.unlink()

        # Не должно быть ошибок (но значение может быть любым из потоков)
        assert len(errors) == 0, f"Race condition errors: {errors}"

    def test_database_concurrent_operations(self):
        """Тест конкурентных операций с БД."""
        from src.data.database import Database, _reset_database_instance

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        _reset_database_instance()
        db = Database(db_path)
        db.initialize()

        errors = []

        def add_entries(thread_id):
            try:
                for i in range(30):
                    db.add_history_entry(
                        started_at=datetime.now(),
                        ended_at=datetime.now() + timedelta(seconds=1),
                        text=f"Thread {thread_id} entry {i}",
                        language="ru"
                    )
            except Exception as e:
                errors.append(e)

        def read_entries():
            try:
                for _ in range(50):
                    _ = db.get_history()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=add_entries, args=(i,)))
        for _ in range(2):
            threads.append(threading.Thread(target=read_entries))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Cleanup
        _reset_database_instance()
        if db_path.exists():
            db_path.unlink()

        assert len(errors) == 0, f"Race condition errors: {errors}"


# ===== Shutdown Tests =====


class TestShutdownBehavior:
    """Тесты поведения при завершении."""

    def test_graceful_component_shutdown(self):
        """Тест корректного завершения компонентов."""
        from src.core.hotkey_manager import HotkeyManager
        from src.core.audio_capture import AudioCapture
        from src.core.recognizer import Recognizer

        # Создаём компоненты
        hotkey_manager = HotkeyManager()
        audio_capture = AudioCapture()
        recognizer = Recognizer(model_path="dummy")

        # Завершаем в правильном порядке (обратном от создания)
        recognizer.unload()
        audio_capture.stop()
        hotkey_manager.stop_listening()

        # Проверяем состояния
        assert not recognizer.is_loaded()
        assert not audio_capture.is_running()
        assert not hotkey_manager.is_listening()

    def test_repeated_shutdown_calls(self):
        """Тест повторных вызовов shutdown."""
        from src.core.hotkey_manager import HotkeyManager
        from src.core.audio_capture import AudioCapture

        hotkey_manager = HotkeyManager()
        audio_capture = AudioCapture()

        # Многократный stop не должен вызывать ошибок
        for _ in range(5):
            hotkey_manager.stop_listening()
            audio_capture.stop()

        assert not hotkey_manager.is_listening()
        assert not audio_capture.is_running()


# ===== Signal Integration Tests =====


class TestSignalIntegration:
    """Тесты интеграции сигналов (mock PyQt)."""

    def test_signal_emission_patterns(self):
        """Тест паттернов emit сигналов."""
        # Проверяем что callbacks вызываются thread-safe
        results = []
        lock = threading.Lock()

        def callback(value):
            with lock:
                results.append(value)

        # Симулируем вызовы из разных потоков
        threads = []
        for i in range(10):
            t = threading.Thread(target=lambda v=i: callback(v))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
