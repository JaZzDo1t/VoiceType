"""
VoiceType - Database Tests
Тесты для модуля базы данных.
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.data.database import Database


class TestDatabase:
    """Тесты класса Database."""

    @pytest.fixture
    def temp_db(self):
        """Создать временную БД."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        db = Database(db_path)
        db.initialize()
        yield db

        # Cleanup
        if db_path.exists():
            db_path.unlink()

    # === Тесты истории ===

    def test_add_history_entry(self, temp_db):
        """Добавление записи в историю."""
        started = datetime.now()
        ended = started + timedelta(seconds=30)

        entry_id = temp_db.add_history_entry(
            started_at=started,
            ended_at=ended,
            text="Тестовый текст",
            language="ru"
        )

        assert entry_id is not None
        assert entry_id > 0

    def test_get_history(self, temp_db):
        """Получение истории."""
        # Добавляем записи
        for i in range(3):
            started = datetime.now()
            ended = started + timedelta(seconds=10)
            temp_db.add_history_entry(
                started_at=started,
                ended_at=ended,
                text=f"Текст {i}",
                language="ru"
            )

        history = temp_db.get_history()

        assert len(history) == 3
        # Проверяем что все записи получены (порядок может варьироваться
        # при быстром добавлении из-за одинакового created_at)
        texts = [h["text"] for h in history]
        assert "Текст 0" in texts
        assert "Текст 1" in texts
        assert "Текст 2" in texts

    def test_history_limit(self, temp_db):
        """Ограничение количества записей."""
        # Добавляем 20 записей
        for i in range(20):
            started = datetime.now()
            ended = started + timedelta(seconds=10)
            temp_db.add_history_entry(
                started_at=started,
                ended_at=ended,
                text=f"Текст {i}",
                language="ru"
            )

        history = temp_db.get_history()

        # Должно быть максимум 15
        assert len(history) <= 15

    def test_delete_history_entry(self, temp_db):
        """Удаление записи."""
        started = datetime.now()
        ended = started + timedelta(seconds=10)

        entry_id = temp_db.add_history_entry(
            started_at=started,
            ended_at=ended,
            text="Удаляемый текст",
            language="ru"
        )

        # Удаляем
        result = temp_db.delete_history_entry(entry_id)

        assert result is True

        # Проверяем что удалилась
        history = temp_db.get_history()
        assert len(history) == 0

    def test_clear_history(self, temp_db):
        """Очистка истории."""
        # Добавляем записи
        for i in range(5):
            started = datetime.now()
            ended = started + timedelta(seconds=10)
            temp_db.add_history_entry(
                started_at=started,
                ended_at=ended,
                text=f"Текст {i}",
                language="ru"
            )

        # Очищаем
        count = temp_db.clear_history()

        assert count == 5

        history = temp_db.get_history()
        assert len(history) == 0

    # === Тесты статистики ===

    def test_add_stats_entry(self, temp_db):
        """Добавление записи статистики."""
        temp_db.add_stats_entry(cpu_percent=25.5, ram_mb=150.3)

        stats = temp_db.get_stats_24h()

        assert len(stats) == 1
        assert stats[0]["cpu_percent"] == 25.5
        assert stats[0]["ram_mb"] == 150.3

    def test_stats_24h_filter(self, temp_db):
        """Фильтрация статистики за 24 часа."""
        # Добавляем записи
        for i in range(5):
            temp_db.add_stats_entry(cpu_percent=i * 10, ram_mb=100 + i * 10)

        stats = temp_db.get_stats_24h()

        assert len(stats) == 5

    def test_get_today_recognition_time(self, temp_db):
        """Получение времени распознавания за сегодня."""
        # Добавляем записи с разной длительностью
        for duration in [30, 60, 45]:  # Всего 135 секунд
            started = datetime.now()
            ended = started + timedelta(seconds=duration)
            temp_db.add_history_entry(
                started_at=started,
                ended_at=ended,
                text="Текст",
                language="ru"
            )

        total = temp_db.get_today_recognition_time()

        assert total == 135
