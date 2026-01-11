"""
VoiceType - Database Manager
SQLite база данных для истории распознавания и статистики.
"""
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from contextlib import contextmanager
from loguru import logger

from src.utils.constants import (
    DATABASE_FILE, MAX_HISTORY_ENTRIES,
    STATS_RETENTION_HOURS
)


class Database:
    """
    SQLite база данных для истории и статистики.
    Потокобезопасная через контекстный менеджер.
    """

    def __init__(self, db_path: Path = None):
        """
        Args:
            db_path: Путь к файлу БД. По умолчанию DATABASE_FILE.
        """
        self.db_path = db_path or DATABASE_FILE
        self._initialized = False

    def initialize(self) -> None:
        """Инициализировать БД, создать таблицы если не существуют."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Таблица истории распознавания
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at DATETIME NOT NULL,
                    ended_at DATETIME NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица статистики ресурсов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    cpu_percent REAL NOT NULL,
                    ram_mb REAL NOT NULL
                )
            """)

            # Индексы для быстрого поиска
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_created
                ON history(created_at DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stats_timestamp
                ON stats(timestamp DESC)
            """)

            conn.commit()

        self._initialized = True
        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Контекстный менеджер для подключения к БД."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_initialized(self) -> None:
        """Убедиться что БД инициализирована."""
        if not self._initialized:
            self.initialize()

    # ==================== История ====================

    def add_history_entry(
        self,
        started_at: datetime,
        ended_at: datetime,
        text: str,
        language: str
    ) -> int:
        """
        Добавить запись в историю.
        Автоматически удаляет старые записи (>MAX_HISTORY_ENTRIES).

        Returns:
            ID новой записи
        """
        self._ensure_initialized()

        duration = int((ended_at - started_at).total_seconds())

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Добавляем новую запись
            cursor.execute("""
                INSERT INTO history (started_at, ended_at, duration_seconds, text, language)
                VALUES (?, ?, ?, ?, ?)
            """, (started_at, ended_at, duration, text, language))

            entry_id = cursor.lastrowid

            # Удаляем старые записи, оставляя только MAX_HISTORY_ENTRIES
            cursor.execute("""
                DELETE FROM history
                WHERE id NOT IN (
                    SELECT id FROM history
                    ORDER BY created_at DESC
                    LIMIT ?
                )
            """, (MAX_HISTORY_ENTRIES,))

            conn.commit()

        logger.debug(f"History entry added: {entry_id}")
        return entry_id

    def get_history(self, limit: int = MAX_HISTORY_ENTRIES) -> List[Dict]:
        """
        Получить историю распознавания.

        Args:
            limit: Максимальное количество записей

        Returns:
            Список словарей с записями истории
        """
        self._ensure_initialized()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, started_at, ended_at, duration_seconds, text, language, created_at
                FROM history
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_history_entry(self, entry_id: int) -> bool:
        """
        Удалить запись из истории.

        Returns:
            True если запись была удалена
        """
        self._ensure_initialized()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM history WHERE id = ?", (entry_id,))
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.debug(f"History entry deleted: {entry_id}")
        return deleted

    def clear_history(self) -> int:
        """
        Очистить всю историю.

        Returns:
            Количество удалённых записей
        """
        self._ensure_initialized()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM history")
            conn.commit()
            count = cursor.rowcount

        logger.info(f"History cleared: {count} entries deleted")
        return count

    # ==================== Статистика ====================

    def add_stats_entry(self, cpu_percent: float, ram_mb: float) -> None:
        """Добавить запись статистики."""
        self._ensure_initialized()

        timestamp = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO stats (timestamp, cpu_percent, ram_mb)
                VALUES (?, ?, ?)
            """, (timestamp, cpu_percent, ram_mb))
            conn.commit()

    def get_stats_24h(self) -> List[Dict]:
        """
        Получить статистику за последние STATS_RETENTION_HOURS часов.

        Returns:
            Список словарей с записями статистики
        """
        self._ensure_initialized()

        cutoff = datetime.now() - timedelta(hours=STATS_RETENTION_HOURS)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, cpu_percent, ram_mb
                FROM stats
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """, (cutoff,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def cleanup_old_stats(self) -> int:
        """
        Удалить статистику старше 24 часов.

        Returns:
            Количество удалённых записей
        """
        self._ensure_initialized()

        cutoff = datetime.now() - timedelta(hours=STATS_RETENTION_HOURS)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM stats WHERE timestamp < ?", (cutoff,))
            conn.commit()
            count = cursor.rowcount

        if count > 0:
            logger.debug(f"Old stats cleaned up: {count} entries")
        return count

    def get_today_recognition_time(self) -> int:
        """
        Получить общее время распознавания за сегодня (в секундах).

        Returns:
            Суммарное время в секундах
        """
        self._ensure_initialized()

        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COALESCE(SUM(duration_seconds), 0) as total
                FROM history
                WHERE created_at >= ?
            """, (today_start,))

            row = cursor.fetchone()
            return row["total"] if row else 0


# Глобальный экземпляр БД (thread-safe singleton)
_db_instance: Optional[Database] = None
_db_lock = threading.Lock()


def get_database() -> Database:
    """Получить глобальный экземпляр БД (thread-safe singleton)."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = Database()
    return _db_instance


def _reset_database_instance() -> None:
    """
    Сбросить глобальный singleton экземпляр БД.
    Используется только для тестов.
    """
    global _db_instance
    _db_instance = None
