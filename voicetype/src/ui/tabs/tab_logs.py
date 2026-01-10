"""
VoiceType - Logs Tab
Вкладка 'Логи' для просмотра журнала событий.
"""
import os
import subprocess
import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from loguru import logger

from src.utils.constants import LOGS_DIR


class TabLogs(QWidget):
    """
    Вкладка 'Логи'.
    Просмотр журнала событий приложения.
    """

    # Уровни логирования
    LOG_LEVELS = [
        ("Все", None),
        ("DEBUG", "DEBUG"),
        ("INFO", "INFO"),
        ("WARNING", "WARNING"),
        ("ERROR", "ERROR"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_filter = None
        self._setup_ui()
        self._load_logs()

        # Автообновление каждые 5 секунд
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._load_logs)
        self._refresh_timer.start(5000)

    def _setup_ui(self):
        """Настроить интерфейс."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Верхняя панель
        top_layout = QHBoxLayout()

        # Фильтр по уровню
        top_layout.addWidget(QLabel("Фильтр:"))

        self._filter_combo = QComboBox()
        for name, level in self.LOG_LEVELS:
            self._filter_combo.addItem(name, level)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._filter_combo.setMinimumWidth(120)
        top_layout.addWidget(self._filter_combo)

        top_layout.addStretch()

        # Кнопка обновления
        refresh_btn = QPushButton("Обновить")
        refresh_btn.setObjectName("secondaryButton")
        refresh_btn.clicked.connect(self._load_logs)
        top_layout.addWidget(refresh_btn)

        # Кнопка открытия папки
        open_folder_btn = QPushButton("Открыть папку")
        open_folder_btn.setObjectName("secondaryButton")
        open_folder_btn.clicked.connect(self._open_logs_folder)
        top_layout.addWidget(open_folder_btn)

        # Кнопка очистки файла
        clear_btn = QPushButton("Очистить")
        clear_btn.setObjectName("dangerButton")
        clear_btn.clicked.connect(self._clear_log_file)
        top_layout.addWidget(clear_btn)

        layout.addLayout(top_layout)

        # Информация о файле
        self._file_label = QLabel("Файл: -")
        self._file_label.setObjectName("secondaryLabel")
        layout.addWidget(self._file_label)

        # Поле с логами
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        # Моноширинный шрифт
        font = QFont("Consolas", 9)
        if not font.exactMatch():
            font = QFont("Courier New", 9)
        self._log_view.setFont(font)

        layout.addWidget(self._log_view)

        # Статус
        self._status_label = QLabel("Строк: 0")
        self._status_label.setObjectName("secondaryLabel")
        layout.addWidget(self._status_label)

    def _get_log_file(self) -> Path:
        """Получить путь к файлу логов."""
        return LOGS_DIR / "voicetype.log"

    def _load_logs(self):
        """Загрузить логи из файла."""
        log_file = self._get_log_file()

        if not log_file.exists():
            self._file_label.setText("Файл: -")
            self._log_view.setPlainText("Файл логов не найден.\nЛоги появятся после запуска приложения.")
            self._status_label.setText("Строк: 0")
            return

        self._file_label.setText(f"Файл: {log_file.name}")

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Фильтруем по уровню если нужно
            if self._current_filter:
                filtered_lines = []
                for line in lines:
                    # Формат: HH:mm:ss [LEVEL] message
                    if f"[{self._current_filter}]" in line:
                        filtered_lines.append(line)
                    # Добавляем строки без уровня (продолжение предыдущей)
                    elif not any(f"[{level}]" in line for _, level in self.LOG_LEVELS if level):
                        if filtered_lines:
                            filtered_lines.append(line)
                lines = filtered_lines

            # Показываем все строки (новые сверху)
            display_lines = list(reversed(lines))

            text = "".join(display_lines)
            self._log_view.setPlainText(text)

            # Прокручиваем вверх (к новым записям)
            scrollbar = self._log_view.verticalScrollBar()
            scrollbar.setValue(0)

            self._status_label.setText(f"Строк: {len(display_lines)}")

        except Exception as e:
            self._log_view.setPlainText(f"Ошибка чтения логов: {e}")
            self._status_label.setText("Ошибка")
            logger.error(f"Failed to read logs: {e}")

    def _on_filter_changed(self, index):
        """Обработчик изменения фильтра."""
        self._current_filter = self._filter_combo.currentData()
        self._load_logs()

    def _open_logs_folder(self):
        """Открыть папку с логами в проводнике."""
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)

            if sys.platform == "win32":
                os.startfile(str(LOGS_DIR))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(LOGS_DIR)])
            else:
                subprocess.run(["xdg-open", str(LOGS_DIR)])

        except Exception as e:
            logger.error(f"Failed to open logs folder: {e}")

    def _clear_log_file(self):
        """Очистить файл логов."""
        log_file = self._get_log_file()
        try:
            if log_file.exists():
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("")
            self._load_logs()
            logger.info("Log file cleared by user")
        except Exception as e:
            logger.error(f"Failed to clear log file: {e}")

    def refresh(self):
        """Обновить вкладку."""
        self._load_logs()

    def append_log(self, message: str):
        """
        Добавить сообщение в отображение.
        Используется для live-логирования.
        """
        self._log_view.appendPlainText(message)

        # Прокручиваем вниз
        scrollbar = self._log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
