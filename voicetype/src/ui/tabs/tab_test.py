"""
VoiceType - Test Tab
Вкладка 'Тест' для тестирования микрофона и распознавания.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from loguru import logger

from src.data.config import get_config
from src.ui.widgets.level_meter import LevelMeterWithLabel
from src.utils.system_info import get_microphone_by_id


class TabTest(QWidget):
    """
    Вкладка 'Тест'.
    Тестирование микрофона и распознавания.
    """

    # Сигналы
    test_started = pyqtSignal()
    test_stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = get_config()
        self._is_testing = False
        self._models_ready = False
        self._final_text = ""  # Накопленный финальный текст с пунктуацией
        self._setup_ui()

    def _setup_ui(self):
        """Настроить интерфейс."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Информация о микрофоне
        mic_group = QGroupBox("МИКРОФОН")
        mic_layout = QVBoxLayout(mic_group)

        mic_info_layout = QHBoxLayout()
        mic_info_layout.addWidget(QLabel("Текущий микрофон:"))
        self._mic_label = QLabel("-")
        self._mic_label.setStyleSheet("font-weight: bold;")
        mic_info_layout.addWidget(self._mic_label)
        mic_info_layout.addStretch()
        mic_layout.addLayout(mic_info_layout)

        # Индикатор уровня
        self._level_meter = LevelMeterWithLabel("Уровень сигнала")
        mic_layout.addWidget(self._level_meter)

        layout.addWidget(mic_group)

        # Тестирование
        test_group = QGroupBox("ТЕСТИРОВАНИЕ")
        test_layout = QVBoxLayout(test_group)

        # Описание
        description = QLabel(
            "Нажмите кнопку 'Начать тест' и говорите в микрофон.\n"
            "Результат распознавания появится в поле ниже."
        )
        description.setObjectName("secondaryLabel")
        description.setWordWrap(True)
        test_layout.addWidget(description)

        # Кнопка теста
        btn_layout = QHBoxLayout()

        self._test_btn = QPushButton("Начать тест")
        self._test_btn.setMinimumWidth(150)
        self._test_btn.setMinimumHeight(40)
        self._test_btn.clicked.connect(self._toggle_test)
        btn_layout.addWidget(self._test_btn)

        btn_layout.addStretch()

        # Статус
        self._status_label = QLabel("Готов к тестированию")
        self._status_label.setObjectName("secondaryLabel")
        btn_layout.addWidget(self._status_label)

        test_layout.addLayout(btn_layout)

        # Результат
        result_label = QLabel("Результат распознавания:")
        test_layout.addWidget(result_label)

        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMinimumHeight(150)
        self._result_text.setPlaceholderText("Здесь появится распознанный текст...")
        test_layout.addWidget(self._result_text)

        # Кнопки управления результатом
        result_btn_layout = QHBoxLayout()
        result_btn_layout.addStretch()

        clear_btn = QPushButton("Очистить")
        clear_btn.setObjectName("secondaryButton")
        clear_btn.clicked.connect(self._clear_result)
        result_btn_layout.addWidget(clear_btn)

        copy_btn = QPushButton("Копировать")
        copy_btn.setObjectName("secondaryButton")
        copy_btn.clicked.connect(self._copy_result)
        result_btn_layout.addWidget(copy_btn)

        test_layout.addLayout(result_btn_layout)

        layout.addWidget(test_group)

        # Spacer
        layout.addStretch()

        # Обновляем информацию о микрофоне
        self._update_mic_info()

        # Изначально кнопка неактивна до загрузки моделей
        self._test_btn.setEnabled(False)
        self._status_label.setText("Загрузка модели...")

    def _update_mic_info(self):
        """Обновить информацию о микрофоне."""
        mic_id = self._config.get("audio.microphone_id", "default")
        mic_info = get_microphone_by_id(mic_id)

        if mic_info:
            self._mic_label.setText(mic_info["name"])
        else:
            self._mic_label.setText(f"ID: {mic_id}")

    def _toggle_test(self):
        """Переключить тестирование."""
        if self._is_testing:
            self._stop_test()
        else:
            self._start_test()

    def _start_test(self):
        """Начать тестирование."""
        if self._is_testing:
            return  # Защита от повторных вызовов

        self._is_testing = True
        self._final_text = ""  # Сбрасываем накопленный текст
        self._test_btn.setText("Остановить тест")
        self._test_btn.setStyleSheet("background-color: #EF4444;")
        self._status_label.setText("Идёт запись...")
        self._result_text.clear()

        self.test_started.emit()
        logger.info("Test recording started")

    def _stop_test(self):
        """Остановить тестирование."""
        if not self._is_testing:
            return  # Защита от повторных вызовов

        # ВАЖНО: сначала emit сигнал (пока _is_testing ещё True),
        # чтобы финальный результат с пунктуацией попал в тест
        self.test_stopped.emit()

        # Теперь меняем состояние UI
        self._is_testing = False
        self._test_btn.setText("Начать тест")
        self._test_btn.setStyleSheet("")
        self._status_label.setText("Тест завершён")
        self._level_meter.reset()

        logger.info("Test recording stopped")

    def _clear_result(self):
        """Очистить результат."""
        self._final_text = ""
        self._result_text.clear()

    def _copy_result(self):
        """Копировать результат в буфер обмена."""
        text = self._result_text.toPlainText()
        if text:
            try:
                import pyperclip
                pyperclip.copy(text)
                self._status_label.setText("Скопировано!")
            except Exception as e:
                logger.error(f"Failed to copy: {e}")

    def update_level(self, level: float):
        """Обновить индикатор уровня."""
        self._level_meter.set_level(level)

    def append_partial_result(self, text: str):
        """Добавить промежуточный результат (показывает после финального текста)."""
        # Показываем финальный текст + текущий промежуточный
        if self._final_text:
            display = self._final_text + " " + text if text else self._final_text
        else:
            display = text if text else ""
        self._result_text.setPlainText(display)

    def append_final_result(self, text: str):
        """Добавить финальный результат (с пунктуацией)."""
        if text:
            if self._final_text:
                self._final_text += " " + text
            else:
                self._final_text = text
            self._result_text.setPlainText(self._final_text)

    def is_testing(self) -> bool:
        """Проверить, идёт ли тестирование."""
        return self._is_testing

    def refresh(self):
        """Обновить вкладку."""
        self._update_mic_info()

    def set_models_ready(self, ready: bool, model_name: str = ""):
        """
        Установить статус готовности моделей.

        Args:
            ready: Модели готовы к использованию
            model_name: Имя загруженной модели
        """
        self._models_ready = ready
        self._test_btn.setEnabled(ready)
        if ready:
            if model_name:
                self._status_label.setText(f"Готов ({model_name})")
            else:
                self._status_label.setText("Готов к тестированию")
        else:
            self._status_label.setText("Загрузка модели...")
