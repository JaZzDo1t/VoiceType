"""
VoiceType - Hotkeys Settings Tab
Вкладка 'Хоткеи' для настройки горячих клавиш.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal
from loguru import logger

from src.data.config import get_config
from src.ui.widgets.hotkey_edit import HotkeyEdit
from src.utils.constants import DEFAULT_HOTKEY_START, DEFAULT_HOTKEY_STOP


class TabHotkeys(QWidget):
    """
    Вкладка 'Хоткеи'.
    Настройка горячих клавиш для управления записью.
    """

    # Сигналы
    start_hotkey_changed = pyqtSignal(str)
    stop_hotkey_changed = pyqtSignal(str)
    hotkeys_reset = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = get_config()
        self._setup_ui()
        self._load_settings()
        self._connect_signals()

    def _setup_ui(self):
        """Настроить интерфейс."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Описание
        description = QLabel(
            "Настройте глобальные горячие клавиши для управления записью.\n"
            "Горячие клавиши работают даже когда приложение не в фокусе."
        )
        description.setObjectName("secondaryLabel")
        description.setWordWrap(True)
        layout.addWidget(description)

        # Секция ГОРЯЧИЕ КЛАВИШИ
        hotkeys_group = QGroupBox("ГОРЯЧИЕ КЛАВИШИ")
        hotkeys_layout = QVBoxLayout(hotkeys_group)
        hotkeys_layout.setSpacing(16)

        # Начать запись
        start_layout = QHBoxLayout()
        start_label = QLabel("Начать запись:")
        start_label.setMinimumWidth(150)
        start_layout.addWidget(start_label)

        self._start_hotkey = HotkeyEdit()
        start_layout.addWidget(self._start_hotkey)
        start_layout.addStretch()
        hotkeys_layout.addLayout(start_layout)

        # Остановить запись
        stop_layout = QHBoxLayout()
        stop_label = QLabel("Остановить запись:")
        stop_label.setMinimumWidth(150)
        stop_layout.addWidget(stop_label)

        self._stop_hotkey = HotkeyEdit()
        stop_layout.addWidget(self._stop_hotkey)
        stop_layout.addStretch()
        hotkeys_layout.addLayout(stop_layout)

        layout.addWidget(hotkeys_group)

        # Кнопка сброса
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()

        self._reset_btn = QPushButton("Сбросить по умолчанию")
        self._reset_btn.setObjectName("secondaryButton")
        reset_layout.addWidget(self._reset_btn)

        layout.addLayout(reset_layout)

        # Предупреждение
        warning = QLabel(
            "⚠️ Некоторые комбинации клавиш могут быть заняты другими приложениями "
            "или системой. Если хоткей не работает, попробуйте другую комбинацию."
        )
        warning.setObjectName("secondaryLabel")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        # Spacer
        layout.addStretch()

    def _load_settings(self):
        """Загрузить настройки из конфига."""
        start_hotkey = self._config.get("hotkeys.start_recording", DEFAULT_HOTKEY_START)
        stop_hotkey = self._config.get("hotkeys.stop_recording", DEFAULT_HOTKEY_STOP)

        self._start_hotkey.set_hotkey(start_hotkey)
        self._stop_hotkey.set_hotkey(stop_hotkey)

    def _connect_signals(self):
        """Подключить сигналы."""
        self._start_hotkey.hotkey_changed.connect(self._on_start_hotkey_changed)
        self._stop_hotkey.hotkey_changed.connect(self._on_stop_hotkey_changed)
        self._reset_btn.clicked.connect(self._on_reset_clicked)

    def _on_start_hotkey_changed(self, hotkey: str):
        """Обработчик изменения хоткея старта."""
        if hotkey:
            self._config.set("hotkeys.start_recording", hotkey)
            self.start_hotkey_changed.emit(hotkey)
            logger.info(f"Start hotkey changed: {hotkey}")

    def _on_stop_hotkey_changed(self, hotkey: str):
        """Обработчик изменения хоткея стопа."""
        if hotkey:
            self._config.set("hotkeys.stop_recording", hotkey)
            self.stop_hotkey_changed.emit(hotkey)
            logger.info(f"Stop hotkey changed: {hotkey}")

    def _on_reset_clicked(self):
        """Сбросить хоткеи по умолчанию."""
        self._config.set("hotkeys.start_recording", DEFAULT_HOTKEY_START)
        self._config.set("hotkeys.stop_recording", DEFAULT_HOTKEY_STOP)

        self._start_hotkey.set_hotkey(DEFAULT_HOTKEY_START)
        self._stop_hotkey.set_hotkey(DEFAULT_HOTKEY_STOP)

        self.start_hotkey_changed.emit(DEFAULT_HOTKEY_START)
        self.stop_hotkey_changed.emit(DEFAULT_HOTKEY_STOP)
        self.hotkeys_reset.emit()

        logger.info("Hotkeys reset to defaults")

    def get_start_hotkey(self) -> str:
        """Получить хоткей старта."""
        return self._start_hotkey.get_hotkey()

    def get_stop_hotkey(self) -> str:
        """Получить хоткей стопа."""
        return self._stop_hotkey.get_hotkey()

    def refresh(self):
        """Обновить вкладку."""
        self._load_settings()
