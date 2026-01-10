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
from src.utils.constants import DEFAULT_HOTKEY_TOGGLE


class TabHotkeys(QWidget):
    """
    Вкладка 'Хоткеи'.
    Настройка горячих клавиш для управления записью.
    Используется один toggle хоткей для старт/стоп записи.
    """

    # Сигналы
    toggle_hotkey_changed = pyqtSignal(str)
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
            "Настройте глобальную горячую клавишу для управления записью.\n"
            "Одна клавиша для старта и остановки записи (toggle).\n"
            "Горячие клавиши работают даже когда приложение не в фокусе."
        )
        description.setObjectName("secondaryLabel")
        description.setWordWrap(True)
        layout.addWidget(description)

        # Секция ГОРЯЧИЕ КЛАВИШИ
        hotkeys_group = QGroupBox("ГОРЯЧАЯ КЛАВИША ЗАПИСИ")
        hotkeys_layout = QVBoxLayout(hotkeys_group)
        hotkeys_layout.setSpacing(16)

        # Toggle запись (старт/стоп одной кнопкой)
        toggle_layout = QHBoxLayout()
        toggle_label = QLabel("Запись (старт/стоп):")
        toggle_label.setMinimumWidth(150)
        toggle_layout.addWidget(toggle_label)

        self._toggle_hotkey = HotkeyEdit()
        toggle_layout.addWidget(self._toggle_hotkey)
        toggle_layout.addStretch()
        hotkeys_layout.addLayout(toggle_layout)

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
        toggle_hotkey = self._config.get("hotkeys.toggle_recording", DEFAULT_HOTKEY_TOGGLE)
        self._toggle_hotkey.set_hotkey(toggle_hotkey)

    def _connect_signals(self):
        """Подключить сигналы."""
        self._toggle_hotkey.hotkey_changed.connect(self._on_toggle_hotkey_changed)
        self._reset_btn.clicked.connect(self._on_reset_clicked)

    def _on_toggle_hotkey_changed(self, hotkey: str):
        """Обработчик изменения toggle хоткея."""
        if hotkey:
            self._config.set("hotkeys.toggle_recording", hotkey)
            self.toggle_hotkey_changed.emit(hotkey)
            logger.info(f"Toggle hotkey changed: {hotkey}")

    def _on_reset_clicked(self):
        """Сбросить хоткей по умолчанию."""
        self._config.set("hotkeys.toggle_recording", DEFAULT_HOTKEY_TOGGLE)
        self._toggle_hotkey.set_hotkey(DEFAULT_HOTKEY_TOGGLE)
        self.toggle_hotkey_changed.emit(DEFAULT_HOTKEY_TOGGLE)
        self.hotkeys_reset.emit()
        logger.info("Toggle hotkey reset to default")

    def get_toggle_hotkey(self) -> str:
        """Получить toggle хоткей."""
        return self._toggle_hotkey.get_hotkey()

    def refresh(self):
        """Обновить вкладку."""
        self._load_settings()
