"""
VoiceType - System Tray Icon
Иконка приложения в системном трее Windows.
"""
import sys
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction
from PyQt6.QtCore import pyqtSignal, QSize
from loguru import logger

from src.utils.constants import (
    APP_NAME,
    TRAY_STATE_READY, TRAY_STATE_RECORDING,
    TRAY_STATE_LOADING, TRAY_STATE_ERROR
)


class TrayIcon(QSystemTrayIcon):
    """
    Иконка приложения в трее Windows.
    Показывает состояние и предоставляет контекстное меню.
    """

    # Сигналы
    start_recording_clicked = pyqtSignal()
    stop_recording_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    exit_clicked = pyqtSignal()

    # Цвета состояний
    STATE_COLORS = {
        TRAY_STATE_READY: "#22C55E",      # Зелёный
        TRAY_STATE_RECORDING: "#EF4444",   # Красный
        TRAY_STATE_LOADING: "#EAB308",     # Жёлтый
        TRAY_STATE_ERROR: "#6B7280",       # Серый
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_state = TRAY_STATE_LOADING
        self._is_recording = False
        self._icons_cache = {}

        # Создаём меню
        self._setup_menu()

        # Устанавливаем начальную иконку
        self._update_icon()

        # Обработка клика по иконке
        self.activated.connect(self._on_activated)

        # Tooltip
        self.setToolTip(f"{APP_NAME} - Загрузка...")

    def _setup_menu(self):
        """Настроить контекстное меню."""
        menu = QMenu()

        # Действие записи
        self._recording_action = QAction("Начать запись", menu)
        self._recording_action.triggered.connect(self._on_recording_action)
        menu.addAction(self._recording_action)

        menu.addSeparator()

        # Настройки
        settings_action = QAction("Настройки", menu)
        settings_action.triggered.connect(self.settings_clicked.emit)
        menu.addAction(settings_action)

        menu.addSeparator()

        # Выход
        exit_action = QAction("Выход", menu)
        exit_action.triggered.connect(self.exit_clicked.emit)
        menu.addAction(exit_action)

        self.setContextMenu(menu)

    def _on_recording_action(self):
        """Обработчик действия записи."""
        if self._is_recording:
            self.stop_recording_clicked.emit()
        else:
            self.start_recording_clicked.emit()

    def _on_activated(self, reason):
        """Обработчик активации иконки."""
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            # Одиночный клик - открыть настройки
            self.settings_clicked.emit()
        elif reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            # Двойной клик - переключить запись
            self._on_recording_action()

    def set_state(self, state: str) -> None:
        """
        Установить состояние иконки.

        Args:
            state: 'ready', 'recording', 'loading', 'error'
        """
        if state not in self.STATE_COLORS:
            logger.warning(f"Unknown tray state: {state}")
            state = TRAY_STATE_ERROR

        self._current_state = state
        self._is_recording = (state == TRAY_STATE_RECORDING)

        # Обновляем иконку
        self._update_icon()

        # Обновляем текст меню
        if self._is_recording:
            self._recording_action.setText("Остановить запись")
        else:
            self._recording_action.setText("Начать запись")

        # Обновляем tooltip
        tooltips = {
            TRAY_STATE_READY: f"{APP_NAME} - Готов",
            TRAY_STATE_RECORDING: f"{APP_NAME} - Запись...",
            TRAY_STATE_LOADING: f"{APP_NAME} - Загрузка...",
            TRAY_STATE_ERROR: f"{APP_NAME} - Ошибка",
        }
        self.setToolTip(tooltips.get(state, APP_NAME))

        logger.info(f"Tray state changed to: {state}")

    def _update_icon(self):
        """Обновить иконку в трее."""
        icon = self._create_icon(self._current_state)
        self.setIcon(icon)

        # Workaround для Windows: иногда иконка не обновляется
        # hide()/show() работают лучше чем setVisible для обновления из других потоков
        if self.isVisible():
            self.hide()
            self.show()

    def _create_icon(self, state: str) -> QIcon:
        """
        Создать иконку для состояния.
        Использует кэширование.

        Args:
            state: Состояние

        Returns:
            QIcon
        """
        if state in self._icons_cache:
            return self._icons_cache[state]

        # Пытаемся загрузить иконку из файла
        icon = self._load_icon_from_file(state)

        if icon is None:
            # Генерируем программно
            icon = self._generate_icon(state)

        self._icons_cache[state] = icon
        return icon

    def _load_icon_from_file(self, state: str) -> Optional[QIcon]:
        """Попытаться загрузить иконку из файла."""
        # Определяем путь к ресурсам
        if getattr(sys, 'frozen', False):
            base_dir = Path(sys.executable).parent / "_internal"
        else:
            base_dir = Path(__file__).parent.parent.parent

        icon_names = {
            TRAY_STATE_READY: "tray_ready.png",
            TRAY_STATE_RECORDING: "tray_recording.png",
            TRAY_STATE_LOADING: "tray_loading.png",
            TRAY_STATE_ERROR: "tray_error.png",
        }

        icon_path = base_dir / "resources" / "icons" / icon_names.get(state, "tray_ready.png")

        if icon_path.exists():
            return QIcon(str(icon_path))

        return None

    def _generate_icon(self, state: str) -> QIcon:
        """
        Программно сгенерировать иконку.

        Args:
            state: Состояние

        Returns:
            QIcon
        """
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))  # Прозрачный фон

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Основной фон (серый круг)
        bg_color = QColor("#6B7280")
        painter.setBrush(bg_color)
        painter.setPen(bg_color)
        painter.drawEllipse(2, 2, size - 4, size - 4)

        # Текст "VT"
        painter.setPen(QColor("#FFFFFF"))
        font = QFont("Arial", 18, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), 0x0084, "VT")  # AlignCenter

        # Индикатор состояния (маленький кружок в правом нижнем углу)
        indicator_size = 16
        indicator_x = size - indicator_size - 2
        indicator_y = size - indicator_size - 2

        indicator_color = QColor(self.STATE_COLORS.get(state, "#6B7280"))
        painter.setBrush(indicator_color)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawEllipse(indicator_x, indicator_y, indicator_size, indicator_size)

        painter.end()

        return QIcon(pixmap)

    def show_notification(self, title: str, message: str, icon_type=None) -> None:
        """
        Показать всплывающее уведомление.

        Args:
            title: Заголовок
            message: Текст сообщения
            icon_type: Тип иконки (Information, Warning, Critical)
        """
        if icon_type is None:
            icon_type = QSystemTrayIcon.MessageIcon.Information

        self.showMessage(title, message, icon_type, 3000)

    def is_recording(self) -> bool:
        """Проверить, идёт ли запись."""
        return self._is_recording

    def get_state(self) -> str:
        """Получить текущее состояние."""
        return self._current_state
