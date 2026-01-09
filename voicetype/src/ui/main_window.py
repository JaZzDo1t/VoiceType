"""
VoiceType - Main Window
Главное окно приложения с настройками.
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QApplication, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QByteArray
from PyQt6.QtGui import QIcon, QCloseEvent
from loguru import logger

from src.data.config import get_config
from src.ui.themes import get_theme, apply_theme
from src.ui.tabs.tab_main import TabMain
from src.ui.tabs.tab_hotkeys import TabHotkeys
from src.ui.tabs.tab_history import TabHistory
from src.ui.tabs.tab_stats import TabStats
from src.ui.tabs.tab_logs import TabLogs
from src.ui.tabs.tab_test import TabTest
from src.ui.widgets.loading_overlay import LoadingOverlay
from src.utils.constants import (
    APP_NAME, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT,
    THEME_DARK
)


class MainWindow(QMainWindow):
    """
    Главное окно приложения.
    Содержит TabWidget с вкладками настроек.
    """

    # Сигналы
    window_closed = pyqtSignal()
    theme_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._config = get_config()
        self._is_loading = True  # Начинаем в режиме загрузки
        self._setup_ui()
        self._apply_current_theme()
        self._restore_geometry()
        self._connect_signals()

        logger.info("Main window initialized")

    def _setup_ui(self):
        """Настроить интерфейс."""
        # Основные параметры окна
        self.setWindowTitle(f"{APP_NAME} - Настройки")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stacked widget для переключения между загрузкой и вкладками
        self._stacked_widget = QStackedWidget()
        layout.addWidget(self._stacked_widget)

        # Страница загрузки (индекс 0)
        self._loading_overlay = LoadingOverlay()
        self._stacked_widget.addWidget(self._loading_overlay)

        # Контейнер с вкладками (индекс 1)
        tabs_container = QWidget()
        tabs_layout = QVBoxLayout(tabs_container)
        tabs_layout.setContentsMargins(8, 8, 8, 8)

        # Tab widget
        self._tab_widget = QTabWidget()
        tabs_layout.addWidget(self._tab_widget)

        self._stacked_widget.addWidget(tabs_container)

        # Создаём вкладки
        self._setup_tabs()

        # Начинаем с экрана загрузки
        self._stacked_widget.setCurrentIndex(0)

    def _setup_tabs(self):
        """Создать вкладки."""
        # Основные настройки
        self._tab_main = TabMain()
        self._tab_widget.addTab(self._tab_main, "Основные")

        # Хоткеи
        self._tab_hotkeys = TabHotkeys()
        self._tab_widget.addTab(self._tab_hotkeys, "Хоткеи")

        # История
        self._tab_history = TabHistory()
        self._tab_widget.addTab(self._tab_history, "История")

        # Статистика
        self._tab_stats = TabStats()
        self._tab_widget.addTab(self._tab_stats, "Статистика")

        # Логи
        self._tab_logs = TabLogs()
        self._tab_widget.addTab(self._tab_logs, "Логи")

        # Тест
        self._tab_test = TabTest()
        self._tab_widget.addTab(self._tab_test, "Тест")

    def _connect_signals(self):
        """Подключить сигналы."""
        # Смена темы
        self._tab_main.theme_changed.connect(self._on_theme_changed)

    def _apply_current_theme(self):
        """Применить текущую тему."""
        theme = self._config.get("system.theme", THEME_DARK)
        self._apply_theme(theme)

    def _apply_theme(self, theme_name: str):
        """Применить тему к окну."""
        stylesheet = get_theme(theme_name)
        self.setStyleSheet(stylesheet)
        logger.debug(f"Theme applied: {theme_name}")

    def _on_theme_changed(self, theme_name: str):
        """Обработчик смены темы."""
        self._apply_theme(theme_name)
        self.theme_changed.emit(theme_name)

    def _restore_geometry(self):
        """Восстановить размер и позицию окна."""
        geometry = self._config.get("internal.window_geometry")
        if geometry:
            try:
                self.restoreGeometry(QByteArray.fromBase64(geometry.encode()))
                logger.debug("Window geometry restored")
            except Exception as e:
                logger.warning(f"Failed to restore geometry: {e}")
                self._center_on_screen()
        else:
            self._center_on_screen()

    def _save_geometry(self):
        """Сохранить размер и позицию окна."""
        geometry = self.saveGeometry().toBase64().data().decode()
        self._config.set("internal.window_geometry", geometry)
        logger.debug("Window geometry saved")

    def _center_on_screen(self):
        """Центрировать окно на экране."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 2
            self.move(x, y)

    def closeEvent(self, event: QCloseEvent):
        """Обработчик закрытия окна."""
        # Сохраняем геометрию
        self._save_geometry()

        # Скрываем окно вместо закрытия (приложение работает в трее)
        event.ignore()
        self.hide()

        self.window_closed.emit()
        logger.debug("Main window hidden")

    def show_and_activate(self):
        """Показать и активировать окно."""
        self.show()
        self.raise_()
        self.activateWindow()

        # Обновляем вкладки
        self._tab_main.refresh()
        self._tab_history.refresh()
        self._tab_stats.refresh()
        self._tab_logs.refresh()

        logger.debug("Main window shown")

    # === Доступ к вкладкам ===

    @property
    def tab_main(self) -> TabMain:
        """Вкладка основных настроек."""
        return self._tab_main

    @property
    def tab_hotkeys(self) -> TabHotkeys:
        """Вкладка хоткеев."""
        return self._tab_hotkeys

    @property
    def tab_history(self) -> TabHistory:
        """Вкладка истории."""
        return self._tab_history

    @property
    def tab_stats(self) -> TabStats:
        """Вкладка статистики."""
        return self._tab_stats

    @property
    def tab_logs(self) -> TabLogs:
        """Вкладка логов."""
        return self._tab_logs

    @property
    def tab_test(self) -> TabTest:
        """Вкладка тестирования."""
        return self._tab_test

    def select_tab(self, index: int):
        """Выбрать вкладку по индексу."""
        if 0 <= index < self._tab_widget.count():
            self._tab_widget.setCurrentIndex(index)

    def select_tab_by_name(self, name: str):
        """Выбрать вкладку по имени."""
        tab_names = {
            "main": 0, "основные": 0,
            "hotkeys": 1, "хоткеи": 1,
            "history": 2, "история": 2,
            "stats": 3, "статистика": 3,
            "logs": 4, "логи": 4,
            "test": 5, "тест": 5,
        }
        index = tab_names.get(name.lower(), 0)
        self.select_tab(index)

    # === Управление состоянием загрузки ===

    def set_loading(self, loading: bool, model_name: str = ""):
        """
        Установить состояние загрузки.

        Args:
            loading: True - показать экран загрузки, False - показать вкладки
            model_name: Имя загружаемой модели
        """
        self._is_loading = loading

        if loading:
            self._loading_overlay.reset()
            if model_name:
                self._loading_overlay.set_status("Загрузка модели", model_name)
            else:
                self._loading_overlay.set_status("Загрузка")
            self._stacked_widget.setCurrentIndex(0)
            self.setWindowTitle(f"{APP_NAME} - Загрузка...")
        else:
            self._loading_overlay.stop_animation()
            self._stacked_widget.setCurrentIndex(1)
            self.setWindowTitle(f"{APP_NAME} - Настройки")
            # Обновляем статус в tab_test
            self._tab_test.set_models_ready(True, model_name)

        logger.debug(f"Loading state: {loading}, model: {model_name}")

    def set_loading_status(self, status: str, model_name: str = ""):
        """
        Обновить статус загрузки.

        Args:
            status: Текст статуса
            model_name: Имя модели
        """
        self._loading_overlay.set_status(status, model_name)

    def set_loading_progress(self, value: int, maximum: int = 100):
        """
        Установить прогресс загрузки.

        Args:
            value: Текущее значение
            maximum: Максимальное значение (0 для indeterminate)
        """
        self._loading_overlay.set_progress(value, maximum)

    def show_loading_error(self, error_text: str):
        """
        Показать ошибку загрузки.

        Args:
            error_text: Текст ошибки
        """
        self._loading_overlay.show_error(error_text)
        self.setWindowTitle(f"{APP_NAME} - Ошибка загрузки")

    def is_loading(self) -> bool:
        """Проверить, идёт ли загрузка."""
        return self._is_loading

    @property
    def loading_overlay(self) -> LoadingOverlay:
        """Доступ к оверлею загрузки."""
        return self._loading_overlay
