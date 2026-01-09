"""
VoiceType - Loading Overlay Widget
Виджет отображения загрузки модели с прогресс-баром.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont


class LoadingOverlay(QWidget):
    """
    Оверлей загрузки модели.
    Показывает прогресс-бар и текст статуса.
    """

    # Сигналы для кнопок действий
    retry_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    minimize_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dots_count = 0
        self._base_status = "Загрузка"
        self._model_name = ""
        self._elapsed_seconds = 0
        self._setup_ui()
        self._setup_animation()

    def _setup_ui(self):
        """Настроить интерфейс."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Spacer сверху
        layout.addStretch(2)

        # Контейнер для контента
        container = QFrame()
        container.setObjectName("loadingContainer")
        container.setStyleSheet("""
            #loadingContainer {
                background-color: rgba(31, 41, 55, 0.95);
                border-radius: 16px;
                padding: 30px;
            }
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(16)

        # Иконка/заголовок
        title_label = QLabel("VoiceType")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #3B82F6;")
        container_layout.addWidget(title_label)

        # Статус загрузки
        self._status_label = QLabel("Загрузка модели...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(14)
        self._status_label.setFont(status_font)
        self._status_label.setStyleSheet("color: #F9FAFB;")
        container_layout.addWidget(self._status_label)

        # Имя модели
        self._model_label = QLabel("")
        self._model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._model_label.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        container_layout.addWidget(self._model_label)

        # Прогресс-бар
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(0)  # Indeterminate mode
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #374151;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #3B82F6;
                border-radius: 4px;
            }
        """)
        container_layout.addWidget(self._progress_bar)

        # Таймер загрузки
        self._timer_label = QLabel("0 сек")
        self._timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._timer_label.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        container_layout.addWidget(self._timer_label)

        # Подсказка (реальные замеры: small 0.7 сек, large 132 сек)
        self._hint_label = QLabel("Large модель: ~2 мин | Small модель: ~1 сек")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setStyleSheet("color: #6B7280; font-size: 11px;")
        container_layout.addWidget(self._hint_label)

        # Кнопки действий (скрыты по умолчанию, показываются при ошибке)
        self._buttons_container = QWidget()
        buttons_layout = QHBoxLayout(self._buttons_container)
        buttons_layout.setContentsMargins(0, 16, 0, 0)
        buttons_layout.setSpacing(12)

        # Кнопка "Повторить"
        self._retry_btn = QPushButton("Повторить")
        self._retry_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
        """)
        self._retry_btn.clicked.connect(self.retry_clicked.emit)
        buttons_layout.addWidget(self._retry_btn)

        # Кнопка "Настройки"
        self._settings_btn = QPushButton("Настройки")
        self._settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #374151;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4B5563;
            }
        """)
        self._settings_btn.clicked.connect(self.settings_clicked.emit)
        buttons_layout.addWidget(self._settings_btn)

        # Кнопка "Свернуть"
        self._minimize_btn = QPushButton("Свернуть")
        self._minimize_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #9CA3AF;
                border: 1px solid #4B5563;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #374151;
                color: white;
            }
        """)
        self._minimize_btn.clicked.connect(self.minimize_clicked.emit)
        buttons_layout.addWidget(self._minimize_btn)

        self._buttons_container.hide()  # Скрыты по умолчанию
        container_layout.addWidget(self._buttons_container)

        layout.addWidget(container)

        # Spacer снизу
        layout.addStretch(3)

    def _setup_animation(self):
        """Настроить анимацию точек и таймер."""
        # Анимация точек
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._animate_dots)
        self._animation_timer.start(400)

        # Таймер секунд
        self._seconds_timer = QTimer()
        self._seconds_timer.timeout.connect(self._update_elapsed)
        self._seconds_timer.start(1000)

    def _update_elapsed(self):
        """Обновить счётчик времени."""
        self._elapsed_seconds += 1
        if self._elapsed_seconds < 60:
            self._timer_label.setText(f"{self._elapsed_seconds} сек")
        else:
            mins = self._elapsed_seconds // 60
            secs = self._elapsed_seconds % 60
            self._timer_label.setText(f"{mins} мин {secs} сек")

    def _animate_dots(self):
        """Анимировать точки в статусе."""
        self._dots_count = (self._dots_count + 1) % 4
        dots = "." * self._dots_count
        self._status_label.setText(f"{self._base_status}{dots}")

    def set_status(self, status: str, model_name: str = ""):
        """
        Установить статус загрузки.

        Args:
            status: Текст статуса (без точек)
            model_name: Имя загружаемой модели
        """
        self._base_status = status
        self._model_name = model_name

        if model_name:
            self._model_label.setText(f"Модель: {model_name}")
        else:
            self._model_label.setText("")

        self._status_label.setText(status)

    def set_progress(self, value: int, maximum: int = 100):
        """
        Установить значение прогресса.

        Args:
            value: Текущее значение (0-100)
            maximum: Максимальное значение
        """
        if maximum > 0:
            self._progress_bar.setMaximum(maximum)
            self._progress_bar.setValue(value)
        else:
            # Indeterminate mode
            self._progress_bar.setMaximum(0)

    def set_indeterminate(self, indeterminate: bool = True):
        """Установить режим неопределённого прогресса."""
        if indeterminate:
            self._progress_bar.setMaximum(0)
        else:
            self._progress_bar.setMaximum(100)

    def show_error(self, error_text: str, details: str = ""):
        """Показать ошибку загрузки."""
        self._animation_timer.stop()
        self._seconds_timer.stop()
        self._status_label.setText(error_text)
        self._status_label.setStyleSheet("color: #EF4444; font-size: 14px;")
        self._timer_label.setStyleSheet("color: #EF4444; font-size: 12px;")

        # Показываем детали ошибки вместо подсказки
        if details:
            self._hint_label.setText(details)
            self._hint_label.setStyleSheet("color: #F87171; font-size: 11px;")
        else:
            self._hint_label.setText("Проверьте настройки или попробуйте снова")
            self._hint_label.setStyleSheet("color: #F87171; font-size: 11px;")

        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #374151;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #EF4444;
                border-radius: 4px;
            }
        """)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(100)

        # Показываем кнопки действий
        self._buttons_container.show()

    def reset(self):
        """Сбросить состояние."""
        self._dots_count = 0
        self._elapsed_seconds = 0
        self._base_status = "Загрузка"
        self._status_label.setStyleSheet("color: #F9FAFB; font-size: 14px;")
        self._timer_label.setText("0 сек")
        self._timer_label.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #374151;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #3B82F6;
                border-radius: 4px;
            }
        """)
        self.set_indeterminate(True)

        # Восстанавливаем подсказку
        self._hint_label.setText("Large модель: ~2 мин | Small модель: ~1 сек")
        self._hint_label.setStyleSheet("color: #6B7280; font-size: 11px;")

        # Скрываем кнопки действий
        self._buttons_container.hide()

        if not self._animation_timer.isActive():
            self._animation_timer.start(400)
        if not self._seconds_timer.isActive():
            self._seconds_timer.start(1000)

    def stop_animation(self):
        """Остановить анимацию и таймер."""
        self._animation_timer.stop()
        self._seconds_timer.stop()

    def hideEvent(self, event):
        """При скрытии останавливаем анимацию."""
        self._animation_timer.stop()
        self._seconds_timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        """При показе запускаем анимацию."""
        self._animation_timer.start(400)
        self._seconds_timer.start(1000)
        super().showEvent(event)
