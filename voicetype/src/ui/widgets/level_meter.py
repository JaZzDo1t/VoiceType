"""
VoiceType - Level Meter Widget
Виджет для отображения уровня аудио сигнала в реальном времени.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtProperty
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QPen


class LevelMeter(QWidget):
    """
    Индикатор уровня аудио сигнала.
    Показывает горизонтальную полосу с градиентом от зелёного к красному.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._level: float = 0.0  # 0.0 - 1.0
        self._peak_level: float = 0.0
        self._peak_decay: float = 0.02  # Скорость затухания пика

        # Цвета градиента
        self._color_low = QColor("#22C55E")     # Зелёный
        self._color_mid = QColor("#EAB308")     # Жёлтый
        self._color_high = QColor("#EF4444")    # Красный
        self._color_bg = QColor("#374151")      # Фон
        self._color_peak = QColor("#FFFFFF")    # Пик

        # Настройки
        self.setMinimumHeight(20)
        self.setMaximumHeight(30)
        self.setMinimumWidth(100)

        # Таймер для анимации пика
        self._peak_timer = QTimer(self)
        self._peak_timer.timeout.connect(self._decay_peak)
        self._peak_timer.start(50)  # 20 fps

    def set_level(self, level: float):
        """
        Установить текущий уровень.

        Args:
            level: Значение от 0.0 до 1.0
        """
        self._level = max(0.0, min(1.0, level))

        # Обновляем пик если текущий уровень выше
        if self._level > self._peak_level:
            self._peak_level = self._level

        self.update()

    def get_level(self) -> float:
        """Получить текущий уровень."""
        return self._level

    def _decay_peak(self):
        """Затухание индикатора пика."""
        if self._peak_level > self._level:
            self._peak_level -= self._peak_decay
            if self._peak_level < self._level:
                self._peak_level = self._level
            self.update()

    def reset(self):
        """Сбросить индикатор."""
        self._level = 0.0
        self._peak_level = 0.0
        self.update()

    def paintEvent(self, event):
        """Отрисовка виджета."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        margin = 2
        bar_rect = rect.adjusted(margin, margin, -margin, -margin)

        # Фон
        painter.fillRect(bar_rect, self._color_bg)

        # Градиент для полосы уровня
        gradient = QLinearGradient(bar_rect.left(), 0, bar_rect.right(), 0)
        gradient.setColorAt(0.0, self._color_low)
        gradient.setColorAt(0.5, self._color_mid)
        gradient.setColorAt(1.0, self._color_high)

        # Полоса уровня
        if self._level > 0:
            level_width = int(bar_rect.width() * self._level)
            level_rect = bar_rect.adjusted(0, 0, -(bar_rect.width() - level_width), 0)
            painter.fillRect(level_rect, gradient)

        # Индикатор пика (вертикальная линия)
        if self._peak_level > 0.01:
            peak_x = bar_rect.left() + int(bar_rect.width() * self._peak_level)
            painter.setPen(QPen(self._color_peak, 2))
            painter.drawLine(peak_x, bar_rect.top(), peak_x, bar_rect.bottom())

        # Рамка
        painter.setPen(QPen(self._color_bg.darker(120), 1))
        painter.drawRect(bar_rect)

    # Qt Property для анимации
    level = pyqtProperty(float, get_level, set_level)


class LevelMeterWithLabel(QWidget):
    """
    Индикатор уровня с подписью и значением в процентах.
    """

    def __init__(self, label: str = "Уровень", parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Верхняя строка: label + значение
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._label.setObjectName("secondaryLabel")
        top_layout.addWidget(self._label)

        top_layout.addStretch()

        self._value_label = QLabel("0%")
        self._value_label.setObjectName("secondaryLabel")
        top_layout.addWidget(self._value_label)

        layout.addLayout(top_layout)

        # Индикатор
        self._meter = LevelMeter()
        layout.addWidget(self._meter)

    def set_level(self, level: float):
        """Установить уровень."""
        self._meter.set_level(level)
        self._value_label.setText(f"{int(level * 100)}%")

    def get_level(self) -> float:
        """Получить уровень."""
        return self._meter.get_level()

    def reset(self):
        """Сбросить."""
        self._meter.reset()
        self._value_label.setText("0%")

    def set_label(self, text: str):
        """Установить текст подписи."""
        self._label.setText(text)
