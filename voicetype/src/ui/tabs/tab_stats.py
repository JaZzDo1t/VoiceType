"""
VoiceType - Statistics Tab
Вкладка 'Статистика' с мониторингом ресурсов.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QProgressBar, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QPainterPath, QFont
from loguru import logger

from src.data.database import get_database
from src.utils.system_info import get_process_cpu, get_process_memory, get_memory_usage


class SimpleGraph(QWidget):
    """Простой график для отображения временного ряда."""

    def __init__(self, color: str = "#3B82F6", parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._data = []
        self._max_points = 60  # 60 точек (1 час при интервале 1 минута)
        self._max_value = 100.0

        self.setMinimumHeight(100)
        self.setMinimumWidth(200)

    def set_data(self, data: list):
        """Установить данные графика."""
        self._data = data[-self._max_points:]
        if self._data:
            self._max_value = max(max(self._data) * 1.2, 10)
        self.update()

    def add_point(self, value: float):
        """Добавить точку."""
        self._data.append(value)
        if len(self._data) > self._max_points:
            self._data.pop(0)
        self._max_value = max(max(self._data) * 1.2, 10) if self._data else 100
        self.update()

    def clear(self):
        """Очистить график."""
        self._data.clear()
        self.update()

    def paintEvent(self, event):
        """Отрисовка графика с подписями."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        left_margin = 40  # Увеличено для подписей Y
        right_margin = 10
        top_margin = 10
        bottom_margin = 20  # Для подписей X

        # Фон
        painter.fillRect(rect, QColor("#1F2937"))

        graph_width = rect.width() - left_margin - right_margin
        graph_height = rect.height() - top_margin - bottom_margin

        # Подписи Y-оси и сетка
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor("#6B7280"))
        for i in range(5):
            y = top_margin + graph_height * i / 4
            value = self._max_value * (4 - i) / 4
            # Подпись значения
            painter.drawText(2, int(y + 4), f"{value:.0f}")
            # Линия сетки
            painter.setPen(QPen(QColor("#374151"), 1))
            painter.drawLine(left_margin, int(y), rect.width() - right_margin, int(y))
            painter.setPen(QColor("#6B7280"))

        if not self._data:
            # Нет данных
            painter.setPen(QColor("#6B7280"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Нет данных")
            return

        # Подписи X-оси (время)
        painter.setPen(QColor("#6B7280"))
        painter.drawText(left_margin, rect.height() - 3, "60м")
        painter.drawText(rect.width() - right_margin - 15, rect.height() - 3, "0м")

        # График
        path = QPainterPath()

        for i, value in enumerate(self._data):
            x = left_margin + (i / max(len(self._data) - 1, 1)) * graph_width
            y = rect.height() - bottom_margin - (value / self._max_value) * graph_height

            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        painter.setPen(QPen(self._color, 2))
        painter.drawPath(path)

        # Заливка под графиком
        fill_path = QPainterPath(path)
        fill_path.lineTo(rect.width() - right_margin, rect.height() - bottom_margin)
        fill_path.lineTo(left_margin, rect.height() - bottom_margin)
        fill_path.closeSubpath()

        fill_color = QColor(self._color)
        fill_color.setAlpha(50)
        painter.fillPath(fill_path, fill_color)

        # Текущее значение в правом верхнем углу
        if self._data:
            current = self._data[-1]
            painter.setPen(self._color)
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(rect.width() - 60, 18, f"{current:.1f}")


class TabStats(QWidget):
    """
    Вкладка 'Статистика'.
    Графики CPU/RAM за 24 часа.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = get_database()
        self._setup_ui()
        self._load_stats()

        # Таймер обновления текущих значений
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_current)
        self._update_timer.start(10000)  # Каждые 10 секунд

    def _setup_ui(self):
        """Настроить интерфейс."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Текущие значения
        current_group = QGroupBox("ТЕКУЩЕЕ СОСТОЯНИЕ (обновление: 10 сек)")
        current_layout = QHBoxLayout(current_group)

        # CPU
        cpu_layout = QVBoxLayout()
        cpu_label = QLabel("CPU")
        cpu_label.setObjectName("secondaryLabel")
        cpu_layout.addWidget(cpu_label)

        self._cpu_value = QLabel("0%")
        self._cpu_value.setStyleSheet("font-size: 24px; font-weight: bold;")
        cpu_layout.addWidget(self._cpu_value)

        self._cpu_bar = QProgressBar()
        self._cpu_bar.setMaximum(100)
        self._cpu_bar.setValue(0)
        self._cpu_bar.setTextVisible(False)
        cpu_layout.addWidget(self._cpu_bar)

        current_layout.addLayout(cpu_layout)

        # RAM
        ram_layout = QVBoxLayout()
        ram_label = QLabel("RAM")
        ram_label.setObjectName("secondaryLabel")
        ram_layout.addWidget(ram_label)

        self._ram_value = QLabel("0 МБ")
        self._ram_value.setStyleSheet("font-size: 24px; font-weight: bold;")
        ram_layout.addWidget(self._ram_value)

        self._ram_bar = QProgressBar()
        # Используем системную память как максимум (в МБ)
        sys_mem = get_memory_usage()
        self._system_ram_mb = int(sys_mem.get("total_mb", 16000))
        self._ram_bar.setMaximum(self._system_ram_mb)
        self._ram_bar.setValue(0)
        self._ram_bar.setTextVisible(False)
        ram_layout.addWidget(self._ram_bar)

        current_layout.addLayout(ram_layout)

        layout.addWidget(current_group)

        # График CPU
        cpu_graph_group = QGroupBox("ИСТОРИЯ CPU (24 часа)")
        cpu_graph_layout = QVBoxLayout(cpu_graph_group)
        self._cpu_graph = SimpleGraph(color="#3B82F6")
        cpu_graph_layout.addWidget(self._cpu_graph)
        layout.addWidget(cpu_graph_group)

        # График RAM
        ram_graph_group = QGroupBox("ИСТОРИЯ RAM (24 часа)")
        ram_graph_layout = QVBoxLayout(ram_graph_group)
        self._ram_graph = SimpleGraph(color="#22C55E")
        ram_graph_layout.addWidget(self._ram_graph)
        layout.addWidget(ram_graph_group)

        # Статистика распознавания
        recog_group = QGroupBox("РАСПОЗНАВАНИЕ")
        recog_layout = QHBoxLayout(recog_group)

        recog_layout.addWidget(QLabel("Всего распознано сегодня:"))
        self._recog_time = QLabel("0 мин 0 сек")
        self._recog_time.setStyleSheet("font-weight: bold;")
        recog_layout.addWidget(self._recog_time)
        recog_layout.addStretch()

        layout.addWidget(recog_group)

        # Кнопка обновления
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        refresh_btn = QPushButton("Обновить")
        refresh_btn.setObjectName("secondaryButton")
        refresh_btn.clicked.connect(self._load_stats)
        btn_layout.addWidget(refresh_btn)

        layout.addLayout(btn_layout)

    def _update_current(self):
        """Обновить текущие значения."""
        cpu = get_process_cpu()
        ram = get_process_memory()

        self._cpu_value.setText(f"{cpu:.1f}%")
        self._cpu_bar.setValue(int(min(cpu, 100)))

        # Показываем RAM в контексте системной памяти
        self._ram_value.setText(f"{ram:.0f} / {self._system_ram_mb:.0f} МБ")
        self._ram_bar.setValue(int(min(ram, self._system_ram_mb)))

    def _load_stats(self):
        """Загрузить статистику из БД."""
        # Получаем статистику за 24 часа
        stats = self._db.get_stats_24h()

        if stats:
            cpu_data = [s["cpu_percent"] for s in stats]
            ram_data = [s["ram_mb"] for s in stats]

            self._cpu_graph.set_data(cpu_data)
            self._ram_graph.set_data(ram_data)
        else:
            self._cpu_graph.clear()
            self._ram_graph.clear()

        # Время распознавания
        total_seconds = self._db.get_today_recognition_time()
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        self._recog_time.setText(f"{minutes} мин {seconds} сек")

        # Обновляем текущие значения
        self._update_current()

    def refresh(self):
        """Обновить вкладку."""
        self._load_stats()

    def update_graphs(self, cpu: float, ram: float):
        """Обновить графики новыми данными (вызывается по сигналу из app)."""
        self._cpu_graph.add_point(cpu)
        self._ram_graph.add_point(ram)
        # Обновляем и текущие значения
        self._cpu_value.setText(f"{cpu:.1f}%")
        self._cpu_bar.setValue(int(min(cpu, 100)))
        self._ram_value.setText(f"{ram:.0f} / {self._system_ram_mb:.0f} МБ")
        self._ram_bar.setValue(int(min(ram, self._system_ram_mb)))

    def record_stats(self, cpu: float, ram: float):
        """Записать статистику в БД (устаревший метод, используйте update_graphs)."""
        self._db.add_stats_entry(cpu, ram)
        self.update_graphs(cpu, ram)
