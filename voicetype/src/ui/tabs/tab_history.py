"""
VoiceType - History Tab
Вкладка 'История' со списком сессий распознавания.
"""
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QScrollArea, QFrame, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from loguru import logger
import pyperclip

from src.data.database import get_database


class DateHeader(QWidget):
    """Styled date header with separator line."""

    def __init__(self, date_str: str, parent=None):
        super().__init__(parent)
        self._setup_ui(date_str)

    def _setup_ui(self, date_str: str):
        """Setup the date header UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 8)
        layout.setSpacing(12)

        # Format date nicely (e.g., "8 января 2026")
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            months_ru = [
                "", "января", "февраля", "марта", "апреля", "мая", "июня",
                "июля", "августа", "сентября", "октября", "ноября", "декабря"
            ]
            formatted_date = f"{date_obj.day} {months_ru[date_obj.month]} {date_obj.year}"
        except ValueError:
            formatted_date = date_str

        # Date label with icon
        date_label = QLabel(f"  {formatted_date}")
        date_label.setStyleSheet("""
            QLabel {
                color: #9CA3AF;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
        """)
        layout.addWidget(date_label)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("""
            QFrame {
                background-color: #374151;
                max-height: 1px;
            }
        """)
        line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(line)


class HistoryItem(QFrame):
    """Элемент истории с улучшенным дизайном."""

    deleted = pyqtSignal(int)  # ID записи

    # Alternating colors for cards
    COLORS = ["#2D3748", "#374151"]

    def __init__(self, entry: dict, index: int = 0, parent=None):
        super().__init__(parent)
        self._entry = entry
        self._index = index
        self._expanded = False
        self._full_text = entry.get("text", "")
        self._setup_ui()

    def _setup_ui(self):
        """Настроить интерфейс."""
        bg_color = self.COLORS[self._index % 2]

        self.setStyleSheet(f"""
            HistoryItem {{
                background-color: {bg_color};
                border-radius: 10px;
                border: 1px solid #4B5563;
            }}
            HistoryItem:hover {{
                border: 1px solid #6B7280;
                background-color: #3D4A5C;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # Верхняя строка: метаданные (мелким шрифтом, приглушённо)
        meta_layout = QHBoxLayout()
        meta_layout.setSpacing(16)

        # Время (только время, без даты - дата уже в заголовке)
        started_at = self._entry.get("started_at", "")
        if isinstance(started_at, str):
            try:
                dt = datetime.fromisoformat(started_at[:19])
                time_str = dt.strftime("%H:%M")
            except ValueError:
                time_str = started_at[11:16] if len(started_at) > 16 else started_at
        else:
            time_str = started_at.strftime("%H:%M") if started_at else ""

        time_label = QLabel(time_str)
        time_label.setStyleSheet("""
            QLabel {
                color: #6B7280;
                font-size: 11px;
                font-weight: 500;
            }
        """)
        meta_layout.addWidget(time_label)

        # Длительность
        duration = self._entry.get("duration_seconds", 0)
        minutes = duration // 60
        seconds = duration % 60
        if minutes > 0:
            duration_str = f"{minutes} мин {seconds} сек"
        else:
            duration_str = f"{seconds} сек"

        duration_label = QLabel(duration_str)
        duration_label.setStyleSheet("""
            QLabel {
                color: #6B7280;
                font-size: 11px;
            }
        """)
        meta_layout.addWidget(duration_label)

        # Язык (компактный бейдж)
        language = self._entry.get("language", "ru")
        lang_label = QLabel(language.upper())
        lang_label.setStyleSheet("""
            QLabel {
                color: #9CA3AF;
                background-color: #4B5563;
                font-size: 10px;
                font-weight: 600;
                padding: 2px 6px;
                border-radius: 4px;
            }
        """)
        lang_label.setFixedHeight(18)
        meta_layout.addWidget(lang_label)

        meta_layout.addStretch()
        layout.addLayout(meta_layout)

        # Текст (основной контент, более крупный и заметный)
        text = self._full_text
        max_preview_len = 150

        # Text container
        text_container = QVBoxLayout()
        text_container.setSpacing(6)

        # Determine if text needs truncation
        needs_expansion = len(text) > max_preview_len
        display_text = text[:max_preview_len] + "..." if needs_expansion and not self._expanded else text

        self._text_label = QLabel(display_text)
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._text_label.setStyleSheet("""
            QLabel {
                color: #F3F4F6;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        text_container.addWidget(self._text_label)

        # "Show more" link if text is truncated
        if needs_expansion:
            self._expand_btn = QPushButton("Показать полностью")
            self._expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._expand_btn.setStyleSheet("""
                QPushButton {
                    color: #60A5FA;
                    background: transparent;
                    border: none;
                    font-size: 12px;
                    text-align: left;
                    padding: 0;
                }
                QPushButton:hover {
                    color: #93C5FD;
                    text-decoration: underline;
                }
            """)
            self._expand_btn.clicked.connect(self._toggle_expand)
            text_container.addWidget(self._expand_btn)

        layout.addLayout(text_container)

        # Кнопки (компактнее, справа)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        btn_layout.addStretch()

        copy_btn = QPushButton("Копировать")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #4B5563;
                color: #D1D5DB;
                border: none;
                padding: 6px 14px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #6B7280;
                color: #F9FAFB;
            }
        """)
        copy_btn.clicked.connect(lambda: self._copy_text(self._full_text))
        btn_layout.addWidget(copy_btn)

        delete_btn = QPushButton("Удалить")
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #F87171;
                border: 1px solid #F87171;
                padding: 5px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #F87171;
                color: #FFFFFF;
            }
        """)
        delete_btn.clicked.connect(self._on_delete)
        btn_layout.addWidget(delete_btn)

        layout.addLayout(btn_layout)

    def _toggle_expand(self):
        """Toggle text expansion."""
        self._expanded = not self._expanded
        if self._expanded:
            self._text_label.setText(self._full_text)
            self._expand_btn.setText("Свернуть")
        else:
            self._text_label.setText(self._full_text[:150] + "...")
            self._expand_btn.setText("Показать полностью")

    def _copy_text(self, text: str):
        """Копировать текст в буфер обмена."""
        try:
            pyperclip.copy(text)
            logger.debug("Text copied to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy: {e}")

    def _on_delete(self):
        """Удалить запись."""
        entry_id = self._entry.get("id")
        if entry_id:
            self.deleted.emit(entry_id)


class TabHistory(QWidget):
    """
    Вкладка 'История'.
    Список последних 15 сессий распознавания.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = get_database()
        self._setup_ui()
        self._load_history()

    def _setup_ui(self):
        """Настроить интерфейс."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Верхняя панель (заголовок и кнопки)
        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)

        # Заголовок с иконкой
        title = QLabel("История распознавания")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #F9FAFB;
            }
        """)
        top_layout.addWidget(title)

        # Счётчик записей (рядом с заголовком)
        self._count_label = QLabel("0 записей")
        self._count_label.setStyleSheet("""
            QLabel {
                color: #6B7280;
                font-size: 13px;
                background-color: #374151;
                padding: 4px 10px;
                border-radius: 12px;
            }
        """)
        top_layout.addWidget(self._count_label)

        top_layout.addStretch()

        # Кнопка обновить (иконка)
        self._refresh_btn = QPushButton("Обновить")
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4B5563;
                color: #D1D5DB;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #6B7280;
                color: #F9FAFB;
            }
        """)
        self._refresh_btn.clicked.connect(self._load_history)
        top_layout.addWidget(self._refresh_btn)

        # Кнопка очистить
        self._clear_btn = QPushButton("Очистить всё")
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #F87171;
                border: 1px solid #F87171;
                padding: 7px 14px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #F87171;
                color: #FFFFFF;
            }
        """)
        self._clear_btn.clicked.connect(self._on_clear_all)
        top_layout.addWidget(self._clear_btn)

        layout.addLayout(top_layout)

        # Scroll area для списка
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
        """)

        self._list_widget = QWidget()
        self._list_widget.setStyleSheet("background-color: transparent;")
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._list_layout.setContentsMargins(0, 0, 8, 0)
        self._list_layout.setSpacing(10)

        scroll.setWidget(self._list_widget)
        layout.addWidget(scroll)

        # Пустое состояние (улучшенный дизайн)
        self._empty_container = QWidget()
        empty_layout = QVBoxLayout(self._empty_container)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.setSpacing(12)

        empty_icon = QLabel("(empty)")
        empty_icon.setStyleSheet("""
            QLabel {
                color: #4B5563;
                font-size: 48px;
            }
        """)
        empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_icon)

        empty_text = QLabel("История пуста")
        empty_text.setStyleSheet("""
            QLabel {
                color: #6B7280;
                font-size: 16px;
                font-weight: 500;
            }
        """)
        empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_text)

        empty_hint = QLabel("Распознанные тексты будут отображаться здесь")
        empty_hint.setStyleSheet("""
            QLabel {
                color: #4B5563;
                font-size: 13px;
            }
        """)
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_hint)

        self._empty_container.hide()
        layout.addWidget(self._empty_container)

    def _load_history(self):
        """Загрузить историю из БД."""
        # Очищаем текущий список
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Загружаем данные
        entries = self._db.get_history()

        # Обновляем счётчик
        count = len(entries)
        if count == 0:
            self._count_label.setText("0 записей")
        elif count == 1:
            self._count_label.setText("1 запись")
        elif count < 5:
            self._count_label.setText(f"{count} записи")
        else:
            self._count_label.setText(f"{count} записей")

        if not entries:
            self._empty_container.show()
            return

        self._empty_container.hide()

        # Группировка по дням
        current_date = None
        item_index = 0

        for entry in entries:
            # Парсим дату
            created_at = entry.get("created_at", "")
            if isinstance(created_at, str):
                try:
                    entry_date = datetime.fromisoformat(created_at[:10])
                except ValueError:
                    entry_date = datetime.now()
            else:
                entry_date = created_at if created_at else datetime.now()

            # Добавляем разделитель даты если нужно
            date_str = entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, 'strftime') else str(entry_date)[:10]

            if date_str != current_date:
                current_date = date_str
                # Используем новый DateHeader компонент
                date_header = DateHeader(date_str)
                self._list_layout.addWidget(date_header)

            # Добавляем элемент с индексом для чередования цветов
            item = HistoryItem(entry, index=item_index)
            item.deleted.connect(self._on_delete_entry)
            self._list_layout.addWidget(item)
            item_index += 1

        # Spacer в конце
        self._list_layout.addStretch()

    def _on_delete_entry(self, entry_id: int):
        """Удалить запись."""
        self._db.delete_history_entry(entry_id)
        self._load_history()
        logger.info(f"History entry deleted: {entry_id}")

    def _on_clear_all(self):
        """Очистить всю историю."""
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Вы уверены, что хотите очистить всю историю?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._db.clear_history()
            self._load_history()
            logger.info("History cleared")

    def refresh(self):
        """Обновить вкладку."""
        self._load_history()

    def add_entry(self, started_at, ended_at, text: str, language: str):
        """Добавить новую запись в историю."""
        self._db.add_history_entry(started_at, ended_at, text, language)
        self._load_history()
