"""
VoiceType - Hotkey Edit Widget
Виджет для ввода и отображения горячих клавиш.
"""
from typing import Optional, Set
from PyQt6.QtWidgets import QLineEdit, QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeyEvent, QFocusEvent
from loguru import logger


class HotkeyEdit(QWidget):
    """
    Виджет для ввода горячей клавиши.
    Показывает текущий хоткей и позволяет записать новый.
    """

    # Сигнал при изменении хоткея
    hotkey_changed = pyqtSignal(str)  # новый хоткей как строка

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_hotkey: str = ""
        self._is_recording: bool = False
        self._pressed_keys: Set[str] = set()
        self._max_pressed_keys: Set[str] = set()  # Максимальный набор нажатых клавиш

        self._setup_ui()

    def _setup_ui(self):
        """Настроить интерфейс."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Поле с текущим хоткеем
        self._line_edit = QLineEdit()
        self._line_edit.setReadOnly(True)
        self._line_edit.setPlaceholderText("Нажмите для записи...")
        self._line_edit.setMinimumWidth(150)
        layout.addWidget(self._line_edit)

        # Кнопка записи
        self._record_btn = QPushButton("Изменить")
        self._record_btn.setObjectName("secondaryButton")
        self._record_btn.clicked.connect(self._toggle_recording)
        self._record_btn.setFixedWidth(100)
        layout.addWidget(self._record_btn)

        # Кнопка очистки
        self._clear_btn = QPushButton("X")
        self._clear_btn.setObjectName("dangerButton")
        self._clear_btn.clicked.connect(self._clear_hotkey)
        self._clear_btn.setFixedWidth(30)
        self._clear_btn.setToolTip("Очистить")
        layout.addWidget(self._clear_btn)

    def _toggle_recording(self):
        """Переключить режим записи."""
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Начать запись хоткея."""
        self._is_recording = True
        self._pressed_keys.clear()
        self._max_pressed_keys.clear()
        self._record_btn.setText("Готово")
        self._line_edit.setText("Нажмите комбинацию...")
        self._line_edit.setFocus()
        self._line_edit.setReadOnly(False)

        # Перехватываем события клавиатуры
        self._line_edit.keyPressEvent = self._on_key_press
        self._line_edit.keyReleaseEvent = self._on_key_release

    def _stop_recording(self):
        """Остановить запись хоткея."""
        self._is_recording = False
        self._record_btn.setText("Изменить")
        self._line_edit.setReadOnly(True)

        # Восстанавливаем стандартные обработчики
        self._line_edit.keyPressEvent = lambda e: None
        self._line_edit.keyReleaseEvent = lambda e: None

        # Обновляем отображение
        self._update_display()

    def _on_key_press(self, event: QKeyEvent):
        """Обработчик нажатия клавиши."""
        if not self._is_recording:
            return

        key = event.key()
        key_str = self._key_to_string(key, event.modifiers())

        if key_str:
            self._pressed_keys.add(key_str)
            # Сохраняем максимальный набор нажатых клавиш
            if len(self._pressed_keys) > len(self._max_pressed_keys):
                self._max_pressed_keys = self._pressed_keys.copy()
            self._update_recording_display()

        event.accept()

    def _on_key_release(self, event: QKeyEvent):
        """Обработчик отпускания клавиши."""
        if not self._is_recording:
            return

        key = event.key()
        key_str = self._key_to_string(key, event.modifiers())

        if key_str:
            self._pressed_keys.discard(key_str)

        # Сохраняем когда ВСЕ клавиши отпущены
        if not self._pressed_keys and self._max_pressed_keys:
            self._current_hotkey = self._format_hotkey(self._max_pressed_keys)
            self._stop_recording()
            self.hotkey_changed.emit(self._current_hotkey)

        event.accept()

    def _key_to_string(self, key: int, modifiers) -> Optional[str]:
        """Преобразовать код клавиши в строку."""
        from PyQt6.QtCore import Qt

        # Модификаторы
        if key == Qt.Key.Key_Control:
            return "ctrl"
        elif key == Qt.Key.Key_Shift:
            return "shift"
        elif key == Qt.Key.Key_Alt:
            return "alt"
        elif key == Qt.Key.Key_Meta:
            return "win"

        # Функциональные клавиши
        if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F12:
            return f"f{key - Qt.Key.Key_F1 + 1}"

        # Специальные клавиши
        special_keys = {
            Qt.Key.Key_Space: "space",
            Qt.Key.Key_Return: "enter",
            Qt.Key.Key_Enter: "enter",
            Qt.Key.Key_Tab: "tab",
            Qt.Key.Key_Escape: "esc",
            Qt.Key.Key_Backspace: "backspace",
            Qt.Key.Key_Delete: "delete",
            Qt.Key.Key_Insert: "insert",
            Qt.Key.Key_Home: "home",
            Qt.Key.Key_End: "end",
            Qt.Key.Key_PageUp: "pageup",
            Qt.Key.Key_PageDown: "pagedown",
            Qt.Key.Key_Up: "up",
            Qt.Key.Key_Down: "down",
            Qt.Key.Key_Left: "left",
            Qt.Key.Key_Right: "right",
        }

        if key in special_keys:
            return special_keys[key]

        # Обычные символы
        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            return chr(key).lower()

        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            return chr(key)

        return None

    def _format_hotkey(self, keys: Set[str]) -> str:
        """Форматировать множество клавиш в строку хоткея."""
        # Порядок модификаторов
        mod_order = ["ctrl", "alt", "shift", "win"]

        modifiers = []
        main_keys = []  # Поддерживаем несколько обычных клавиш

        for k in keys:
            if k in mod_order:
                modifiers.append(k)
            else:
                main_keys.append(k)

        # Сортируем модификаторы по порядку
        modifiers.sort(key=lambda x: mod_order.index(x) if x in mod_order else 99)

        # Сортируем обычные клавиши алфавитно
        main_keys.sort()

        # Формируем строку: модификаторы + обычные клавиши
        parts = modifiers + main_keys
        return "+".join(parts)

    def _update_recording_display(self):
        """Обновить отображение во время записи."""
        # Показываем максимальный набор (полную комбинацию)
        if self._max_pressed_keys:
            display = self._format_hotkey(self._max_pressed_keys)
            self._line_edit.setText(display.upper())
        elif self._pressed_keys:
            display = self._format_hotkey(self._pressed_keys)
            self._line_edit.setText(display.upper())
        else:
            self._line_edit.setText("Нажмите комбинацию...")

    def _update_display(self):
        """Обновить отображение текущего хоткея."""
        if self._current_hotkey:
            self._line_edit.setText(self._current_hotkey.upper())
        else:
            self._line_edit.setText("")

    def _clear_hotkey(self):
        """Очистить хоткей."""
        self._current_hotkey = ""
        self._pressed_keys.clear()
        self._max_pressed_keys.clear()
        self._update_display()
        self.hotkey_changed.emit("")

    def set_hotkey(self, hotkey: str):
        """
        Установить хоткей.

        Args:
            hotkey: Строка хоткея, например "ctrl+shift+s"
        """
        self._current_hotkey = hotkey.lower() if hotkey else ""
        self._update_display()

    def get_hotkey(self) -> str:
        """
        Получить текущий хоткей.

        Returns:
            Строка хоткея
        """
        return self._current_hotkey

    def is_recording(self) -> bool:
        """Проверить, идёт ли запись."""
        return self._is_recording
