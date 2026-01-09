"""
VoiceType - UI Themes
Тёмная и светлая темы оформления для PyQt6.
"""
from src.utils.constants import THEME_DARK, THEME_LIGHT


# Тёмная тема
DARK_THEME = """
QMainWindow, QWidget {
    background-color: #1F2937;
    color: #F9FAFB;
}

QTabWidget::pane {
    border: 1px solid #374151;
    background-color: #1F2937;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #374151;
    color: #D1D5DB;
    padding: 8px 16px;
    border: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #4B5563;
    color: #F9FAFB;
}

QTabBar::tab:hover:!selected {
    background-color: #4B5563;
}

QPushButton {
    background-color: #3B82F6;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2563EB;
}

QPushButton:pressed {
    background-color: #1D4ED8;
}

QPushButton:disabled {
    background-color: #4B5563;
    color: #9CA3AF;
}

QPushButton#dangerButton {
    background-color: #EF4444;
}

QPushButton#dangerButton:hover {
    background-color: #DC2626;
}

QPushButton#secondaryButton {
    background-color: #4B5563;
}

QPushButton#secondaryButton:hover {
    background-color: #6B7280;
}

QComboBox {
    background-color: #374151;
    color: #F9FAFB;
    border: 1px solid #4B5563;
    padding: 6px 12px;
    border-radius: 4px;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #3B82F6;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #374151;
    color: #F9FAFB;
    selection-background-color: #3B82F6;
    border: 1px solid #4B5563;
}

QLineEdit {
    background-color: #374151;
    color: #F9FAFB;
    border: 1px solid #4B5563;
    padding: 6px 12px;
    border-radius: 4px;
}

QLineEdit:focus {
    border-color: #3B82F6;
}

QLineEdit:read-only {
    background-color: #1F2937;
    color: #9CA3AF;
}

QTextEdit, QPlainTextEdit {
    background-color: #374151;
    color: #F9FAFB;
    border: 1px solid #4B5563;
    border-radius: 4px;
    padding: 8px;
}

QTextEdit:read-only, QPlainTextEdit:read-only {
    background-color: #1F2937;
}

QSlider::groove:horizontal {
    background: #374151;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #3B82F6;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #2563EB;
}

QSlider::sub-page:horizontal {
    background: #3B82F6;
    border-radius: 3px;
}

QCheckBox {
    color: #F9FAFB;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 2px solid #4B5563;
    background-color: #374151;
}

QCheckBox::indicator:checked {
    background-color: #3B82F6;
    border-color: #3B82F6;
}

QCheckBox::indicator:hover {
    border-color: #3B82F6;
}

QRadioButton {
    color: #F9FAFB;
    spacing: 8px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #4B5563;
    background-color: #374151;
}

QRadioButton::indicator:checked {
    background-color: #3B82F6;
    border-color: #3B82F6;
}

QGroupBox {
    border: 1px solid #374151;
    border-radius: 4px;
    margin-top: 16px;
    padding-top: 16px;
    font-weight: 600;
    color: #9CA3AF;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #9CA3AF;
}

QLabel {
    color: #F9FAFB;
}

QLabel#secondaryLabel {
    color: #9CA3AF;
}

QLabel#titleLabel {
    font-size: 14px;
    font-weight: 600;
}

QScrollBar:vertical {
    background-color: #1F2937;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #4B5563;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #6B7280;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #1F2937;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #4B5563;
    border-radius: 6px;
    min-width: 30px;
}

QListWidget {
    background-color: #374151;
    color: #F9FAFB;
    border: 1px solid #4B5563;
    border-radius: 4px;
}

QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #4B5563;
}

QListWidget::item:selected {
    background-color: #3B82F6;
}

QListWidget::item:hover:!selected {
    background-color: #4B5563;
}

QProgressBar {
    background-color: #374151;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #3B82F6;
    border-radius: 4px;
}

QToolTip {
    background-color: #374151;
    color: #F9FAFB;
    border: 1px solid #4B5563;
    padding: 4px 8px;
    border-radius: 4px;
}

QMenu {
    background-color: #374151;
    color: #F9FAFB;
    border: 1px solid #4B5563;
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #3B82F6;
}

QMenu::separator {
    height: 1px;
    background-color: #4B5563;
    margin: 4px 8px;
}
"""


# Светлая тема
LIGHT_THEME = """
QMainWindow, QWidget {
    background-color: #F9FAFB;
    color: #1F2937;
}

QTabWidget::pane {
    border: 1px solid #E5E7EB;
    background-color: #FFFFFF;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #E5E7EB;
    color: #4B5563;
    padding: 8px 16px;
    border: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #1F2937;
}

QTabBar::tab:hover:!selected {
    background-color: #D1D5DB;
}

QPushButton {
    background-color: #3B82F6;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2563EB;
}

QPushButton:pressed {
    background-color: #1D4ED8;
}

QPushButton:disabled {
    background-color: #D1D5DB;
    color: #9CA3AF;
}

QPushButton#dangerButton {
    background-color: #EF4444;
}

QPushButton#dangerButton:hover {
    background-color: #DC2626;
}

QPushButton#secondaryButton {
    background-color: #E5E7EB;
    color: #1F2937;
}

QPushButton#secondaryButton:hover {
    background-color: #D1D5DB;
}

QComboBox {
    background-color: #FFFFFF;
    color: #1F2937;
    border: 1px solid #D1D5DB;
    padding: 6px 12px;
    border-radius: 4px;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #3B82F6;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    color: #1F2937;
    selection-background-color: #3B82F6;
    selection-color: white;
    border: 1px solid #D1D5DB;
}

QLineEdit {
    background-color: #FFFFFF;
    color: #1F2937;
    border: 1px solid #D1D5DB;
    padding: 6px 12px;
    border-radius: 4px;
}

QLineEdit:focus {
    border-color: #3B82F6;
}

QLineEdit:read-only {
    background-color: #F3F4F6;
    color: #6B7280;
}

QTextEdit, QPlainTextEdit {
    background-color: #FFFFFF;
    color: #1F2937;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    padding: 8px;
}

QTextEdit:read-only, QPlainTextEdit:read-only {
    background-color: #F3F4F6;
}

QSlider::groove:horizontal {
    background: #E5E7EB;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #3B82F6;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #2563EB;
}

QSlider::sub-page:horizontal {
    background: #3B82F6;
    border-radius: 3px;
}

QCheckBox {
    color: #1F2937;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 2px solid #D1D5DB;
    background-color: #FFFFFF;
}

QCheckBox::indicator:checked {
    background-color: #3B82F6;
    border-color: #3B82F6;
}

QCheckBox::indicator:hover {
    border-color: #3B82F6;
}

QRadioButton {
    color: #1F2937;
    spacing: 8px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #D1D5DB;
    background-color: #FFFFFF;
}

QRadioButton::indicator:checked {
    background-color: #3B82F6;
    border-color: #3B82F6;
}

QGroupBox {
    border: 1px solid #E5E7EB;
    border-radius: 4px;
    margin-top: 16px;
    padding-top: 16px;
    font-weight: 600;
    color: #6B7280;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #6B7280;
}

QLabel {
    color: #1F2937;
}

QLabel#secondaryLabel {
    color: #6B7280;
}

QLabel#titleLabel {
    font-size: 14px;
    font-weight: 600;
}

QScrollBar:vertical {
    background-color: #F3F4F6;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #D1D5DB;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #9CA3AF;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #F3F4F6;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #D1D5DB;
    border-radius: 6px;
    min-width: 30px;
}

QListWidget {
    background-color: #FFFFFF;
    color: #1F2937;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
}

QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #E5E7EB;
}

QListWidget::item:selected {
    background-color: #3B82F6;
    color: white;
}

QListWidget::item:hover:!selected {
    background-color: #F3F4F6;
}

QProgressBar {
    background-color: #E5E7EB;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #3B82F6;
    border-radius: 4px;
}

QToolTip {
    background-color: #FFFFFF;
    color: #1F2937;
    border: 1px solid #D1D5DB;
    padding: 4px 8px;
    border-radius: 4px;
}

QMenu {
    background-color: #FFFFFF;
    color: #1F2937;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #3B82F6;
    color: white;
}

QMenu::separator {
    height: 1px;
    background-color: #E5E7EB;
    margin: 4px 8px;
}
"""


# Словарь тем для быстрого доступа
THEMES = {
    THEME_DARK: DARK_THEME,
    THEME_LIGHT: LIGHT_THEME
}


def get_theme(theme_name: str) -> str:
    """
    Получить stylesheet для темы.

    Args:
        theme_name: "dark" или "light"

    Returns:
        QSS stylesheet строка
    """
    return THEMES.get(theme_name, DARK_THEME)


def apply_theme(widget, theme_name: str) -> None:
    """
    Применить тему к виджету.

    Args:
        widget: QWidget или QApplication
        theme_name: "dark" или "light"
    """
    stylesheet = get_theme(theme_name)
    widget.setStyleSheet(stylesheet)
