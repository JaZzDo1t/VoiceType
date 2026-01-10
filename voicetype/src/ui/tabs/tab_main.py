"""
VoiceType - Main Settings Tab
Вкладка 'Основные' с основными настройками приложения.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QCheckBox, QRadioButton,
    QButtonGroup
)
from PyQt6.QtCore import pyqtSignal
from loguru import logger

from src.data.config import get_config
from src.utils.system_info import get_microphones
from src.utils.constants import (
    OUTPUT_MODE_KEYBOARD, OUTPUT_MODE_CLIPBOARD,
    THEME_DARK, THEME_LIGHT
)


class TabMain(QWidget):
    """
    Вкладка 'Основные'.
    Настройки микрофона, языка, модели, вывода, темы.
    Все изменения сохраняются автоматически.
    """

    # Сигналы
    microphone_changed = pyqtSignal(str)      # ID микрофона
    language_changed = pyqtSignal(str)        # ru/en
    model_changed = pyqtSignal(str)           # small/large
    punctuation_changed = pyqtSignal(bool)    # enabled
    output_mode_changed = pyqtSignal(str)     # keyboard/clipboard
    theme_changed = pyqtSignal(str)           # dark/light
    autostart_changed = pyqtSignal(bool)      # enabled

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

        # Секция АУДИО
        audio_group = QGroupBox("АУДИО")
        audio_layout = QVBoxLayout(audio_group)

        # Микрофон
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("Микрофон:"))
        self._mic_combo = QComboBox()
        self._mic_combo.setMinimumWidth(250)
        mic_layout.addWidget(self._mic_combo)
        mic_layout.addStretch()
        audio_layout.addLayout(mic_layout)

        # Язык
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Язык:"))
        self._lang_combo = QComboBox()
        self._lang_combo.addItem("Русский", "ru")
        self._lang_combo.addItem("English", "en")
        self._lang_combo.setMinimumWidth(150)
        lang_layout.addWidget(self._lang_combo)
        lang_layout.addStretch()
        audio_layout.addLayout(lang_layout)

        # Модель
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Модель:"))
        self._model_combo = QComboBox()
        self._model_combo.addItem("Малая (быстрее)", "small")
        self._model_combo.addItem("Большая (точнее)", "large")
        self._model_combo.setMinimumWidth(150)
        model_layout.addWidget(self._model_combo)
        model_layout.addStretch()
        audio_layout.addLayout(model_layout)

        layout.addWidget(audio_group)

        # Секция РАСПОЗНАВАНИЕ
        recog_group = QGroupBox("РАСПОЗНАВАНИЕ")
        recog_layout = QVBoxLayout(recog_group)

        # Автопунктуация
        self._punctuation_check = QCheckBox("Автоматическая пунктуация")
        recog_layout.addWidget(self._punctuation_check)

        # Секция МОДЕЛИ (статус загрузки)
        models_group = QGroupBox("МОДЕЛИ")
        models_layout = QVBoxLayout(models_group)

        # Статус Vosk
        vosk_layout = QHBoxLayout()
        vosk_layout.addWidget(QLabel("Vosk:"))
        self._vosk_status = QLabel("выгружена")
        self._vosk_status.setStyleSheet("color: #9CA3AF;")  # серый
        vosk_layout.addWidget(self._vosk_status)
        vosk_layout.addStretch()
        models_layout.addLayout(vosk_layout)

        # Статус пунктуации (RUPunct ONNX)
        punct_layout = QHBoxLayout()
        punct_layout.addWidget(QLabel("Пунктуация:"))
        self._punct_status = QLabel("выгружена")
        self._punct_status.setStyleSheet("color: #9CA3AF;")  # серый
        punct_layout.addWidget(self._punct_status)
        punct_layout.addStretch()
        models_layout.addLayout(punct_layout)

        recog_layout.addWidget(models_group)

        layout.addWidget(recog_group)

        # Секция ВЫВОД
        output_group = QGroupBox("ВЫВОД ТЕКСТА")
        output_layout = QVBoxLayout(output_group)

        self._output_btn_group = QButtonGroup(self)

        self._output_keyboard_radio = QRadioButton("В активное окно (эмуляция клавиатуры)")
        self._output_keyboard_radio.setChecked(True)
        self._output_btn_group.addButton(self._output_keyboard_radio)
        output_layout.addWidget(self._output_keyboard_radio)

        self._output_clipboard_radio = QRadioButton("В буфер обмена")
        self._output_btn_group.addButton(self._output_clipboard_radio)
        output_layout.addWidget(self._output_clipboard_radio)

        layout.addWidget(output_group)

        # Секция СИСТЕМА
        system_group = QGroupBox("СИСТЕМА")
        system_layout = QVBoxLayout(system_group)

        # Автозапуск
        self._autostart_check = QCheckBox("Запускать при старте Windows")
        system_layout.addWidget(self._autostart_check)

        # Тема
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Тема:"))
        self._theme_combo = QComboBox()
        self._theme_combo.addItem("Тёмная", THEME_DARK)
        self._theme_combo.addItem("Светлая", THEME_LIGHT)
        self._theme_combo.setMinimumWidth(150)
        theme_layout.addWidget(self._theme_combo)
        theme_layout.addStretch()
        system_layout.addLayout(theme_layout)

        layout.addWidget(system_group)

        # Растягивающийся spacer
        layout.addStretch()

    def _load_settings(self):
        """Загрузить настройки из конфига."""
        # Микрофоны
        self._refresh_microphones()

        # Язык
        language = self._config.get("audio.language", "ru")
        index = self._lang_combo.findData(language)
        if index >= 0:
            self._lang_combo.setCurrentIndex(index)

        # Модель
        model = self._config.get("audio.model", "small")
        index = self._model_combo.findData(model)
        if index >= 0:
            self._model_combo.setCurrentIndex(index)

        # Пунктуация
        punctuation = self._config.get("recognition.punctuation_enabled", True)
        self._punctuation_check.setChecked(punctuation)

        # Вывод
        output_mode = self._config.get("output.mode", OUTPUT_MODE_KEYBOARD)
        if output_mode == OUTPUT_MODE_CLIPBOARD:
            self._output_clipboard_radio.setChecked(True)
        else:
            self._output_keyboard_radio.setChecked(True)

        # Автозапуск
        autostart = self._config.get("system.autostart", False)
        self._autostart_check.setChecked(autostart)

        # Тема
        theme = self._config.get("system.theme", THEME_DARK)
        index = self._theme_combo.findData(theme)
        if index >= 0:
            self._theme_combo.setCurrentIndex(index)

    def _connect_signals(self):
        """Подключить сигналы."""
        self._mic_combo.currentIndexChanged.connect(self._on_mic_changed)
        self._lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._punctuation_check.toggled.connect(self._on_punctuation_changed)
        self._output_btn_group.buttonToggled.connect(self._on_output_changed)
        self._autostart_check.toggled.connect(self._on_autostart_changed)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)

    def _refresh_microphones(self):
        """Обновить список микрофонов."""
        self._mic_combo.clear()

        microphones = get_microphones()
        current_mic = self._config.get("audio.microphone_id", "default")

        for mic in microphones:
            self._mic_combo.addItem(mic["name"], mic["id"])

        # Выбираем текущий микрофон
        index = self._mic_combo.findData(current_mic)
        if index >= 0:
            self._mic_combo.setCurrentIndex(index)

    def _on_mic_changed(self, index):
        """Обработчик изменения микрофона."""
        mic_id = self._mic_combo.currentData()
        if mic_id:
            self._config.set("audio.microphone_id", mic_id)
            self.microphone_changed.emit(mic_id)
            logger.debug(f"Microphone changed: {mic_id}")

    def _on_lang_changed(self, index):
        """Обработчик изменения языка."""
        language = self._lang_combo.currentData()
        if language:
            self._config.set("audio.language", language)
            self.language_changed.emit(language)
            logger.debug(f"Language changed: {language}")

    def _on_model_changed(self, index):
        """Обработчик изменения модели."""
        model = self._model_combo.currentData()
        if model:
            self._config.set("audio.model", model)
            self.model_changed.emit(model)
            logger.debug(f"Model changed: {model}")

    def _on_punctuation_changed(self, checked):
        """Обработчик изменения пунктуации."""
        self._config.set("recognition.punctuation_enabled", checked)
        self.punctuation_changed.emit(checked)
        logger.debug(f"Punctuation changed: {checked}")

    def _on_output_changed(self, button, checked):
        """Обработчик изменения режима вывода."""
        if not checked:
            return

        if button == self._output_keyboard_radio:
            mode = OUTPUT_MODE_KEYBOARD
        else:
            mode = OUTPUT_MODE_CLIPBOARD

        self._config.set("output.mode", mode)
        self.output_mode_changed.emit(mode)
        logger.debug(f"Output mode changed: {mode}")

    def _on_autostart_changed(self, checked):
        """Обработчик изменения автозапуска."""
        self._config.set("system.autostart", checked)
        self.autostart_changed.emit(checked)
        logger.debug(f"Autostart changed: {checked}")

    def _on_theme_changed(self, index):
        """Обработчик изменения темы."""
        theme = self._theme_combo.currentData()
        if theme:
            self._config.set("system.theme", theme)
            self.theme_changed.emit(theme)
            logger.debug(f"Theme changed: {theme}")

    def refresh(self):
        """Обновить все данные вкладки."""
        self._refresh_microphones()

    def set_vosk_status(self, loaded: bool, model_name: str = ""):
        """
        Установить статус загрузки Vosk модели.

        Args:
            loaded: True если модель загружена
            model_name: Название модели (опционально)
        """
        if loaded:
            text = f"загружена ({model_name})" if model_name else "загружена"
            self._vosk_status.setText(text)
            self._vosk_status.setStyleSheet("color: #10B981;")  # зелёный
        else:
            self._vosk_status.setText("выгружена")
            self._vosk_status.setStyleSheet("color: #9CA3AF;")  # серый

    def set_punctuation_status(self, loaded: bool, model_name: str = ""):
        """
        Установить статус загрузки модели пунктуации.

        Args:
            loaded: True если модель загружена
            model_name: Название модели (опционально)
        """
        if loaded:
            text = f"загружена ({model_name})" if model_name else "загружена"
            self._punct_status.setText(text)
            self._punct_status.setStyleSheet("color: #10B981;")  # зелёный
        else:
            self._punct_status.setText("выгружена")
            self._punct_status.setStyleSheet("color: #9CA3AF;")  # серый

    def set_silero_status(self, loaded: bool):
        """
        Установить статус загрузки Silero TE (для обратной совместимости).
        Теперь просто вызывает set_punctuation_status.

        Args:
            loaded: True если модель загружена
        """
        self.set_punctuation_status(loaded, "Silero TE" if loaded else "")
