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
    THEME_DARK, THEME_LIGHT,
    ENGINE_VOSK, ENGINE_WHISPER,
    WHISPER_MODEL_SIZES, WHISPER_DEFAULT_MODEL,
    WHISPER_DEFAULT_VAD_THRESHOLD
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
    engine_changed = pyqtSignal(str)          # vosk/whisper
    whisper_model_changed = pyqtSignal(str)   # tiny/small/medium etc.
    whisper_device_changed = pyqtSignal(str)  # cpu/cuda

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

        # Секция ДВИЖОК
        engine_group = QGroupBox("ДВИЖОК РАСПОЗНАВАНИЯ")
        engine_layout = QVBoxLayout(engine_group)

        # Engine selection
        engine_select_layout = QHBoxLayout()
        engine_select_layout.addWidget(QLabel("Движок:"))
        self._engine_combo = QComboBox()
        self._engine_combo.addItem("Vosk (стриминг)", ENGINE_VOSK)
        self._engine_combo.addItem("Whisper (качество)", ENGINE_WHISPER)
        self._engine_combo.setMinimumWidth(200)
        engine_select_layout.addWidget(self._engine_combo)
        engine_select_layout.addStretch()
        engine_layout.addLayout(engine_select_layout)

        # Whisper settings (shown only when Whisper selected)
        self._whisper_settings = QWidget()
        whisper_layout = QVBoxLayout(self._whisper_settings)
        whisper_layout.setContentsMargins(20, 10, 0, 0)  # indent

        # Whisper model
        whisper_model_layout = QHBoxLayout()
        whisper_model_layout.addWidget(QLabel("Модель Whisper:"))
        self._whisper_model_combo = QComboBox()
        self._whisper_model_combo.addItem("tiny (75MB, быстро)", "tiny")
        self._whisper_model_combo.addItem("base (145MB)", "base")
        self._whisper_model_combo.addItem("small (500MB) ⭐", "small")
        self._whisper_model_combo.addItem("medium (1.5GB, точно)", "medium")
        self._whisper_model_combo.addItem("large-v3 (3GB, макс)", "large-v3")
        self._whisper_model_combo.setMinimumWidth(200)
        whisper_model_layout.addWidget(self._whisper_model_combo)
        whisper_model_layout.addStretch()
        whisper_layout.addLayout(whisper_model_layout)

        # Whisper device
        whisper_device_layout = QHBoxLayout()
        whisper_device_layout.addWidget(QLabel("Устройство:"))
        self._whisper_device_combo = QComboBox()
        self._whisper_device_combo.addItem("CPU", "cpu")
        self._whisper_device_combo.addItem("GPU (CUDA)", "cuda")
        self._whisper_device_combo.setMinimumWidth(150)
        whisper_device_layout.addWidget(self._whisper_device_combo)
        whisper_device_layout.addStretch()
        whisper_layout.addLayout(whisper_device_layout)

        engine_layout.addWidget(self._whisper_settings)
        self._whisper_settings.hide()  # Hidden by default

        layout.addWidget(engine_group)

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

        # Статус Vosk/Whisper (label changes based on engine)
        vosk_layout = QHBoxLayout()
        self._engine_status_label = QLabel("Vosk:")
        vosk_layout.addWidget(self._engine_status_label)
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
        # Engine
        engine = self._config.get("audio.engine", ENGINE_VOSK)
        index = self._engine_combo.findData(engine)
        if index >= 0:
            self._engine_combo.setCurrentIndex(index)
        self._update_engine_ui(engine)

        # Whisper model
        whisper_model = self._config.get("audio.whisper.model", WHISPER_DEFAULT_MODEL)
        index = self._whisper_model_combo.findData(whisper_model)
        if index >= 0:
            self._whisper_model_combo.setCurrentIndex(index)

        # Whisper device
        whisper_device = self._config.get("audio.whisper.device", "cpu")
        index = self._whisper_device_combo.findData(whisper_device)
        if index >= 0:
            self._whisper_device_combo.setCurrentIndex(index)

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
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self._whisper_model_combo.currentIndexChanged.connect(self._on_whisper_model_changed)
        self._whisper_device_combo.currentIndexChanged.connect(self._on_whisper_device_changed)
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

    def _on_engine_changed(self, index):
        """Обработчик изменения движка."""
        engine = self._engine_combo.currentData()
        if engine:
            self._config.set("audio.engine", engine)
            self._update_engine_ui(engine)
            self.engine_changed.emit(engine)
            logger.debug(f"Engine changed: {engine}")

    def _update_engine_ui(self, engine: str):
        """Обновить UI в зависимости от выбранного движка."""
        is_whisper = (engine == ENGINE_WHISPER)
        self._whisper_settings.setVisible(is_whisper)

        # Update engine status label
        self._engine_status_label.setText("Whisper:" if is_whisper else "Vosk:")

        # Vosk-specific settings
        self._model_combo.setEnabled(not is_whisper)  # Disable Vosk model for Whisper

        # Punctuation - Whisper has built-in, disable checkbox
        self._punctuation_check.setEnabled(not is_whisper)
        if is_whisper:
            self._punctuation_check.setChecked(True)  # Whisper always has punctuation

        # Reset status when switching engines
        self._vosk_status.setText("выгружена")
        self._vosk_status.setStyleSheet("color: #9CA3AF;")

    def _on_whisper_model_changed(self, index):
        """Обработчик изменения модели Whisper."""
        model = self._whisper_model_combo.currentData()
        if model:
            self._config.set("audio.whisper.model", model)
            self.whisper_model_changed.emit(model)
            logger.debug(f"Whisper model changed: {model}")

    def _on_whisper_device_changed(self, index):
        """Обработчик изменения устройства Whisper."""
        device = self._whisper_device_combo.currentData()
        if device:
            self._config.set("audio.whisper.device", device)
            self.whisper_device_changed.emit(device)
            logger.debug(f"Whisper device changed: {device}")

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

    def set_whisper_status(self, loaded: bool, model_name: str = ""):
        """
        Set Whisper model status.

        Args:
            loaded: True if model is loaded
            model_name: Name of the model (optional)
        """
        # Update Vosk status label to show Whisper when in Whisper mode
        engine = self._config.get("audio.engine", ENGINE_VOSK)
        if engine == ENGINE_WHISPER:
            if loaded:
                text = f"загружена ({model_name})" if model_name else "загружена"
                self._vosk_status.setText(text)
                self._vosk_status.setStyleSheet("color: #10B981;")
            else:
                self._vosk_status.setText("выгружена")
                self._vosk_status.setStyleSheet("color: #9CA3AF;")
