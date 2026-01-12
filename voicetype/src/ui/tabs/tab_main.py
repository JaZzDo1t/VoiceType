"""
VoiceType - Main Settings Tab
Вкладка 'Основные' с основными настройками приложения.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QRadioButton, QButtonGroup, QCheckBox, QSlider
)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from loguru import logger

from src.data.config import get_config
from src.utils.system_info import get_microphones
from src.utils.constants import (
    OUTPUT_MODE_KEYBOARD, OUTPUT_MODE_CLIPBOARD,
    THEME_DARK, THEME_LIGHT,
    ENGINE_WHISPER,
    WHISPER_DEFAULT_MODEL
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
    output_mode_changed = pyqtSignal(str)     # keyboard/clipboard
    theme_changed = pyqtSignal(str)           # dark/light
    autostart_changed = pyqtSignal(bool)      # enabled
    whisper_model_changed = pyqtSignal(str)   # base/small
    vad_threshold_changed = pyqtSignal(float) # 0.0-1.0
    noise_floor_changed = pyqtSignal(int)     # 200-2000

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

        # Секция РАСПОЗНАВАНИЕ (Whisper)
        recog_group = QGroupBox("РАСПОЗНАВАНИЕ")
        recog_layout = QVBoxLayout(recog_group)

        # Whisper model selection
        whisper_model_layout = QHBoxLayout()
        whisper_model_layout.addWidget(QLabel("Модель Whisper:"))
        self._whisper_model_combo = QComboBox()
        self._whisper_model_combo.addItem("base (~150MB)", "base")
        self._whisper_model_combo.addItem("small (~500MB) - рекомендуемая", "small")
        self._whisper_model_combo.setMinimumWidth(280)
        whisper_model_layout.addWidget(self._whisper_model_combo)
        whisper_model_layout.addStretch()
        recog_layout.addLayout(whisper_model_layout)

        # Статус модели Whisper
        whisper_status_layout = QHBoxLayout()
        whisper_status_layout.addWidget(QLabel("Whisper:"))
        self._whisper_status = QLabel("выгружена")
        self._whisper_status.setStyleSheet("color: #9CA3AF;")  # серый
        whisper_status_layout.addWidget(self._whisper_status)
        whisper_status_layout.addStretch()
        recog_layout.addLayout(whisper_status_layout)

        # Статус Silero VAD
        vad_status_layout = QHBoxLayout()
        vad_status_layout.addWidget(QLabel("Silero VAD:"))
        self._vad_status = QLabel("выгружен")
        self._vad_status.setStyleSheet("color: #9CA3AF;")  # серый
        vad_status_layout.addWidget(self._vad_status)
        vad_status_layout.addStretch()
        recog_layout.addLayout(vad_status_layout)

        # Порог VAD (чувствительность)
        vad_threshold_layout = QHBoxLayout()
        vad_threshold_layout.addWidget(QLabel("Порог VAD:"))
        self._vad_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._vad_threshold_slider.setMinimum(30)  # 0.3
        self._vad_threshold_slider.setMaximum(90)  # 0.9
        self._vad_threshold_slider.setValue(70)    # 0.7 default
        self._vad_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._vad_threshold_slider.setTickInterval(10)
        self._vad_threshold_slider.setMinimumWidth(150)
        vad_threshold_layout.addWidget(self._vad_threshold_slider)
        self._vad_threshold_label = QLabel("0.7")
        self._vad_threshold_label.setMinimumWidth(30)
        vad_threshold_layout.addWidget(self._vad_threshold_label)
        vad_threshold_layout.addStretch()
        recog_layout.addLayout(vad_threshold_layout)

        # Подсказка
        vad_hint = QLabel("← чувствительнее | строже →")
        vad_hint.setStyleSheet("color: #6B7280; font-size: 11px;")
        recog_layout.addWidget(vad_hint)

        layout.addWidget(recog_group)

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

        # Порог шума (для индикатора уровня)
        noise_floor_layout = QHBoxLayout()
        noise_floor_layout.addWidget(QLabel("Порог шума:"))
        self._noise_floor_slider = QSlider(Qt.Orientation.Horizontal)
        self._noise_floor_slider.setMinimum(200)   # 200
        self._noise_floor_slider.setMaximum(2000)  # 2000
        self._noise_floor_slider.setValue(800)     # 800 default
        self._noise_floor_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._noise_floor_slider.setTickInterval(200)
        self._noise_floor_slider.setMinimumWidth(150)
        noise_floor_layout.addWidget(self._noise_floor_slider)
        self._noise_floor_label = QLabel("800")
        self._noise_floor_label.setMinimumWidth(40)
        noise_floor_layout.addWidget(self._noise_floor_label)
        noise_floor_layout.addStretch()
        audio_layout.addLayout(noise_floor_layout)

        # Подсказка для порога шума
        noise_hint = QLabel("← чувствительнее | фильтрует шумы →")
        noise_hint.setStyleSheet("color: #6B7280; font-size: 11px;")
        audio_layout.addWidget(noise_hint)

        layout.addWidget(audio_group)

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
        # Устанавливаем Whisper как движок по умолчанию
        self._config.set("audio.engine", ENGINE_WHISPER)

        # Whisper model
        whisper_model = self._config.get("audio.whisper.model", WHISPER_DEFAULT_MODEL)
        # Проверяем, что модель в допустимых пределах (base, small)
        if whisper_model not in ["base", "small"]:
            whisper_model = "small"  # default
        index = self._whisper_model_combo.findData(whisper_model)
        if index >= 0:
            self._whisper_model_combo.setCurrentIndex(index)

        # Микрофоны
        self._refresh_microphones()

        # Язык
        language = self._config.get("audio.language", "ru")
        index = self._lang_combo.findData(language)
        if index >= 0:
            self._lang_combo.setCurrentIndex(index)

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

        # VAD threshold
        vad_threshold = self._config.get("audio.whisper.vad_threshold", 0.7)
        slider_value = int(vad_threshold * 100)
        self._vad_threshold_slider.setValue(slider_value)
        self._vad_threshold_label.setText(f"{vad_threshold:.1f}")

        # Noise floor
        noise_floor = self._config.get("audio.noise_floor", 800)
        self._noise_floor_slider.setValue(noise_floor)
        self._noise_floor_label.setText(str(noise_floor))

    def _connect_signals(self):
        """Подключить сигналы."""
        self._whisper_model_combo.currentIndexChanged.connect(self._on_whisper_model_changed)
        self._mic_combo.currentIndexChanged.connect(self._on_mic_changed)
        self._lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        self._output_btn_group.buttonToggled.connect(self._on_output_changed)
        self._autostart_check.toggled.connect(self._on_autostart_changed)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self._vad_threshold_slider.valueChanged.connect(self._on_vad_threshold_changed)
        self._noise_floor_slider.valueChanged.connect(self._on_noise_floor_changed)

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

    def _on_whisper_model_changed(self, index):
        """Обработчик изменения модели Whisper."""
        model = self._whisper_model_combo.currentData()
        if model:
            self._config.set("audio.whisper.model", model)
            self.whisper_model_changed.emit(model)
            logger.debug(f"Whisper model changed: {model}")

    def _on_vad_threshold_changed(self, value):
        """Обработчик изменения порога VAD."""
        threshold = value / 100.0
        self._vad_threshold_label.setText(f"{threshold:.1f}")
        self._config.set("audio.whisper.vad_threshold", threshold)
        self.vad_threshold_changed.emit(threshold)
        logger.debug(f"VAD threshold changed: {threshold}")

    def _on_noise_floor_changed(self, value):
        """Обработчик изменения порога шума."""
        self._noise_floor_label.setText(str(value))
        self._config.set("audio.noise_floor", value)
        self.noise_floor_changed.emit(value)
        logger.debug(f"Noise floor changed: {value}")

    def refresh(self):
        """Обновить все данные вкладки."""
        self._refresh_microphones()

    def set_whisper_status(self, loaded: bool, model_name: str = ""):
        """
        Установить статус загрузки модели Whisper.

        Args:
            loaded: True если модель загружена
            model_name: Название модели (опционально)
        """
        if loaded:
            text = f"загружена ({model_name})" if model_name else "загружена"
            self._whisper_status.setText(text)
            self._whisper_status.setStyleSheet("color: #10B981;")  # зелёный
        else:
            self._whisper_status.setText("выгружена")
            self._whisper_status.setStyleSheet("color: #9CA3AF;")  # серый

    def set_vad_status(self, loaded: bool):
        """
        Установить статус загрузки Silero VAD.

        Args:
            loaded: True если VAD загружен
        """
        if loaded:
            self._vad_status.setText("загружен")
            self._vad_status.setStyleSheet("color: #10B981;")  # зелёный
        else:
            self._vad_status.setText("выгружен")
            self._vad_status.setStyleSheet("color: #9CA3AF;")  # серый
