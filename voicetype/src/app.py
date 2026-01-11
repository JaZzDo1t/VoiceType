"""
VoiceType - Application Controller
Главный класс приложения, связывающий все компоненты.
Только Whisper движок для распознавания речи.
"""
import sys
import gc
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QObject, pyqtSignal
from loguru import logger

from src.data.config import get_config
from src.data.database import get_database
from src.data.models_manager import get_models_manager
from src.core.audio_capture import AudioCapture
from src.core.whisper_recognizer import WhisperRecognizer
from src.core.output_manager import OutputManager
from src.core.hotkey_manager import HotkeyManager
from src.ui.tray_icon import TrayIcon
from src.ui.main_window import MainWindow
from src.utils.logger import setup_logger
from src.utils.autostart import Autostart
from src.utils.system_info import get_process_cpu, get_process_memory
from src.utils.constants import (
    TRAY_STATE_READY, TRAY_STATE_RECORDING,
    TRAY_STATE_LOADING, TRAY_STATE_ERROR,
    STATS_INTERVAL_SECONDS,
    WHISPER_DEFAULT_MODEL, WHISPER_DEFAULT_VAD_THRESHOLD,
    WHISPER_DEFAULT_MIN_SILENCE_MS, WHISPER_DEFAULT_UNLOAD_TIMEOUT
)


class VoiceTypeApp(QObject):
    """
    Главный класс приложения.
    Управляет всеми компонентами и их взаимодействием.
    """

    # Сигналы для thread-safe обновления UI
    _update_level_signal = pyqtSignal(float)
    _partial_result_signal = pyqtSignal(str)
    _final_result_signal = pyqtSignal(str)
    _recognition_finished_signal = pyqtSignal()
    _models_loaded_signal = pyqtSignal(str, str)  # (state, model_name)
    _loading_status_signal = pyqtSignal(str, str)  # (status_text, model_name)
    _hotkey_triggered_signal = pyqtSignal(str)  # action: "toggle"
    _stats_collected_signal = pyqtSignal(float, float)  # (cpu, ram) для обновления графиков
    _whisper_status_signal = pyqtSignal(bool, str)  # (loaded, model_name) для обновления статуса Whisper
    _vad_status_signal = pyqtSignal(bool)  # (loaded) для обновления статуса VAD

    def __init__(self):
        super().__init__()

        self._config = get_config()
        self._db = get_database()
        self._models_manager = get_models_manager()

        # Компоненты (создаются позже)
        self._audio_capture: Optional[AudioCapture] = None
        self._recognizer: Optional[WhisperRecognizer] = None
        self._output_manager: Optional[OutputManager] = None
        self._hotkey_manager: Optional[HotkeyManager] = None

        # UI
        self._tray_icon: Optional[TrayIcon] = None
        self._main_window: Optional[MainWindow] = None

        # Состояние
        self._is_recording = False
        self._is_initialized = False
        self._models_loaded = False  # Флаг загрузки моделей
        self._recognition_thread: Optional[threading.Thread] = None
        self._session_text = ""
        self._session_start: Optional[datetime] = None

        # Таймеры
        self._stats_timer: Optional[QTimer] = None
        self._level_timer: Optional[QTimer] = None

    def initialize(self) -> bool:
        """
        Инициализировать приложение.

        Returns:
            True если успешно
        """
        try:
            logger.info("Initializing VoiceType application...")

            # Конфигурация
            self._config.load()

            # База данных
            self._db.initialize()

            # Создаём UI
            self._create_ui()

            # Показываем окно загрузки СРАЗУ при старте
            self._main_window.set_loading(True)
            self._main_window.show()

            # Показываем статус загрузки в трее
            self._tray_icon.set_state(TRAY_STATE_LOADING)
            self._tray_icon.show()

            # Загружаем модели в фоне
            self._load_models_async()

            # Инициализируем остальные компоненты
            self._init_output_manager()
            self._init_hotkeys()
            self._start_stats_timer()

            # Подключаем сигналы UI
            self._connect_ui_signals()

            # Применяем настройки автозапуска
            self._apply_autostart_setting()

            # КРИТИЧНО: Запускаем hotkey listener ПОСЛЕ того, как event loop начнёт работать.
            # QTimer.singleShot() сработает только когда app.exec() уже запущен,
            # это гарантирует что PyQt сигналы из pynput thread будут доставлены.
            QTimer.singleShot(100, self._start_hotkeys_deferred)

            self._is_initialized = True
            logger.info("VoiceType initialized successfully (hotkeys will start after event loop)")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            if self._tray_icon:
                self._tray_icon.set_state(TRAY_STATE_ERROR)
            return False

    def _create_ui(self):
        """Создать UI компоненты."""
        # Tray icon
        self._tray_icon = TrayIcon()
        self._tray_icon.start_recording_clicked.connect(self.start_recording)
        self._tray_icon.stop_recording_clicked.connect(self.stop_recording)
        self._tray_icon.settings_clicked.connect(self._show_settings)
        self._tray_icon.exit_clicked.connect(self.quit)

        # Main window
        self._main_window = MainWindow()

        # Подключаем сигналы обновления UI
        self._update_level_signal.connect(self._on_level_update)
        self._partial_result_signal.connect(self._on_partial_result)
        self._final_result_signal.connect(self._on_final_result)
        self._recognition_finished_signal.connect(self._on_recognition_finished)
        self._models_loaded_signal.connect(self._on_models_loaded)
        self._loading_status_signal.connect(self._on_loading_status)
        self._hotkey_triggered_signal.connect(self._on_hotkey_triggered)
        self._whisper_status_signal.connect(self._on_whisper_status_changed)
        self._vad_status_signal.connect(self._on_vad_status_changed)

    def _connect_ui_signals(self):
        """Подключить сигналы UI."""
        # TabMain
        self._main_window.tab_main.microphone_changed.connect(self._on_microphone_changed)
        self._main_window.tab_main.language_changed.connect(self._on_language_changed)
        self._main_window.tab_main.output_mode_changed.connect(self._on_output_mode_changed)
        self._main_window.tab_main.autostart_changed.connect(self._on_autostart_changed)
        # Whisper model changed signal
        if hasattr(self._main_window.tab_main, 'whisper_model_changed'):
            self._main_window.tab_main.whisper_model_changed.connect(self._on_whisper_model_changed)

        # VAD threshold changed signal
        if hasattr(self._main_window.tab_main, 'vad_threshold_changed'):
            self._main_window.tab_main.vad_threshold_changed.connect(self._on_vad_threshold_changed)

        # Noise floor changed signal
        if hasattr(self._main_window.tab_main, 'noise_floor_changed'):
            self._main_window.tab_main.noise_floor_changed.connect(self._on_noise_floor_changed)

        # TabHotkeys - один toggle хоткей вместо двух отдельных
        self._main_window.tab_hotkeys.toggle_hotkey_changed.connect(self._on_toggle_hotkey_changed)

        # TabTest
        self._main_window.tab_test.test_started.connect(self._on_test_started)
        self._main_window.tab_test.test_stopped.connect(self._on_test_stopped)

        # LoadingOverlay action buttons
        self._main_window.loading_overlay.retry_clicked.connect(self._on_retry_loading)
        self._main_window.loading_overlay.settings_clicked.connect(self._on_loading_settings)
        self._main_window.loading_overlay.minimize_clicked.connect(self._on_loading_minimize)

        # TabStats - подключаем сигнал для обновления графиков
        self._stats_collected_signal.connect(self._main_window.tab_stats.update_graphs)

    def _unload_models(self) -> None:
        """Выгрузить текущие модели для освобождения памяти."""
        if self._recognizer is not None:
            if hasattr(self._recognizer, 'unload'):
                self._recognizer.unload()
            self._recognizer = None

        gc.collect()
        logger.debug("Models unloaded, memory freed")

        # Обновляем UI статус (thread-safe через сигналы)
        self._whisper_status_signal.emit(False, "")
        self._vad_status_signal.emit(False)

    def _create_recognizer(self) -> bool:
        """
        Create Whisper recognizer.

        Returns:
            True if recognizer created successfully
        """
        return self._create_whisper_recognizer()

    def _create_whisper_recognizer(self) -> bool:
        """
        Create Whisper recognizer with VAD.

        Returns:
            True if successful (Whisper loads lazily on first process_audio)
        """
        model_size = self._config.get("audio.whisper.model", WHISPER_DEFAULT_MODEL)
        device = self._config.get("audio.whisper.device", "cpu")
        language = self._config.get("audio.language", "ru")
        vad_threshold = self._config.get("audio.whisper.vad_threshold", WHISPER_DEFAULT_VAD_THRESHOLD)
        min_silence_ms = self._config.get("audio.whisper.min_silence_ms", WHISPER_DEFAULT_MIN_SILENCE_MS)
        unload_timeout = self._config.get("audio.whisper.unload_timeout", WHISPER_DEFAULT_UNLOAD_TIMEOUT)

        # Обновляем статус загрузки
        self._loading_status_signal.emit("Подготовка Whisper", model_size)

        self._recognizer = WhisperRecognizer(
            model_size=model_size,
            device=device,
            language=language,
            vad_threshold=vad_threshold,
            min_silence_duration_ms=min_silence_ms,
            unload_timeout_sec=unload_timeout,
        )

        # Connect callbacks
        self._recognizer.on_partial_result = lambda t: self._partial_result_signal.emit(t)
        self._recognizer.on_final_result = lambda t: self._final_result_signal.emit(t)

        # Connect model status callbacks (thread-safe via signals)
        self._recognizer.on_model_loaded = lambda model_name: self._on_recognizer_model_loaded(model_name)
        self._recognizer.on_model_unloaded = lambda: self._on_recognizer_model_unloaded()

        logger.info(f"Whisper recognizer created: model={model_size}, device={device}, language={language}")
        return True  # Whisper loads lazily

    def _load_models_async(self):
        """Загрузить модели в фоновом потоке."""
        def load():
            try:
                # Выгружаем старые модели перед загрузкой новых
                self._unload_models()

                # Создаём Whisper recognizer
                if not self._create_recognizer():
                    self._models_loaded_signal.emit(TRAY_STATE_ERROR, "")
                    return

                model_name = self._config.get("audio.whisper.model", WHISPER_DEFAULT_MODEL)

                # Загружаем модели сразу (не лениво)
                logger.info(f"Загрузка моделей при старте...")
                if self._recognizer and self._recognizer.load_model():
                    # Модели загружены успешно
                    self._models_loaded_signal.emit(TRAY_STATE_READY, model_name)
                    logger.info(f"Whisper и VAD загружены: {model_name}")
                else:
                    # Ошибка загрузки
                    self._models_loaded_signal.emit(TRAY_STATE_ERROR, "")
                    logger.error("Не удалось загрузить модели")

            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                self._models_loaded_signal.emit(TRAY_STATE_ERROR, "")

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def _init_output_manager(self):
        """Инициализировать менеджер вывода."""
        mode = self._config.get("output.mode", "keyboard")
        language = self._config.get("audio.language", "ru")
        self._output_manager = OutputManager(mode, language)

    def _init_hotkeys(self):
        """
        Инициализировать хоткеи (только регистрация, НЕ запуск listener).

        ВАЖНО: start_listening() вызывается отложенно через _start_hotkeys_deferred(),
        потому что Qt event loop должен быть запущен ДО того, как pynput начнёт
        emit сигналы. Иначе сигналы будут потеряны.
        """
        self._hotkey_manager = HotkeyManager()

        # Toggle hotkey - одна кнопка для старт/стоп записи
        toggle_hotkey = self._config.get("hotkeys.toggle_recording")

        # Thread-safe callbacks: emit signals instead of direct method calls
        # because pynput listener runs in a separate thread
        if toggle_hotkey:
            self._hotkey_manager.register(
                toggle_hotkey,
                lambda: self._hotkey_triggered_signal.emit("toggle")
            )

        # НЕ вызываем start_listening() здесь!
        # Listener будет запущен отложенно через QTimer после старта event loop
        logger.debug("Toggle hotkey registered, listener will start after event loop")

    def _start_hotkeys_deferred(self):
        """
        Запустить hotkey listener ПОСЛЕ того, как Qt event loop начал работать.

        Этот метод вызывается через QTimer.singleShot() из initialize(),
        что гарантирует, что event loop уже запущен и PyQt сигналы будут доставлены.
        """
        if not self._hotkey_manager:
            logger.error("HotkeyManager not initialized, cannot start listener")
            return

        logger.info("Starting hotkey listener (deferred, event loop running)...")

        # Проверяем флаг pynput runtime hook (в frozen builds)
        if getattr(sys, 'frozen', False):
            pynput_hook_success = getattr(sys, '_pynput_rthook_success', None)
            if pynput_hook_success is False:
                logger.error("pynput runtime hook failed, hotkeys may not work")
            elif pynput_hook_success is True:
                logger.debug("pynput runtime hook succeeded")

        # Запускаем listener
        success = self._hotkey_manager.start_listening()

        if not success:
            logger.error("Failed to start hotkey listener")
            if self._tray_icon:
                self._tray_icon.show_notification(
                    "VoiceType - Предупреждение",
                    "Не удалось запустить глобальные хоткеи"
                )
            return

        # Проверяем listener через небольшую задержку
        QTimer.singleShot(200, self._validate_hotkey_listener)

    def _validate_hotkey_listener(self):
        """Проверить, что hotkey listener действительно работает."""
        if not self._hotkey_manager:
            return

        is_valid = self._hotkey_manager.validate_listener()

        if is_valid:
            logger.info("Hotkey listener validated successfully")
        else:
            logger.error("Hotkey listener validation failed - listener not running")
            if self._tray_icon:
                self._tray_icon.show_notification(
                    "VoiceType - Ошибка",
                    "Глобальные хоткеи не работают. Попробуйте перезапустить приложение."
                )

    def _start_stats_timer(self):
        """Запустить таймер сбора статистики."""
        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._collect_stats)
        self._stats_timer.start(STATS_INTERVAL_SECONDS * 1000)

    def _collect_stats(self):
        """Собрать и записать статистику."""
        cpu = get_process_cpu()
        ram = get_process_memory()
        self._db.add_stats_entry(cpu, ram)
        self._db.cleanup_old_stats()
        # Уведомляем TabStats для обновления графиков
        self._stats_collected_signal.emit(cpu, ram)

    # === Управление записью ===

    def toggle_recording(self):
        """
        Переключить состояние записи (toggle).
        Если записывает - остановить, если не записывает - начать.
        """
        if self._is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Начать запись и распознавание."""
        if self._is_recording:
            logger.warning("Already recording")
            return

        # Если модели ещё ни разу не загружались - ждём
        if not self._models_loaded:
            logger.warning("Models not loaded yet, ignoring start_recording")
            return

        logger.info("Starting recording...")

        # Whisper: проверяем что recognizer создан
        if not self._recognizer:
            logger.info("Re-creating Whisper recognizer...")
            if not self._create_whisper_recognizer():
                logger.error("Failed to create Whisper recognizer")
                self._tray_icon.show_notification("Ошибка", "Не удалось создать Whisper")
                return

        # Проверяем загружены ли модели (могли выгрузиться по таймауту)
        if not self._recognizer.is_loaded():
            logger.info("Models unloaded, reloading asynchronously...")
            self._reload_models_then_start()
            return

        self._do_start_recording()

    def _reload_models_then_start(self):
        """Перезагрузить модели асинхронно и начать запись."""
        # Показываем статус загрузки в test tab
        self._main_window.tab_test.set_models_ready(False)
        self._main_window.tab_test._status_label.setText("Загрузка модели...")

        def reload():
            try:
                model_name = self._config.get("audio.whisper.model", WHISPER_DEFAULT_MODEL)
                logger.info(f"Перезагрузка моделей: {model_name}")

                if self._recognizer.load_model():
                    # Модели загружены - сигналим UI
                    self._whisper_status_signal.emit(True, model_name)
                    self._vad_status_signal.emit(True)
                    # Запускаем запись через сигнал (thread-safe)
                    QTimer.singleShot(0, self._do_start_recording)
                    logger.info("Модели перезагружены, запускаем запись")
                else:
                    logger.error("Не удалось перезагрузить модели")
                    # Восстанавливаем кнопку теста
                    QTimer.singleShot(0, lambda: self._main_window.tab_test.set_models_ready(True))

            except Exception as e:
                logger.error(f"Ошибка перезагрузки моделей: {e}")
                QTimer.singleShot(0, lambda: self._main_window.tab_test.set_models_ready(True))

        thread = threading.Thread(target=reload, daemon=True)
        thread.start()

    def _do_start_recording(self):
        """Внутренний метод - непосредственный старт записи (модели уже загружены)."""
        if self._is_recording:
            return

        self._is_recording = True
        self._session_text = ""
        self._session_start = datetime.now()

        # Сбрасываем recognizer
        self._recognizer.reset()

        # Запускаем захват аудио
        mic_id = self._config.get("audio.microphone_id", "default")
        self._audio_capture = AudioCapture(device_id=mic_id)

        # Устанавливаем порог шума из конфига
        noise_floor = self._config.get("audio.noise_floor", 800)
        self._audio_capture.set_noise_floor(noise_floor)

        if not self._audio_capture.start():
            self._is_recording = False
            self._tray_icon.show_notification("Ошибка", "Не удалось запустить микрофон")
            return

        # Запускаем поток распознавания
        self._recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._recognition_thread.start()

        # Запускаем таймер обновления уровня
        self._level_timer = QTimer()
        self._level_timer.timeout.connect(self._update_audio_level)
        self._level_timer.start(50)  # 20 fps

        # Обновляем UI
        self._tray_icon.set_state(TRAY_STATE_RECORDING)

        # Обновляем test tab
        self._main_window.tab_test.set_models_ready(True)

        logger.info("Recording started")

    def stop_recording(self):
        """Остановить запись."""
        if not self._is_recording:
            return

        logger.info("Stopping recording...")

        self._is_recording = False

        # Останавливаем таймер уровня
        if self._level_timer:
            self._level_timer.stop()
            self._level_timer = None

        # Останавливаем захват аудио
        if self._audio_capture:
            self._audio_capture.stop()

        # Получаем финальный результат
        if self._recognizer:
            final = self._recognizer.get_final_result()
            if final:
                self._process_final_text(final)

        # Сохраняем в историю
        if self._session_text and self._session_start:
            self._db.add_history_entry(
                started_at=self._session_start,
                ended_at=datetime.now(),
                text=self._session_text,
                language=self._config.get("audio.language", "ru")
            )

        # Обновляем UI
        self._tray_icon.set_state(TRAY_STATE_READY)
        self._recognition_finished_signal.emit()

        logger.info("Recording stopped")

    def _recognition_loop(self):
        """Цикл распознавания (в отдельном потоке)."""
        audio_queue = self._audio_capture.get_audio_queue()

        while self._is_recording:
            try:
                audio_data = audio_queue.get(timeout=0.1)
                self._recognizer.process_audio(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Recognition error: {e}")
                break

    def _update_audio_level(self):
        """Обновить уровень аудио."""
        if self._audio_capture:
            level = self._audio_capture.get_level()
            self._update_level_signal.emit(level)

    def _on_level_update(self, level: float):
        """Обработчик обновления уровня (в UI потоке)."""
        if self._main_window.tab_test.is_testing():
            self._main_window.tab_test.update_level(level)

    def _on_partial_result(self, text: str):
        """Обработчик промежуточного результата."""
        if self._main_window.tab_test.is_testing():
            self._main_window.tab_test.append_partial_result(text)

    def _on_final_result(self, text: str):
        """Обработчик финального результата."""
        self._process_final_text(text)

    def _process_final_text(self, text: str):
        """Обработать финальный текст."""
        if not text:
            return

        logger.debug(f"Processing final text: '{text[:50]}...'")

        # Добавляем к сессии
        if self._session_text:
            self._session_text += " " + text
        else:
            self._session_text = text

        # Выводим текст
        if not self._main_window.tab_test.is_testing():
            self._output_manager.output(text + " ")
        else:
            self._main_window.tab_test.append_final_result(text)

    def _on_recognition_finished(self):
        """Обработчик завершения распознавания."""
        self._main_window.tab_history.refresh()

    def _on_models_loaded(self, state: str, model_name: str):
        """Обработчик завершения загрузки моделей (в UI потоке)."""
        self._tray_icon.set_state(state)
        if state == TRAY_STATE_READY:
            logger.info(f"Models loaded successfully (signal): {model_name}")
            self._models_loaded = True
            # Переключаем на вкладки
            self._main_window.set_loading(False, model_name)

            # Включаем кнопку теста
            self._main_window.tab_test.set_models_ready(True, model_name)

            logger.info(f"Whisper recognizer ready: {model_name}")

            # Показываем уведомление о готовности
            self._tray_icon.show_notification(
                "VoiceType",
                f"Whisper {model_name} готов"
            )
        elif state == TRAY_STATE_ERROR:
            logger.error("Models loading failed (signal)")
            self._models_loaded = False
            self._main_window.show_loading_error("Ошибка загрузки модели")
            self._main_window.tab_test.set_models_ready(False)
            self._main_window.tab_main.set_whisper_status(False)
            self._main_window.tab_main.set_vad_status(False)
            self._tray_icon.show_notification(
                "VoiceType - Ошибка",
                "Не удалось загрузить модель распознавания."
            )

    def _on_loading_status(self, status: str, model_name: str):
        """Обработчик обновления статуса загрузки (в UI потоке)."""
        self._main_window.set_loading_status(status, model_name)

    def _on_hotkey_triggered(self, action: str):
        """Обработчик нажатия хоткея (в UI потоке)."""
        if action == "toggle":
            self.toggle_recording()

    def _on_whisper_status_changed(self, loaded: bool, model_name: str):
        """Обработчик изменения статуса Whisper (в UI потоке)."""
        self._main_window.tab_main.set_whisper_status(loaded, model_name)
        logger.debug(f"Whisper status updated: loaded={loaded}, model={model_name}")

    def _on_vad_status_changed(self, loaded: bool):
        """Обработчик изменения статуса VAD (в UI потоке)."""
        self._main_window.tab_main.set_vad_status(loaded)
        logger.debug(f"VAD status updated: loaded={loaded}")

    def _on_recognizer_model_loaded(self, model_name: str):
        """Callback от WhisperRecognizer при загрузке модели (может вызываться из другого потока)."""
        # Thread-safe обновление UI через сигналы (PyQt сигналы безопасны из любого потока)
        self._whisper_status_signal.emit(True, model_name)
        self._vad_status_signal.emit(True)
        logger.debug(f"Recognizer model loaded callback: {model_name}")

    def _on_recognizer_model_unloaded(self):
        """Callback от WhisperRecognizer при выгрузке модели (может вызываться из другого потока)."""
        # Thread-safe обновление UI через сигналы (PyQt сигналы безопасны из любого потока)
        self._whisper_status_signal.emit(False, "")
        self._vad_status_signal.emit(False)
        logger.debug("Recognizer model unloaded callback")

    # === Обработчики настроек ===

    def _on_microphone_changed(self, mic_id: str):
        """Микрофон изменён."""
        logger.info(f"Microphone changed to: {mic_id}")

    def _on_language_changed(self, language: str):
        """Язык изменён - нужно перезагрузить модель."""
        logger.info(f"Language changed to: {language}")

        # Обновляем язык в OutputManager для правильной раскладки
        if self._output_manager:
            self._output_manager.set_language(language)

        self._models_loaded = False
        self._tray_icon.set_state(TRAY_STATE_LOADING)
        self._main_window.set_loading(True)
        self._load_models_async()

    def _on_whisper_model_changed(self, model: str):
        """Whisper модель изменена - нужно перезагрузить."""
        logger.info(f"Whisper model changed to: {model}")
        self._models_loaded = False
        self._tray_icon.set_state(TRAY_STATE_LOADING)
        self._main_window.set_loading(True)
        self._load_models_async()

    def _on_vad_threshold_changed(self, threshold: float):
        """Порог VAD изменён - применяем в реальном времени."""
        logger.info(f"VAD threshold changed to: {threshold}")
        if self._recognizer and hasattr(self._recognizer, 'set_vad_threshold'):
            self._recognizer.set_vad_threshold(threshold)

    def _on_noise_floor_changed(self, noise_floor: int):
        """Порог шума изменён - применяем в реальном времени."""
        logger.info(f"Noise floor changed to: {noise_floor}")
        if self._audio_capture and hasattr(self._audio_capture, 'set_noise_floor'):
            self._audio_capture.set_noise_floor(noise_floor)

    def _on_output_mode_changed(self, mode: str):
        """Режим вывода изменён."""
        if self._output_manager:
            self._output_manager.set_mode(mode)

    def _on_autostart_changed(self, enabled: bool):
        """Автозапуск изменён."""
        if enabled:
            Autostart.enable()
        else:
            Autostart.disable()

    def _on_toggle_hotkey_changed(self, hotkey: str):
        """Toggle хоткей изменён."""
        if self._hotkey_manager:
            # Удаляем старый
            old_hotkey = self._config.get("hotkeys.toggle_recording")
            if old_hotkey:
                self._hotkey_manager.unregister(old_hotkey)
            # Регистрируем новый с thread-safe callback
            if hotkey:
                self._hotkey_manager.register(
                    hotkey,
                    lambda: self._hotkey_triggered_signal.emit("toggle")
                )

    def _on_test_started(self):
        """Тест начат."""
        self.start_recording()

    def _on_test_stopped(self):
        """Тест остановлен."""
        self.stop_recording()

    def _on_retry_loading(self):
        """Повторить загрузку моделей."""
        logger.info("Retrying model loading...")
        self._main_window.loading_overlay.reset()
        self._models_loaded = False
        self._tray_icon.set_state(TRAY_STATE_LOADING)
        self._load_models_async()

    def _on_loading_settings(self):
        """Открыть настройки из экрана загрузки."""
        logger.info("Opening settings from loading screen")
        # Переключаемся на основной интерфейс и показываем вкладку настроек
        self._main_window.set_loading(False)
        self._main_window.tabs.setCurrentWidget(self._main_window.tab_main)
        self._main_window.show()
        self._main_window.activateWindow()

    def _on_loading_minimize(self):
        """Свернуть окно в трей."""
        logger.info("Minimizing to tray from loading screen")
        self._main_window.hide()

    def _apply_autostart_setting(self):
        """Применить настройку автозапуска."""
        if self._config.get("system.autostart", False):
            Autostart.enable()

    def _show_settings(self):
        """Показать окно настроек."""
        self._main_window.show_and_activate()

    def quit(self):
        """Завершить приложение."""
        logger.info("Shutting down VoiceType...")

        # Останавливаем запись если идёт
        if self._is_recording:
            self.stop_recording()

        # Останавливаем таймеры
        if self._stats_timer:
            self._stats_timer.stop()

        # Останавливаем хоткеи
        if self._hotkey_manager:
            self._hotkey_manager.stop_listening()

        # Выгружаем модели для освобождения памяти
        self._unload_models()

        # Скрываем tray icon
        if self._tray_icon:
            self._tray_icon.hide()

        # Завершаем приложение
        QApplication.quit()

        logger.info("VoiceType shutdown complete")
