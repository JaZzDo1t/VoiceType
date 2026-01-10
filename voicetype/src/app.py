"""
VoiceType - Application Controller
Главный класс приложения, связывающий все компоненты.
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
from src.core.recognizer import Recognizer
from src.core.punctuation import Punctuation, PunctuationDisabled, RUPunctONNX
from src.core.lazy_model_manager import LazyModelManager
from src.core.output_manager import OutputManager
from src.core.hotkey_manager import HotkeyManager
from src.ui.tray_icon import TrayIcon
from src.ui.main_window import MainWindow
from src.ui.themes import apply_theme
from src.utils.logger import setup_logger
from src.utils.autostart import Autostart
from src.utils.system_info import get_process_cpu, get_process_memory
from src.utils.constants import (
    TRAY_STATE_READY, TRAY_STATE_RECORDING,
    TRAY_STATE_LOADING, TRAY_STATE_ERROR,
    STATS_INTERVAL_SECONDS
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

    def __init__(self):
        super().__init__()

        self._config = get_config()
        self._db = get_database()
        self._models_manager = get_models_manager()

        # Компоненты (создаются позже)
        self._audio_capture: Optional[AudioCapture] = None
        self._recognizer: Optional[Recognizer] = None
        self._punctuation = None
        self._output_manager: Optional[OutputManager] = None
        self._hotkey_manager: Optional[HotkeyManager] = None

        # Lazy Model Manager (для экономии RAM)
        self._lazy_model_manager: Optional[LazyModelManager] = None

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
        self._vosk_model_path: Optional[str] = None  # Путь к модели Vosk для перезагрузки

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

        # Lazy Model Manager (создаём здесь для правильной работы с Qt thread)
        self._lazy_model_manager = LazyModelManager(self)
        self._lazy_model_manager.vosk_status_changed.connect(self._on_vosk_status_changed)
        self._lazy_model_manager.punctuation_status_changed.connect(self._on_punctuation_status_changed)

        # Подключаем сигналы обновления UI
        self._update_level_signal.connect(self._on_level_update)
        self._partial_result_signal.connect(self._on_partial_result)
        self._final_result_signal.connect(self._on_final_result)
        self._recognition_finished_signal.connect(self._on_recognition_finished)
        self._models_loaded_signal.connect(self._on_models_loaded)
        self._loading_status_signal.connect(self._on_loading_status)
        self._hotkey_triggered_signal.connect(self._on_hotkey_triggered)

    def _connect_ui_signals(self):
        """Подключить сигналы UI."""
        # TabMain
        self._main_window.tab_main.microphone_changed.connect(self._on_microphone_changed)
        self._main_window.tab_main.language_changed.connect(self._on_language_changed)
        self._main_window.tab_main.model_changed.connect(self._on_model_changed)
        self._main_window.tab_main.output_mode_changed.connect(self._on_output_mode_changed)
        self._main_window.tab_main.autostart_changed.connect(self._on_autostart_changed)

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
        
        if self._punctuation is not None:
            if hasattr(self._punctuation, 'unload'):
                self._punctuation.unload()
            self._punctuation = None
        
        gc.collect()

        # Очищаем CUDA cache только если torch уже загружен runtime hook'ом
        # НЕ делаем import torch - это вызывает двойную инициализацию в frozen build!
        if getattr(sys, '_torch_rthook_success', False) and 'torch' in sys.modules:
            try:
                torch_module = sys.modules['torch']
                if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
            except Exception:
                pass

        logger.debug("Models unloaded, memory freed")

    def _load_vosk_model(self) -> tuple[bool, str]:
        """
        Загрузить модель Vosk.

        Returns:
            Tuple (success, model_name)
        """
        language = self._config.get("audio.language", "ru")
        model_size = self._config.get("audio.model", "small")

        model_path = self._models_manager.get_vosk_model_path(language, model_size)
        model_name = model_path.name if model_path else f"{language}/{model_size}"

        if not model_path:
            logger.error(f"Model not found: {language}/{model_size}")
            return False, model_name

        # Сохраняем путь для перезагрузки
        self._vosk_model_path = str(model_path)

        # Обновляем статус загрузки
        self._loading_status_signal.emit("Загрузка Vosk", model_name)

        # Загружаем Vosk
        self._recognizer = Recognizer(self._vosk_model_path)
        self._recognizer.on_partial_result = lambda t: self._partial_result_signal.emit(t)
        self._recognizer.on_final_result = lambda t: self._final_result_signal.emit(t)

        if not self._recognizer.load_model():
            return False, model_name

        return True, model_name

    def _load_models_async(self):
        """Загрузить модели в фоновом потоке."""
        def load():
            try:
                # Выгружаем старые модели перед загрузкой новых
                self._unload_models()

                # Загружаем Vosk
                success, model_name = self._load_vosk_model()
                if not success:
                    self._models_loaded_signal.emit(TRAY_STATE_ERROR, model_name)
                    return

                # Загружаем пунктуацию через LazyModelManager
                if self._config.get("recognition.punctuation_enabled", True):
                    self._loading_status_signal.emit("Загрузка RUPunct ONNX", model_name)
                    if self._lazy_model_manager.load_punctuation():
                        self._punctuation = self._lazy_model_manager._punctuation
                    else:
                        logger.warning("LazyModelManager punctuation failed, using basic")
                        self._punctuation = PunctuationDisabled()
                else:
                    self._punctuation = PunctuationDisabled()

                # Готово
                self._models_loaded_signal.emit(TRAY_STATE_READY, model_name)
                logger.info("Models loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                self._models_loaded_signal.emit(TRAY_STATE_ERROR, "")

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def _reload_vosk_only_async(self):
        """
        Перезагрузить только Vosk модель, не трогая Silero TE.
        Используется при смене размера модели (small/large) без смены языка.
        """
        def load():
            try:
                # Выгружаем только Vosk, сохраняя Silero TE
                if self._recognizer is not None:
                    if hasattr(self._recognizer, 'unload'):
                        self._recognizer.unload()
                    self._recognizer = None

                gc.collect()

                # Загружаем Vosk
                success, model_name = self._load_vosk_model()
                if not success:
                    self._models_loaded_signal.emit(TRAY_STATE_ERROR, model_name)
                    return

                # Silero TE уже загружен, не трогаем его
                logger.debug(f"Loading state: Vosk reloaded, Silero TE preserved (loaded: {self._punctuation.is_loaded() if self._punctuation else False})")

                # Готово
                self._models_loaded_signal.emit(TRAY_STATE_READY, model_name)
                logger.info(f"Vosk model reloaded successfully: {model_name}")

            except Exception as e:
                logger.error(f"Failed to reload Vosk model: {e}")
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

        # Отменяем таймер автовыгрузки если запущен
        if self._lazy_model_manager:
            self._lazy_model_manager.cancel_auto_unload_timer()

            # Загружаем Vosk если был выгружен
            if not self._recognizer or not self._recognizer.is_loaded():
                if self._vosk_model_path:
                    logger.info("Loading Vosk model for recording...")
                    self._recognizer = Recognizer(self._vosk_model_path)
                    self._recognizer.on_partial_result = lambda t: self._partial_result_signal.emit(t)
                    self._recognizer.on_final_result = lambda t: self._final_result_signal.emit(t)
                    if self._recognizer.load_model():
                        logger.info("Vosk model loaded")
                        self._main_window.tab_main.set_vosk_status(True, Path(self._vosk_model_path).name)
                    else:
                        logger.error("Failed to load Vosk model")
                        self._tray_icon.show_notification("Ошибка", "Не удалось загрузить модель")
                        return
                else:
                    logger.error("No Vosk model path saved")
                    return

            # Загружаем пунктуацию если была выгружена
            if not self._lazy_model_manager.is_punctuation_loaded():
                logger.info("Loading punctuation model for recording...")
                if self._lazy_model_manager.load_punctuation():
                    self._punctuation = self._lazy_model_manager._punctuation
                    logger.info("Punctuation model loaded")

        self._is_recording = True
        self._session_text = ""
        self._session_start = datetime.now()

        # Сбрасываем recognizer
        self._recognizer.reset()

        # Запускаем захват аудио
        mic_id = self._config.get("audio.microphone_id", "default")
        self._audio_capture = AudioCapture(device_id=mic_id)

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

        # Запускаем таймер автовыгрузки моделей для экономии RAM
        if self._lazy_model_manager:
            self._lazy_model_manager.start_auto_unload_timer(timeout_sec=30)

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

        # Применяем пунктуацию
        logger.debug(f"Processing final text: '{text[:50]}...'")
        logger.debug(f"Punctuation: {self._punctuation}, is_loaded: {self._punctuation.is_loaded() if self._punctuation else 'N/A'}")

        if self._punctuation and self._punctuation.is_loaded():
            original = text
            text = self._punctuation.enhance(text)
            logger.info(f"Punctuation applied: '{original[:30]}...' -> '{text[:30]}...'")

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

            # Обновляем статус Vosk
            vosk_loaded = self._recognizer and self._recognizer.is_loaded()
            self._main_window.tab_main.set_vosk_status(vosk_loaded, model_name if vosk_loaded else "")
            logger.info(f"Vosk status: {'loaded' if vosk_loaded else 'not loaded'}")

            # Обновляем статус пунктуации
            punct_loaded = self._punctuation and self._punctuation.is_loaded()
            punct_name = ""
            if punct_loaded:
                info = self._punctuation.get_model_info()
                punct_name = info.get("model_name", info.get("type", "RUPunct"))
            self._main_window.tab_main.set_punctuation_status(punct_loaded, punct_name)
            logger.info(f"Punctuation status: {'loaded' if punct_loaded else 'not loaded'} ({punct_name})")

            # Запускаем таймер автовыгрузки моделей для экономии RAM
            if self._lazy_model_manager:
                self._lazy_model_manager.start_auto_unload_timer(timeout_sec=30)

            # Показываем уведомление о готовности
            self._tray_icon.show_notification(
                "VoiceType",
                "Модель загружена"
            )
        elif state == TRAY_STATE_ERROR:
            logger.error("Models loading failed (signal)")
            self._models_loaded = False
            self._main_window.show_loading_error("Ошибка загрузки модели")
            self._main_window.tab_test.set_models_ready(False)
            self._main_window.tab_main.set_vosk_status(False)
            self._main_window.tab_main.set_punctuation_status(False)
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

    def _on_vosk_status_changed(self, loaded: bool, model_name: str):
        """Обработчик изменения статуса Vosk модели (от LazyModelManager)."""
        if not loaded and self._recognizer:
            # Выгружаем recognizer при автовыгрузке
            self._recognizer.unload()
            self._recognizer = None
            logger.info("Vosk recognizer unloaded via auto-unload")
        self._main_window.tab_main.set_vosk_status(loaded, model_name)
        logger.debug(f"Vosk status changed: loaded={loaded}, name={model_name}")

    def _on_punctuation_status_changed(self, loaded: bool, model_name: str):
        """Обработчик изменения статуса модели пунктуации (от LazyModelManager)."""
        self._main_window.tab_main.set_punctuation_status(loaded, model_name)
        logger.debug(f"Punctuation status changed: loaded={loaded}, name={model_name}")

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

    def _on_model_changed(self, model: str):
        """Модель изменена - нужно перезагрузить только Vosk, Silero TE не трогаем."""
        logger.info(f"Model changed to: {model}")
        self._models_loaded = False
        self._tray_icon.set_state(TRAY_STATE_LOADING)
        self._main_window.set_loading(True)
        self._reload_vosk_only_async()

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

        # Выгружаем модели через LazyModelManager
        if self._lazy_model_manager:
            self._lazy_model_manager.unload_all()

        # Выгружаем модели для освобождения памяти (fallback)
        self._unload_models()

        # Скрываем tray icon
        if self._tray_icon:
            self._tray_icon.hide()

        # Завершаем приложение
        QApplication.quit()

        logger.info("VoiceType shutdown complete")
