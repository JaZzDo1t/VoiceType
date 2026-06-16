"""
Загрузка/выгрузка моделей Whisper + Silero VAD.

Помощник VoiceTypeApp: получает ссылку на app, эмитит его сигналы и
работает с общим состоянием (app._recognizer и т.д.) через self._app.
Загрузка моделей идёт в главном потоке (требование CTranslate2 + Qt).
"""
import gc
from loguru import logger
from PyQt6.QtCore import QTimer

from src.utils.constants import (
    TRAY_STATE_READY, TRAY_STATE_ERROR,
    WHISPER_DEFAULT_MODEL, WHISPER_DEFAULT_VAD_THRESHOLD,
    WHISPER_DEFAULT_MIN_SILENCE_MS, WHISPER_DEFAULT_UNLOAD_TIMEOUT,
)
from src.core.whisper_recognizer import WhisperRecognizer
from src.utils.system_info import reset_vram_baseline


class ModelLoader:
    def __init__(self, app):
        self._app = app

    def _unload_models(self) -> None:
        """Выгрузить текущие модели для освобождения памяти."""
        if self._app._recognizer is not None:
            if hasattr(self._app._recognizer, 'unload'):
                self._app._recognizer.unload()
            self._app._recognizer = None

        gc.collect()
        logger.debug("Models unloaded, memory freed")

        # Сбрасываем baseline VRAM после выгрузки
        reset_vram_baseline()

        # Обновляем UI статус (thread-safe через сигналы)
        self._app._whisper_status_signal.emit(False, "")
        self._app._vad_status_signal.emit(False)

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
        model_size = self._app._config.get("audio.whisper.model", WHISPER_DEFAULT_MODEL)
        device = self._app._config.get("audio.whisper.device", "cuda")
        language = self._app._config.get("audio.language", "ru")
        vad_threshold = self._app._config.get("audio.whisper.vad_threshold", WHISPER_DEFAULT_VAD_THRESHOLD)
        min_silence_ms = self._app._config.get("audio.whisper.min_silence_ms", WHISPER_DEFAULT_MIN_SILENCE_MS)
        unload_timeout = self._app._config.get("audio.whisper.unload_timeout", WHISPER_DEFAULT_UNLOAD_TIMEOUT)

        # Обновляем статус загрузки
        self._app._loading_status_signal.emit("Подготовка Whisper", model_size)

        self._app._recognizer = WhisperRecognizer(
            model_size=model_size,
            device=device,
            language=language,
            vad_threshold=vad_threshold,
            min_silence_duration_ms=min_silence_ms,
            unload_timeout_sec=unload_timeout,
        )

        # Connect callbacks
        self._app._recognizer.on_partial_result = lambda t: self._app._partial_result_signal.emit(t)
        self._app._recognizer.on_final_result = lambda t: self._app._final_result_signal.emit(t)

        # Connect model status callbacks (thread-safe via signals)
        self._app._recognizer.on_model_loaded = lambda model_name: self._app._on_recognizer_model_loaded(model_name)
        self._app._recognizer.on_model_unloaded = lambda: self._app._on_recognizer_model_unloaded()

        # Ошибки во время работы (вызывается из recognition-потока) → UI через сигнал
        self._app._recognizer.on_error = lambda e: self._app._error_signal.emit(
            "Ошибка распознавания", f"{type(e).__name__}: {e}")

        # Connect loading progress callback
        self._app._recognizer.on_loading_progress = lambda progress: self._app._main_window.set_loading_progress(progress, 100)

        logger.info(f"Whisper recognizer created: model={model_size}, device={device}, language={language}")
        return True  # Whisper loads lazily

    def _load_models_async(self):
        """Запланировать загрузку моделей после запуска event loop."""
        # Используем QTimer.singleShot чтобы загрузить модели ПОСЛЕ того как
        # event loop запустится. Загрузка будет в главном потоке, что избегает
        # crash от CTranslate2 + threading + Qt event loop.
        QTimer.singleShot(50, self._do_load_models)

    def _check_environment(self, model_name: str, device: str) -> bool:
        """Проверить окружение перед загрузкой. True = всё ок; иначе показать диагноз и False.

        Вызывается из главного потока (_do_load_models / _do_reload_models_then_start),
        поэтому UI-методы зовём напрямую.
        """
        from src.utils.diagnostics import diagnose
        issues = diagnose(model_name, device)
        if not issues:
            return True
        title = issues[0].title
        detail = "\n".join(i.detail for i in issues)
        logger.error(f"Диагностика выявила проблемы: {detail}")
        self._app._main_window.show_loading_error(title, detail)
        self._app._tray_icon.show_notification("VoiceType - Ошибка", title)
        self._app._tray_icon.set_state(TRAY_STATE_ERROR)
        return False

    def _do_load_models(self):
        """Фактическая загрузка моделей (в главном потоке)."""
        from PyQt6.QtCore import QCoreApplication

        try:
            # Выгружаем старые модели перед загрузкой новых
            self._unload_models()
            QCoreApplication.processEvents()

            # Создаём Whisper recognizer
            if not self._create_recognizer():
                self._app._models_loaded_signal.emit(TRAY_STATE_ERROR, "")
                return

            model_name = self._app._config.get("audio.whisper.model", WHISPER_DEFAULT_MODEL)
            QCoreApplication.processEvents()

            # Проактивная диагностика окружения перед загрузкой
            device = self._app._config.get("audio.whisper.device", "cuda")
            if not self._check_environment(model_name, device):
                self._app._models_loaded = False
                return

            # Загружаем модели
            logger.info(f"Загрузка моделей при старте...")
            if self._app._recognizer and self._app._recognizer.load_model():
                # Модели загружены успешно
                self._app._models_loaded_signal.emit(TRAY_STATE_READY, model_name)
                logger.info(f"Whisper и VAD загружены: {model_name}")
            else:
                # Ошибка загрузки
                self._app._models_loaded_signal.emit(TRAY_STATE_ERROR, "")
                logger.error("Не удалось загрузить модели")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self._app._models_loaded_signal.emit(TRAY_STATE_ERROR, "")

    def _reload_models_then_start(self):
        """Перезагрузить модели и начать запись."""
        # Показываем статус загрузки в test tab
        self._app._main_window.tab_test.set_models_ready(False)
        self._app._main_window.tab_test.set_loading_status("Загрузка модели...")

        # Загружаем в главном потоке через singleShot (избегаем crash с CTranslate2)
        QTimer.singleShot(50, self._do_reload_models_then_start)

    def _do_reload_models_then_start(self):
        """Фактическая перезагрузка моделей (в главном потоке)."""
        from PyQt6.QtCore import QCoreApplication

        try:
            model_name = self._app._config.get("audio.whisper.model", WHISPER_DEFAULT_MODEL)
            logger.info(f"Перезагрузка моделей: {model_name}")

            device = self._app._config.get("audio.whisper.device", "cuda")
            if not self._check_environment(model_name, device):
                if self._app._recognizer is not None:
                    self._app._recognizer.set_processing(False)
                return

            # ВАЖНО: Сначала блокируем auto-unload, потом отменяем таймер
            # Это предотвращает race condition если таймер уже сработал
            self._app._recognizer.set_processing(True)
            self._app._recognizer.cancel_unload_timer()

            QCoreApplication.processEvents()

            if self._app._recognizer.load_model():
                # Модели загружены - сигналим UI
                self._app._whisper_status_signal.emit(True, model_name)
                self._app._vad_status_signal.emit(True)
                logger.info("Модели перезагружены, запускаем запись")
                # Запускаем запись (set_processing уже True)
                self._app._do_start_recording()
            else:
                logger.error("Не удалось перезагрузить модели")
                self._app._recognizer.set_processing(False)  # Разрешаем auto-unload
                self._app._main_window.tab_test.set_models_ready(True)

        except Exception as e:
            logger.error(f"Ошибка перезагрузки моделей: {e}")
            self._app._recognizer.set_processing(False)  # Разрешаем auto-unload
            self._app._main_window.tab_test.set_models_ready(True)
