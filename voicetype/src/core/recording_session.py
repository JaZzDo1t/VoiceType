"""
Логика записи и поток распознавания.

Помощник VoiceTypeApp: обращается к общему состоянию (app._recognizer,
app._audio_capture, app._output_manager) и эмитит сигналы app через
self._app. Cross-thread общение — только через сигналы app (как требует
CLAUDE.md). Флаг записи — threading.Event (потокобезопасно, K2).
"""
import threading
import queue
from datetime import datetime
from typing import Optional
from loguru import logger

from src.core.audio_capture import AudioCapture
from src.utils.constants import TRAY_STATE_READY, TRAY_STATE_RECORDING
from PyQt6.QtCore import QTimer


class RecordingSession:
    def __init__(self, app):
        self._app = app
        self._is_recording = threading.Event()          # K2: было bool
        self._recognition_thread: Optional[threading.Thread] = None
        self._session_text = ""
        self._session_start: Optional[datetime] = None

    def is_recording(self) -> bool:
        return self._is_recording.is_set()

    def start(self):
        """Начать запись и распознавание."""
        if self._is_recording.is_set():
            logger.warning("Already recording")
            return

        # Если модели ещё ни разу не загружались - ждём
        if not self._app._models_loaded:
            logger.warning("Models not loaded yet, ignoring start_recording")
            return

        logger.info("Starting recording...")

        # Whisper: проверяем что recognizer создан
        if not self._app._recognizer:
            logger.info("Re-creating Whisper recognizer...")
            if not self._app._models._create_whisper_recognizer():
                logger.error("Failed to create Whisper recognizer")
                self._app._tray_icon.show_notification("Ошибка", "Не удалось создать Whisper")
                return

        # Проверяем загружены ли модели (могли выгрузиться по таймауту)
        if not self._app._recognizer.is_loaded():
            logger.info("Models unloaded, reloading asynchronously...")
            self._app._models._reload_models_then_start()
            return

        self._do_start_recording()

    def _do_start_recording(self):
        """Внутренний метод - непосредственный старт записи (модели уже загружены)."""
        logger.debug("_do_start_recording: начало")

        if self._is_recording.is_set():
            logger.debug("_do_start_recording: уже записываем, выход")
            return

        self._is_recording.set()
        self._session_text = ""
        self._session_start = datetime.now()
        logger.debug("_do_start_recording: флаги установлены")

        # ВАЖНО: Включаем режим обработки - блокирует auto-unload
        # (может быть уже True если вызвано из _do_reload_models_then_start)
        if not self._app._recognizer.is_processing:
            self._app._recognizer.set_processing(True)
        logger.debug("_do_start_recording: set_processing done")

        # Сбрасываем recognizer
        logger.debug("_do_start_recording: вызов reset()...")
        self._app._recognizer.reset()
        logger.debug("_do_start_recording: reset() завершён")

        # Запускаем захват аудио
        logger.debug("_do_start_recording: создание AudioCapture...")
        mic_id = self._app._config.get("audio.microphone_id", "default")
        self._app._audio_capture = AudioCapture(device_id=mic_id)

        # Устанавливаем порог шума из конфига
        noise_floor = self._app._config.get("audio.noise_floor", 800)
        self._app._audio_capture.set_noise_floor(noise_floor)
        logger.debug("_do_start_recording: AudioCapture создан, запуск...")

        if not self._app._audio_capture.start():
            self._is_recording.clear()
            self._app._tray_icon.show_notification("Ошибка", "Не удалось запустить микрофон")
            logger.error("_do_start_recording: не удалось запустить микрофон")
            return

        logger.debug("_do_start_recording: микрофон запущен, создание recognition thread...")

        # Запускаем поток распознавания
        self._recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._recognition_thread.start()
        logger.debug("_do_start_recording: recognition thread запущен")

        # Запускаем таймер обновления уровня
        self._app._level_timer = QTimer()
        self._app._level_timer.timeout.connect(self._update_audio_level)
        self._app._level_timer.start(50)  # 20 fps

        # Обновляем UI
        self._app._tray_icon.set_state(TRAY_STATE_RECORDING)

        # Обновляем test tab
        self._app._main_window.tab_test.set_models_ready(True)

        logger.info("Recording started successfully")

    def stop(self):
        """Остановить запись."""
        if not self._is_recording.is_set():
            logger.debug("stop_recording() called but not recording, ignoring")
            return

        logger.info(f"stop_recording() called, setting _is_recording=False")

        self._is_recording.clear()

        # Останавливаем таймер уровня
        if self._app._level_timer:
            self._app._level_timer.stop()
            self._app._level_timer = None

        # Останавливаем захват аудио
        if self._app._audio_capture:
            self._app._audio_capture.stop()

        # Получаем финальный результат
        if self._app._recognizer:
            final = self._app._recognizer.get_final_result()
            if final:
                self._app._process_final_text(final)

            # ВАЖНО: Выключаем режим обработки - разрешает auto-unload
            self._app._recognizer.set_processing(False)

        # Сохраняем в историю
        if self._session_text and self._session_start:
            self._app._db.add_history_entry(
                started_at=self._session_start,
                ended_at=datetime.now(),
                text=self._session_text,
                language=self._app._config.get("audio.language", "ru")
            )

        # Обновляем UI
        self._app._tray_icon.set_state(TRAY_STATE_READY)
        self._app._recognition_finished_signal.emit()

        logger.info("Recording stopped")

    def _recognition_loop(self):
        """Цикл распознавания (в отдельном потоке)."""
        audio_queue = self._app._audio_capture.get_audio_queue()

        while self._is_recording.is_set():
            try:
                audio_data = audio_queue.get(timeout=0.1)
                self._app._recognizer.process_audio(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Recognition error: {e}")
                break

    def _update_audio_level(self):
        """Обновить уровень аудио."""
        if self._app._audio_capture:
            level = self._app._audio_capture.get_level()
            self._app._update_level_signal.emit(level)

    def append_session_text(self, text: str):
        """Добавить текст к сессии (вызывается из app._process_final_text)."""
        if self._session_text:
            self._session_text += " " + text
        else:
            self._session_text = text
