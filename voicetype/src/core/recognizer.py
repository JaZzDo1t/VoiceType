"""
VoiceType - Speech Recognizer
Обёртка над Vosk для streaming распознавания речи.
"""
import json
import threading
import queue
from pathlib import Path
from typing import Optional, Callable
from loguru import logger

from src.utils.constants import SAMPLE_RATE


class Recognizer:
    """
    Обёртка над Vosk для streaming распознавания.
    Поддерживает partial results (промежуточные) и final results.
    """

    def __init__(self, model_path: str, sample_rate: int = SAMPLE_RATE):
        """
        Args:
            model_path: Путь к папке с моделью Vosk
            sample_rate: Частота дискретизации (должна соответствовать аудио)
        """
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate

        self._model = None
        self._recognizer = None
        self._is_loaded = False
        self._lock = threading.Lock()

        # Callbacks
        self.on_partial_result: Optional[Callable[[str], None]] = None
        self.on_final_result: Optional[Callable[[str], None]] = None
        self.on_loading_progress: Optional[Callable[[int], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    def load_model(self) -> bool:
        """
        Загрузить модель Vosk.

        Returns:
            True если успешно загружена
        """
        if self._is_loaded:
            logger.warning("Model already loaded")
            return True

        try:
            from vosk import Model, KaldiRecognizer, SetLogLevel

            # Отключаем логи Vosk (слишком много спама)
            SetLogLevel(-1)

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            if self.on_loading_progress:
                self.on_loading_progress(10)

            logger.info(f"Loading Vosk model from {self.model_path}...")

            # Загрузка модели
            self._model = Model(str(self.model_path))

            if self.on_loading_progress:
                self.on_loading_progress(80)

            # Создаём recognizer
            self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
            self._recognizer.SetWords(True)  # Включаем информацию о словах

            if self.on_loading_progress:
                self.on_loading_progress(100)

            self._is_loaded = True
            logger.info("Vosk model loaded successfully")
            return True

        except ImportError:
            logger.error("Vosk library not installed")
            if self.on_error:
                self.on_error(ImportError("Vosk library not installed"))
            return False

        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            if self.on_error:
                self.on_error(e)
            return False

    def process_audio(self, audio_data: bytes) -> Optional[str]:
        """
        Обработать порцию аудио.

        Args:
            audio_data: Аудио данные (PCM int16)

        Returns:
            Распознанный текст (partial или final) или None
        """
        if not self._is_loaded or self._recognizer is None:
            logger.warning("Model not loaded, call load_model() first")
            return None

        with self._lock:
            try:
                # AcceptWaveform возвращает True если это конец фразы (final result)
                if self._recognizer.AcceptWaveform(audio_data):
                    # Final result
                    result = json.loads(self._recognizer.Result())
                    text = result.get("text", "").strip()

                    if text and self.on_final_result:
                        self.on_final_result(text)

                    return text if text else None
                else:
                    # Partial result
                    result = json.loads(self._recognizer.PartialResult())
                    text = result.get("partial", "").strip()

                    if text and self.on_partial_result:
                        self.on_partial_result(text)

                    return None  # Не возвращаем partial, только через callback

            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                return None

    def get_final_result(self) -> Optional[str]:
        """
        Получить финальный результат (вызывать после остановки аудио).

        Returns:
            Финальный текст или None
        """
        if not self._is_loaded or self._recognizer is None:
            return None

        with self._lock:
            try:
                result = json.loads(self._recognizer.FinalResult())
                text = result.get("text", "").strip()

                if text and self.on_final_result:
                    self.on_final_result(text)

                return text if text else None

            except Exception as e:
                logger.error(f"Error getting final result: {e}")
                return None

    def reset(self) -> None:
        """Сбросить состояние распознавателя для новой сессии."""
        if self._is_loaded and self._model:
            with self._lock:
                try:
                    from vosk import KaldiRecognizer
                    self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
                    self._recognizer.SetWords(True)
                    logger.debug("Recognizer reset")
                except Exception as e:
                    logger.error(f"Error resetting recognizer: {e}")

    def is_loaded(self) -> bool:
        """Проверить, загружена ли модель."""
        return self._is_loaded

    def unload(self) -> None:
        """Выгрузить модель для освобождения памяти."""
        with self._lock:
            self._recognizer = None
            self._model = None
            self._is_loaded = False
            logger.info("Vosk model unloaded")

    def __del__(self):
        """Деструктор."""
        self.unload()
