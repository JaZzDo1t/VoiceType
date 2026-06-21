"""
VoiceType - Whisper Speech Recognizer
Распознаватель речи на основе faster-whisper с Silero VAD (ONNX).

Использует ONNX версию Silero VAD - не требует PyTorch!
"""
import os
import threading
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from loguru import logger

from src.utils.constants import SAMPLE_RATE
from src.core.audio_buffer import AudioBuffer
from src.core.vad_processor import VadProcessor


class WhisperRecognizer:
    """
    Распознаватель речи на основе faster-whisper.

    Использует Silero VAD для определения пауз в речи.
    Поддерживает автоматическую выгрузку модели после периода неактивности.

    Пример использования:
        recognizer = WhisperRecognizer(model_size="small", device="cuda")
        recognizer.on_final_result = lambda text: print(f"Результат: {text}")
        recognizer.load_model()

        # В цикле обработки аудио
        recognizer.process_audio(audio_chunk)

        # После завершения записи
        text = recognizer.get_final_result()
    """

    # Поддерживаемые размеры моделей Whisper (small, medium)
    SUPPORTED_MODEL_SIZES = ["small", "medium"]

    # Поддерживаемые устройства (cuda предпочтителен, cpu как fallback)
    SUPPORTED_DEVICES = ["cuda", "cpu"]

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        language: str = "ru",
        sample_rate: int = SAMPLE_RATE,
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        unload_timeout_sec: int = 60,
    ):
        """
        Инициализация распознавателя Whisper.

        Args:
            model_size: Размер модели Whisper (small, medium)
            device: Устройство для вычислений (cuda)
            language: Язык распознавания (ru, en и т.д.)
            sample_rate: Частота дискретизации аудио (по умолчанию 16000)
            vad_threshold: Порог срабатывания VAD (0.0-1.0, выше = менее чувствительный)
            min_silence_duration_ms: Минимальная длительность паузы для срабатывания транскрипции (мс)
            unload_timeout_sec: Время неактивности до автоматической выгрузки модели (секунды)
        """
        # Валидация параметров
        if model_size not in self.SUPPORTED_MODEL_SIZES:
            raise ValueError(
                f"Неподдерживаемый размер модели: {model_size}. "
                f"Поддерживаются: {self.SUPPORTED_MODEL_SIZES}"
            )

        if device not in self.SUPPORTED_DEVICES:
            raise ValueError(
                f"Неподдерживаемое устройство: {device}. "
                f"Поддерживаются: {self.SUPPORTED_DEVICES}"
            )

        self.model_size = model_size
        self.device = device
        self.language = language
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.unload_timeout_sec = unload_timeout_sec

        # Внутреннее состояние
        self._model = None
        self._vad = VadProcessor(sample_rate, vad_threshold)
        self._is_loaded = False
        self._lock = threading.Lock()

        # Буфер аудио с триггером транскрипции по тишине
        self._buffer = AudioBuffer(sample_rate, min_silence_duration_ms)

        # Для автоматической выгрузки
        self._last_activity_time: float = 0
        self._unload_timer: Optional[threading.Timer] = None
        self._unload_lock = threading.Lock()
        self._is_processing = False  # Флаг активной обработки (блокирует auto-unload)

        # Callbacks
        self.on_partial_result: Optional[Callable[[str], None]] = None
        self.on_final_result: Optional[Callable[[str], None]] = None
        self.on_loading_progress: Optional[Callable[[int], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_model_loaded: Optional[Callable[[str], None]] = None  # called with model_name after loading
        self.on_model_unloaded: Optional[Callable[[], None]] = None  # called after unloading
        self._last_load_error: Optional[str] = None  # текст последней ошибки загрузки (для UI)

        logger.info(
            f"WhisperRecognizer инициализирован: model={model_size}, "
            f"device={device}, language={language}"
        )

        # Предзагружаем пути к CUDA DLL при инициализации
        self._add_cuda_dll_paths()

    def _add_cuda_dll_paths(self) -> None:
        """Добавить пути к CUDA DLL для Windows. Тонкая обёртка над src.utils.cuda_paths."""
        from src.utils.cuda_paths import add_cuda_dll_paths
        add_cuda_dll_paths()

    def load_model(self) -> bool:
        """
        Загрузить модели Whisper и Silero VAD.

        Returns:
            True если модели успешно загружены, False в случае ошибки
        """
        if self._is_loaded:
            logger.warning("Модели уже загружены")
            return True

        try:
            if self.on_loading_progress:
                self.on_loading_progress(5)

            # Добавляем пути к CUDA DLL (нужно для Windows после unload/reload)
            self._add_cuda_dll_paths()

            # Загрузка faster-whisper
            logger.info(f"Загрузка модели Whisper ({self.model_size})...")
            from faster_whisper import WhisperModel

            if self.on_loading_progress:
                self.on_loading_progress(20)

            # GPU: float16, CPU: int8
            compute_type = "float16" if self.device == "cuda" else "int8"

            # Без молчаливого CUDA→CPU fallback: при device=cuda и сбое ошибка
            # пробрасывается наверх, чтобы пользователь увидел честный диагноз.
            # Переключение на CPU делается осознанно через настройки.
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )

            if self.on_loading_progress:
                self.on_loading_progress(60)

            # Загрузка Silero VAD
            logger.info("Загрузка Silero VAD...")
            self._vad.load()

            if self.on_loading_progress:
                self.on_loading_progress(90)

            self._is_loaded = True
            self._update_activity_time()

            if self.on_loading_progress:
                self.on_loading_progress(100)

            logger.info("Модели Whisper и VAD успешно загружены")

            # Invoke callback for model loaded
            if self.on_model_loaded:
                self.on_model_loaded(self.model_size)

            return True

        except ImportError as e:
            error_msg = f"Библиотека не установлена: {e}"
            logger.error(error_msg)
            self._last_load_error = error_msg
            if self.on_error:
                self.on_error(ImportError(error_msg))
            return False

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self._last_load_error = str(e)
            if self.on_error:
                self.on_error(e)
            return False

    def _update_activity_time(self) -> None:
        """Обновить время последней активности и перезапустить таймер выгрузки."""
        self._last_activity_time = time.time()
        self._restart_unload_timer()

    def _restart_unload_timer(self) -> None:
        """Перезапустить таймер автоматической выгрузки."""
        with self._unload_lock:
            # Отменяем текущий таймер
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None

            # Создаём новый таймер только если не в процессе обработки
            if self._is_loaded and self.unload_timeout_sec > 0 and not self._is_processing:
                self._unload_timer = threading.Timer(
                    self.unload_timeout_sec,
                    self._auto_unload
                )
                self._unload_timer.daemon = True
                self._unload_timer.start()

    def cancel_unload_timer(self) -> None:
        """Отменить таймер автоматической выгрузки (вызывать перед загрузкой/записью)."""
        with self._unload_lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None
                logger.debug("Таймер выгрузки отменён")

    @property
    def is_processing(self) -> bool:
        """Флаг активной обработки (True = auto-unload заблокирован)."""
        return self._is_processing

    def set_processing(self, is_processing: bool) -> None:
        """
        Установить флаг активной обработки.

        Пока флаг True, auto-unload не будет срабатывать.
        Вызывать перед началом и после завершения записи.
        """
        self._is_processing = is_processing
        if is_processing:
            # Отменяем таймер при начале обработки
            self.cancel_unload_timer()
            logger.debug("Режим обработки включён, таймер выгрузки отменён")
        else:
            # Перезапускаем таймер после завершения обработки
            self._update_activity_time()
            logger.debug("Режим обработки выключен, таймер выгрузки перезапущен")

    def _auto_unload(self) -> None:
        """Автоматическая выгрузка модели после периода неактивности."""
        # Проверяем с lock для thread-safety
        with self._unload_lock:
            # Не выгружаем если идёт обработка
            if self._is_processing:
                logger.debug("Auto-unload пропущен: идёт обработка")
                return

            elapsed = time.time() - self._last_activity_time
            should_unload = elapsed >= self.unload_timeout_sec and self._is_loaded

        if should_unload:
            logger.info(
                f"Автоматическая выгрузка модели после {elapsed:.0f}с неактивности"
            )
            self.unload()

    def process_audio(self, audio_data: bytes) -> Optional[str]:
        """
        Обработать порцию аудио данных.

        Аудио накапливается в буфере. При обнаружении паузы (через VAD)
        происходит транскрипция накопленного аудио.

        Args:
            audio_data: Аудио данные в формате PCM int16

        Returns:
            None (результат возвращается через callback on_final_result)
        """
        # Проверка загрузки с lock для thread-safety
        with self._lock:
            is_loaded = self._is_loaded
            model_exists = self._model is not None

        # Автозагрузка модели при необходимости (вне lock, т.к. load_model долгий)
        if not is_loaded or not model_exists:
            logger.info("Модель не загружена, выполняется автозагрузка...")
            if not self.load_model():
                return None

        self._update_activity_time()

        with self._lock:
            try:
                # Конвертация bytes в numpy array (int16 -> float32)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio_np = audio_np / 32768.0  # Нормализация в диапазон [-1, 1]

                # Определение речи/тишины через VAD
                is_speech = self._vad.detect_speech(audio_np)

                if is_speech and self.on_partial_result:
                    self.on_partial_result("...")

                if self._buffer.add(audio_np, is_speech):
                    result = self._transcribe(self._buffer.get_audio())
                    self._buffer.reset()
                    self._reset_vad_state()
                    if result and self.on_final_result:
                        self.on_final_result(result)
                    return result

                return None

            except Exception as e:
                logger.error(f"Ошибка обработки аудио: {e}")
                if self.on_error:
                    self.on_error(e)
                return None

    def _transcribe(self, audio_concat: np.ndarray) -> Optional[str]:
        """Транскрибировать переданный аудио-массив. Вернуть текст или None."""
        if audio_concat is None or len(audio_concat) == 0 or self._model is None:
            return None
        try:
            self._add_cuda_dll_paths()
            min_samples = int(0.5 * self.sample_rate)
            if len(audio_concat) < min_samples:
                logger.debug("Аудио слишком короткое для транскрипции")
                return None
            logger.debug(f"Транскрипция {len(audio_concat) / self.sample_rate:.2f}с аудио...")
            segments, info = self._model.transcribe(
                audio_concat, language=self.language, beam_size=5, best_of=5,
                temperature=0.0, vad_filter=False, without_timestamps=True,
                # Антигаллюцинации Whisper (титры YouTube на тишине/шуме):
                # не переносим контекст между сегментами — иначе галлюцинация
                # "заражает" следующие; и режем неуверенные/не-речевые сегменты.
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            texts = []
            for segment in segments:
                # Фильтр по уверенности сегмента: типичные галлюцинации идут
                # с высоким no_speech_prob и низким avg_logprob.
                no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
                avg_logprob = getattr(segment, "avg_logprob", 0.0)
                if no_speech_prob > 0.6 and avg_logprob < -0.5:
                    logger.debug(
                        f"Сегмент отброшен (no_speech={no_speech_prob:.2f}, "
                        f"logprob={avg_logprob:.2f}): {segment.text.strip()[:60]}"
                    )
                    continue
                text = segment.text.strip()
                if self._is_hallucination(text):
                    logger.debug(f"Сегмент-галлюцинация отброшен: {text[:60]}")
                    continue
                texts.append(text)
            result = " ".join(texts).strip()
            if result:
                logger.info(f"Транскрипция: {result[:100]}...")
                return result
            return None
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {e}")
            if self.on_error:
                self.on_error(e)
            return None

    # Характерные маркеры галлюцинаций Whisper (титры из обучающих данных).
    # Сравнение регистронезависимое по вхождению подстроки.
    _HALLUCINATION_MARKERS = (
        "редактор субтитров",
        "корректор",
        "субтитры сделал",
        "субтитры создавал",
        "субтитры подготовил",
        "продолжение следует",
        "спасибо за просмотр",
        "спасибо за внимание",
        "подписывайтесь на канал",
        "ставьте лайк",
        "subscribe",
        "thanks for watching",
        "amara.org",
        "субтитры и перевод",
        "редактор",
    )

    def _is_hallucination(self, text: str) -> bool:
        """Проверить, похож ли текст на типичную галлюцинацию Whisper."""
        if not text:
            return False
        lowered = text.lower()
        return any(marker in lowered for marker in self._HALLUCINATION_MARKERS)

    def _reset_buffer(self) -> None:
        """Сбросить буфер аудио и состояние VAD."""
        self._buffer.reset()
        self._reset_vad_state()

    def _reset_vad_state(self) -> None:
        """Сбросить внутреннее состояние VAD для новой сессии."""
        self._vad.reset()

    def reset_for_new_session(self) -> None:
        """
        Подготовить распознаватель к новой сессии записи.
        Сбрасывает VAD state и буфер.
        """
        logger.debug("Сброс состояния для новой сессии записи")
        self._reset_buffer()

    def get_final_result(self) -> Optional[str]:
        """
        Получить финальный результат (вызывать после остановки записи).

        Транскрибирует весь оставшийся буфер независимо от пауз.

        Returns:
            Распознанный текст или None
        """
        logger.debug(f"get_final_result called: is_loaded={self._is_loaded}, has_audio={self._buffer.has_audio}")

        if not self._is_loaded:
            logger.warning("get_final_result: модель не загружена!")
            return None

        with self._lock:
            if not self._buffer.has_audio:
                logger.debug("get_final_result: буфер пуст, нечего транскрибировать")
                return None

            audio = self._buffer.get_audio()
            logger.info(
                f"get_final_result: транскрибируем {len(audio) / self.sample_rate:.2f}с аудио"
            )
            result = self._transcribe(audio)
            self._reset_buffer()

            return result

    def reset(self) -> None:
        """Сбросить состояние распознавателя для новой сессии."""
        with self._lock:
            self._reset_buffer()
            logger.debug("Распознаватель сброшен")

    def is_loaded(self) -> bool:
        """
        Проверить, загружены ли модели.

        Returns:
            True если модели загружены
        """
        return self._is_loaded

    def unload(self) -> None:
        """Выгрузить модели для освобождения памяти."""
        # Сначала отменяем таймер (отдельный lock, без вложенности)
        with self._unload_lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None

        # Теперь основная выгрузка
        with self._lock:
            # Проверяем что не в процессе обработки
            if self._is_processing:
                logger.warning("unload() вызван во время обработки, пропускаем")
                return

            if not self._is_loaded:
                logger.debug("unload(): модели уже выгружены")
                return

            # Очищаем модели - используем del для вызова деструкторов
            if self._model is not None:
                del self._model
            self._model = None

            self._vad.unload()

            # Очищаем буфер
            self._buffer.reset()

            self._is_loaded = False

            # Агрессивная очистка памяти
            import gc
            gc.collect()

            # Пробуем освободить CUDA память если доступно
            try:
                import ctranslate2
                if hasattr(ctranslate2, 'cuda') and hasattr(ctranslate2.cuda, 'empty_cache'):
                    ctranslate2.cuda.empty_cache()
            except Exception as e:
                logger.debug(f"ctranslate2 empty_cache недоступен: {e}")

            logger.info("Модели Whisper и VAD ONNX выгружены")

        # Invoke callback OUTSIDE of lock to avoid deadlock with UI
        if self.on_model_unloaded:
            self.on_model_unloaded()

    def set_language(self, language: str) -> None:
        """
        Установить язык распознавания.

        Args:
            language: Код языка (например, 'ru', 'en')
        """
        self.language = language
        logger.info(f"Язык распознавания установлен: {language}")

    def set_vad_threshold(self, threshold: float) -> None:
        """
        Установить порог срабатывания VAD.

        Args:
            threshold: Порог (0.0-1.0, выше = менее чувствительный)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Порог VAD должен быть в диапазоне [0.0, 1.0]")

        self.vad_threshold = threshold
        logger.info(f"Порог VAD установлен: {threshold}")

    def set_min_silence_duration(self, duration_ms: int) -> None:
        """
        Установить минимальную длительность паузы для транскрипции.

        Args:
            duration_ms: Длительность в миллисекундах
        """
        if duration_ms < 100:
            raise ValueError("Минимальная длительность паузы должна быть >= 100мс")

        self.min_silence_duration_ms = duration_ms
        self._buffer.set_silence_threshold(duration_ms)
        logger.info(f"Минимальная пауза установлена: {duration_ms}мс")

    def get_model_info(self) -> dict:
        """
        Получить информацию о модели.

        Returns:
            Словарь с информацией о модели
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "vad_threshold": self.vad_threshold,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "unload_timeout_sec": self.unload_timeout_sec,
            "is_loaded": self._is_loaded,
        }

    def __del__(self):
        """Деструктор - выгружаем модели."""
        try:
            self.unload()
        except Exception:
            pass
