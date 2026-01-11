"""
VoiceType - Whisper Speech Recognizer
Распознаватель речи на основе faster-whisper с Silero VAD (ONNX).

Использует ONNX версию Silero VAD - не требует PyTorch!
"""
import os
import threading
import time
import urllib.request
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from loguru import logger

from src.utils.constants import SAMPLE_RATE

# URL для скачивания ONNX модели Silero VAD
SILERO_VAD_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


class WhisperRecognizer:
    """
    Распознаватель речи на основе faster-whisper.

    Использует Silero VAD для определения пауз в речи.
    Поддерживает автоматическую выгрузку модели после периода неактивности.

    Пример использования:
        recognizer = WhisperRecognizer(model_size="small", device="cpu")
        recognizer.on_final_result = lambda text: print(f"Результат: {text}")
        recognizer.load_model()

        # В цикле обработки аудио
        recognizer.process_audio(audio_chunk)

        # После завершения записи
        text = recognizer.get_final_result()
    """

    # Поддерживаемые размеры моделей Whisper
    SUPPORTED_MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

    # Поддерживаемые устройства
    SUPPORTED_DEVICES = ["cpu", "cuda"]

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        language: str = "ru",
        sample_rate: int = SAMPLE_RATE,
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        unload_timeout_sec: int = 60,
    ):
        """
        Инициализация распознавателя Whisper.

        Args:
            model_size: Размер модели Whisper (tiny, base, small, medium, large-v3)
            device: Устройство для вычислений (cpu или cuda)
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
        self._vad_session = None  # ONNX InferenceSession для VAD
        self._vad_state = None  # Combined state для VAD ONNX [2, 1, 128]
        self._is_loaded = False
        self._lock = threading.Lock()

        # Буфер аудио для накопления до паузы
        self._audio_buffer: List[np.ndarray] = []
        self._speech_started = False
        self._silence_samples = 0
        self._silence_threshold_samples = int(
            (min_silence_duration_ms / 1000) * sample_rate
        )

        # Для автоматической выгрузки
        self._last_activity_time: float = 0
        self._unload_timer: Optional[threading.Timer] = None
        self._unload_lock = threading.Lock()

        # Callbacks
        self.on_partial_result: Optional[Callable[[str], None]] = None
        self.on_final_result: Optional[Callable[[str], None]] = None
        self.on_loading_progress: Optional[Callable[[int], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

        logger.info(
            f"WhisperRecognizer инициализирован: model={model_size}, "
            f"device={device}, language={language}"
        )

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

            # Загрузка faster-whisper
            logger.info(f"Загрузка модели Whisper ({self.model_size})...")
            from faster_whisper import WhisperModel

            if self.on_loading_progress:
                self.on_loading_progress(20)

            # Определение типа вычислений
            compute_type = "float16" if self.device == "cuda" else "int8"

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )

            if self.on_loading_progress:
                self.on_loading_progress(60)

            # Загрузка Silero VAD
            logger.info("Загрузка Silero VAD...")
            self._load_silero_vad()

            if self.on_loading_progress:
                self.on_loading_progress(90)

            self._is_loaded = True
            self._update_activity_time()

            if self.on_loading_progress:
                self.on_loading_progress(100)

            logger.info("Модели Whisper и VAD успешно загружены")
            return True

        except ImportError as e:
            error_msg = f"Библиотека не установлена: {e}"
            logger.error(error_msg)
            if self.on_error:
                self.on_error(ImportError(error_msg))
            return False

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            if self.on_error:
                self.on_error(e)
            return False

    def _load_silero_vad(self) -> None:
        """Загрузить модель Silero VAD (ONNX версия - без PyTorch!)."""
        import onnxruntime as ort

        # Путь к ONNX модели VAD
        vad_cache_dir = Path.home() / ".cache" / "silero-vad"
        vad_cache_dir.mkdir(parents=True, exist_ok=True)
        vad_onnx_path = vad_cache_dir / "silero_vad.onnx"

        # Скачиваем модель если не существует
        if not vad_onnx_path.exists():
            logger.info("Скачивание Silero VAD ONNX модели...")
            try:
                urllib.request.urlretrieve(SILERO_VAD_ONNX_URL, str(vad_onnx_path))
                logger.info(f"VAD модель сохранена: {vad_onnx_path}")
            except Exception as e:
                logger.error(f"Ошибка скачивания VAD модели: {e}")
                raise

        # Создаём ONNX сессию
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1

        self._vad_session = ort.InferenceSession(
            str(vad_onnx_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        # Инициализируем state (2 слоя, batch=1, 128 units)
        # ONNX модель использует combined state вместо отдельных h и c
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)

        logger.debug("Silero VAD ONNX загружен (без PyTorch!)")

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

            # Создаём новый таймер
            if self._is_loaded and self.unload_timeout_sec > 0:
                self._unload_timer = threading.Timer(
                    self.unload_timeout_sec,
                    self._auto_unload
                )
                self._unload_timer.daemon = True
                self._unload_timer.start()

    def _auto_unload(self) -> None:
        """Автоматическая выгрузка модели после периода неактивности."""
        with self._unload_lock:
            elapsed = time.time() - self._last_activity_time
            if elapsed >= self.unload_timeout_sec and self._is_loaded:
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
        # Автозагрузка модели при необходимости
        if not self._is_loaded:
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
                is_speech = self._detect_speech(audio_np)

                if is_speech:
                    # Речь обнаружена - добавляем в буфер
                    self._audio_buffer.append(audio_np)
                    self._speech_started = True
                    self._silence_samples = 0

                    # Отправляем частичный результат (индикация что идёт запись)
                    if self.on_partial_result and len(self._audio_buffer) > 0:
                        # Показываем что идёт запись
                        self.on_partial_result("...")

                elif self._speech_started:
                    # Тишина после речи - накапливаем паузу
                    self._audio_buffer.append(audio_np)
                    self._silence_samples += len(audio_np)

                    # Проверяем достигнут ли порог тишины
                    if self._silence_samples >= self._silence_threshold_samples:
                        # Пауза достигнута - транскрибируем
                        result = self._transcribe_buffer()

                        # Сбрасываем состояние
                        self._reset_buffer()

                        if result and self.on_final_result:
                            self.on_final_result(result)

                        return result

                return None

            except Exception as e:
                logger.error(f"Ошибка обработки аудио: {e}")
                if self.on_error:
                    self.on_error(e)
                return None

    def _detect_speech(self, audio_np: np.ndarray) -> bool:
        """
        Определить наличие речи в аудио через Silero VAD (ONNX).

        Args:
            audio_np: Аудио в формате numpy array (float32, [-1, 1])

        Returns:
            True если обнаружена речь, False если тишина
        """
        if self._vad_session is None:
            return True  # Если VAD не загружен - считаем что есть речь

        try:
            # Silero VAD ONNX требует ровно 512 семплов для 16kHz
            # Обрабатываем аудио окнами по 512 семплов
            WINDOW_SIZE = 512 if self.sample_rate == 16000 else 256
            sr_input = np.array(self.sample_rate, dtype=np.int64)

            max_speech_prob = 0.0

            # Обрабатываем все полные окна
            for i in range(0, len(audio_np) - WINDOW_SIZE + 1, WINDOW_SIZE):
                window = audio_np[i:i + WINDOW_SIZE]
                audio_input = window.reshape(1, -1).astype(np.float32)

                ort_inputs = {
                    'input': audio_input,
                    'state': self._vad_state,
                    'sr': sr_input,
                }

                output, state_out = self._vad_session.run(None, ort_inputs)
                self._vad_state = state_out

                speech_prob = output[0][0]
                max_speech_prob = max(max_speech_prob, speech_prob)

                # Ранний выход если точно речь
                if speech_prob >= self.vad_threshold:
                    return True

            return max_speech_prob >= self.vad_threshold

        except Exception as e:
            logger.warning(f"Ошибка VAD: {e}")
            return True  # При ошибке считаем что есть речь

    def _transcribe_buffer(self) -> Optional[str]:
        """
        Транскрибировать накопленный аудио буфер.

        Returns:
            Распознанный текст или None
        """
        if not self._audio_buffer or self._model is None:
            return None

        try:
            # Объединяем все чанки в один массив
            audio_concat = np.concatenate(self._audio_buffer)

            # Минимальная длина аудио (0.5 секунды)
            min_samples = int(0.5 * self.sample_rate)
            if len(audio_concat) < min_samples:
                logger.debug("Аудио слишком короткое для транскрипции")
                return None

            logger.debug(f"Транскрипция {len(audio_concat) / self.sample_rate:.2f}с аудио...")

            # Транскрипция через faster-whisper
            segments, info = self._model.transcribe(
                audio_concat,
                language=self.language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=False,  # Мы уже используем свой VAD
                without_timestamps=True,
            )

            # Собираем текст из сегментов
            texts = [segment.text.strip() for segment in segments]
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

    def _reset_buffer(self) -> None:
        """Сбросить буфер аудио и состояние VAD."""
        self._audio_buffer.clear()
        self._speech_started = False
        self._silence_samples = 0
        # Сбрасываем state VAD ONNX для нового сегмента
        if self._vad_state is not None:
            self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)

    def get_final_result(self) -> Optional[str]:
        """
        Получить финальный результат (вызывать после остановки записи).

        Транскрибирует весь оставшийся буфер независимо от пауз.

        Returns:
            Распознанный текст или None
        """
        if not self._is_loaded:
            return None

        with self._lock:
            if not self._audio_buffer:
                return None

            result = self._transcribe_buffer()
            self._reset_buffer()

            if result and self.on_final_result:
                self.on_final_result(result)

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
        with self._lock:
            # Отменяем таймер
            with self._unload_lock:
                if self._unload_timer is not None:
                    self._unload_timer.cancel()
                    self._unload_timer = None

            # Очищаем модели
            self._model = None
            self._vad_session = None
            self._vad_state = None

            # Очищаем буфер
            self._audio_buffer.clear()
            self._speech_started = False
            self._silence_samples = 0

            self._is_loaded = False

            import gc
            gc.collect()

            logger.info("Модели Whisper и VAD ONNX выгружены")

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
        self._silence_threshold_samples = int(
            (duration_ms / 1000) * self.sample_rate
        )
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
