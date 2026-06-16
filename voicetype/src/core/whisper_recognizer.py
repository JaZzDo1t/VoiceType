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
from src.core.audio_buffer import AudioBuffer

# URL для скачивания ONNX модели Silero VAD
SILERO_VAD_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


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
        self._vad_session = None  # ONNX InferenceSession для VAD
        self._vad_state = None  # Combined state для VAD ONNX [2, 1, 128]
        self._vad_context = None  # Context для Silero VAD V5+ (64 сэмпла для 16kHz)
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
        """
        Добавить пути к CUDA DLL для Windows.

        Нужно вызывать перед каждой загрузкой модели, т.к. после unload()
        DLL могут стать недоступны.

        Использует два метода для совместимости:
        1. os.add_dll_directory() - официальный способ для Python 3.8+
        2. PATH environment variable - для библиотек использующих LoadLibrary напрямую
        """
        if os.name != 'nt':  # Только для Windows
            return

        try:
            import sys

            # sys.prefix указывает на venv когда он активирован
            # Это надёжнее чем искать в sys.path
            site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')

            if not os.path.exists(site_packages):
                logger.warning(f"Не найден site-packages: {site_packages}")
                return

            logger.debug(f"Используем site-packages: {site_packages}")

            # Пути к nvidia DLL (проверяем разные структуры пакетов)
            nvidia_paths = []
            nvidia_base = os.path.join(site_packages, 'nvidia')
            if os.path.exists(nvidia_base):
                for pkg in ['cublas', 'cudnn', 'cufft', 'cuda_runtime', 'nvjitlink']:
                    # Проверяем оба варианта: bin и lib/x64
                    for subdir in ['bin', os.path.join('lib', 'x64'), 'lib']:
                        path = os.path.join(nvidia_base, pkg, subdir)
                        if os.path.exists(path):
                            nvidia_paths.append(path)

            added_count = 0
            current_path = os.environ.get('PATH', '')
            path_additions = []

            for path in nvidia_paths:
                try:
                    # Метод 1: os.add_dll_directory (Python 3.8+)
                    os.add_dll_directory(path)
                    added_count += 1
                    logger.debug(f"add_dll_directory: {path}")
                except Exception as e:
                    logger.debug(f"add_dll_directory failed для {path}: {e}")

                # Метод 2: добавляем в PATH для LoadLibrary
                if path not in current_path:
                    path_additions.append(path)

            # Обновляем PATH
            if path_additions:
                new_path = ';'.join(path_additions) + ';' + current_path
                os.environ['PATH'] = new_path
                logger.debug(f"Добавлено в PATH: {len(path_additions)} путей")

            if added_count > 0:
                logger.info(f"Добавлено {added_count} путей к CUDA DLL")
            else:
                logger.warning("Не найдены пути к CUDA DLL в nvidia packages")

        except Exception as e:
            logger.warning(f"Ошибка добавления CUDA DLL путей: {e}")

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
            self._load_silero_vad()

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

        # Инициализируем context для Silero VAD V5+
        # 64 сэмпла для 16kHz, 32 сэмпла для 8kHz
        context_size = 64 if self.sample_rate == 16000 else 32
        self._vad_context = np.zeros(context_size, dtype=np.float32)

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
                is_speech = self._detect_speech(audio_np)

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

    def _detect_speech(self, audio_np: np.ndarray) -> bool:
        """
        Определить наличие речи в аудио через Silero VAD (ONNX).

        Silero VAD V5+ требует context - 64 сэмпла от предыдущего чанка для 16kHz.
        Без context модель возвращает prob ~0.001 даже при явной речи.

        Args:
            audio_np: Аудио в формате numpy array (float32, [-1, 1])

        Returns:
            True если обнаружена речь, False если тишина
        """
        if self._vad_session is None:
            return True  # Если VAD не загружен - считаем что есть речь

        try:
            # Silero VAD ONNX требует 512 семплов для 16kHz (256 для 8kHz)
            # Плюс 64 сэмпла context от предыдущего чанка
            WINDOW_SIZE = 512 if self.sample_rate == 16000 else 256
            CONTEXT_SIZE = 64 if self.sample_rate == 16000 else 32
            sr_input = np.array(self.sample_rate, dtype=np.int64)

            max_speech_prob = 0.0
            chunk_count = 0

            # Silero VAD V5+ требует context (64 сэмпла) + audio (512 сэмплов) = 576 на входе
            # Обрабатываем аудио окнами, каждый раз добавляя context в начало
            FULL_INPUT_SIZE = CONTEXT_SIZE + WINDOW_SIZE  # 64 + 512 = 576

            # Обрабатываем все полные окна
            offset = 0
            while offset + WINDOW_SIZE <= len(audio_np):
                chunk_count += 1

                # Берём context + следующие 512 сэмплов
                window = audio_np[offset:offset + WINDOW_SIZE]
                audio_with_context = np.concatenate([self._vad_context, window])
                audio_input = audio_with_context.reshape(1, -1).astype(np.float32)

                ort_inputs = {
                    'input': audio_input,
                    'state': self._vad_state,
                    'sr': sr_input,
                }

                output, state_out = self._vad_session.run(None, ort_inputs)
                self._vad_state = state_out

                speech_prob = output[0][0]
                max_speech_prob = max(max_speech_prob, speech_prob)

                # Обновляем context - последние 64 сэмпла текущего окна
                self._vad_context = window[-CONTEXT_SIZE:].copy()

                # Ранний выход если точно речь
                if speech_prob >= self.vad_threshold:
                    return True

                offset += WINDOW_SIZE

            # Сохраняем context для следующего вызова (последние CONTEXT_SIZE сэмплов)
            if len(audio_np) >= CONTEXT_SIZE:
                self._vad_context = audio_np[-CONTEXT_SIZE:].copy()
            else:
                # Если audio_np короче CONTEXT_SIZE, берём часть из старого context
                keep_from_old = CONTEXT_SIZE - len(audio_np)
                self._vad_context = np.concatenate([
                    self._vad_context[-keep_from_old:],
                    audio_np
                ])

            # Логируем каждый чанк для отладки
            if not hasattr(self, '_vad_log_counter'):
                self._vad_log_counter = 0
            self._vad_log_counter += 1

            # Вычисляем RMS для диагностики
            rms = np.sqrt(np.mean(audio_np ** 2))
            max_val = np.max(np.abs(audio_np))

            # Логируем каждый чанк пока отлаживаем
            logger.debug(f"VAD: prob={max_speech_prob:.3f}, thr={self.vad_threshold}, rms={rms:.4f}, max={max_val:.4f}")

            return max_speech_prob >= self.vad_threshold

        except Exception as e:
            logger.warning(f"Ошибка VAD: {e}")
            return True  # При ошибке считаем что есть речь

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
            )
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
        self._buffer.reset()
        self._reset_vad_state()

    def _reset_vad_state(self) -> None:
        """Сбросить внутреннее состояние VAD для новой сессии."""
        if self._vad_state is not None:
            self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        # Сбрасываем context для Silero VAD V5+
        if self._vad_context is not None:
            context_size = 64 if self.sample_rate == 16000 else 32
            self._vad_context = np.zeros(context_size, dtype=np.float32)
        self._vad_log_counter = 0

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

            if self._vad_session is not None:
                del self._vad_session
            self._vad_session = None

            self._vad_state = None
            self._vad_context = None

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
