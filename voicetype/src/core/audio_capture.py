"""
VoiceType - Audio Capture
Захват аудио с микрофона в реальном времени.
"""
import threading
import queue
import time
import numpy as np
from typing import Optional, Callable, List, Dict, Tuple
from loguru import logger

from src.utils.constants import SAMPLE_RATE, CHANNELS, CHUNK_SIZE

# Альтернативные sample rates для fallback
FALLBACK_SAMPLE_RATES = [16000, 44100, 48000, 22050, 8000]


class AudioDeviceError(Exception):
    """Ошибка аудио устройства."""
    pass


class AudioCapture:
    """
    Захватывает аудио с выбранного микрофона.
    Работает в отдельном потоке.
    Отправляет данные в очередь для распознавания.
    """

    def __init__(
        self,
        device_id: str = "default",
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SIZE
    ):
        """
        Args:
            device_id: ID микрофона или "default"
            sample_rate: Частота дискретизации (по умолчанию 16000)
            chunk_size: Размер буфера
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self._audio_queue: queue.Queue = queue.Queue()
        self._stream = None
        self._pyaudio = None
        self._audio = None  # Alias для совместимости
        self._capture_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._current_level: float = 0.0
        self._actual_sample_rate: int = sample_rate  # Реальный используемый sample rate
        self._actual_device_id: Optional[int] = None  # Реальный используемый device index
        self._noise_floor: int = 800  # Порог шума для индикатора уровня

        # Callbacks
        self.on_audio_data: Optional[Callable[[bytes], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    def _get_device_index(self) -> Optional[int]:
        """Получить индекс устройства по ID."""
        if self.device_id == "default":
            return None  # PyAudio использует устройство по умолчанию

        try:
            return int(self.device_id)
        except ValueError:
            logger.warning(f"Invalid device ID: {self.device_id}, using default")
            return None

    def _init_pyaudio(self) -> bool:
        """
        Инициализация PyAudio с обработкой ошибок.

        Returns:
            True если PyAudio успешно инициализирован
        """
        if self._pyaudio is not None:
            return True

        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
            self._audio = self._pyaudio  # Alias для совместимости
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            return False

    def validate_device(self, device_id: Optional[int] = None) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Валидирует аудио устройство перед использованием.

        Проверяет:
        - Существует ли устройство
        - Доступно ли оно для записи (input)
        - Поддерживает ли нужный sample rate

        Args:
            device_id: Индекс устройства для проверки.
                      None = устройство по умолчанию

        Returns:
            Tuple[bool, Optional[int], Optional[str]]:
                - success: True если устройство валидно
                - device_index: Индекс валидного устройства (или None для default)
                - error_message: Сообщение об ошибке (или None при успехе)
        """
        try:
            import pyaudio

            # Инициализируем PyAudio если еще не сделали
            if not self._init_pyaudio():
                return False, None, "Failed to initialize PyAudio"

            # Получаем информацию об устройствах
            device_count = self._pyaudio.get_device_count()

            if device_count == 0:
                return False, None, "No audio devices found"

            # Если device_id не указан, используем default
            if device_id is None:
                try:
                    default_info = self._pyaudio.get_default_input_device_info()
                    default_index = default_info.get('index')
                    logger.debug(f"Using default input device: {default_info.get('name')} (index: {default_index})")
                    return True, None, None  # None означает использовать default
                except IOError as e:
                    return False, None, f"No default input device available: {e}"

            # Проверяем что device_id в допустимом диапазоне
            if device_id < 0 or device_id >= device_count:
                logger.warning(f"Device index {device_id} out of range (0-{device_count-1})")
                return False, None, f"Device index {device_id} out of range"

            # Получаем информацию об устройстве
            try:
                device_info = self._pyaudio.get_device_info_by_index(device_id)
            except IOError as e:
                logger.warning(f"Cannot get info for device {device_id}: {e}")
                return False, None, f"Cannot access device {device_id}: {e}"

            # Проверяем что это input устройство
            max_input_channels = device_info.get('maxInputChannels', 0)
            if max_input_channels < 1:
                logger.warning(f"Device {device_id} ({device_info.get('name')}) is not an input device")
                return False, None, f"Device is not an input device (no input channels)"

            # Проверяем поддержку sample rate
            device_name = device_info.get('name', 'Unknown')
            if not self._check_sample_rate_support(device_id, self.sample_rate):
                logger.warning(f"Device {device_id} ({device_name}) may not support sample rate {self.sample_rate}")
                # Это warning, не critical error - PyAudio может сделать resampling

            logger.debug(f"Device {device_id} ({device_name}) validated successfully")
            return True, device_id, None

        except Exception as e:
            logger.error(f"Error validating device {device_id}: {e}")
            return False, None, str(e)

    def _check_sample_rate_support(self, device_index: Optional[int], sample_rate: int) -> bool:
        """
        Проверяет поддержку sample rate устройством.

        Args:
            device_index: Индекс устройства (None для default)
            sample_rate: Частота дискретизации для проверки

        Returns:
            True если sample rate поддерживается
        """
        try:
            import pyaudio

            if self._pyaudio is None:
                return False

            # Проверяем поддержку формата
            is_supported = self._pyaudio.is_format_supported(
                sample_rate,
                input_device=device_index,
                input_channels=CHANNELS,
                input_format=pyaudio.paInt16
            )
            return is_supported
        except ValueError:
            # ValueError означает что формат не поддерживается
            return False
        except Exception as e:
            logger.debug(f"Error checking sample rate support: {e}")
            return False

    def _find_working_sample_rate(self, device_index: Optional[int]) -> Optional[int]:
        """
        Находит рабочий sample rate для устройства.

        Args:
            device_index: Индекс устройства

        Returns:
            Рабочий sample rate или None если ни один не работает
        """
        # Сначала пробуем запрошенный
        if self._check_sample_rate_support(device_index, self.sample_rate):
            return self.sample_rate

        # Пробуем fallback варианты
        for rate in FALLBACK_SAMPLE_RATES:
            if rate != self.sample_rate and self._check_sample_rate_support(device_index, rate):
                logger.info(f"Using fallback sample rate: {rate}")
                return rate

        # Ничего не подошло - возвращаем запрошенный, PyAudio может сделать resampling
        logger.warning(f"No confirmed sample rate support, trying {self.sample_rate}")
        return self.sample_rate

    def _try_open_stream(
        self,
        device_index: Optional[int],
        sample_rate: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Пытается открыть аудио stream с обработкой ошибок.

        Args:
            device_index: Индекс устройства (None для default)
            sample_rate: Частота дискретизации

        Returns:
            Tuple[success, error_message]
        """
        try:
            import pyaudio

            if self._pyaudio is None:
                return False, "PyAudio not initialized"

            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            self._actual_sample_rate = sample_rate
            self._actual_device_id = device_index
            return True, None

        except OSError as e:
            # OSError часто возникает при проблемах с устройством
            error_msg = f"OS error opening stream: {e}"
            logger.error(error_msg)
            return False, error_msg

        except IOError as e:
            # IOError при проблемах ввода/вывода
            error_msg = f"IO error opening stream: {e}"
            logger.error(error_msg)
            return False, error_msg

        except Exception as e:
            # Любые другие ошибки
            error_msg = f"Error opening stream: {e}"
            logger.error(error_msg)
            return False, error_msg

    def start(self) -> bool:
        """
        Начать захват аудио.

        Выполняет валидацию устройства перед открытием stream.
        При ошибке пытается использовать fallback на default устройство.
        При ошибке sample rate пробует альтернативные частоты.

        Returns:
            True если успешно запущен
        """
        if self._is_running:
            logger.warning("Audio capture already running")
            return False

        try:
            # Шаг 1: Инициализация PyAudio
            if not self._init_pyaudio():
                error = AudioDeviceError("Failed to initialize PyAudio")
                self._notify_error(error)
                return False

            # Шаг 2: Получаем и валидируем device index
            device_index = self._get_device_index()

            # Шаг 3: Валидация устройства
            is_valid, validated_device, error_msg = self.validate_device(device_index)

            if not is_valid:
                logger.warning(f"Device validation failed: {error_msg}")

                # Fallback на default устройство
                if device_index is not None:
                    logger.info("Falling back to default audio device")
                    is_valid, validated_device, error_msg = self.validate_device(None)

                    if not is_valid:
                        error = AudioDeviceError(f"No valid audio device available: {error_msg}")
                        self._notify_error(error)
                        self._cleanup()
                        return False
                else:
                    error = AudioDeviceError(f"Default device not available: {error_msg}")
                    self._notify_error(error)
                    self._cleanup()
                    return False

            # Шаг 4: Находим рабочий sample rate
            working_sample_rate = self._find_working_sample_rate(validated_device)

            # Шаг 5: Пытаемся открыть stream
            success, stream_error = self._try_open_stream(validated_device, working_sample_rate)

            if not success:
                # Пробуем с default устройством если еще не пробовали
                if validated_device is not None:
                    logger.warning(f"Failed to open stream on device {validated_device}, trying default")
                    working_sample_rate = self._find_working_sample_rate(None)
                    success, stream_error = self._try_open_stream(None, working_sample_rate)

                if not success:
                    # Последняя попытка - пробуем разные sample rates с default
                    for rate in FALLBACK_SAMPLE_RATES:
                        logger.debug(f"Trying sample rate {rate} with default device")
                        success, stream_error = self._try_open_stream(None, rate)
                        if success:
                            break

            if not success:
                error = AudioDeviceError(f"Failed to open audio stream: {stream_error}")
                self._notify_error(error)
                self._cleanup()
                return False

            # Шаг 6: Запускаем stream
            try:
                self._is_running = True
                self._stream.start_stream()
            except Exception as e:
                logger.error(f"Failed to start stream: {e}")
                error = AudioDeviceError(f"Failed to start audio stream: {e}")
                self._notify_error(error)
                self._is_running = False
                self._cleanup()
                return False

            # Логируем успешный запуск
            actual_device = self._actual_device_id if self._actual_device_id is not None else "default"
            if self._actual_sample_rate != self.sample_rate:
                logger.info(
                    f"Audio capture started (device: {actual_device}, "
                    f"sample_rate: {self._actual_sample_rate} [requested: {self.sample_rate}])"
                )
            else:
                logger.info(f"Audio capture started (device: {actual_device}, sample_rate: {self._actual_sample_rate})")

            return True

        except Exception as e:
            # Catch-all для непредвиденных ошибок
            logger.error(f"Unexpected error starting audio capture: {e}")
            self._notify_error(e)
            self._cleanup()
            return False

    def _notify_error(self, error: Exception) -> None:
        """
        Уведомляет об ошибке через callback.

        Args:
            error: Исключение для передачи в callback
        """
        logger.error(f"Audio capture error: {error}")
        if self.on_error:
            try:
                self.on_error(error)
            except Exception as callback_error:
                logger.error(f"Error in on_error callback: {callback_error}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback для PyAudio stream."""
        import pyaudio

        if status:
            logger.warning(f"Audio stream status: {status}")

        if self._is_running:
            # Помещаем данные в очередь
            self._audio_queue.put(in_data)

            # Вычисляем уровень сигнала
            self._update_level(in_data)

            # Вызываем callback если установлен
            if self.on_audio_data:
                self.on_audio_data(in_data)

        return (None, pyaudio.paContinue)

    def _update_level(self, audio_data: bytes) -> None:
        """Обновить текущий уровень сигнала."""
        try:
            # Конвертируем bytes в numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Вычисляем RMS (root mean square) как меру громкости
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

            # Порог шума - значения ниже этого считаются тишиной
            if rms < self._noise_floor:
                self._current_level = 0.0
            else:
                # Линейная шкала от noise_floor до ~10000 (громкая речь)
                # Выше 10000 = 100%
                effective_rms = rms - self._noise_floor
                MAX_SPEECH_RMS = 10000.0 - self._noise_floor
                level = effective_rms / MAX_SPEECH_RMS
                self._current_level = max(0.0, min(1.0, level))

        except Exception as e:
            logger.debug(f"Error calculating audio level: {e}")

    def set_noise_floor(self, noise_floor: int) -> None:
        """Установить порог шума для индикатора уровня."""
        self._noise_floor = noise_floor
        logger.debug(f"Noise floor set to: {noise_floor}")

    def stop(self) -> None:
        """Остановить захват."""
        if not self._is_running:
            return

        self._is_running = False
        self._cleanup()
        logger.info("Audio capture stopped")

    def _cleanup(self) -> None:
        """Очистить ресурсы PyAudio и stream."""
        if self._stream:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")
            self._stream = None

        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception as e:
                logger.debug(f"Error terminating PyAudio: {e}")
            self._pyaudio = None
            self._audio = None  # Очищаем alias

        # Очищаем очередь
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Сбрасываем уровень сигнала
        self._current_level = 0.0

    def get_audio_queue(self) -> queue.Queue:
        """Получить очередь с аудио-данными."""
        return self._audio_queue

    def get_level(self) -> float:
        """
        Получить текущий уровень сигнала.

        Returns:
            Уровень от 0.0 до 1.0
        """
        return self._current_level

    def is_running(self) -> bool:
        """Проверить, запущен ли захват."""
        return self._is_running

    def get_actual_sample_rate(self) -> int:
        """
        Получить реальный используемый sample rate.

        Может отличаться от запрошенного если был применен fallback.

        Returns:
            Реальная частота дискретизации
        """
        return self._actual_sample_rate

    def get_actual_device_id(self) -> Optional[int]:
        """
        Получить реальный используемый device index.

        Может отличаться от запрошенного если был применен fallback.

        Returns:
            Индекс устройства или None для default
        """
        return self._actual_device_id

    @staticmethod
    def list_devices() -> List[Dict]:
        """
        Получить список доступных микрофонов.

        Returns:
            Список словарей с информацией о устройствах
        """
        from src.utils.system_info import get_microphones
        return get_microphones()

    def __enter__(self):
        """Контекстный менеджер - вход."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер - выход."""
        self.stop()
        return False
