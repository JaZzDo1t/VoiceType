"""
VoiceType - System Information
Утилиты для получения информации о системе и устройствах.
"""
from typing import List, Dict, Optional
import psutil
from loguru import logger


def get_microphones() -> List[Dict]:
    """
    Получить список доступных микрофонов.

    Returns:
        Список словарей с информацией о микрофонах:
        [{"id": "0", "name": "Microphone Name", "is_default": True}, ...]
    """
    microphones = []

    try:
        import pyaudio

        p = pyaudio.PyAudio()

        # Получаем индекс устройства по умолчанию
        try:
            default_input = p.get_default_input_device_info()
            default_index = default_input.get("index", -1)
        except IOError:
            default_index = -1

        # Перебираем все устройства
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)

                # Только устройства с входными каналами (микрофоны)
                if info.get("maxInputChannels", 0) > 0:
                    microphones.append({
                        "id": str(i),
                        "name": info.get("name", f"Device {i}"),
                        "is_default": i == default_index,
                        "sample_rate": int(info.get("defaultSampleRate", 16000)),
                        "channels": info.get("maxInputChannels", 1)
                    })
            except Exception as e:
                logger.warning(f"Failed to get info for device {i}: {e}")

        p.terminate()

    except ImportError:
        logger.error("PyAudio not installed")
    except Exception as e:
        logger.error(f"Failed to enumerate microphones: {e}")

    # Добавляем виртуальное устройство "default" в начало
    microphones.insert(0, {
        "id": "default",
        "name": "System Default",
        "is_default": True,
        "sample_rate": 16000,
        "channels": 1
    })

    logger.debug(f"Found {len(microphones)} microphones")
    return microphones


def get_microphone_by_id(device_id: str) -> Optional[Dict]:
    """
    Получить информацию о микрофоне по ID.

    Args:
        device_id: ID устройства или "default"

    Returns:
        Словарь с информацией или None если не найден
    """
    microphones = get_microphones()

    for mic in microphones:
        if mic["id"] == device_id:
            return mic

    return None


def get_cpu_usage() -> float:
    """
    Получить текущую загрузку CPU (%).

    Returns:
        Процент загрузки CPU (0.0 - 100.0)
    """
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception as e:
        logger.error(f"Failed to get CPU usage: {e}")
        return 0.0


def get_memory_usage() -> Dict:
    """
    Получить информацию об использовании памяти.

    Returns:
        {"total_mb": float, "used_mb": float, "percent": float}
    """
    try:
        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total / (1024 * 1024),
            "used_mb": mem.used / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "percent": mem.percent
        }
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return {"total_mb": 0, "used_mb": 0, "available_mb": 0, "percent": 0}


def get_process_memory() -> float:
    """
    Получить использование памяти текущим процессом (МБ).

    Returns:
        Использование памяти в МБ
    """
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception as e:
        logger.error(f"Failed to get process memory: {e}")
        return 0.0


class _ProcessMonitor:
    """Синглтон для мониторинга процесса с сохранением базовой линии CPU."""
    _instance = None

    def __init__(self):
        self._process = psutil.Process()
        # Инициализируем базовую линию - первый вызов всегда 0%
        self._process.cpu_percent(interval=None)
        self._last_cpu = 0.0
        self._cpu_count = psutil.cpu_count(logical=True) or 1

    @classmethod
    def get_instance(cls) -> "_ProcessMonitor":
        if cls._instance is None:
            cls._instance = _ProcessMonitor()
        return cls._instance

    def get_cpu(self) -> float:
        """Получить CPU% нормализованный до 0-100% (делим на количество ядер)."""
        try:
            # interval=None использует время с прошлого вызова
            cpu = self._process.cpu_percent(interval=None)
            # Нормализуем: psutil может вернуть до N*100% где N = кол-во ядер
            cpu_normalized = cpu / self._cpu_count
            self._last_cpu = cpu_normalized
            return cpu_normalized
        except Exception as e:
            logger.error(f"Failed to get process CPU: {e}")
            return self._last_cpu

    def get_memory(self) -> float:
        """Получить использование памяти в МБ."""
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.error(f"Failed to get process memory: {e}")
            return 0.0


def get_process_cpu() -> float:
    """
    Получить использование CPU текущим процессом (%).

    Returns:
        Процент CPU (может быть > 100 на многоядерных системах)
    """
    return _ProcessMonitor.get_instance().get_cpu()


def get_system_info() -> Dict:
    """
    Получить общую информацию о системе.

    Returns:
        Словарь с информацией о системе
    """
    import platform

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
    }
