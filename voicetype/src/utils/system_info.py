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


class _VRAMMonitor:
    """Синглтон для мониторинга VRAM с отслеживанием baseline."""
    _instance = None

    def __init__(self):
        self._baseline_mb = None
        self._total_mb = 0
        self._available = False
        self._init_baseline()

    def _init_baseline(self):
        """Инициализировать baseline при первом запуске."""
        # Отложенная инициализация - не делаем ничего сразу
        # Baseline будет установлен при первом вызове get_app_vram()
        pass

    def _do_init_baseline(self):
        """Фактическая инициализация baseline."""
        if self._baseline_mb is not None:
            return  # Уже инициализировано
        info = self._query_nvidia_smi()
        if info["available"]:
            self._baseline_mb = info["used_mb"]
            self._total_mb = info["total_mb"]
            self._available = True
            logger.debug(f"VRAM baseline set: {self._baseline_mb:.0f} MB")
        else:
            self._available = False

    def _query_nvidia_smi(self) -> Dict:
        """Запросить данные от nvidia-smi."""
        try:
            import subprocess
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) == 3:
                    return {
                        "total_mb": float(parts[0]),
                        "used_mb": float(parts[1]),
                        "free_mb": float(parts[2]),
                        "available": True
                    }
        except FileNotFoundError:
            logger.debug("nvidia-smi not found - no NVIDIA GPU")
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timeout")
        except Exception as e:
            logger.error(f"Failed to get VRAM usage: {e}")
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "available": False}

    @classmethod
    def get_instance(cls) -> "_VRAMMonitor":
        if cls._instance is None:
            cls._instance = _VRAMMonitor()
        return cls._instance

    def get_app_vram(self, lazy: bool = False) -> Dict:
        """Получить использование VRAM приложением (разница от baseline).

        Args:
            lazy: Если True, не инициализировать baseline сразу, вернуть "неизвестно"
        """
        # Ленивая инициализация baseline
        if self._baseline_mb is None:
            if lazy:
                # При ленивом режиме возвращаем "недоступно" без инициализации
                return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "available": False}
            self._do_init_baseline()

        if not self._available or self._baseline_mb is None:
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "available": False}

        info = self._query_nvidia_smi()
        if not info["available"]:
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "available": False}

        # Использование приложением = текущее - baseline
        app_used = max(0, info["used_mb"] - self._baseline_mb)
        return {
            "total_mb": self._total_mb,
            "used_mb": app_used,
            "free_mb": info["free_mb"],
            "available": True
        }

    def reset_baseline(self):
        """Сбросить baseline (вызывать после выгрузки моделей)."""
        self._baseline_mb = None
        self._do_init_baseline()


def get_vram_usage(lazy: bool = False) -> Dict:
    """
    Получить информацию об использовании видеопамяти приложением.

    Args:
        lazy: Если True, не инициализировать baseline сразу

    Returns:
        {"total_mb": float, "used_mb": float, "free_mb": float, "available": bool}
        used_mb - только использование этим приложением (относительно baseline при старте)
    """
    return _VRAMMonitor.get_instance().get_app_vram(lazy=lazy)


def reset_vram_baseline():
    """Сбросить baseline VRAM (вызывать после выгрузки моделей)."""
    _VRAMMonitor.get_instance().reset_baseline()


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
