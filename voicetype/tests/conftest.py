"""
VoiceType - Pytest Configuration
Общие fixtures для тестов.
"""
import sys
from pathlib import Path
import pytest

# Добавляем корень проекта в путь
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Регистрация custom markers."""
    config.addinivalue_line(
        "markers", "hardware: tests that require real hardware (microphone)"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take more than 5 seconds"
    )
    config.addinivalue_line(
        "markers", "e2e: end-to-end integration tests"
    )


# =============================================================================
# Hardware Detection Helpers
# =============================================================================

def is_microphone_available() -> bool:
    """
    Проверить доступность микрофона в системе.

    Returns:
        True если микрофон доступен
    """
    try:
        from src.core.audio_capture import AudioCapture
        devices = AudioCapture.list_devices()
        # list_devices() всегда добавляет "default" первым
        # Проверяем есть ли реальные устройства помимо "default"
        real_devices = [d for d in devices if d['id'] != 'default']
        return len(real_devices) > 0
    except Exception:
        return False


# =============================================================================
# Session-Scoped Fixtures (shared across all tests)
# =============================================================================

@pytest.fixture(scope="session")
def check_microphone_available():
    """
    Skip all hardware tests if no microphone is available.
    Session-scoped: runs once per test session.
    """
    if not is_microphone_available():
        pytest.skip("No microphone available - skipping hardware tests")


# =============================================================================
# Function-Scoped Fixtures (fresh instance per test)
# =============================================================================

@pytest.fixture
def audio_capture():
    """
    Создать AudioCapture с default device.
    Автоматически останавливает и очищает ресурсы после теста.
    """
    from src.core.audio_capture import AudioCapture
    capture = AudioCapture(device_id="default")
    yield capture
    # Cleanup
    if capture.is_running():
        capture.stop()


@pytest.fixture
def started_audio_capture(check_microphone_available):
    """
    Создать и запустить AudioCapture.
    Уже готов к записи. Требует наличия микрофона.
    """
    from src.core.audio_capture import AudioCapture
    capture = AudioCapture(device_id="default")
    result = capture.start()
    if not result:
        pytest.skip("Failed to start audio capture")
    yield capture
    # Cleanup
    capture.stop()


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def temp_audio_file(tmp_path):
    """
    Создать временный audio файл для тестов.
    """
    import numpy as np
    from src.utils.constants import SAMPLE_RATE

    # Генерируем 1 секунду тишины (16-bit PCM)
    duration = 1.0
    samples = int(SAMPLE_RATE * duration)
    audio_data = np.zeros(samples, dtype=np.int16).tobytes()

    audio_file = tmp_path / "test_audio.raw"
    audio_file.write_bytes(audio_data)

    return audio_file


@pytest.fixture
def project_root():
    """
    Возвращает путь к корню проекта.
    """
    return ROOT_DIR
