import numpy as np
from unittest.mock import MagicMock
from src.core.vad_processor import VadProcessor


def test_detect_speech_true_when_session_none():
    """Без загруженной сессии считаем, что речь есть (как в текущем коде)."""
    vad = VadProcessor(sample_rate=16000, vad_threshold=0.5)
    assert vad.is_loaded is False
    assert vad.detect_speech(np.zeros(512, dtype=np.float32)) is True


def test_detect_speech_uses_threshold():
    vad = VadProcessor(sample_rate=16000, vad_threshold=0.5)
    vad._state = np.zeros((2, 1, 128), dtype=np.float32)
    vad._context = np.zeros(64, dtype=np.float32)
    # Мок ONNX-сессии: возвращает (output, state). output[0][0] — вероятность речи.
    sess = MagicMock()
    sess.run.return_value = (np.array([[0.9]], dtype=np.float32),
                             np.zeros((2, 1, 128), dtype=np.float32))
    vad._session = sess
    assert vad.detect_speech(np.zeros(512, dtype=np.float32)) is True   # 0.9 >= 0.5

    sess.run.return_value = (np.array([[0.1]], dtype=np.float32),
                             np.zeros((2, 1, 128), dtype=np.float32))
    vad._context = np.zeros(64, dtype=np.float32)
    assert vad.detect_speech(np.zeros(512, dtype=np.float32)) is False  # 0.1 < 0.5


def test_reset_zeroes_state_and_counter():
    vad = VadProcessor(sample_rate=16000, vad_threshold=0.5)
    vad._state = np.ones((2, 1, 128), dtype=np.float32)
    vad._context = np.ones(64, dtype=np.float32)
    vad._log_counter = 99
    vad.reset()
    assert np.count_nonzero(vad._state) == 0
    assert np.count_nonzero(vad._context) == 0
    assert vad._log_counter == 0
