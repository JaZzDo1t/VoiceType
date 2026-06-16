import numpy as np
from src.core.audio_buffer import AudioBuffer


def _chunk(n): return np.ones(n, dtype=np.float32)


def test_speech_accumulates_no_trigger():
    buf = AudioBuffer(sample_rate=16000, min_silence_duration_ms=300)
    assert buf.add(_chunk(4000), is_speech=True) is False
    assert buf.add(_chunk(4000), is_speech=True) is False
    assert buf.has_audio is True
    assert len(buf.get_audio()) == 8000


def test_silence_before_speech_ignored():
    buf = AudioBuffer(sample_rate=16000, min_silence_duration_ms=300)
    assert buf.add(_chunk(4000), is_speech=False) is False
    assert buf.has_audio is False
    assert len(buf.get_audio()) == 0


def test_silence_after_speech_triggers_at_threshold():
    buf = AudioBuffer(sample_rate=16000, min_silence_duration_ms=300)  # порог = 4800 сэмплов
    buf.add(_chunk(4000), is_speech=True)
    assert buf.add(_chunk(4000), is_speech=False) is False   # silence=4000 < 4800
    assert buf.add(_chunk(4000), is_speech=False) is True    # silence=8000 >= 4800


def test_reset_clears_state():
    buf = AudioBuffer(sample_rate=16000, min_silence_duration_ms=300)
    buf.add(_chunk(4000), is_speech=True)
    buf.add(_chunk(4000), is_speech=False)
    buf.reset()
    assert buf.has_audio is False
    assert len(buf.get_audio()) == 0
    # после reset тишина снова игнорируется (speech_started сброшен)
    assert buf.add(_chunk(4000), is_speech=False) is False


def test_set_silence_threshold_changes_trigger():
    buf = AudioBuffer(sample_rate=16000, min_silence_duration_ms=300)  # порог 4800
    buf.set_silence_threshold(100)  # порог 1600
    buf.add(_chunk(4000), is_speech=True)
    assert buf.add(_chunk(2000), is_speech=False) is True  # silence=2000 >= 1600
