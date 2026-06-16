"""Тест: при device=cuda и сбое НЕТ молчаливого перехода на CPU."""
from unittest.mock import patch, MagicMock

import numpy as np

from src.core.whisper_recognizer import WhisperRecognizer


def test_cuda_failure_does_not_fallback_to_cpu():
    rec = WhisperRecognizer(model_size="small", device="cuda", language="ru")
    errors = []
    rec.on_error = lambda e: errors.append(e)

    # Патчим WhisperModel в его источнике (faster_whisper), а не в модуле recognizer:
    # load_model делает локальный `from faster_whisper import WhisperModel` при каждом
    # вызове (намеренно — чтобы CUDA DLL-пути добавлялись ДО импорта ctranslate2),
    # поэтому подмена атрибута на модуле-источнике корректно перехватывается импортом.
    with patch("faster_whisper.WhisperModel",
               side_effect=RuntimeError("cudnn missing")) as mock_model:
        ok = rec.load_model()

    assert ok is False                 # загрузка провалилась
    assert rec.device == "cuda"        # device НЕ сменился на cpu
    assert mock_model.call_count == 1  # CPU-попытки не было (был бы 2-й вызов)
    assert len(errors) == 1            # on_error вызван
    assert rec._last_load_error        # ошибка сохранена для UI


def _pcm(n_samples: int) -> bytes:
    """n_samples int16-нулей как PCM-байты (длина важна, содержимое — нет: VAD замокан)."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _ready_recognizer():
    """Recognizer с обойдённой загрузкой: модель замокана, VAD загруженным считается."""
    rec = WhisperRecognizer(model_size="small", device="cuda", language="ru",
                            min_silence_duration_ms=300, sample_rate=16000)
    rec._is_loaded = True
    rec._model = MagicMock()
    # _model.transcribe возвращает один сегмент с текстом "привет"
    seg = MagicMock()
    seg.text = "привет"
    rec._model.transcribe.return_value = ([seg], MagicMock())
    rec._vad_session = MagicMock()  # чтобы _detect_speech не считал VAD выгруженным
    return rec


def test_process_audio_accumulates_speech_without_transcribing():
    rec = _ready_recognizer()
    finals = []
    rec.on_final_result = lambda t: finals.append(t)
    # Всё — речь: 5 чанков по 4000 сэмплов
    with patch.object(rec, "_detect_speech", return_value=True):
        for _ in range(5):
            rec.process_audio(_pcm(4000))
    assert rec._model.transcribe.call_count == 0   # транскрипции ещё не было
    assert finals == []
    assert len(rec._audio_buffer) == 5             # всё накоплено


def test_process_audio_transcribes_after_silence():
    rec = _ready_recognizer()
    finals = []
    rec.on_final_result = lambda t: finals.append(t)
    # Речь (1 чанк), затем тишина пока не наберётся порог (300мс*16000/1000 = 4800 сэмплов)
    with patch.object(rec, "_detect_speech", side_effect=[True, False, False]):
        rec.process_audio(_pcm(4000))   # речь
        rec.process_audio(_pcm(4000))   # тишина: 4000 < 4800
        rec.process_audio(_pcm(4000))   # тишина: 8000 >= 4800 -> транскрипция
    assert rec._model.transcribe.call_count == 1
    assert finals == ["привет"]
    assert len(rec._audio_buffer) == 0             # буфер сброшен после транскрипции


def test_process_audio_ignores_silence_before_speech():
    rec = _ready_recognizer()
    with patch.object(rec, "_detect_speech", return_value=False):
        for _ in range(3):
            rec.process_audio(_pcm(4000))
    assert rec._model.transcribe.call_count == 0
    assert len(rec._audio_buffer) == 0             # тишина до речи не копится


def test_get_final_result_transcribes_remaining_buffer():
    rec = _ready_recognizer()
    with patch.object(rec, "_detect_speech", return_value=True):
        rec.process_audio(_pcm(16000))            # 1с речи накоплено
    result = rec.get_final_result()
    assert result == "привет"
    assert rec._model.transcribe.call_count == 1
    assert len(rec._audio_buffer) == 0
