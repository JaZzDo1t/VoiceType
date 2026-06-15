"""Тест: при device=cuda и сбое НЕТ молчаливого перехода на CPU."""
from unittest.mock import patch

from src.core.whisper_recognizer import WhisperRecognizer


def test_cuda_failure_does_not_fallback_to_cpu():
    rec = WhisperRecognizer(model_size="small", device="cuda", language="ru")
    errors = []
    rec.on_error = lambda e: errors.append(e)

    with patch("faster_whisper.WhisperModel",
               side_effect=RuntimeError("cudnn missing")) as mock_model:
        ok = rec.load_model()

    assert ok is False                 # загрузка провалилась
    assert rec.device == "cuda"        # device НЕ сменился на cpu
    assert mock_model.call_count == 1  # CPU-попытки не было (был бы 2-й вызов)
    assert len(errors) == 1            # on_error вызван
    assert rec._last_load_error        # ошибка сохранена для UI
