"""
Буфер аудио с триггером транскрипции по тишине.

Накапливает чанки речи; после начала речи копит и тишину, и когда тишина
достигает порога — сигналит, что пора транскрибировать. Тишина до начала
речи игнорируется. Не потокобезопасен — вызывается под локом оркестратора.
"""
from typing import List
import numpy as np


class AudioBuffer:
    def __init__(self, sample_rate: int, min_silence_duration_ms: int):
        self._sample_rate = sample_rate
        self._chunks: List[np.ndarray] = []
        self._speech_started = False
        self._silence_samples = 0
        self._silence_threshold_samples = int(
            (min_silence_duration_ms / 1000) * sample_rate
        )

    def add(self, audio_np: np.ndarray, is_speech: bool) -> bool:
        """Добавить чанк. Вернуть True, если достигнут порог тишины (пора транскрибировать)."""
        if is_speech:
            self._chunks.append(audio_np)
            self._speech_started = True
            self._silence_samples = 0
            return False
        if self._speech_started:
            self._chunks.append(audio_np)
            self._silence_samples += len(audio_np)
            return self._silence_samples >= self._silence_threshold_samples
        return False  # тишина до начала речи — игнорируем

    def get_audio(self) -> np.ndarray:
        if not self._chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._chunks)

    @property
    def has_audio(self) -> bool:
        return len(self._chunks) > 0

    def reset(self) -> None:
        self._chunks.clear()
        self._speech_started = False
        self._silence_samples = 0

    def set_silence_threshold(self, min_silence_duration_ms: int) -> None:
        """Обновить порог тишины (не сбрасывая накопленное)."""
        self._silence_threshold_samples = int(
            (min_silence_duration_ms / 1000) * self._sample_rate
        )
