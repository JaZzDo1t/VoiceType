"""
Silero VAD (ONNX) — определение наличия речи в аудио-чанке.

ONNX-версия Silero VAD V5+ требует context (64 сэмпла от предыдущего чанка
для 16kHz) — без него модель возвращает prob ~0.001 даже при явной речи.
Не потокобезопасен — вызывается под локом оркестратора.
"""
import urllib.request
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger

# URL для скачивания ONNX модели Silero VAD
SILERO_VAD_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


class VadProcessor:
    def __init__(self, sample_rate: int, vad_threshold: float):
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self._session = None
        self._state: Optional[np.ndarray] = None
        self._context: Optional[np.ndarray] = None
        self._log_counter = 0  # V6: инициализация в __init__, без hasattr

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        """Загрузить ONNX-модель Silero VAD (скачать при отсутствии)."""
        import onnxruntime as ort

        vad_cache_dir = Path.home() / ".cache" / "silero-vad"
        vad_cache_dir.mkdir(parents=True, exist_ok=True)
        vad_onnx_path = vad_cache_dir / "silero_vad.onnx"

        if not vad_onnx_path.exists():
            logger.info("Скачивание Silero VAD ONNX модели...")
            try:
                urllib.request.urlretrieve(SILERO_VAD_ONNX_URL, str(vad_onnx_path))
                logger.info(f"VAD модель сохранена: {vad_onnx_path}")
            except Exception as e:
                logger.error(f"Ошибка скачивания VAD модели: {e}")
                raise

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(vad_onnx_path), sess_options=sess_options,
            providers=['CPUExecutionProvider'],
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        context_size = 64 if self.sample_rate == 16000 else 32
        self._context = np.zeros(context_size, dtype=np.float32)
        logger.debug("Silero VAD ONNX загружен (без PyTorch!)")

    def unload(self) -> None:
        """Выгрузить ONNX-сессию для освобождения памяти."""
        self._session = None

    def detect_speech(self, audio_np: np.ndarray) -> bool:
        """True, если в чанке обнаружена речь."""
        if self._session is None:
            return True  # VAD не загружен — считаем что речь есть

        try:
            WINDOW_SIZE = 512 if self.sample_rate == 16000 else 256
            CONTEXT_SIZE = 64 if self.sample_rate == 16000 else 32
            sr_input = np.array(self.sample_rate, dtype=np.int64)
            max_speech_prob = 0.0

            offset = 0
            while offset + WINDOW_SIZE <= len(audio_np):
                window = audio_np[offset:offset + WINDOW_SIZE]
                audio_with_context = np.concatenate([self._context, window])
                audio_input = audio_with_context.reshape(1, -1).astype(np.float32)
                ort_inputs = {'input': audio_input, 'state': self._state, 'sr': sr_input}
                output, state_out = self._session.run(None, ort_inputs)
                self._state = state_out
                speech_prob = output[0][0]
                max_speech_prob = max(max_speech_prob, speech_prob)
                self._context = window[-CONTEXT_SIZE:].copy()
                if speech_prob >= self.vad_threshold:
                    return True
                offset += WINDOW_SIZE

            if len(audio_np) >= CONTEXT_SIZE:
                self._context = audio_np[-CONTEXT_SIZE:].copy()
            else:
                keep_from_old = CONTEXT_SIZE - len(audio_np)
                self._context = np.concatenate([self._context[-keep_from_old:], audio_np])

            self._log_counter += 1
            rms = np.sqrt(np.mean(audio_np ** 2))
            max_val = np.max(np.abs(audio_np))
            logger.debug(f"VAD: prob={max_speech_prob:.3f}, thr={self.vad_threshold}, "
                         f"rms={rms:.4f}, max={max_val:.4f}")
            return bool(max_speech_prob >= self.vad_threshold)

        except Exception as e:
            logger.warning(f"Ошибка VAD: {e}")
            return True  # при ошибке считаем что есть речь

    def reset(self) -> None:
        """Сбросить state и context для нового сегмента речи."""
        if self._state is not None:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        if self._context is not None:
            context_size = 64 if self.sample_rate == 16000 else 32
            self._context = np.zeros(context_size, dtype=np.float32)
        self._log_counter = 0
