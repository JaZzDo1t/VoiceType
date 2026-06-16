# Whisper Recognizer Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Разбить `whisper_recognizer.py` (784 строки) на `VadProcessor` + `AudioBuffer` + тонкое ядро-оркестратор, не меняя поведение.

**Architecture:** Выделить две обособленные ответственности (Silero VAD и буфер с триггером по тишине) в самостоятельные Qt-независимые классы в `src/core/`. `WhisperRecognizer` держит их экземпляры и оркеструет. Безопасность — характеризующие тесты на текущее поведение ДО рефактора, затем рефактор под зелёные.

**Tech Stack:** Python 3.11, numpy, onnxruntime (Silero VAD), faster-whisper, pytest.

---

## File Structure

| Файл | Ответственность |
|------|------------------|
| `voicetype/src/core/vad_processor.py` **(новый)** | Silero VAD: загрузка ONNX, `detect_speech`, state/context, reset. |
| `voicetype/src/core/audio_buffer.py` **(новый)** | Накопление чанков, подсчёт тишины, триггер транскрипции. |
| `voicetype/src/core/whisper_recognizer.py` *(правка)* | Ядро-оркестратор: Whisper load/unload/`_transcribe`, `process_audio`, auto-unload, CUDA DLL. |
| `voicetype/tests/test_whisper_recognizer.py` *(дополнить)* | Характеризующие тесты `process_audio`/`get_final_result`. |
| `voicetype/tests/test_audio_buffer.py` **(новый)** | Юнит-тесты `AudioBuffer`. |
| `voicetype/tests/test_vad_processor.py` **(новый)** | Юнит-тесты `VadProcessor`. |

**Все команды pytest — из каталога `voicetype/`.** Тестовый python: `voicetype\venv\Scripts\python`.

---

## Task 1: Характеризующие тесты на текущее поведение

Фиксируем baseline ДО любого рефактора. Тесты идут через публичный `process_audio`/`get_final_result`, мокая VAD (`_detect_speech`) и транскрипцию (`_model.transcribe`).

**Files:**
- Modify: `voicetype/tests/test_whisper_recognizer.py`

- [ ] **Step 1: Написать характеризующие тесты**

Добавить в конец `voicetype/tests/test_whisper_recognizer.py`:

```python
import numpy as np
from unittest.mock import patch, MagicMock


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
```

- [ ] **Step 2: Прогнать — убедиться, что проходят на ТЕКУЩЕМ коде**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_whisper_recognizer.py -v` (из `voicetype/`)
Expected: PASS — старый тест + 4 новых характеризующих (baseline зафиксирован). Если какой-то падает — значит мок не совпал с поведением; разобраться ДО рефактора.

- [ ] **Step 3: Commit**

```bash
git add voicetype/tests/test_whisper_recognizer.py
git commit -m "test: characterization tests for process_audio/get_final_result before split"
```

---

## Task 2: Выделить AudioBuffer

**Files:**
- Create: `voicetype/src/core/audio_buffer.py`
- Create: `voicetype/tests/test_audio_buffer.py`
- Modify: `voicetype/src/core/whisper_recognizer.py`

- [ ] **Step 1: Написать юнит-тесты AudioBuffer (упадут — модуля нет)**

`voicetype/tests/test_audio_buffer.py`:

```python
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
```

- [ ] **Step 2: Прогнать — убедиться, что падает**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_audio_buffer.py -v` (из `voicetype/`)
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.audio_buffer'`

- [ ] **Step 3: Реализовать AudioBuffer**

`voicetype/src/core/audio_buffer.py`:

```python
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
```

- [ ] **Step 4: Прогнать юнит-тесты AudioBuffer**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_audio_buffer.py -v` (из `voicetype/`)
Expected: PASS — 4 passed

- [ ] **Step 5: Интегрировать AudioBuffer в WhisperRecognizer**

В `voicetype/src/core/whisper_recognizer.py`:

(a) Импорт вверху (после `from src.utils.constants import SAMPLE_RATE`):
```python
from src.core.audio_buffer import AudioBuffer
```

(b) В `__init__` заменить три поля буфера:
```python
        # Буфер аудио для накопления до паузы
        self._audio_buffer: List[np.ndarray] = []
        self._speech_started = False
        self._silence_samples = 0
        self._silence_threshold_samples = int(
            (min_silence_duration_ms / 1000) * sample_rate
        )
```
на:
```python
        # Буфер аудио с триггером транскрипции по тишине
        self._buffer = AudioBuffer(sample_rate, min_silence_duration_ms)
```

(c) В `process_audio` заменить блок накопления (текущие строки ~420-445, от `if is_speech:` до `return None` перед `except`) на:
```python
                if is_speech and self.on_partial_result:
                    self.on_partial_result("...")

                if self._buffer.add(audio_np, is_speech):
                    result = self._transcribe(self._buffer.get_audio())
                    self._buffer.reset()
                    self._reset_vad_state()
                    if result and self.on_final_result:
                        self.on_final_result(result)
                    return result

                return None
```

(d) Переименовать `_transcribe_buffer(self)` в `_transcribe(self, audio_concat: np.ndarray)` и убрать внутреннее чтение буфера. Тело становится:
```python
    def _transcribe(self, audio_concat: np.ndarray) -> Optional[str]:
        """Транскрибировать переданный аудио-массив. Вернуть текст или None."""
        if audio_concat is None or len(audio_concat) == 0 or self._model is None:
            return None
        try:
            self._add_cuda_dll_paths()
            min_samples = int(0.5 * self.sample_rate)
            if len(audio_concat) < min_samples:
                logger.debug("Аудио слишком короткое для транскрипции")
                return None
            logger.debug(f"Транскрипция {len(audio_concat) / self.sample_rate:.2f}с аудио...")
            segments, info = self._model.transcribe(
                audio_concat, language=self.language, beam_size=5, best_of=5,
                temperature=0.0, vad_filter=False, without_timestamps=True,
            )
            texts = [segment.text.strip() for segment in segments]
            result = " ".join(texts).strip()
            if result:
                logger.info(f"Транскрипция: {result[:100]}...")
                return result
            return None
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {e}")
            if self.on_error:
                self.on_error(e)
            return None
```

(e) Заменить `_reset_buffer` (использует поля буфера) на делегирование:
```python
    def _reset_buffer(self) -> None:
        """Сбросить буфер аудио и состояние VAD."""
        self._buffer.reset()
        self._reset_vad_state()
```

(f) В `get_final_result` заменить тело внутри `with self._lock:` (использование `self._audio_buffer` и `_transcribe_buffer`):
```python
        with self._lock:
            if not self._buffer.has_audio:
                logger.debug("get_final_result: буфер пуст, нечего транскрибировать")
                return None
            audio = self._buffer.get_audio()
            logger.info(
                f"get_final_result: транскрибируем {len(audio) / self.sample_rate:.2f}с аудио"
            )
            result = self._transcribe(audio)
            self._reset_buffer()
            return result
```
(И аналогично убрать ссылку `len(self._audio_buffer)` в логе на строке ~633 — заменить на `self._buffer.has_audio`.)

- [ ] **Step 6: Обновить характеризующие тесты под новую структуру**

В `voicetype/tests/test_whisper_recognizer.py` характеризующие тесты обращаются к `rec._audio_buffer` и считают, что мок `_detect_speech` существует. Поскольку `_audio_buffer` больше нет, заменить проверки буфера на `rec._buffer`:
- `len(rec._audio_buffer) == 5` → `rec._buffer.has_audio is True` (и при необходимости проверка длины через `len(rec._buffer.get_audio())`)
- `len(rec._audio_buffer) == 0` → `rec._buffer.has_audio is False`

`_detect_speech` пока остаётся (VAD выделим в Task 3), поэтому patch на него работает.

- [ ] **Step 7: Прогнать весь набор**

Run: `voicetype\venv\Scripts\python -m pytest tests/ -v` (из `voicetype/`)
Expected: PASS — характеризующие (4) + AudioBuffer (4) + старые (14) = 22 passed

- [ ] **Step 8: Commit**

```bash
git add voicetype/src/core/audio_buffer.py voicetype/tests/test_audio_buffer.py voicetype/src/core/whisper_recognizer.py voicetype/tests/test_whisper_recognizer.py
git commit -m "refactor: extract AudioBuffer from whisper_recognizer"
```

---

## Task 3: Выделить VadProcessor (+ V6)

**Files:**
- Create: `voicetype/src/core/vad_processor.py`
- Create: `voicetype/tests/test_vad_processor.py`
- Modify: `voicetype/src/core/whisper_recognizer.py`

- [ ] **Step 1: Написать юнит-тесты VadProcessor (упадут)**

`voicetype/tests/test_vad_processor.py`:

```python
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
```

- [ ] **Step 2: Прогнать — убедиться, что падает**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_vad_processor.py -v` (из `voicetype/`)
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.vad_processor'`

- [ ] **Step 3: Реализовать VadProcessor (перенос _load_silero_vad + _detect_speech + reset, V6 в __init__)**

`voicetype/src/core/vad_processor.py`:

```python
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
            return max_speech_prob >= self.vad_threshold

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
```

- [ ] **Step 4: Прогнать юнит-тесты VadProcessor**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_vad_processor.py -v` (из `voicetype/`)
Expected: PASS — 3 passed

- [ ] **Step 5: Интегрировать VadProcessor в WhisperRecognizer**

В `voicetype/src/core/whisper_recognizer.py`:

(a) Импорт вверху:
```python
from src.core.vad_processor import VadProcessor
```
И удалить старую константу `SILERO_VAD_ONNX_URL` из этого файла (она теперь в vad_processor.py).

(b) В `__init__` заменить VAD-поля:
```python
        self._vad_session = None  # ONNX InferenceSession для VAD
        self._vad_state = None  # Combined state для VAD ONNX [2, 1, 128]
        self._vad_context = None  # Context для Silero VAD V5+ (64 сэмпла для 16kHz)
```
на:
```python
        self._vad = VadProcessor(sample_rate, vad_threshold)
```
Также убрать строку `import urllib.request` сверху, если она больше нигде не используется (проверить Grep'ом).

(c) В `load_model` заменить вызов `self._load_silero_vad()` на `self._vad.load()`. Удалить метод `_load_silero_vad` целиком.

(d) В `process_audio` заменить `is_speech = self._detect_speech(audio_np)` на `is_speech = self._vad.detect_speech(audio_np)`. Удалить метод `_detect_speech` целиком.

(e) Заменить `_reset_vad_state` на делегирование (и обновить `_reset_buffer`, который его зовёт):
```python
    def _reset_vad_state(self) -> None:
        """Сбросить внутреннее состояние VAD для новой сессии."""
        self._vad.reset()
```
(`_reset_buffer` из Task 2 уже зовёт `self._reset_vad_state()` — оставить как есть; либо заменить прямой вызов на `self._vad.reset()`.)

(f) В `unload` — там, где сбрасывались `self._vad_session`/`self._vad_state` (Grep `_vad_session`), заменить на работу через `self._vad` (обнулить сессию: `self._vad._session = None` или добавить `VadProcessor.unload()` метод, обнуляющий `_session`). Предпочтительно: добавить в VadProcessor метод `unload(self)` (`self._session = None`) и звать его. Проверить Grep'ом ВСЕ оставшиеся обращения к `_vad_session`/`_vad_state`/`_vad_context` в whisper_recognizer и перевести их на `self._vad`.

- [ ] **Step 6: Обновить характеризующие тесты под VadProcessor**

В `voicetype/tests/test_whisper_recognizer.py`:
- `_ready_recognizer`: заменить `rec._vad_session = MagicMock()` на `rec._vad._session = MagicMock()`.
- patch-точку `patch.object(rec, "_detect_speech", ...)` заменить на `patch.object(rec._vad, "detect_speech", ...)`.
Логика тестов (кейсы) не меняется — фиксируем то же поведение.

- [ ] **Step 7: Прогнать весь набор**

Run: `voicetype\venv\Scripts\python -m pytest tests/ -v` (из `voicetype/`)
Expected: PASS — характеризующие (4) + AudioBuffer (4) + VadProcessor (3) + остальные (14) = 25 passed

- [ ] **Step 8: Commit**

```bash
git add voicetype/src/core/vad_processor.py voicetype/tests/test_vad_processor.py voicetype/src/core/whisper_recognizer.py voicetype/tests/test_whisper_recognizer.py
git commit -m "refactor: extract VadProcessor from whisper_recognizer (+V6 log counter in __init__)"
```

---

## Task 4: Чистка ядра и финальная верификация

**Files:**
- Modify: `voicetype/src/core/whisper_recognizer.py`

- [ ] **Step 1: Проверить отсутствие осиротевших ссылок**

Run (из `voicetype/`):
`voicetype\venv\Scripts\python -c "import ast; ast.parse(open('src/core/whisper_recognizer.py', encoding='utf-8').read()); print('syntax OK')"`
Затем Grep в `whisper_recognizer.py` по `_audio_buffer`, `_vad_session`, `_vad_state`, `_vad_context`, `_detect_speech`, `_load_silero_vad`, `_transcribe_buffer`, `_speech_started`, `_silence_samples` — **не должно остаться ни одного** обращения (всё переведено на `self._buffer` / `self._vad` / `self._transcribe`). Если что-то осталось — перевести.

- [ ] **Step 2: Проверить размер файла**

Файл `whisper_recognizer.py` должен сократиться примерно до ~450-500 строк (было 784). Это индикатор, что VAD и буфер действительно выехали.

- [ ] **Step 3: Полный прогон тестов**

Run: `voicetype\venv\Scripts\python -m pytest tests/ -v` (из `voicetype/`)
Expected: PASS — 25 passed

- [ ] **Step 4: Ручная проверка запуском**

Run (из `voicetype/`): `voicetype\venv\Scripts\python run.py`
Expected: приложение запускается, модель грузится на GPU; надиктовать фразу по хоткею — текст распознаётся и вставляется как прежде. VAD реагирует на речь/паузы корректно. Закрыть после проверки.

- [ ] **Step 5: Commit (если в Step 1 были доработки)**

```bash
git add voicetype/src/core/whisper_recognizer.py
git commit -m "refactor: remove orphaned VAD/buffer references from whisper_recognizer core"
```

---

## Замечания по реализации

- **Кодировка:** Windows; при чтении в проверках синтаксиса использовать `encoding="utf-8"`.
- **Запуск тестов всегда из `voicetype/`** (conftest.py добавляет корень в sys.path). pytest требует `Pygments` в venv (установлен).
- **Поведение должно остаться идентичным** — характеризующие тесты Task 1 не должны меняться по смыслу (только patch-точки и обращения к буферу при интеграции). Если характеризующий тест начинает требовать изменения ОЖИДАЕМОГО результата — значит рефактор изменил поведение, это ошибка.
- **Вне scope:** разбиение app.py (часть 2 Фазы B), V1 (cuda_paths), V11 (diagnostics→utils), K2 (_is_recording→Event).
