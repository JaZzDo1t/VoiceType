# Дизайн: разбиение whisper_recognizer (Фаза B, под-проект 1)

**Дата:** 2026-06-16
**Статус:** утверждён (brainstorming)
**Затрагивает:** `voicetype/src/core/whisper_recognizer.py` (784 строки) → + `vad_processor.py`, `audio_buffer.py` (новые)

## Проблема и цель

`whisper_recognizer.py` — God-object на 784 строки с тремя ответственностями: жизненный цикл модели Whisper, VAD-конвейер (Silero ONNX) и буферизация аудио с триггером по тишине. Цель — выделить две обособленные ответственности в самостоятельные тестируемые единицы. **Рефакторинг чисто структурный: поведение должно остаться идентичным.**

Это первый из двух под-проектов Фазы B аудита (второй — разбиение `app.py`, отдельным заходом).

## Принятые решения

- **Подход A:** выделить `VadProcessor` + `AudioBuffer`; ядро `WhisperRecognizer` остаётся оркестратором (~450 строк).
- **Верификация:** характеризующие тесты ДО рефактора (фиксируют текущее поведение `process_audio`), затем рефактор под зелёные тесты.
- **Попутно:** V6 — `_vad_log_counter` инициализируется в `__init__` нового `VadProcessor`, а не через `hasattr`.
- **Вне scope:** разбиение `app.py`; V1 (единый `cuda_paths`); V11 (`diagnostics`→`utils`); выделение ModelLifecycle. Поведение НЕ меняется.

## Архитектура

Два новых модуля в `src/core/`, не зависящих от Qt. `WhisperRecognizer` держит их экземпляры и оркеструет. Швы проходят по существующим границам: `_detect_speech` (VAD) и поля буфера (`_audio_buffer`/`_speech_started`/`_silence_samples`) — уже почти изолированы, выделение следует реальной структуре.

## Компоненты

### `src/core/vad_processor.py` (новый) — `VadProcessor`

Инкапсулирует Silero VAD (ONNX). Переносится из `whisper_recognizer`: `_load_silero_vad`, `_detect_speech`, поля `_vad_session`/`_vad_state`/`_vad_context`, сброс VAD-состояния, константа `SILERO_VAD_ONNX_URL`.

```python
class VadProcessor:
    def __init__(self, sample_rate: int, vad_threshold: float): ...
    def load(self) -> None:          # бывш. _load_silero_vad (скачивание + ONNX сессия)
    def detect_speech(self, audio_np: np.ndarray) -> bool:  # бывш. _detect_speech
    def reset(self) -> None:         # сброс _state и _context для нового сегмента
    @property
    def is_loaded(self) -> bool: ...
```

Владеет: `_session`, `_state`, `_context`, `_log_counter` (V6 — в `__init__`, не `hasattr`), `sample_rate`, `vad_threshold`. Если `_session is None`, `detect_speech` возвращает `True` (как сейчас — «считаем что речь есть»).

### `src/core/audio_buffer.py` (новый) — `AudioBuffer`

Инкапсулирует накопление аудио и триггер транскрипции по тишине. Переносится: `_audio_buffer`, `_speech_started`, `_silence_samples`, `_silence_threshold_samples`, логика накопления из `process_audio` (строки 420-445) и `_reset_buffer`.

```python
class AudioBuffer:
    def __init__(self, sample_rate: int, min_silence_duration_ms: int): ...
    def add(self, audio_np: np.ndarray, is_speech: bool) -> bool:
        # Инкапсулирует текущую логику:
        #   is_speech            -> append, speech_started=True, silence=0; return False
        #   not is_speech & started -> append, silence += len; return (silence >= threshold)
        #   not is_speech & not started -> игнор (тишина до начала речи); return False
    def get_audio(self) -> np.ndarray:  # np.concatenate(chunks)
    def reset(self) -> None:            # бывш. _reset_buffer (chunks/started/silence)
    @property
    def has_audio(self) -> bool: ...
```

### `WhisperRecognizer` после (~450 строк)

Держит `self._vad: VadProcessor` и `self._buffer: AudioBuffer`. `load_model` создаёт и грузит `VadProcessor` (вместо `_load_silero_vad`). Остаётся: Whisper-модель load/unload, `_transcribe` (адаптируется — принимает `audio_np` вместо чтения `self._audio_buffer`), auto-unload таймер, `_add_cuda_dll_paths`, `set_processing`/`is_processing`, callbacks.

## Поток данных (process_audio после рефактора)

```python
with self._lock:
    audio_np = (np.frombuffer(audio_data, np.int16).astype(np.float32)) / 32768.0
    is_speech = self._vad.detect_speech(audio_np)

    if is_speech and self.on_partial_result:
        self.on_partial_result("...")

    if self._buffer.add(audio_np, is_speech):   # True = достигнут порог тишины
        result = self._transcribe(self._buffer.get_audio())
        self._buffer.reset()
        self._vad.reset()                        # сброс VAD для нового сегмента
        if result and self.on_final_result:
            self.on_final_result(result)
        return result
    return None
```

Поведение тождественно текущему: партиал шлётся при речи, транскрипция — при достижении порога тишины после речи, сброс буфера и VAD-состояния после сегмента (сейчас `_reset_buffer` сбрасывает и буфер, и VAD-state — теперь это два явных вызова).

## Обработка ошибок

Без изменений: `process_audio` сохраняет внешний `try/except` с `on_error`. `VadProcessor.detect_speech` сохраняет внутренний `try/except` (как в `_detect_speech`). Граница потоков не меняется — `WhisperRecognizer._lock` остаётся в `process_audio`; `VadProcessor`/`AudioBuffer` не потокобезопасны сами по себе (вызываются под `_lock` оркестратора), что документируется.

## Тестирование

**Характеризующие тесты ДО рефактора** (`tests/test_whisper_recognizer.py`, дополнить) — на текущем `process_audio` через публичный API, фиксируют поведение, переживают рефактор без изменений:
- Подготовка: recognizer с `_is_loaded=True`, `_model`=мок, метод транскрипции замокан (текущее имя `_transcribe_buffer`; после рефактора `_transcribe(audio_np)`) — не грузить GPU; VAD замокан (текущий `_detect_speech`; после — `VadProcessor.detect_speech`) через `patch`, чтобы управлять speech/silence.
- Кейсы: (1) серия speech-чанков накапливается, транскрипция НЕ вызвана; (2) speech затем silence ≥ порога → транскрипция вызвана один раз с накопленным аудио, буфер сброшен; (3) silence до начала речи игнорируется.

**Юнит-тесты новых классов** (после выделения):
- `tests/test_audio_buffer.py`: `add` (три ветки), `get_audio` (конкатенация), `reset`, порог тишины из `min_silence_duration_ms`. Чистая numpy — без моков.
- `tests/test_vad_processor.py`: `detect_speech` с управляемым входом (мок ONNX-сессии или фикстура), `reset` обнуляет state/context, `is_loaded`.

## Порядок реализации

1. Характеризующие тесты на текущий `process_audio` → зелёные (фиксируют baseline).
2. Выделить `AudioBuffer` (проще, чистая логика), переключить `process_audio` на него → характеризующие зелёные + юнит `AudioBuffer`.
3. Выделить `VadProcessor` (+V6), переключить `process_audio`/`load_model` → характеризующие зелёные + юнит `VadProcessor`.
4. Почистить ядро `WhisperRecognizer`, финальный прогон тестов + ручная проверка (запуск приложения, надиктовать фразу, убедиться в распознавании).

## Критерий готовности

Все тесты зелёные (характеризующие + новые юнит), приложение запускается и распознаёт речь на GPU как прежде, `whisper_recognizer.py` ужат до ~450 строк, VAD и буфер — самостоятельные модули с юнит-тестами.
