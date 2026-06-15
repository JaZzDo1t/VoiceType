# Дизайн: устойчивая обработка ошибок и диагностика окружения

**Дата:** 2026-06-15
**Статус:** утверждён (brainstorming)
**Затрагивает:** `voicetype/src/core/diagnostics.py` (новый), `whisper_recognizer.py`, `app.py`, `models_manager.py`, `constants.py`

## Проблема

VoiceType постоянно висит в трее. Когда модель Whisper или CUDA-библиотеки удалены/повреждены (например, после чистки диска), приложение ведёт себя плохо:

1. **`is_whisper_model_cached()` не видит повреждение.** Проверяет только наличие папки `models--Systran--faster-whisper-{size}` и непустой `snapshots`, но не реальные веса. Битая модель (пустые файлы `.incomplete` по 0 байт) считается валидной.
2. **Молчаливый fallback CUDA→CPU.** При сбое CUDA `load_model()` тихо переключается на CPU (`logger.warning` только в лог). Пользователь не знает, что потерял GPU и работает медленно.
3. **`recognizer.on_error` не подключён к UI.** Колбэк вызывается при сбоях VAD и транскрипции, но в `app.py` не назначен — ошибки во время работы молча уходят в лог.
4. **Сообщение об ошибке общее.** «Ошибка загрузки модели» без диагноза: непонятно, что именно не так (нет модели / нет cuDNN / нет сети) и что чинить.

Цель: GPU-режим основной; при проблемах приложение должно честно сообщать, что не так, а не зависать молча.

## Принятые решения

- **Объём:** все 4 дыры.
- **Подход:** C — гибрид. Лёгкая проактивная проверка двух надёжно проверяемых вещей (целостность модели, наличие cuDNN/nvJitLink) + реактивный перехват как страховка от неожиданных ошибок.
- **Битая/отсутствующая модель:** показать понятный диагноз, без встроенного скачивания (пользователь качает сам).
- **CUDA недоступен при `device=cuda`:** стоп с диагнозом, **без** молчаливого перехода на CPU.

## Архитектура

Новый модуль **`src/core/diagnostics.py`** — чистые функции проверки окружения без побочных эффектов и без зависимости от Qt/GPU. Возвращает структурированный результат. Тестируется в отрыве от приложения. Остальные изменения — точечная интеграция в существующий поток загрузки.

## Компоненты

### `src/core/diagnostics.py` (новый)

```python
class IssueCode(str, Enum):
    MODEL_MISSING = "model_missing"
    MODEL_CORRUPT = "model_corrupt"
    CUDA_CUDNN_MISSING = "cuda_cudnn_missing"
    CUDA_NVJITLINK_MISSING = "cuda_nvjitlink_missing"

@dataclass
class Issue:
    code: IssueCode
    title: str    # короткий заголовок для overlay/трея
    detail: str   # понятное объяснение + что делать

def diagnose(model_size: str, device: str) -> list[Issue]:
    """Пустой список = окружение в порядке. Иначе — список проблем."""
```

Логика проверок:

- **Модель** (`_check_model`): найти `snapshots/<hash>/model.bin` в кэше Whisper (`Path.home() / ".cache" / "huggingface" / "hub"` — путь вычисляется внутри `diagnostics`, без импорта `models_manager`, чтобы избежать цикла).
  - папки модели / `snapshots` нет, либо `snapshots` пуст → `MODEL_MISSING`
  - есть файлы `*.incomplete` в `blobs/`, либо `model.bin` весит меньше `MIN_MODEL_BIN_SIZE` → `MODEL_CORRUPT`
  - иначе ок
  - **Только метаданные файлов** (`Path.exists`, `stat().st_size`, `iterdir`). Никакого чтения содержимого, никаких контрольных сумм — проверка занимает <1 мс.
- **CUDA** (`_check_cuda`, только если `device == "cuda"`): искать DLL в `Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / <pkg> / "bin"` (тот же механизм расположения, что использует `WhisperRecognizer._add_cuda_dll_paths`).
  - нет `cudnn64_9.dll` → `CUDA_CUDNN_MISSING`
  - нет `nvJitLink64_12.dll` → `CUDA_NVJITLINK_MISSING` (без него не грузится уже установленный `cublasLt64_12.dll`)

Порядок: сначала модель, затем CUDA. Собираются **все** найденные проблемы (а не первая) — чтобы пользователь сразу видел полную картину.

### `src/utils/constants.py`

- `MIN_MODEL_BIN_SIZE = 50 * 1024 * 1024` — нижний порог размера `model.bin` (small ≈ 0.48 ГБ, medium ≈ 1.53 ГБ; 50 МБ надёжно отсекает 0-байтные пустышки).
- `CUDA_REQUIRED_DLLS` — сопоставление имени DLL → имя nvidia-пакета и подпапки (`cudnn64_9.dll` → `cudnn`, `nvJitLink64_12.dll` → `nvjitlink`).

### `src/data/models_manager.py`

Усилить `is_whisper_model_cached()`: помимо наличия `snapshots`, проверять реальный `model.bin` (> `MIN_MODEL_BIN_SIZE`) и отсутствие `.incomplete`. Чтобы не дублировать логику — `models_manager` импортирует и вызывает проверку модели из `diagnostics`. Зависимость односторонняя: `models_manager` → `diagnostics`; `diagnostics` не импортирует `models_manager`, поэтому цикла нет. Это чинит ложноположительный результат для всех потребителей (включая UI-список скачанных моделей).

### `src/core/whisper_recognizer.py`

Убрать молчаливый CUDA→CPU fallback (текущие строки 225–242). При `device == "cuda"` и сбое `WhisperModel(...)` — **не** пробовать CPU, а пробросить исключение наверх (его поймает реактивная страховка в `load_model`/`app.py`). Если пользователь сам выбрал `device == "cpu"` в конфиге — работаем на CPU как обычно. Вызовы `on_error` (строки 271, 277, 453, 596) сохраняются.

### `src/app.py`

1. **Проактивная проверка перед загрузкой.** В `_do_load_models` и `_do_reload_models_then_start` перед `load_model()` вызвать `diagnose(model, device)`. Если список непуст → `_models_loaded_signal.emit(TRAY_STATE_ERROR, "")`, показать диагноз (см. ниже), `load_model()` **не** вызывается.
2. **Подключить `on_error`.** Новый сигнал `_error_signal = pyqtSignal(str, str)` (title, detail) для thread-safety — `on_error` вызывается из recognition-потока (требование CLAUDE.md: cross-thread только через сигналы). Назначить `self._recognizer.on_error = lambda e: self._error_signal.emit("Ошибка распознавания", str(e))`. Обработчик в UI-потоке → трей-уведомление + лог.
3. **Показ диагноза.** Собрать заголовок (title первой проблемы) и детали (detail всех проблем списком), передать в `self._main_window.show_loading_error(title, detail)` (overlay уже поддерживает `show_error(error_text, details="")`) + трей-уведомление. Кнопки overlay переиспользуются: **Повторить** (перезапуск диагностики и загрузки), **Настройки** (сменить device на CPU), **Свернуть**.

## Поток данных

**Загрузка (старт / reload / смена модели):**

```
_do_load_models / _do_reload_models_then_start
  → diagnose(model, device)
      ├─ [] (ок)        → load_model() как раньше
      └─ [issues...]    → emit TRAY_STATE_ERROR
                          → overlay.show_error(title, detail-список)
                          → трей-уведомление
                          (load_model НЕ вызывается)
```

**Ошибка во время работы:**

```
recognition thread: on_error(e)
  → _error_signal.emit("Ошибка распознавания", str(e))   # thread-safe
  → UI thread: трей-уведомление + logger.error
```

## Тексты диагноза

Текст хранится в `Issue.detail` (формируется в `diagnostics`; `app.py` деталей не знает, просто отображает). Размер модели в тексте — приблизительная константа внутри `diagnostics` (small ≈ 0.5 ГБ, medium ≈ 1.5 ГБ); при отсутствии данных подстановка опускается.

- `MODEL_MISSING` → «Модель {size} не найдена. Скачайте её (≈{size} ГБ) и перезапустите приложение.»
- `MODEL_CORRUPT` → «Модель {size} повреждена (незавершённая загрузка). Удалите кэш модели и скачайте заново.»
- `CUDA_CUDNN_MISSING` → «GPU недоступен: не найден cuDNN. Установите nvidia-cudnn-cu12 или переключитесь на CPU в настройках.»
- `CUDA_NVJITLINK_MISSING` → «GPU недоступен: не найден nvJitLink (нужен для cuBLAS). Установите nvidia-nvjitlink-cu12 или переключитесь на CPU в настройках.»

## Тестирование

`diagnose()` — чистые функции, тестируются без GPU и без реальных моделей:

- временная папка модели с `.incomplete` → `MODEL_CORRUPT`
- пустая папка / нет `snapshots` → `MODEL_MISSING`
- фейковый `model.bin` размером > порога → ок
- CUDA-проверка: мок наличия/отсутствия DLL-путей → соответствующие коды
- интеграционный кейс на текущей машине: `diagnose("medium", "cuda")` должен вернуть `MODEL_CORRUPT` и `CUDA_CUDNN_MISSING` (реальное состояние после чистки диска)

Наличие pytest и структуру тестов уточнить на этапе плана реализации.

## Вне объёма (YAGNI)

- Встроенное скачивание модели с прогресс-баром (пользователь качает сам).
- Проверка контрольных сумм / целостности весов чтением файла (слишком дорого; достаточно проверки размера и `.incomplete`).
- Кнопки выбора «Повторить / Перейти на CPU» отдельным диалогом для CUDA (переключение device делается через существующие Настройки).
- Периодическая / фоновая проверка окружения (только разово при загрузке моделей).
