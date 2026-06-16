# Аудит проекта VoiceType

**Дата:** 2026-06-16
**Объём:** ~8260 LOC, 31 файл, `src/{core,data,ui,utils}` + `run.py`
**Метод:** 3 параллельных агента (зависимости / код / архитектура) + ручная верификация находок.

---

## Критичные проблемы (исправить сейчас)

### K1. `self._main_window.tabs` не существует — краш кнопки «Настройки» на экране ошибки ✅ подтверждён
`src/app.py:849` (`_on_loading_settings`):
```python
self._main_window.tabs.setCurrentWidget(self._main_window.tab_main)
```
`MainWindow` имеет `_tab_widget` (приватный) и `select_tab_by_name()`, но `.tabs` нет → `AttributeError`. Путь достижим из loading-overlay кнопкой «Настройки» — а её показывает новая диагностика при проблемах. То есть баг бьёт ровно по только что добавленной фиче.
**Фикс:** `self._main_window.select_tab_by_name("main")` (+ убрать второй некорректный аргумент).

### K2. Гонка данных на `_is_recording`
`src/app.py` — обычный `bool`, пишется из главного потока (`stop_recording`), читается из recognition-потока (`_recognition_loop`). Спасает только GIL; архитектурно некорректно, ломается при расширении логики чтения (TOCTOU).
**Фикс:** заменить на `threading.Event` (`is_set()`/`set()`/`clear()`).

### K3. `except Exception: pass` прячет ошибки очистки CUDA-памяти
`src/core/whisper_recognizer.py:~713` — в `unload()` молчаливый `pass` при `ctranslate2.cuda.empty_cache()`. После unload/reload-циклов CUDA-ошибка проглатывается, следующая загрузка падает с непонятным сообщением.
**Фикс:** заменить `pass` на `logger.debug(...)`.

---

## Важные улучшения (запланировать)

### V1. Дублирование логики CUDA-DLL-путей в трёх местах
`whisper_recognizer._add_cuda_dll_paths` + `download_model.py` (почти дословная копия) + список пакетов `[cublas,cudnn,cufft,cuda_runtime,nvjitlink]` ещё и в `constants.CUDA_REQUIRED_DLLS` (другой формат). Рассинхрон при добавлении новой CUDA-зависимости даст «диагностика зелёная, а рантайм падает».
**Фикс:** вынести в `src/utils/cuda_paths.py` единый источник списка пакетов + функцию регистрации. `download_model.py` (standalone) — импортирует оттуда или оставить копию осознанно.

### V2. Нарушения инкапсуляции (доступ к приватным членам чужих классов)
- `app.py:541` → `self._recognizer._is_processing` → добавить `WhisperRecognizer.is_processing` (property).
- `app.py:483` → `self._main_window.tab_test._status_label.setText(...)` → добавить `TabTest.set_loading_status(text)`.

### V3. `HOTKEY_DEBOUNCE_INTERVAL` определён дважды с разными значениями
`constants.py:86` = 0.3, `hotkey_manager.py:15` = 0.5 (локальное переопределение, реально работает 0.5). CLAUDE.md ссылается на 0.3 — документация врёт.
**Фикс:** импортировать из constants, оставить одно значение, поправить CLAUDE.md.

### V4. Мёртвый код
- `tab_stats.py:327` `record_stats()` — помечен deprecated, не вызывается → удалить.
- `tab_logs.py:195` `append_log()` — не подключён → удалить или подключить loguru sink.
- `system_info.py:88` `get_cpu_usage()` — не используется → удалить.
- `system_info.py` `get_memory_usage()` — один вызов в tab_stats → заменить на `psutil.virtual_memory().total`, удалить.

### V5. Дублирование пути кэша HuggingFace
`diagnostics.py:104` и `models_manager.py:43` — оба хардкодят `~/.cache/huggingface/hub`.
**Фикс:** константа `HF_CACHE_DIR` в constants.py; `_check_environment` может передавать `get_whisper_cache_dir()` в `diagnose()` (параметр уже есть).

### V6. `_vad_log_counter` инициализируется через `hasattr`, а не в `__init__`
`whisper_recognizer.py:523` — антипаттерн, атрибут добавлен постфактум.
**Фикс:** `self._vad_log_counter = 0` в `__init__`.

### V7. `app.py` — God-Object (893 строки, 4 ответственности)
Жизненный цикл + запись + управление моделями + 18 UI-обработчиков.
**Фикс (план):** выделить `RecordingSession` (start/stop/`_recognition_loop`/level) и `ModelLoader` (`_do_load_models`/`_do_reload`/`_check_environment`/`_create_recognizer`/`_unload`). `VoiceTypeApp` ужмётся до ~350-400 строк.

### V8. `whisper_recognizer.py` — God-Object (784 строки, 3 ответственности)
**Фикс (план):** выделить `VadProcessor` (Silero VAD: load/detect/state) и `AudioBuffer` (буфер + триггер по тишине). Ядро (`load_model`/`unload`/`process_audio`/`_transcribe`) остаётся.

### V9. Засорённый venv — заброшенные эксперименты вайбкодинга
Установлены руками и не используются кодом: `vosk`, `sounddevice`, `soundfile`, `keyboard`, `pyttsx3`, `PyAutoGUI`(+MouseInfo/PyGetWindow/PyRect/PyScreeze/pytweening), `optimum`/`optimum-onnx`/`onnxscript`, `scipy`, `Pygments`, `pyreadline3`. Это альтернативные ASR/audio/automation-движки, которые пробовались и брошены.
**Фикс:** `pip uninstall` перечисленного. (`nvidia-*` — нужны, не трогать.)

### V10. `pytest` используется, но не в зависимостях
Тесты есть, pytest стоит в venv, но в `requirements.txt` его нет и нет `requirements-dev.txt`.
**Фикс:** создать `requirements-dev.txt` (`pytest`, `pytest-cov`).

### V11. Инверсия слоёв: `data` → `core`
`data/models_manager.py:60` импортирует `core/diagnostics`. По архитектуре `data` не должен зависеть от `core`. (Внесено новой фичей.)
**Фикс:** перенести `diagnostics.py` в `src/utils/` (он Qt/GPU-независимый — там ему и место).

---

## Рекомендации (при возможности)

- **Воспроизводимость окружения:** нет `.python-version` (реально 3.11.9) и lock-файла. Добавить `.python-version` и `pip freeze > requirements.lock`. Уточнить `numpy>=1.26,<2.0`.
- **Дублирующий таймер:** `tab_stats.py` держит свой `QTimer` на 10 с, дублируя `_stats_timer` из `app.py` → данные расходятся. Убрать локальный, полагаться на `_stats_collected_signal`.
- **Инлайн-стили:** hex-цвета (`#9CA3AF`, `#EF4444`…) разбросаны по виджетам мимо `themes.py` → при смене темы перекрашивается не всё. Вынести в тему.
- **`format_duration`:** дублируется в tab_history / tab_stats / whisper_recognizer → вынести в utils.
- **Отладочный `import time`** внутри `app.py:729` (`_on_hotkey_triggered`) — убрать или поднять в модуль.
- **Синглтоны `system_info`** (`_ProcessMonitor`/`_VRAMMonitor`) без `threading.Lock`, в отличие от config/db/models — привести к общему паттерну.
- **`tab_main._load_settings`** делает `config.set()` в `__init__` — побочная запись на диск при создании виджета. Для инициализации достаточно `get`.
- **Порядок в `initialize()`:** `_load_models_async()` ставится до `_connect_ui_signals()`. Поменять местами для надёжности.

---

## Здоровые части (не трогать)

- **Слоистая структура** `core/data/ui/utils` — чистая, не «свалка файлов».
- **Threading через PyQt-сигналы** — соблюдён почти везде (cross-thread callbacks эмитят сигналы).
- **Обработка ошибок единообразна** — `except Exception as e: logger.error(...)`, bare `except:` нет нигде.
- **Синглтоны** config/database/models — double-checked locking, есть `_reset_*` для тестов.
- **Нет циклических импортов.**
- **Версии зависимостей** зафиксированы диапазонами (`>=x,<y`).
- `themes.py` (669) и `audio_capture.py` (536) — размер оправдан (QSS-блоки / многоуровневый fallback), не God-objects.

---

## План рефакторинга (по приоритету)

1. **Критичные (быстро, точечно):** K1 (`.tabs`→`select_tab_by_name`), K3 (`pass`→`logger.debug`), K2 (`_is_recording`→`Event`).
2. **Инкапсуляция и дубли (полдня):** V2 (публичные методы), V3 (debounce), V5 (HF cache константа), V6 (`_vad_log_counter`), V11 (diagnostics→utils).
3. **Чистка (быстро):** V4 (мёртвый код), V9 (venv), V10 (requirements-dev), отладочный import.
4. **Дедуп CUDA-путей (V1):** единый `cuda_paths.py`.
5. **Разбиение God-objects (отдельная сессия):** V7 (app.py), V8 (whisper_recognizer) — крупные, делать по плану через brainstorming.
6. **Косметика:** стили в тему, дублирующий таймер, lock-файл, синглтоны system_info.
