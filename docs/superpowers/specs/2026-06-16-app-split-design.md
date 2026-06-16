# Дизайн: разбиение app.py (Фаза B.2)

**Дата:** 2026-06-16
**Статус:** утверждён (brainstorming)
**Затрагивает:** `voicetype/src/app.py` (893 строки) → + `recording_session.py`, `model_loader.py` (новые)

## Проблема и цель

`app.py` — God-object `VoiceTypeApp` (893 строки) с четырьмя ответственностями: оркестрация жизненного цикла, логика записи + поток распознавания, загрузка/выгрузка моделей, маршрутизация ~18 UI-обработчиков. Цель — выделить запись и загрузку моделей в отдельные классы. **Рефакторинг структурный: поведение идентично.** Это вторая (финальная) часть Фазы B аудита.

## Принятые решения

- **Подход «хелперы со ссылкой на app»** (минимальный риск): `RecordingSession` и `ModelLoader` — обычные классы (НЕ QObject), получают в конструкторе ссылку на `VoiceTypeApp`. Все `pyqtSignal` остаются атрибутами `VoiceTypeApp` (его QObject-природа не трогается); хелперы эмитят их через `self._app._<signal>`. Методы переезжают почти дословно (`self._x` → `self._app._x` для общих полей).
- **K2 попутно:** `_is_recording` (bool) → `threading.Event`, живёт в `RecordingSession`.
- **Верификация:** автотесты на Qt+потоки непрактичны → структурный перенос + обязательная ручная проверка запуском. 26 существующих тестов (whisper/diagnostics) остаются зелёными как страховка соседнего кода.
- **Вне scope:** стили→тема (отдельная косметика); любые изменения поведения.

## Архитектура

Два новых модуля в `src/core/`. `VoiceTypeApp` создаёт их в `__init__` (`self._recording = RecordingSession(self)`, `self._models = ModelLoader(self)`) и делегирует. Общее изменяемое состояние, к которому обращаются оба и UI (`_recognizer`, `_audio_capture`, `_output_manager`, `_tray_icon`, `_main_window`, `_config`, флаг `_models_loaded`), остаётся на `VoiceTypeApp` — хелперы читают/пишут его через `self._app`. Это сохраняет единый «источник истины» и не плодит дублей состояния.

## Компоненты

### `src/core/recording_session.py` (новый) — `RecordingSession`

Переносится из `app.py`: `start_recording`, `stop_recording`, `_do_start_recording`, `_recognition_loop`, `_update_audio_level`. Владеет собственным состоянием записи.

```python
class RecordingSession:
    def __init__(self, app):           # app: VoiceTypeApp
        self._app = app
        self._is_recording = threading.Event()   # K2
        self._recognition_thread: Optional[threading.Thread] = None
        self._session_text = ""
        self._session_start = None
    def start(self) -> None:           # бывш. start_recording (+ _do_start_recording)
    def stop(self) -> None:            # бывш. stop_recording
    def is_recording(self) -> bool:    # self._is_recording.is_set()
    def _recognition_loop(self) -> None:
    def _update_audio_level(self) -> None:
```

Обращения к общему: `self._app._recognizer`, `self._app._audio_capture`, `self._app._output_manager`, `self._app._tray_icon`, сигналы `self._app._partial_result_signal` / `_final_result_signal` / `_update_level_signal` / `_recognition_finished_signal`.

### `src/core/model_loader.py` (новый) — `ModelLoader`

Переносится: `_load_models_async`, `_do_load_models`, `_reload_models_then_start`, `_do_reload_models_then_start`, `_check_environment`, `_create_recognizer`, `_create_whisper_recognizer`, `_unload_models`.

```python
class ModelLoader:
    def __init__(self, app):
        self._app = app
    def load_async(self) -> None:               # _load_models_async → QTimer → _do_load_models
    def reload_then_start(self) -> None:         # _reload_models_then_start → _do_reload...
    def unload(self) -> None:                    # _unload_models
    def _do_load_models(self) -> None:
    def _do_reload_models_then_start(self) -> None:
    def _check_environment(self, model_name, device) -> bool:
    def _create_recognizer(self) -> bool:
    def _create_whisper_recognizer(self) -> bool:
```

Обращения к общему: `self._app._recognizer` (создаёт/обнуляет), `self._app._config`, `self._app._main_window`, `self._app._tray_icon`, `self._app._models_loaded`, сигналы `self._app._models_loaded_signal` / `_whisper_status_signal` / `_vad_status_signal` / `_error_signal`.

### `VoiceTypeApp` после (~400 строк)

Остаётся QObject: объявления `pyqtSignal`, `initialize`/`quit`, `_create_ui`/`_connect_ui_signals`, ~18 `_on_*` обработчиков, общие поля компонентов. Публичные точки делегируют:
```python
def start_recording(self): self._recording.start()
def stop_recording(self):  self._recording.stop()
```
Места, где раньше читался `self._is_recording`, идут через `self._recording.is_recording()`. `_load_models_async()`/`toggle_recording` вызывают `self._models.*` / `self._recording.*`.

## Поток данных

Поведение тождественно. Пример (хоткей записи): `_on_hotkey_triggered` → `toggle_recording` → проверяет `self._recording.is_recording()` → `self._recording.start()` или `.stop()`. Внутри `start` (в RecordingSession) — та же логика, что была в `_do_start_recording`, но обращения к recognizer/audio/сигналам идут через `self._app`. Поток распознавания (`_recognition_loop`) живёт в RecordingSession, эмитит сигналы через `self._app` (thread-safe, как требует CLAUDE.md).

## Обработка ошибок

Без изменений: существующие `try/except` переезжают вместе с методами. Реактивный путь загрузки (`_do_load_models` → `_models_loaded_signal` → `_on_models_loaded`) сохраняется — `_on_models_loaded` остаётся в `VoiceTypeApp` (UI-обработчик), `ModelLoader` лишь эмитит сигнал.

## Верификация

- **Существующие 26 тестов** (whisper/diagnostics) — должны остаться зелёными (страховка, что соседний код не задет).
- **Синтаксис/импорт:** `app.py`, `recording_session.py`, `model_loader.py` импортируются без ошибок.
- **Ручная проверка запуском** (обязательна — основная гарантия поведения; автотесты на Qt+потоки вне scope):
  1. Запуск `python run.py` → приложение стартует, модель грузится на GPU, статус «готов».
  2. Хоткей записи → надиктовать фразу → текст распознаётся и вставляется; партиал/финал работают.
  3. Остановка записи → финальный результат вставлен.
  4. Смена модели в настройках → reload отрабатывает.
  5. (Если просто) дождаться авто-выгрузки по таймауту → повторный хоткей перезагружает модель и стартует запись.

## Критерий готовности

26 тестов зелёные; `app.py` ужат до ~400 строк; запись и загрузка моделей — в отдельных модулях; ручная проверка (пп. 1-4) пройдена; поведение идентично.
