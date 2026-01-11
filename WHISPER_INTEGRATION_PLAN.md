# План интеграции Whisper в VoiceType

**Бранч:** `feature/whisper-integration`
**Тим-лид:** Claude
**Подход:** Агентская разработка по фазам

---

## Архитектура решения

```
┌─────────────────────────────────────────────────────────────┐
│                      VoiceTypeApp                           │
│                         (app.py)                            │
├─────────────────────────────────────────────────────────────┤
│                    RecognizerFactory                        │
│              (создаёт нужный движок)                        │
├──────────────────────┬──────────────────────────────────────┤
│   VoskRecognizer     │      WhisperRecognizer              │
│   (существующий)     │         (новый)                      │
│                      │  ┌─────────────────────┐             │
│                      │  │ faster-whisper      │             │
│                      │  │ + Silero VAD        │             │
│                      │  │ + auto-unload       │             │
│                      │  └─────────────────────┘             │
└──────────────────────┴──────────────────────────────────────┘
```

---

## Фаза 1: WhisperRecognizer с VAD
**Агент:** code-architect → код
**Файлы:** `src/core/whisper_recognizer.py`

### Задачи:
1. Создать класс `WhisperRecognizer` с интерфейсом как у `Recognizer`
2. Интегрировать faster-whisper с VAD (Silero)
3. Реализовать потоковую обработку:
   - Аудио буфер накапливается
   - VAD детектит паузу → транскрибируем буфер
   - Эмитим результат через сигнал
4. Auto-unload модели после N секунд неактивности
5. Поддержка GPU/CPU

### Интерфейс:
```python
class WhisperRecognizer(QObject):
    partial_result = pyqtSignal(str)  # промежуточный (пока говорит)
    final_result = pyqtSignal(str)    # финальный (после паузы)

    def __init__(self, model_size="small", device="cpu", language="ru")
    def start() -> None
    def stop() -> str  # возвращает финальный текст
    def feed_audio(chunk: bytes) -> None
    def unload_model() -> None
    def is_model_loaded() -> bool
```

### Зависимости:
```
faster-whisper>=1.0.0
```

---

## Фаза 2: Обновление ModelsManager
**Агент:** code-architect → код
**Файлы:** `src/data/models_manager.py`

### Задачи:
1. Добавить управление Whisper моделями
2. Автоматическое скачивание при первом запуске
3. Кеширование в `models/whisper-{size}/`
4. Методы:
   - `get_whisper_model_path(size)`
   - `download_whisper_model(size)`
   - `get_available_whisper_models()`

### Модели для поддержки:
- `tiny` - быстрый тест
- `small` - основная (рекомендуемая)
- `medium` - высокое качество
- `medium-russian` - mitchelldehaven/whisper-medium-russian

---

## Фаза 3: Интеграция в app.py
**Агент:** code-architect → код
**Файлы:** `src/app.py`, `src/core/__init__.py`

### Задачи:
1. Создать `RecognizerFactory`:
   ```python
   def create_recognizer(engine: str, **kwargs) -> BaseRecognizer
   ```
2. Модифицировать `VoiceTypeApp`:
   - Читать `config.audio.engine` ("vosk" | "whisper")
   - Создавать соответствующий распознаватель
   - Подключать сигналы одинаково для обоих
3. Сохранить обратную совместимость с Vosk

### Конфиг:
```yaml
audio:
  engine: "whisper"  # или "vosk"
  whisper:
    model: "small"
    device: "cpu"  # или "cuda"
    vad_threshold: 0.3  # чувствительность VAD
    unload_timeout: 60  # секунд до выгрузки
```

---

## Фаза 4: UI настройки
**Агент:** код
**Файлы:** `src/ui/tabs/tab_main.py`

### Задачи:
1. Добавить ComboBox "Движок распознавания":
   - Vosk (оффлайн, стриминг)
   - Whisper (качество, VAD)
2. При выборе Whisper показать доп. настройки:
   - Модель (small/medium/medium-russian)
   - Устройство (CPU/GPU)
   - Чувствительность VAD (слайдер)
3. Кнопка "Скачать модель" если не скачана
4. Индикатор статуса модели

---

## Фаза 5: Тестирование
**Агент:** тестирование
**Файлы:** `tests/test_whisper_recognizer.py`

### Задачи:
1. Unit-тесты WhisperRecognizer (mock модель)
2. Integration тест с реальной моделью (marker: slow)
3. Тест VAD детекции пауз
4. Тест auto-unload
5. Тест переключения движков

---

## Порядок выполнения

```
Фаза 1 ──────────────────────────────────────┐
  │                                          │
  ▼                                          │
Фаза 2 (параллельно с UI частью Фазы 4)     │
  │                                          │
  ▼                                          │
Фаза 3 ◄─────────────────────────────────────┘
  │
  ▼
Фаза 4 (UI часть)
  │
  ▼
Фаза 5
```

---

## Критерии готовности (Definition of Done)

- [ ] Whisper распознаёт русскую речь с пунктуацией
- [ ] VAD вставляет текст на паузах (задержка <1 сек)
- [ ] Модель выгружается после 60 сек неактивности
- [ ] Загрузка модели <3 сек (small на SSD)
- [ ] UI позволяет выбрать движок и модель
- [ ] Vosk продолжает работать как раньше
- [ ] Тесты проходят

---

## Риски и митигация

| Риск | Митигация |
|------|-----------|
| Whisper медленно грузится | Использовать small, lazy loading |
| VAD неточно детектит паузы | Настраиваемый threshold в UI |
| Конфликт с Vosk | Абстракция через Factory |
| Большой размер модели | Скачивание по требованию |

---

## Начинаем с Фазы 1!
