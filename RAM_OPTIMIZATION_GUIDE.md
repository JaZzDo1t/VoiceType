# Руководство по оптимизации RAM для VoiceType

## Текущее потребление памяти

| Конфигурация | RAM |
|--------------|-----|
| Базовый (Python + PyQt6 + утилиты) | 150-200 MB |
| + Vosk Small + Silero TE | **750-850 MB** |
| + Vosk Large + Silero TE | **2,200-2,300 MB** |

### Распределение памяти

```
PyTorch (Silero TE)     ████████████████████████  450-550 MB (60%)
Vosk (распознавание)    ████████                  50-1,500 MB (зависит от модели)
PyQt6 UI                ██                        50-70 MB
Python + зависимости    ██                        50-80 MB
Буферы аудио            ░                         5-20 MB
```

---

## Вариант 1: Переход на ONNX Runtime (РЕКОМЕНДУЕТСЯ)

### Экономия: 450-500 MB (снижение на 60%)

**Описание:**
Заменить PyTorch (400-500 MB) на ONNX Runtime (15-30 MB) для инференса Silero TE. PyTorch используется только для одной задачи — пунктуации текста.

**После оптимизации:**
```
До:  750-850 MB (Vosk Small + PyTorch/Silero)
После: 280-380 MB (Vosk Small + ONNX/Silero)
```

### Реализация

**Шаг 1: Конвертация модели**

```python
import torch

# Загрузить TorchScript модель
model = torch.jit.load("models/silero-te/te_model_jit.pt")

# Создать dummy inputs
x = torch.zeros(1, 512, dtype=torch.long)
att_mask = torch.ones(1, 512, dtype=torch.long)
lan = torch.tensor([[[3]]])  # Russian

# Экспорт в ONNX
torch.onnx.export(
    model, (x, att_mask, lan),
    "models/silero-te/te_model.onnx",
    opset_version=14,
    input_names=['input_ids', 'attention_mask', 'language'],
    output_names=['punctuation', 'capitalization'],
    dynamic_axes={'input_ids': {1: 'seq_len'}}
)
```

**Шаг 2: Создать ONNX-wrapper**

```python
# src/core/punctuation_onnx.py
import onnxruntime as ort
import numpy as np

class TeModelONNX:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)

    def enhance(self, tokens, attention_mask, language):
        inputs = {
            'input_ids': np.array(tokens, dtype=np.int64),
            'attention_mask': np.array(attention_mask, dtype=np.int64),
            'language': np.array(language, dtype=np.int64)
        }
        outputs = self.session.run(None, inputs)
        return outputs[0], outputs[1]  # punctuation, capitalization
```

**Шаг 3: Обновить requirements.txt**

```diff
- torch>=2.0.0,<3.0.0
+ onnxruntime>=1.15.0,<2.0.0
```

### Преимущества

| Метрика | PyTorch | ONNX Runtime |
|---------|---------|--------------|
| Размер библиотеки | 400-500 MB | 15-30 MB |
| Время загрузки модели | 2-3 сек | 0.5-1 сек |
| Время инференса | 100-200 мс | 80-150 мс |
| Размер exe | ~450 MB | ~70 MB |

### Риски и митигации

- **Риск:** Потеря точности при конвертации
- **Митигация:** Сравнительное тестирование выходов

**Сложность:** СРЕДНЯЯ (3-5 дней)
**Приоритет:** ВЫСОКИЙ

---

## Вариант 2: Лёгкая сборка без Silero TE

### Экономия: 450-550 MB (снижение на 65%)

**Описание:**
Создать альтернативную сборку без PyTorch/Silero для пользователей, которым не нужна автоматическая пунктуация.

**После оптимизации:**
```
Полная сборка:  750-850 MB
Лёгкая сборка:  150-250 MB
```

### Реализация

**Создать `build/voicetype_light.spec`:**

```python
# Копия voicetype.spec без torch
# Удалить:
# - collect_all('torch')
# - hiddenimports для torch
# - datas для silero-te

a = Analysis(
    ['../run.py'],
    # ... остальные настройки ...
    hiddenimports=[
        'vosk',
        'PyQt6.QtWidgets',
        # НЕТ torch
    ],
)
```

**Код уже готов:**
- `PunctuationDisabled` класс автоматически используется при отсутствии torch
- Никаких изменений в `src/core/punctuation.py` не требуется

### Команда сборки

```bash
# Полная сборка
pyinstaller build/voicetype.spec

# Лёгкая сборка
pyinstaller build/voicetype_light.spec
```

**Сложность:** НИЗКАЯ (1-2 дня)
**Приоритет:** СРЕДНИЙ

---

## Вариант 3: Выгрузка неиспользуемых моделей

### Экономия: 450-550 MB (динамически)

**Описание:**
Добавить возможность выгружать Silero TE когда пунктуация не нужна, освобождая память в runtime.

### Реализация

**Добавить в `src/app.py`:**

```python
def _unload_punctuation(self) -> None:
    """Выгрузить Silero TE для освобождения памяти."""
    if self._punctuation is not None:
        if hasattr(self._punctuation, 'unload'):
            self._punctuation.unload()
        self._punctuation = None
    gc.collect()
    logger.info("Silero TE unloaded, memory freed")

def _load_punctuation_on_demand(self) -> None:
    """Загрузить Silero TE по требованию."""
    if self._punctuation is None:
        language = self._config.get("audio.language", "ru")
        try:
            self._punctuation = Punctuation(language=language)
            self._punctuation.load_model()
        except Exception as e:
            logger.warning(f"Failed to load Silero TE: {e}")
            self._punctuation = PunctuationDisabled()
```

**Добавить переключатель в UI:**

```python
# src/ui/tabs/tab_main.py
self._punctuation_toggle = QCheckBox("Включить пунктуацию")
self._punctuation_toggle.toggled.connect(self._on_punctuation_toggled)
```

**Сложность:** НИЗКАЯ (1-2 дня)
**Приоритет:** СРЕДНИЙ

---

## Вариант 4: Таймаут выгрузки при простое

### Экономия: 450-550 MB (при простое)

**Описание:**
Автоматически выгружать Silero TE после 60 минут бездействия. Перезагружать при первом использовании.

### Реализация

```python
# src/app.py
class VoiceTypeApp(QObject):
    def __init__(self):
        # ...
        self._idle_timer = QTimer()
        self._idle_timer.timeout.connect(self._on_idle_timeout)
        self._idle_timeout_minutes = 60

    def _reset_idle_timer(self):
        """Сбросить таймер при активности."""
        self._idle_timer.stop()
        self._idle_timer.start(self._idle_timeout_minutes * 60 * 1000)

    def _on_idle_timeout(self):
        """Выгрузить модели при простое."""
        logger.info("Idle timeout reached, unloading models")
        self._unload_punctuation()

    def start_recording(self):
        # При начале записи загрузить модели если нужно
        self._reset_idle_timer()
        if self._punctuation is None:
            self._load_punctuation_on_demand()
        # ...
```

**Сложность:** СРЕДНЯЯ (2-3 дня)
**Приоритет:** НИЗКИЙ

---

## Сравнительная таблица

| Вариант | Экономия | Сложность | Когда использовать |
|---------|----------|-----------|-------------------|
| **ONNX Runtime** | 450-500 MB | СРЕДНЯЯ | Всегда (рекомендуется) |
| Лёгкая сборка | 450-550 MB | НИЗКАЯ | Не нужна пунктуация |
| Выгрузка моделей | 450-550 MB | НИЗКАЯ | Runtime контроль |
| Таймаут простоя | 450-550 MB | СРЕДНЯЯ | Фоновая работа |

---

## Рекомендуемый план действий

### Фаза 1 (неделя 1-2): ONNX Runtime
1. Конвертировать Silero TE в ONNX
2. Создать ONNX-wrapper класс
3. Обновить зависимости и PyInstaller spec
4. Тестирование

**Результат:** Снижение RAM с 750 MB до 280-380 MB

### Фаза 2 (неделя 3): Дополнительные опции
1. Создать лёгкую сборку
2. Добавить переключатель пунктуации в UI
3. Документация

**Результат:** Гибкость для разных сценариев использования

### Фаза 3 (при необходимости): Продвинутые оптимизации
1. Таймаут выгрузки при простое
2. Квантованные модели (int8)
3. Стриминговый инференс

---

## Быстрые победы (без рефакторинга)

### 1. Использовать Vosk Small вместо Large
```
Экономия: ~1,450 MB
Компромисс: Ниже качество распознавания
```

### 2. Отключить пунктуацию в настройках
```yaml
# config.yaml
recognition:
  punctuation_enabled: false
```
```
Экономия: 450-550 MB (Silero TE не загружается)
Компромисс: Текст без знаков препинания
```

### 3. Закрыть другие приложения
```
VoiceType с большой моделью: 2.2 GB
Рекомендуемый минимум RAM: 4 GB
Комфортный объём RAM: 8+ GB
```

---

## Мониторинг памяти

### Встроенная статистика
Вкладка "Статистика" показывает:
- CPU usage (%)
- RAM usage (MB)
- График за 24 часа

### Командная строка
```bash
# Windows
tasklist /fi "imagename eq VoiceType.exe"

# Python (встроено)
import psutil
process = psutil.Process()
print(f"RAM: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

---

## Заключение

**Главный источник потребления RAM — PyTorch (60% от общего объёма)**, который используется только для одной задачи — пунктуации текста через Silero TE.

**Рекомендация:** Переход на ONNX Runtime снизит потребление RAM на 60% без потери функциональности.

| Метрика | До | После ONNX |
|---------|-----|-----------|
| RAM (типичная конфигурация) | 750-850 MB | 280-380 MB |
| Размер exe | ~450 MB | ~70 MB |
| Время запуска | 5-7 сек | 2-3 сек |
