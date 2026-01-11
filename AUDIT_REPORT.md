# Аудит проекта VoiceType

**Дата:** 2026-01-11
**Статус:** Завершён

## Профиль проекта

| Параметр | Значение |
|----------|----------|
| **Тип** | Python Desktop Application |
| **Фреймворк** | PyQt6 |
| **Размер** | ~7700 LOC (32 Python файла в src/) |
| **Архитектура** | MVC с Qt сигналами |
| **Speech Engine** | faster-whisper + Silero VAD (ONNX) |

---

## КРИТИЧНЫЕ ПРОБЛЕМЫ (исправить сейчас)

### 1. Неиспользуемая зависимость `vosk` (~50+ MB)

**Проблема:** Пакет `vosk` (vosk==0.3.45) установлен в venv, но **нигде не используется в коде**. Проект полностью перешёл на Whisper.

**Влияние:**
- +50 MB к размеру venv
- Лишние DLL в билде
- Путаница в документации

**Решение:**
```bash
pip uninstall vosk -y
```

### 2. Мёртвый код и файлы связанные с Vosk

**Файлы для удаления:**

| Файл | Причина |
|------|---------|
| `tests/test_vosk_large_model_timing.py` | Тестирует Vosk который не используется |
| `tools/convert_silero_to_torchscript.py` | Использует PyTorch (проект отказался от него) |

### 3. Мёртвый код в conftest.py

**Файл:** `tests/conftest.py`

**Удалить:**
- Функция `find_vosk_model()` (строки 54-82)
- Fixture `vosk_model_path` (строки 99-108)

### 4. Устаревшие ссылки на Vosk в build/voicetype_onnx.spec

**Удалить/изменить:**
- Функция `find_vosk_dlls()` (строки 21-30)
- `binaries = find_vosk_dlls()` → `binaries = []`
- В `hiddenimports` удалить `'vosk'`
- В `datas` удалить строки с `vosk-model-*`
- В `upx_exclude` удалить Vosk DLLs

### 5. Устаревшие ссылки на Vosk в build/build.py

**Удалить:**
- Проверка наличия Vosk модели (строки 87-92)

---

## ЗДОРОВЫЕ ЧАСТИ (НЕ ТРОГАТЬ!)

- **Архитектура модулей** — core/, data/, ui/, utils/
- **Thread-safe коммуникация** — PyQt сигналы
- **Singleton паттерн** — config, database, models_manager
- **Обработка ошибок в audio_capture**
- **Auto-unload моделей**
- **Все файлы в src/** — не требуют изменений
- **requirements.txt** — не содержит vosk, не трогать

---

## ПЛАН ИСПРАВЛЕНИЙ

### Задача A: Удалить vosk из venv
```bash
cd voicetype
venv\Scripts\pip.exe uninstall vosk -y
```

### Задача B: Удалить мёртвые файлы
- Удалить `tests/test_vosk_large_model_timing.py`
- Удалить `tools/convert_silero_to_torchscript.py`

### Задача C: Очистить conftest.py
- Удалить функцию `find_vosk_model()`
- Удалить fixture `vosk_model_path`

### Задача D: Очистить build/voicetype_onnx.spec
- Удалить функцию `find_vosk_dlls()`
- Заменить `binaries = find_vosk_dlls()` на `binaries = []`
- Удалить `'vosk'` из hiddenimports
- Удалить строки с vosk-model из datas
- Удалить Vosk DLLs из upx_exclude

### Задача E: Очистить build/build.py
- Удалить проверку Vosk модели

---

## ВАЖНО ДЛЯ АГЕНТОВ

1. **НЕ ТРОГАТЬ** файлы в `src/` — они чистые
2. **НЕ ТРОГАТЬ** `requirements.txt` — там нет vosk
3. **НЕ ТРОГАТЬ** `CLAUDE.md` — документация актуальна
4. Каждый агент работает ТОЛЬКО со своей задачей
5. После изменений НЕ запускать тесты/билд — это сделает пользователь
