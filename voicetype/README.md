# VoiceType

Десктопное приложение для Windows, обеспечивающее глобальный голосовой ввод текста в любое активное окно. Работает полностью локально, без отправки данных в облако.

## Возможности

- **Два движка распознавания**:
  - **Vosk** - streaming, текст в реальном времени
  - **Whisper** (faster-whisper) - высокое качество с VAD
- **Автоматическая пунктуация** - точки, запятые, заглавные буквы (ONNX)
- **Глобальные хоткеи** - управление записью из любого приложения
- **Два режима вывода** - эмуляция клавиатуры или буфер обмена
- **История сессий** - последние 15 записей с возможностью копирования
- **Мониторинг ресурсов** - графики CPU/RAM за 24 часа
- **Тёмная и светлая темы**
- **Без PyTorch** - легковесный (~0.7 GB venv)

## Требования

- Windows 10/11
- Python 3.10 или 3.11
- Микрофон
- ~2 GB RAM (с Whisper small)

## Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd VoiceType/voicetype
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Скачивание моделей

#### Vosk (обязательно для движка Vosk)

Скачайте с https://alphacephei.com/vosk/models:
- `vosk-model-small-ru-0.22` (~50 MB) - быстрая
- `vosk-model-ru-0.42` (~1.5 GB) - качественная

Распакуйте в папку `models/`:
```
models/
  vosk-model-small-ru-0.22/
  vosk-model-ru-0.42/
```

#### RUPunct ONNX (для пунктуации)

Скачайте с https://huggingface.co/averkij/rupunct-onnx:
- `rupunct-onnx/` (~680 MB) - полная
- `rupunct-medium-onnx/` (~330 MB) - компактная

```
models/
  rupunct-onnx/
```

#### Whisper и Silero VAD (автоматически)

Модели faster-whisper и Silero VAD скачиваются автоматически при первом запуске:
- Whisper: `~/.cache/huggingface/hub/`
- Silero VAD: `~/.cache/silero-vad/`

### 5. Запуск

```bash
python run.py
```

Или двойной клик на `run.bat`.

## Использование

### Горячие клавиши (по умолчанию)

| Действие | Комбинация |
|----------|------------|
| Начать запись | `Ctrl+Shift+S` |
| Остановить запись | `Ctrl+Shift+X` |

### Системный трей

После запуска приложение сворачивается в системный трей:
- **Одиночный клик** - открыть настройки
- **Двойной клик** - переключить запись
- **Правый клик** - контекстное меню

### Настройки

- **Движок** - Vosk (streaming) или Whisper (качество)
- **Модель** - размер модели распознавания
- **Язык** - ru, en и другие
- **Пунктуация** - автоматическая расстановка знаков
- **Режим вывода** - клавиатура или буфер обмена
- **Тема** - тёмная или светлая

## Структура проекта

```
voicetype/
├── src/
│   ├── main.py              # Точка входа
│   ├── app.py               # Главный контроллер
│   ├── core/
│   │   ├── whisper_recognizer.py # Whisper движок
│   │   ├── audio_capture.py     # Захват аудио
│   │   └── output_manager.py    # Вывод текста
│   ├── ui/                  # PyQt6 интерфейс
│   ├── data/                # Конфиг, БД, модели
│   └── utils/               # Утилиты
├── models/                  # ML модели (скачать отдельно)
├── tests/                   # Тесты
└── requirements.txt
```

## Технологии

- **Vosk** - streaming распознавание речи
- **faster-whisper** - качественное распознавание (CTranslate2)
- **Silero VAD** - детекция голоса (ONNX)
- **RUPunct** - пунктуация (ONNX)
- **PyQt6** - графический интерфейс
- **pynput** - глобальные хоткеи
- **PyAudio** - захват аудио

## Сборка в exe

```bash
pip install pyinstaller
pyinstaller build/voicetype_onnx.spec
```

Результат: `dist/VoiceType/`

## Конфигурация

Файл: `%APPDATA%/VoiceType/config.yaml`

```yaml
audio:
  microphone_id: "default"
  language: "ru"
  engine: "whisper"  # или "vosk"
  model: "small"
  vad_sensitivity: 0.5

whisper:
  model: "small"
  device: "cpu"
  vad_threshold: 0.3
  unload_timeout: 60

recognition:
  punctuation_enabled: true

output:
  mode: "keyboard"

hotkeys:
  start_recording: "ctrl+shift+s"
  stop_recording: "ctrl+shift+x"

system:
  autostart: false
  theme: "dark"
```

## Лицензия

MIT License

## Благодарности

- [Vosk](https://alphacephei.com/vosk/) - streaming распознавание
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - оптимизированный Whisper
- [Silero](https://github.com/snakers4/silero-models) - VAD модель
- [RUPunct](https://huggingface.co/averkij/rupunct-onnx) - пунктуация
