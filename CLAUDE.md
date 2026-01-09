# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiceType is a Windows desktop application for global voice-to-text input. It runs entirely locally without sending data to the cloud. The application uses Vosk for speech recognition and Silero TE for automatic punctuation.

## Build and Run Commands

```bash
# Activate virtual environment
cd voicetype
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for development

# Run the application
python run.py

# Build standalone exe (~450 MB with PyTorch)
pip install pyinstaller
pyinstaller build/voicetype.spec
# Output: dist/VoiceType/
```

## Testing

```bash
# Run all tests (from voicetype directory)
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_database.py::TestDatabase::test_add_history_entry

# Exclude hardware-dependent tests (no microphone)
pytest -m "not hardware"

# Exclude slow tests
pytest -m "not slow"
```

### Test Markers

Tests use custom pytest markers defined in [conftest.py](voicetype/tests/conftest.py):
- `@pytest.mark.hardware` - Requires real microphone
- `@pytest.mark.slow` - Takes >5 seconds
- `@pytest.mark.e2e` - End-to-end integration tests

Tests auto-skip when models or hardware are unavailable via fixtures like `vosk_model_path` and `check_microphone_available`.

## Code Quality

```bash
black src/        # Format code
isort src/        # Sort imports
flake8 src/       # Lint
mypy src/         # Type check
```

## Architecture

### Threading Model

The application uses three main threads:
1. **Main Thread (Qt Event Loop)** - UI updates, tray icon, settings
2. **Audio Thread** - PyAudio capture, writes to queue
3. **Recognition Thread** - Vosk streaming, Silero punctuation, text output

### Thread-Safe Communication

All cross-thread communication uses PyQt signals. The `VoiceTypeApp` class in [app.py](voicetype/src/app.py) defines signals like `_partial_result_signal`, `_final_result_signal` that background threads emit. Never call UI methods directly from background threads.

```python
# Correct: emit signal from background thread
self._partial_result_signal.emit(text)

# Wrong: direct UI call from background thread
self._main_window.update_text(text)  # Will crash
```

### Key Singletons

Three modules use singleton patterns via `get_*` functions:
- `get_config()` - [config.py](voicetype/src/data/config.py) - YAML configuration with dot notation (`config.get("audio.language")`)
- `get_database()` - [database.py](voicetype/src/data/database.py) - SQLite for history (15 max) and stats (24h)
- `get_models_manager()` - [models_manager.py](voicetype/src/data/models_manager.py) - Vosk/Silero model paths

### Module Structure

```
voicetype/src/
├── app.py              # Main controller, connects all components
├── main.py             # Entry point, Qt app setup
├── core/               # Audio capture, recognition, hotkeys, output
├── ui/                 # PyQt6 interface (tabs, widgets, themes)
├── data/               # Config, database, model management
└── utils/              # Logger, autostart, constants
```

### Recording Flow

1. User triggers hotkey → `HotkeyManager` emits signal
2. `VoiceTypeApp.start_recording()` creates `AudioCapture`
3. `AudioCapture` writes audio chunks to queue in separate thread
4. Recognition thread reads queue, feeds `Recognizer`
5. `Recognizer` emits partial/final results via signals
6. Final text processed through `Punctuation` → `OutputManager`
7. `OutputManager` types into active window (pynput) or copies to clipboard

## Configuration

User config stored at `%APPDATA%/VoiceType/config.yaml`. See [config.example.yaml](voicetype/config.example.yaml) for structure.

Key settings:
- `audio.microphone_id` - "default" or device ID
- `audio.language` - "ru" or "en"
- `audio.model` - "small" (~50MB) or "large" (~1.5GB)
- `output.mode` - "keyboard" (emulate typing) or "clipboard"

## Audio Constants

Key audio settings defined in [constants.py](voicetype/src/utils/constants.py):
- `SAMPLE_RATE = 16000` - Required by Vosk models
- `CHUNK_SIZE = 4000` - ~250ms of audio per chunk
- `CHANNELS = 1` - Mono audio

## Models Location

Vosk models should be placed in `voicetype/models/`:
- `vosk-model-small-ru-0.22/` - Small Russian model
- `vosk-model-ru-0.42/` - Large Russian model

Download from https://alphacephei.com/vosk/models

## UI Tabs

The main window ([main_window.py](voicetype/src/ui/main_window.py)) contains tabs:
- **Main (tab_main.py)** - Audio, recognition, output, system settings
- **Hotkeys (tab_hotkeys.py)** - Global hotkey configuration
- **History (tab_history.py)** - Last 15 recognition sessions
- **Stats (tab_stats.py)** - CPU/RAM graphs for 24h
- **Logs (tab_logs.py)** - Application logs viewer
- **Test (tab_test.py)** - Microphone and recognition testing

## Default Hotkeys

- `Ctrl+Shift+S` - Start recording
- `Ctrl+Shift+X` - Stop recording
