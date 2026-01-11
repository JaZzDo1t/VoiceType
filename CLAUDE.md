# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiceType is a Windows desktop application for global voice-to-text input. It runs entirely locally without sending data to the cloud.

**Speech Recognition Engines:**
- **Vosk** - streaming recognition, real-time text output
- **Whisper** (faster-whisper) - high quality with VAD-based segmentation

**Key Features:**
- No PyTorch dependency (~0.7 GB venv instead of ~2.5 GB)
- ONNX-based VAD (Silero) and punctuation (RUPunct)
- Auto-unload models after inactivity

## Quick Start (New PC Deployment)

```bash
# 1. Clone repository
git clone <repository-url>
cd VoiceType/voicetype

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies (~0.7 GB)
pip install -r requirements.txt

# 4. Download models to models/ folder:
#    - Vosk: https://alphacephei.com/vosk/models
#      -> vosk-model-small-ru-0.22/ (~50 MB)
#    - RUPunct: https://huggingface.co/averkij/rupunct-onnx
#      -> rupunct-onnx/ (~680 MB)

# 5. Run (Whisper + Silero VAD download automatically on first use)
python run.py
```

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

# Build standalone exe (~200 MB without PyTorch)
pip install pyinstaller
pyinstaller build/voicetype_onnx.spec
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

### Recognition Engines

Two recognition engines available:
1. **Vosk** ([recognizer.py](voicetype/src/core/recognizer.py)) - streaming, real-time partial results
2. **Whisper** ([whisper_recognizer.py](voicetype/src/core/whisper_recognizer.py)) - VAD-based, higher quality

Whisper uses:
- **faster-whisper** (CTranslate2) - 4x faster than OpenAI Whisper, no PyTorch
- **Silero VAD ONNX** - voice activity detection without PyTorch
- Auto-unload after configurable timeout (default 60s)

### Threading Model

The application uses three main threads:
1. **Main Thread (Qt Event Loop)** - UI updates, tray icon, settings
2. **Audio Thread** - PyAudio capture, writes to queue
3. **Recognition Thread** - Vosk/Whisper processing, text output

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
- `get_models_manager()` - [models_manager.py](voicetype/src/data/models_manager.py) - Vosk/Whisper/RUPunct model paths

### Module Structure

```
voicetype/src/
├── app.py              # Main controller, connects all components
├── main.py             # Entry point, Qt app setup
├── core/
│   ├── recognizer.py        # Vosk engine
│   ├── whisper_recognizer.py # Whisper engine with VAD
│   ├── audio_capture.py     # PyAudio capture
│   ├── punctuation.py       # RUPunct ONNX
│   └── output_manager.py    # Keyboard/clipboard output
├── ui/                 # PyQt6 interface (tabs, widgets, themes)
├── data/               # Config, database, model management
└── utils/              # Logger, autostart, constants
```

### Recording Flow

1. User triggers hotkey → `HotkeyManager` emits signal
2. `VoiceTypeApp.start_recording()` creates `AudioCapture`
3. `AudioCapture` writes audio chunks to queue in separate thread
4. Recognition thread reads queue, feeds `Recognizer` or `WhisperRecognizer`
5. Recognizer emits partial/final results via signals
6. Final text processed through `Punctuation` → `OutputManager`
7. `OutputManager` types into active window (pynput) or copies to clipboard

## Models Location

### Local Models (in `voicetype/models/`)

**Vosk** - download from https://alphacephei.com/vosk/models:
- `vosk-model-small-ru-0.22/` (~50 MB) - fast
- `vosk-model-ru-0.42/` (~1.5 GB) - quality

**RUPunct ONNX** - download from https://huggingface.co/averkij/rupunct-onnx:
- `rupunct-onnx/` (~680 MB) - full model
- `rupunct-medium-onnx/` (~330 MB) - compact

### Auto-downloaded Models

**Whisper** (faster-whisper) - downloads to `~/.cache/huggingface/hub/`:
- tiny, base, small, medium, large-v2, large-v3

**Silero VAD ONNX** - downloads to `~/.cache/silero-vad/`:
- `silero_vad.onnx` (~2 MB)

## Configuration

User config stored at `%APPDATA%/VoiceType/config.yaml`. See [config.example.yaml](voicetype/config.example.yaml) for structure.

Key settings:
- `audio.engine` - "vosk" or "whisper"
- `audio.microphone_id` - "default" or device ID
- `audio.language` - "ru", "en", etc.
- `audio.model` - "small", "medium", "large"
- `whisper.vad_threshold` - VAD sensitivity (0.0-1.0)
- `whisper.unload_timeout` - seconds before model unload
- `output.mode` - "keyboard" (emulate typing) or "clipboard"

## Audio Constants

Key audio settings defined in [constants.py](voicetype/src/utils/constants.py):
- `SAMPLE_RATE = 16000` - Required by Vosk/Whisper models
- `CHUNK_SIZE = 4000` - ~250ms of audio per chunk
- `CHANNELS = 1` - Mono audio

## UI Tabs

The main window ([main_window.py](voicetype/src/ui/main_window.py)) contains tabs:
- **Main (tab_main.py)** - Audio, recognition engine, output, system settings
- **Hotkeys (tab_hotkeys.py)** - Global hotkey configuration
- **History (tab_history.py)** - Last 15 recognition sessions
- **Stats (tab_stats.py)** - CPU/RAM graphs for 24h
- **Logs (tab_logs.py)** - Application logs viewer
- **Test (tab_test.py)** - Microphone and recognition testing

## Default Hotkeys

- `Ctrl+Shift+S` - Start recording
- `Ctrl+Shift+X` - Stop recording
