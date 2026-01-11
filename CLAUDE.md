# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiceType is a Windows desktop application for global voice-to-text input. It runs entirely locally without sending data to the cloud.

**Speech Recognition:** Whisper (faster-whisper) with VAD-based segmentation for high-quality offline recognition.

**Key Features:**
- No PyTorch dependency (~0.7 GB venv instead of ~2.5 GB)
- ONNX-based Silero VAD for voice activity detection
- Auto-unload models after inactivity (default 10s)

## Quick Start

```bash
cd voicetype
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Models download automatically on first use:

- **Whisper** → `~/.cache/huggingface/hub/`
- **Silero VAD** → `~/.cache/silero-vad/`

## Architecture

### Threading Model

1. **Main Thread (Qt Event Loop)** - UI updates, tray icon, settings
2. **Audio Thread** - PyAudio capture, writes to queue
3. **Recognition Thread** - Whisper processing, text output
4. **Hotkey Listener Thread** - pynput global hotkeys

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
- `get_database()` - [database.py](voicetype/src/data/database.py) - SQLite for history (15 max) and stats (2h retention)
- `get_models_manager()` - [models_manager.py](voicetype/src/data/models_manager.py) - Whisper model paths

### Module Structure

```
voicetype/src/
├── app.py              # Main controller, connects all components
├── main.py             # Entry point, Qt app setup
├── core/
│   ├── whisper_recognizer.py # Whisper engine with Silero VAD
│   ├── audio_capture.py      # PyAudio capture
│   ├── output_manager.py     # Keyboard/clipboard output
│   └── hotkey_manager.py     # Global hotkey listener
├── ui/                 # PyQt6 interface (tabs, widgets, themes)
├── data/               # Config, database, model management
└── utils/              # Logger, autostart, constants
```

### Recording Flow

1. User triggers hotkey → `HotkeyManager` emits signal
2. `VoiceTypeApp.toggle_recording()` creates `AudioCapture`
3. `AudioCapture` writes audio chunks to queue in separate thread
4. Recognition thread reads queue, feeds `WhisperRecognizer`
5. VAD detects speech → accumulates audio → detects silence → transcribes
6. `OutputManager` types into active window (pynput) or copies to clipboard

### Whisper + VAD Pipeline

The [whisper_recognizer.py](voicetype/src/core/whisper_recognizer.py) implements:
- **faster-whisper** (CTranslate2) - 4x faster than OpenAI Whisper, no PyTorch
- **Silero VAD ONNX** - voice activity detection (512-sample windows)
- Lazy model loading on first `process_audio()` call
- Auto-unload after configurable timeout to free memory

## Configuration

User config stored at `%APPDATA%/VoiceType/config.yaml`.

Key settings:

- `audio.microphone_id` - "default" or device ID
- `audio.language` - "ru", "en"
- `audio.whisper.model` - "base", "small", "medium"
- `audio.whisper.vad_threshold` - VAD sensitivity (0.0-1.0, default 0.5)
- `audio.whisper.unload_timeout` - seconds before model unload (default 10)
- `output.mode` - "keyboard" or "clipboard"

## Audio Constants

Defined in [constants.py](voicetype/src/utils/constants.py):

- `SAMPLE_RATE = 16000` - Required by Whisper/VAD models
- `CHUNK_SIZE = 4000` - ~250ms of audio per chunk
- `CHANNELS = 1` - Mono audio

## Default Hotkey

- `Ctrl+Shift+S` - Toggle recording (start/stop)
