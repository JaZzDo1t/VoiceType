# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiceType is a Windows desktop application for global voice-to-text input. It runs entirely locally without sending data to the cloud.

**Speech Recognition:** Whisper (faster-whisper) with VAD-based segmentation for high-quality offline recognition.

**Key Features:**
- No PyTorch dependency (~0.7 GB venv instead of ~2.5 GB)
- ONNX-based Silero VAD for voice activity detection
- Auto-unload models after inactivity to free VRAM
- VRAM monitoring with baseline tracking (shows only app's GPU usage)

## Quick Start

```bash
cd voicetype
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Models download automatically on first use:

- **Whisper** → `~/.cache/huggingface/hub/` (Systran/faster-whisper-{size})
- **Silero VAD** → `~/.cache/silero-vad/silero_vad.onnx`

## Architecture

### Threading Model

1. **Main Thread (Qt Event Loop)** — UI updates, tray icon, settings, **model loading** (CTranslate2 crashes if loaded from non-main thread with Qt)
2. **Audio Thread** — PyAudio capture, writes to queue
3. **Recognition Thread** — reads audio queue, feeds WhisperRecognizer
4. **Hotkey Listener Thread** — pynput global hotkeys

### Thread-Safe Communication

All cross-thread communication uses PyQt signals. The `VoiceTypeApp` class in [app.py](voicetype/src/app.py) defines signals that background threads emit. Never call UI methods directly from background threads.

```python
# Correct: emit signal from background thread
self._partial_result_signal.emit(text)

# Wrong: direct UI call from background thread
self._main_window.update_text(text)  # Will crash
```

Key signals in `VoiceTypeApp`:
- `_whisper_status_signal(bool, str)` — model loaded/unloaded status for UI
- `_vad_status_signal(bool)` — VAD loaded/unloaded status
- `_hotkey_triggered_signal(str)` — hotkey actions from pynput thread
- `_stats_collected_signal(float, float, float)` — CPU, RAM, VRAM for graphs

### Key Singletons

- `get_config()` — [config.py](voicetype/src/data/config.py) — YAML configuration with dot notation (`config.get("audio.language")`)
- `get_database()` — [database.py](voicetype/src/data/database.py) — SQLite for history (15 max) and stats (2h retention)
- `get_models_manager()` — [models_manager.py](voicetype/src/data/models_manager.py) — Whisper model info and cache checking
- `_ProcessMonitor` / `_VRAMMonitor` — [system_info.py](voicetype/src/utils/system_info.py) — CPU/RAM/VRAM monitoring singletons

### Recording Flow

1. User triggers hotkey → `HotkeyManager` emits `_hotkey_triggered_signal`
2. `VoiceTypeApp.toggle_recording()` checks if models loaded; if unloaded by timeout, calls `_reload_models_then_start()` (loads in main thread via `QTimer.singleShot`)
3. `AudioCapture` writes audio chunks to queue in separate thread
4. Recognition thread reads queue, feeds `WhisperRecognizer.process_audio()`
5. VAD detects speech → accumulates audio → detects silence → transcribes
6. `OutputManager` types into active window (pynput) or copies to clipboard

### Model Lifecycle

Models load in the **main thread** (CTranslate2 requirement). The flow:
1. **Startup**: `_load_models_async()` → `QTimer.singleShot(50)` → `_do_load_models()` in main thread
2. **Auto-unload**: After `unload_timeout` seconds of inactivity, `WhisperRecognizer._auto_unload()` fires via `threading.Timer`. Blocked during active recording via `set_processing(True)`.
3. **Reload on demand**: If user presses hotkey with models unloaded, `_reload_models_then_start()` loads models then starts recording.
4. **Model change**: Changing model/language in UI triggers full unload → reload cycle via `_load_models_async()`.

Callbacks `on_model_loaded` / `on_model_unloaded` on WhisperRecognizer emit PyQt signals to update UI status indicators.

### CUDA DLL Handling (Windows)

`WhisperRecognizer._add_cuda_dll_paths()` adds nvidia package paths from venv's site-packages to both `os.add_dll_directory()` and `PATH`. Called at init and before each model load/transcription, because DLLs can become unavailable after unload/reload cycles.

### Whisper + VAD Pipeline

[whisper_recognizer.py](voicetype/src/core/whisper_recognizer.py):
- **faster-whisper** (CTranslate2) — GPU: float16, CPU: int8 (auto-fallback from CUDA to CPU on failure)
- **Silero VAD V5+ ONNX** — 512-sample windows + 64-sample context for 16kHz. The context from previous chunk is critical — without it, VAD returns ~0.001 probability even with clear speech.
- Lazy model loading on first `process_audio()` call
- Auto-unload after configurable timeout to free memory

## Configuration

User config stored at `%APPDATA%/VoiceType/config.yaml`.

Key settings:

| Key | Values | Default |
|-----|--------|---------|
| `audio.microphone_id` | "default" or device ID | "default" |
| `audio.language` | "ru", "en" | "ru" |
| `audio.noise_floor` | 200-2000 (level meter threshold) | 800 |
| `audio.whisper.model` | "small", "medium" | "small" |
| `audio.whisper.device` | "cuda", "cpu" | "cuda" |
| `audio.whisper.vad_threshold` | 0.3-0.9 (higher = less sensitive) | 0.7 |
| `audio.whisper.min_silence_ms` | ≥100 (pause before transcription) | 300 |
| `audio.whisper.unload_timeout` | seconds, 0 = disable | 10 |
| `output.mode` | "keyboard", "clipboard" | "keyboard" |
| `hotkeys.toggle_recording` | key combo string | "ctrl+shift+s" |
| `ui.recording_cursor` | true/false (красный курсор при записи) | true |

**Note:** Default values in constants.py may differ from UI defaults (e.g., VAD threshold is 0.5 in constants but 0.7 in UI slider). The UI values take precedence for user-facing defaults.

## Audio Constants

Defined in [constants.py](voicetype/src/utils/constants.py):

- `SAMPLE_RATE = 16000` — Required by Whisper/VAD models
- `CHUNK_SIZE = 4000` — ~250ms of audio per chunk
- `CHANNELS = 1` — Mono audio

## Hotkey System

- Default: `Ctrl+Shift+S` — Toggle recording (start/stop)
- Hotkey listener starts **after** Qt event loop via `QTimer.singleShot(100)` — ensures PyQt signals from pynput thread are delivered
- Debounce interval: 0.3s (`HOTKEY_DEBOUNCE_INTERVAL`)
