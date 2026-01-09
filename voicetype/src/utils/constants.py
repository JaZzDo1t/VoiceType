"""
VoiceType - Application Constants
"""
from pathlib import Path
import os

# Application info
APP_NAME = "VoiceType"
APP_VERSION = "1.0.0"
APP_AUTHOR = "VoiceType Team"

# Paths
APP_DATA_DIR = Path(os.getenv("APPDATA", "")) / APP_NAME
CONFIG_FILE = APP_DATA_DIR / "config.yaml"
DATABASE_FILE = APP_DATA_DIR / "voicetype.db"
LOGS_DIR = APP_DATA_DIR / "logs"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4000
AUDIO_FORMAT_BITS = 16

# Models
SUPPORTED_LANGUAGES = ["ru", "en"]
MODEL_SIZES = ["small", "large"]

# History
MAX_HISTORY_ENTRIES = 15

# Stats
STATS_RETENTION_HOURS = 24
STATS_INTERVAL_SECONDS = 600  # 10 минут

# Logs
LOG_RETENTION_DAYS = 1  # Keep logs for 24 hours only

# Default hotkeys
DEFAULT_HOTKEY_START = "ctrl+shift+s"
DEFAULT_HOTKEY_STOP = "ctrl+shift+x"

# UI
WINDOW_MIN_WIDTH = 700
WINDOW_MIN_HEIGHT = 500

# Tray states
TRAY_STATE_READY = "ready"
TRAY_STATE_RECORDING = "recording"
TRAY_STATE_LOADING = "loading"
TRAY_STATE_ERROR = "error"

# Output modes
OUTPUT_MODE_KEYBOARD = "keyboard"
OUTPUT_MODE_CLIPBOARD = "clipboard"

# Themes
THEME_DARK = "dark"
THEME_LIGHT = "light"

# Output timing
DEFAULT_TYPING_DELAY = 0.01  # Задержка между символами (секунды)
LAYOUT_SWITCH_DELAY = 0.1   # Задержка после смены раскладки (секунды)

# Hotkeys
HOTKEY_DEBOUNCE_INTERVAL = 0.3  # Интервал debounce для хоткеев (секунды)

# UI sizes
MIC_COMBO_MIN_WIDTH = 250  # Минимальная ширина комбобокса микрофона
