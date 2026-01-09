#!/usr/bin/env python
"""
VoiceType - Run Script
Скрипт запуска из корня проекта.
"""
import sys
from pathlib import Path

# Добавляем src в путь
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
