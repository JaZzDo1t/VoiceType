"""Pytest config: добавляем корень voicetype/ в sys.path для импорта src.*"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent  # voicetype/
sys.path.insert(0, str(ROOT))
