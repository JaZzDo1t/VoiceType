"""
Run pytest tests for VoiceType.
Execute this script to run all core tests.
"""
import sys
import os
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Run pytest
import pytest
sys.exit(pytest.main([
    "tests/test_core_production.py",
    "-v",
    "--tb=short",
    "-x"  # Stop on first failure
]))
