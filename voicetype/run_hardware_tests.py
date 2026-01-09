"""
Run Real Hardware Tests for VoiceType.

This script runs tests that use REAL hardware:
- Real microphone input
- Real Vosk model
- No mocks!

Requirements:
- Microphone connected to the system
- Vosk model installed in models/ directory

Usage:
    python run_hardware_tests.py              # Run all hardware tests
    python run_hardware_tests.py --quick      # Run only quick tests
    python run_hardware_tests.py --audio      # Run only audio capture tests
    python run_hardware_tests.py --recognition # Run only recognition tests
    python run_hardware_tests.py --e2e        # Run end-to-end tests
"""
import sys
import os
from pathlib import Path
import argparse

# Set up paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


def main():
    parser = argparse.ArgumentParser(description="Run VoiceType Hardware Tests")
    parser.add_argument("--quick", action="store_true",
                        help="Run only quick tests (skip long recordings)")
    parser.add_argument("--audio", action="store_true",
                        help="Run only audio capture tests")
    parser.add_argument("--recognition", action="store_true",
                        help="Run only speech recognition tests")
    parser.add_argument("--e2e", action="store_true",
                        help="Run only end-to-end pipeline tests")
    parser.add_argument("--stress", action="store_true",
                        help="Run stress tests")
    parser.add_argument("--list", action="store_true",
                        help="List all available tests")
    parser.add_argument("-x", "--exitfirst", action="store_true",
                        help="Exit on first failure")

    args = parser.parse_args()

    import pytest

    pytest_args = ["tests/test_real_hardware.py", "-v", "-s", "--tb=short"]

    if args.list:
        pytest_args.append("--collect-only")
    elif args.quick:
        # Skip slow tests
        pytest_args.extend(["-m", "not slow"])
    elif args.audio:
        pytest_args.extend(["-k", "TestRealAudioCapture"])
    elif args.recognition:
        pytest_args.extend(["-k", "TestRealSpeechRecognition"])
    elif args.e2e:
        pytest_args.extend(["-k", "TestFullPipeline"])
    elif args.stress:
        pytest_args.extend(["-k", "TestStressAndEdgeCases"])

    if args.exitfirst:
        pytest_args.append("-x")

    print("=" * 60)
    print("VoiceType Real Hardware Tests")
    print("=" * 60)
    print(f"Running: pytest {' '.join(pytest_args)}")
    print("=" * 60)
    print()

    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
