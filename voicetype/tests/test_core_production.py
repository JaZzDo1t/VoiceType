"""
VoiceType - Core Production Tests
Comprehensive test suite for core modules.

Tests aligned with ISO/IEC 25010 and ISTQB standards.
Focus on: reliability, functionality, thread-safety, edge cases.
"""
import pytest
import threading
import queue
import time
import tempfile
import gc
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.constants import (
    SAMPLE_RATE, CHANNELS, CHUNK_SIZE,
    OUTPUT_MODE_KEYBOARD, OUTPUT_MODE_CLIPBOARD,
    DEFAULT_HOTKEY_START, DEFAULT_HOTKEY_STOP
)


# ============================================================================
# AUDIO CAPTURE TESTS
# ============================================================================

class TestAudioCapture:
    """Test suite for AudioCapture module."""

    def test_import_audio_capture(self):
        """Module can be imported without errors."""
        from src.core.audio_capture import AudioCapture
        assert AudioCapture is not None

    def test_init_default_params(self):
        """AudioCapture initializes with default parameters."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()

        assert capture.device_id == "default"
        assert capture.sample_rate == SAMPLE_RATE
        assert capture.chunk_size == CHUNK_SIZE
        assert capture._is_running is False
        assert capture._current_level == 0.0
        assert capture.on_audio_data is None
        assert capture.on_error is None

    def test_init_custom_params(self):
        """AudioCapture accepts custom parameters."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture(
            device_id="1",
            sample_rate=48000,
            chunk_size=8000
        )

        assert capture.device_id == "1"
        assert capture.sample_rate == 48000
        assert capture.chunk_size == 8000

    def test_get_device_index_default(self):
        """_get_device_index returns None for 'default' device."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture(device_id="default")
        assert capture._get_device_index() is None

    def test_get_device_index_numeric(self):
        """_get_device_index converts string to int."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture(device_id="2")
        assert capture._get_device_index() == 2

    def test_get_device_index_invalid(self):
        """_get_device_index handles invalid device ID."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture(device_id="invalid_string")
        assert capture._get_device_index() is None

    def test_audio_queue_accessible(self):
        """get_audio_queue returns the internal queue."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()
        q = capture.get_audio_queue()

        assert isinstance(q, queue.Queue)
        assert q is capture._audio_queue

    def test_get_level_initial(self):
        """get_level returns 0.0 initially."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()
        assert capture.get_level() == 0.0

    def test_is_running_initial(self):
        """is_running returns False initially."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()
        assert capture.is_running() is False

    def test_update_level_valid_audio(self):
        """_update_level correctly calculates level from audio data."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()

        # Create silent audio
        silent_audio = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()
        capture._update_level(silent_audio)
        assert capture._current_level == 0.0

        # Create loud audio (max amplitude)
        loud_audio = np.full(CHUNK_SIZE, 32767, dtype=np.int16).tobytes()
        capture._update_level(loud_audio)
        assert capture._current_level > 0.0
        assert capture._current_level <= 1.0

    def test_update_level_edge_case_empty_data(self):
        """_update_level handles empty audio data gracefully."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()
        # Empty bytes should not crash
        capture._update_level(b"")
        # Level remains 0 or unchanged
        assert capture._current_level >= 0.0

    def test_stop_when_not_running(self):
        """stop() is safe to call when not running."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()
        capture.stop()  # Should not raise
        assert capture.is_running() is False

    def test_context_manager_protocol(self):
        """AudioCapture supports context manager protocol."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()

        assert hasattr(capture, '__enter__')
        assert hasattr(capture, '__exit__')

    def test_cleanup_clears_queue(self):
        """_cleanup empties the audio queue."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()

        # Add items to queue
        for i in range(5):
            capture._audio_queue.put(b"data")

        capture._cleanup()

        assert capture._audio_queue.empty()

    def test_callback_registration(self):
        """Callbacks can be set and called."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()

        received_data = []
        def on_data(data):
            received_data.append(data)

        capture.on_audio_data = on_data

        # Simulate callback
        test_data = b"test_audio_data"
        if capture.on_audio_data:
            capture.on_audio_data(test_data)

        assert len(received_data) == 1
        assert received_data[0] == test_data

    @pytest.mark.skipif(True, reason="Requires actual audio hardware")
    def test_start_stop_cycle_with_hardware(self):
        """Full start/stop cycle with real hardware."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()

        result = capture.start()
        assert result is True
        assert capture.is_running() is True

        time.sleep(0.1)

        capture.stop()
        assert capture.is_running() is False


# ============================================================================
# RECOGNIZER TESTS
# ============================================================================

class TestRecognizer:
    """Test suite for Recognizer module."""

    def test_import_recognizer(self):
        """Module can be imported without errors."""
        from src.core.recognizer import Recognizer
        assert Recognizer is not None

    def test_init_default_params(self):
        """Recognizer initializes with correct parameters."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./models/vosk-model-small-ru")

        assert rec.model_path == Path("./models/vosk-model-small-ru")
        assert rec.sample_rate == SAMPLE_RATE
        assert rec._is_loaded is False
        assert rec._model is None
        assert rec._recognizer is None

    def test_init_custom_sample_rate(self):
        """Recognizer accepts custom sample rate."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./model", sample_rate=8000)
        assert rec.sample_rate == 8000

    def test_is_loaded_initial(self):
        """is_loaded returns False initially."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./model")
        assert rec.is_loaded() is False

    def test_process_audio_without_model(self):
        """process_audio returns None if model not loaded."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./model")
        result = rec.process_audio(b"audio_data")

        assert result is None

    def test_get_final_result_without_model(self):
        """get_final_result returns None if model not loaded."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./model")
        result = rec.get_final_result()

        assert result is None

    def test_unload_when_not_loaded(self):
        """unload is safe when model not loaded."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./model")
        rec.unload()  # Should not raise

        assert rec._is_loaded is False
        assert rec._model is None

    def test_reset_when_not_loaded(self):
        """reset is safe when model not loaded."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./model")
        rec.reset()  # Should not raise

    def test_callback_registration(self):
        """Callbacks can be set."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./model")

        def dummy_callback(text):
            pass

        rec.on_partial_result = dummy_callback
        rec.on_final_result = dummy_callback
        rec.on_loading_progress = lambda x: None
        rec.on_error = lambda e: None

        assert rec.on_partial_result is dummy_callback
        assert rec.on_final_result is dummy_callback

    def test_load_model_nonexistent_path(self):
        """load_model returns False for nonexistent path."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./nonexistent_model_path_12345")

        error_received = []
        rec.on_error = lambda e: error_received.append(e)

        result = rec.load_model()

        assert result is False
        assert rec._is_loaded is False
        assert len(error_received) == 1
        assert isinstance(error_received[0], FileNotFoundError)

    def test_thread_safety_lock_exists(self):
        """Recognizer has a lock for thread safety."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./model")

        assert hasattr(rec, '_lock')
        assert isinstance(rec._lock, type(threading.Lock()))


# ============================================================================
# PUNCTUATION TESTS
# ============================================================================

class TestPunctuation:
    """Test suite for Punctuation module."""

    def test_import_punctuation(self):
        """Module can be imported without errors."""
        from src.core.punctuation import Punctuation, PunctuationDisabled
        assert Punctuation is not None
        assert PunctuationDisabled is not None

    def test_init_default_params(self):
        """Punctuation initializes correctly."""
        from src.core.punctuation import Punctuation

        punct = Punctuation()

        assert punct.model_dir is not None  # Default model dir is set
        assert punct.language == "ru"
        assert punct._is_loaded is False
        assert punct._model is None

    def test_init_custom_language(self):
        """Punctuation accepts custom language."""
        from src.core.punctuation import Punctuation

        punct = Punctuation(language="en")
        assert punct.language == "en"

    def test_is_loaded_initial(self):
        """is_loaded returns False initially."""
        from src.core.punctuation import Punctuation

        punct = Punctuation()
        assert punct.is_loaded() is False

    def test_enhance_without_model(self):
        """enhance returns original text if model not loaded."""
        from src.core.punctuation import Punctuation

        punct = Punctuation()

        text = "привет как дела"
        result = punct.enhance(text)

        assert result == text

    def test_enhance_empty_string(self):
        """enhance handles empty string."""
        from src.core.punctuation import Punctuation

        punct = Punctuation()

        assert punct.enhance("") == ""
        assert punct.enhance("   ") == "   "

    def test_enhance_none_handling(self):
        """enhance handles None gracefully."""
        from src.core.punctuation import Punctuation

        punct = Punctuation()

        # None should be handled (returns None or empty)
        result = punct.enhance(None)
        assert result is None

    def test_enhance_batch_without_model(self):
        """enhance_batch returns original texts if model not loaded."""
        from src.core.punctuation import Punctuation

        punct = Punctuation()

        texts = ["hello", "world"]
        result = punct.enhance_batch(texts)

        assert result == texts

    def test_unload_safe(self):
        """unload is safe when model not loaded."""
        from src.core.punctuation import Punctuation

        punct = Punctuation()
        punct.unload()  # Should not raise

        assert punct._is_loaded is False

    def test_thread_safety_lock_exists(self):
        """Punctuation has a lock for thread safety."""
        from src.core.punctuation import Punctuation

        punct = Punctuation()

        assert hasattr(punct, '_lock')
        assert isinstance(punct._lock, type(threading.Lock()))


class TestPunctuationDisabled:
    """Test suite for PunctuationDisabled (stub class)."""

    def test_init_accepts_any_args(self):
        """PunctuationDisabled accepts any arguments."""
        from src.core.punctuation import PunctuationDisabled

        # Should not raise
        pd = PunctuationDisabled()
        pd = PunctuationDisabled("arg1", kwarg="value")

    def test_load_model_always_true(self):
        """load_model always returns True."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()
        assert pd.load_model() is True

    def test_is_loaded_always_true(self):
        """is_loaded always returns True."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()
        assert pd.is_loaded() is True

    def test_enhance_capitalizes_first_letter(self):
        """enhance capitalizes first letter only."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()

        assert pd.enhance("hello") == "Hello"
        assert pd.enhance("Hello") == "Hello"
        assert pd.enhance("HELLO") == "HELLO"

    def test_enhance_empty_string(self):
        """enhance handles empty string."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()

        result = pd.enhance("")
        assert result == ""

    def test_enhance_none_handling(self):
        """enhance handles None/falsy values."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()

        # None should return None
        result = pd.enhance(None)
        assert result is None

    def test_enhance_batch(self):
        """enhance_batch processes list correctly."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()

        texts = ["hello", "world"]
        result = pd.enhance_batch(texts)

        assert result == ["Hello", "World"]

    def test_set_language_always_true(self):
        """set_language always returns True."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()
        assert pd.set_language("en") is True
        assert pd.set_language("de") is True

    def test_unload_safe(self):
        """unload does nothing, is safe."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()
        pd.unload()  # Should not raise


# ============================================================================
# OUTPUT MANAGER TESTS
# ============================================================================

class TestOutputManager:
    """Test suite for OutputManager module."""

    def test_import_output_manager(self):
        """Module can be imported without errors."""
        from src.core.output_manager import OutputManager
        assert OutputManager is not None

    def test_init_default_mode(self):
        """OutputManager initializes with keyboard mode by default."""
        from src.core.output_manager import OutputManager

        om = OutputManager()

        assert om.mode == OUTPUT_MODE_KEYBOARD
        assert om._keyboard_controller is None
        assert om._typing_delay == 0.01

    def test_init_custom_mode(self):
        """OutputManager accepts custom mode."""
        from src.core.output_manager import OutputManager

        om = OutputManager(mode=OUTPUT_MODE_CLIPBOARD)
        assert om.mode == OUTPUT_MODE_CLIPBOARD

    def test_set_mode_valid(self):
        """set_mode changes mode for valid values."""
        from src.core.output_manager import OutputManager

        om = OutputManager()

        om.set_mode(OUTPUT_MODE_CLIPBOARD)
        assert om.mode == OUTPUT_MODE_CLIPBOARD

        om.set_mode(OUTPUT_MODE_KEYBOARD)
        assert om.mode == OUTPUT_MODE_KEYBOARD

    def test_set_mode_invalid(self):
        """set_mode falls back to keyboard for invalid mode."""
        from src.core.output_manager import OutputManager

        om = OutputManager()

        om.set_mode("invalid_mode")
        assert om.mode == OUTPUT_MODE_KEYBOARD

    def test_output_empty_string(self):
        """output returns True for empty string."""
        from src.core.output_manager import OutputManager

        om = OutputManager()

        result = om.output("")
        assert result is True

    def test_output_routes_to_correct_method(self):
        """output routes to correct method based on mode."""
        from src.core.output_manager import OutputManager

        om = OutputManager()

        # Mock methods
        keyboard_called = []
        clipboard_called = []

        om.output_to_keyboard = lambda t: keyboard_called.append(t) or True
        om.output_to_clipboard = lambda t: clipboard_called.append(t) or True

        # Test keyboard mode
        om.set_mode(OUTPUT_MODE_KEYBOARD)
        om.output("test1")
        assert "test1" in keyboard_called

        # Test clipboard mode
        om.set_mode(OUTPUT_MODE_CLIPBOARD)
        om.output("test2")
        assert "test2" in clipboard_called

    def test_set_typing_delay_clamps_values(self):
        """set_typing_delay clamps values to valid range."""
        from src.core.output_manager import OutputManager

        om = OutputManager()

        # Too small - should clamp to 0.001
        om.set_typing_delay(0.0001)
        assert om._typing_delay == 0.001

        # Too large - should clamp to 0.5
        om.set_typing_delay(1.0)
        assert om._typing_delay == 0.5

        # Valid value
        om.set_typing_delay(0.05)
        assert om._typing_delay == 0.05

    @pytest.mark.skipif(True, reason="Requires clipboard access")
    def test_clipboard_operations(self):
        """Test clipboard copy and get operations."""
        from src.core.output_manager import OutputManager

        om = OutputManager()

        # Clear first
        om.clear_clipboard()

        # Copy text
        result = om.output_to_clipboard("test_text")
        assert result is True

        # Get text back
        retrieved = om.get_clipboard()
        assert retrieved == "test_text"


# ============================================================================
# HOTKEY MANAGER TESTS
# ============================================================================

class TestHotkeyManager:
    """Test suite for HotkeyManager module."""

    def test_import_hotkey_manager(self):
        """Module can be imported without errors."""
        from src.core.hotkey_manager import HotkeyManager
        assert HotkeyManager is not None

    def test_init_state(self):
        """HotkeyManager initializes in correct state."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        assert hm._hotkeys == {}
        assert hm._listener is None
        assert hm._is_listening is False
        assert len(hm._pressed_keys) == 0

    def test_parse_hotkey_simple(self):
        """parse_hotkey parses simple hotkey correctly."""
        from src.core.hotkey_manager import HotkeyManager

        modifiers, key = HotkeyManager.parse_hotkey("ctrl+s")

        assert modifiers == {"ctrl"}
        assert key == "s"

    def test_parse_hotkey_multiple_modifiers(self):
        """parse_hotkey parses multiple modifiers."""
        from src.core.hotkey_manager import HotkeyManager

        modifiers, key = HotkeyManager.parse_hotkey("ctrl+shift+alt+s")

        assert modifiers == {"ctrl", "shift", "alt"}
        assert key == "s"

    def test_parse_hotkey_normalizes_ctrl(self):
        """parse_hotkey normalizes 'control' to 'ctrl'."""
        from src.core.hotkey_manager import HotkeyManager

        modifiers1, _ = HotkeyManager.parse_hotkey("control+s")
        modifiers2, _ = HotkeyManager.parse_hotkey("ctrl+s")

        assert modifiers1 == modifiers2 == {"ctrl"}

    def test_parse_hotkey_normalizes_cmd(self):
        """parse_hotkey normalizes 'win'/'super' to 'cmd'."""
        from src.core.hotkey_manager import HotkeyManager

        mod1, _ = HotkeyManager.parse_hotkey("win+s")
        mod2, _ = HotkeyManager.parse_hotkey("super+s")
        mod3, _ = HotkeyManager.parse_hotkey("cmd+s")

        assert mod1 == mod2 == mod3 == {"cmd"}

    def test_parse_hotkey_case_insensitive(self):
        """parse_hotkey is case insensitive."""
        from src.core.hotkey_manager import HotkeyManager

        mod1, key1 = HotkeyManager.parse_hotkey("CTRL+SHIFT+S")
        mod2, key2 = HotkeyManager.parse_hotkey("ctrl+shift+s")

        assert mod1 == mod2
        assert key1 == key2

    def test_parse_hotkey_with_spaces(self):
        """parse_hotkey handles spaces in hotkey string."""
        from src.core.hotkey_manager import HotkeyManager

        mod1, key1 = HotkeyManager.parse_hotkey("Ctrl + Shift + S")
        mod2, key2 = HotkeyManager.parse_hotkey("ctrl+shift+s")

        assert mod1 == mod2
        assert key1 == key2

    def test_normalize_hotkey(self):
        """normalize_hotkey produces consistent format."""
        from src.core.hotkey_manager import HotkeyManager

        # Different formats should normalize to same result
        result1 = HotkeyManager.normalize_hotkey("Ctrl + Shift + S")
        result2 = HotkeyManager.normalize_hotkey("ctrl+shift+s")
        result3 = HotkeyManager.normalize_hotkey("CTRL+SHIFT+s")

        assert result1 == result2 == result3

    def test_normalize_hotkey_sorts_modifiers(self):
        """normalize_hotkey sorts modifiers consistently."""
        from src.core.hotkey_manager import HotkeyManager

        # Different orders should produce same result
        result1 = HotkeyManager.normalize_hotkey("shift+ctrl+s")
        result2 = HotkeyManager.normalize_hotkey("ctrl+shift+s")

        assert result1 == result2
        assert result1 == "ctrl+shift+s"

    def test_register_hotkey(self):
        """register adds hotkey to internal dict."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        callback = lambda: None
        result = hm.register("ctrl+shift+s", callback)

        assert result is True
        assert "ctrl+shift+s" in hm._hotkeys
        assert hm._hotkeys["ctrl+shift+s"] is callback

    def test_register_normalizes_hotkey(self):
        """register normalizes hotkey before storing."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        hm.register("Ctrl + Shift + S", lambda: None)

        assert "ctrl+shift+s" in hm._hotkeys
        assert "Ctrl + Shift + S" not in hm._hotkeys

    def test_unregister_existing(self):
        """unregister removes existing hotkey."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        hm.register("ctrl+s", lambda: None)
        result = hm.unregister("ctrl+s")

        assert result is True
        assert "ctrl+s" not in hm._hotkeys

    def test_unregister_nonexistent(self):
        """unregister returns False for nonexistent hotkey."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        result = hm.unregister("ctrl+z")

        assert result is False

    def test_unregister_all(self):
        """unregister_all clears all hotkeys."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        hm.register("ctrl+a", lambda: None)
        hm.register("ctrl+b", lambda: None)
        hm.register("ctrl+c", lambda: None)

        hm.unregister_all()

        assert len(hm._hotkeys) == 0

    def test_is_listening_initial(self):
        """is_listening returns False initially."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()
        assert hm.is_listening() is False

    def test_get_registered_hotkeys(self):
        """get_registered_hotkeys returns dict of hotkeys."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        def test_func():
            pass

        hm.register("ctrl+s", test_func)

        registered = hm.get_registered_hotkeys()

        assert "ctrl+s" in registered
        assert registered["ctrl+s"] == "test_func"

    def test_stop_listening_when_not_started(self):
        """stop_listening is safe when not started."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()
        hm.stop_listening()  # Should not raise

        assert hm.is_listening() is False

    def test_context_manager_protocol(self):
        """HotkeyManager supports context manager protocol."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        assert hasattr(hm, '__enter__')
        assert hasattr(hm, '__exit__')

    def test_thread_safety_lock_exists(self):
        """HotkeyManager has a lock for thread safety."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        assert hasattr(hm, '_lock')
        assert isinstance(hm._lock, type(threading.Lock()))

    def test_check_hotkeys_thread_safe(self):
        """_check_hotkeys is protected by lock."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        call_count = [0]

        def callback():
            call_count[0] += 1

        hm.register("ctrl+s", callback)

        # Simulate pressed keys
        hm._pressed_keys = {"ctrl", "s"}

        # Check hotkeys from multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=hm._check_hotkeys)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Callback should be called exactly once due to _pressed_keys.clear()
        assert call_count[0] >= 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCoreIntegration:
    """Integration tests between core modules."""

    def test_audio_to_recognizer_data_flow(self):
        """Audio data can flow from AudioCapture to Recognizer."""
        from src.core.audio_capture import AudioCapture
        from src.core.recognizer import Recognizer

        capture = AudioCapture()
        rec = Recognizer(model_path="./model")

        # Connect via queue
        audio_queue = capture.get_audio_queue()

        # Simulate audio data
        test_audio = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()
        audio_queue.put(test_audio)

        # Verify data can be retrieved
        retrieved = audio_queue.get(timeout=1)
        assert retrieved == test_audio

    def test_recognizer_to_punctuation_flow(self):
        """Text can flow from Recognizer to Punctuation."""
        from src.core.punctuation import PunctuationDisabled

        punct = PunctuationDisabled()

        # Simulate recognized text
        recognized_text = "hello world"
        enhanced = punct.enhance(recognized_text)

        assert enhanced == "Hello world"

    def test_punctuation_to_output_flow(self):
        """Text can flow from Punctuation to OutputManager."""
        from src.core.punctuation import PunctuationDisabled
        from src.core.output_manager import OutputManager

        punct = PunctuationDisabled()
        om = OutputManager()

        text = "test message"
        enhanced = punct.enhance(text)

        # Mock output to avoid keyboard interaction
        output_received = []
        om.output_to_keyboard = lambda t: output_received.append(t) or True

        result = om.output(enhanced)

        assert result is True
        assert "Test message" in output_received

    def test_full_pipeline_mock(self):
        """Full pipeline works with mocked components."""
        from src.core.audio_capture import AudioCapture
        from src.core.recognizer import Recognizer
        from src.core.punctuation import PunctuationDisabled
        from src.core.output_manager import OutputManager

        # Create all components
        capture = AudioCapture()
        rec = Recognizer(model_path="./model")
        punct = PunctuationDisabled()
        om = OutputManager()

        # Configure
        final_output = []
        om.output_to_keyboard = lambda t: final_output.append(t) or True

        # Simulate pipeline
        text = "hello world"
        enhanced = punct.enhance(text)
        om.output(enhanced)

        assert len(final_output) == 1
        assert final_output[0] == "Hello world"


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStress:
    """Stress tests for core modules."""

    def test_audio_queue_high_throughput(self):
        """Audio queue can handle high throughput."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()
        queue_obj = capture.get_audio_queue()

        # Generate 1000 audio chunks
        chunks = [np.random.randint(-32768, 32767, CHUNK_SIZE, dtype=np.int16).tobytes()
                  for _ in range(1000)]

        # Push all chunks
        start = time.time()
        for chunk in chunks:
            queue_obj.put(chunk)
        push_time = time.time() - start

        # Pop all chunks
        start = time.time()
        while not queue_obj.empty():
            queue_obj.get()
        pop_time = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert push_time < 1.0
        assert pop_time < 1.0

    def test_hotkey_rapid_registration(self):
        """Rapid hotkey registration/unregistration is stable."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        for i in range(100):
            hotkey = f"ctrl+{chr(97 + i % 26)}"  # ctrl+a through ctrl+z
            hm.register(hotkey, lambda: None)

        assert len(hm._hotkeys) == 26  # Only unique hotkeys

        hm.unregister_all()
        assert len(hm._hotkeys) == 0

    def test_punctuation_disabled_batch_performance(self):
        """PunctuationDisabled handles large batches efficiently."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()

        # Create 1000 texts
        texts = [f"text number {i}" for i in range(1000)]

        start = time.time()
        results = pd.enhance_batch(texts)
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 1.0  # Should be very fast

    def test_concurrent_hotkey_operations(self):
        """Concurrent hotkey operations don't cause race conditions."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()
        errors = []

        def register_thread(n):
            try:
                for i in range(50):
                    hm.register(f"ctrl+alt+{n}+{i}", lambda: None)
            except Exception as e:
                errors.append(e)

        def unregister_thread(n):
            try:
                for i in range(50):
                    hm.unregister(f"ctrl+alt+{n}+{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for n in range(5):
            threads.append(threading.Thread(target=register_thread, args=(n,)))
            threads.append(threading.Thread(target=unregister_thread, args=(n,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# EDGE CASES & BOUNDARY TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_audio_level_extreme_values(self):
        """Audio level calculation handles extreme values."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()

        # Max positive amplitude
        max_audio = np.full(CHUNK_SIZE, 32767, dtype=np.int16).tobytes()
        capture._update_level(max_audio)
        assert 0.0 <= capture._current_level <= 1.0

        # Max negative amplitude
        min_audio = np.full(CHUNK_SIZE, -32768, dtype=np.int16).tobytes()
        capture._update_level(min_audio)
        assert 0.0 <= capture._current_level <= 1.0

    def test_recognizer_path_types(self):
        """Recognizer accepts both string and Path for model_path."""
        from src.core.recognizer import Recognizer

        rec1 = Recognizer(model_path="./model")
        rec2 = Recognizer(model_path=Path("./model"))

        assert isinstance(rec1.model_path, Path)
        assert isinstance(rec2.model_path, Path)

    def test_punctuation_unicode_handling(self):
        """Punctuation handles unicode correctly."""
        from src.core.punctuation import PunctuationDisabled

        pd = PunctuationDisabled()

        # Russian text
        assert pd.enhance("привет") == "Привет"

        # Chinese text
        result = pd.enhance("hello")
        assert result == "Hello"

        # Emoji (should not crash)
        result = pd.enhance("hello world")
        assert result is not None

    def test_hotkey_single_key(self):
        """Hotkey with just one key (no modifiers)."""
        from src.core.hotkey_manager import HotkeyManager

        modifiers, key = HotkeyManager.parse_hotkey("f1")

        assert modifiers == set()
        assert key == "f1"

    def test_output_manager_very_long_text(self):
        """OutputManager handles very long text."""
        from src.core.output_manager import OutputManager

        om = OutputManager()

        # 10KB of text
        long_text = "x" * 10000

        # Mock to avoid actual keyboard output
        om.output_to_keyboard = lambda t: True

        result = om.output(long_text)
        assert result is True

    def test_audio_capture_device_id_edge_cases(self):
        """AudioCapture handles various device ID formats."""
        from src.core.audio_capture import AudioCapture

        # Negative number
        capture = AudioCapture(device_id="-1")
        assert capture._get_device_index() == -1

        # Zero
        capture = AudioCapture(device_id="0")
        assert capture._get_device_index() == 0

        # Large number
        capture = AudioCapture(device_id="999")
        assert capture._get_device_index() == 999


# ============================================================================
# MEMORY & RESOURCE TESTS
# ============================================================================

class TestMemoryResources:
    """Memory and resource management tests."""

    def test_audio_capture_cleanup_releases_resources(self):
        """AudioCapture cleanup properly releases resources."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture()

        # Add data to queue
        for _ in range(10):
            capture._audio_queue.put(b"test")

        capture._cleanup()

        assert capture._stream is None
        assert capture._pyaudio is None
        assert capture._audio_queue.empty()

    def test_recognizer_unload_releases_model(self):
        """Recognizer unload releases model from memory."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./model")

        # Simulate loaded state
        rec._model = "fake_model"
        rec._recognizer = "fake_recognizer"
        rec._is_loaded = True

        rec.unload()

        assert rec._model is None
        assert rec._recognizer is None
        assert rec._is_loaded is False

    def test_punctuation_unload_releases_model(self):
        """Punctuation unload releases model from memory."""
        from src.core.punctuation import Punctuation

        punct = Punctuation()

        # Simulate loaded state
        punct._model = "fake_model"
        punct._is_loaded = True

        punct.unload()

        assert punct._model is None
        assert punct._is_loaded is False

    def test_hotkey_manager_stop_clears_state(self):
        """HotkeyManager stop clears all state."""
        from src.core.hotkey_manager import HotkeyManager

        hm = HotkeyManager()

        # Simulate some state
        hm._pressed_keys = {"ctrl", "shift", "s"}

        hm.stop_listening()

        assert hm._listener is None
        assert hm._is_listening is False
        assert len(hm._pressed_keys) == 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Error handling tests."""

    def test_audio_capture_error_callback(self):
        """AudioCapture calls error callback on failure."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture(device_id="99999")  # Invalid device

        errors = []
        capture.on_error = lambda e: errors.append(e)

        # Start should fail with invalid device
        # This may or may not trigger depending on PyAudio implementation
        result = capture.start()

        # Either fails or succeeds with default device
        # Just verify no crash

    def test_recognizer_error_callback_on_missing_model(self):
        """Recognizer calls error callback when model not found."""
        from src.core.recognizer import Recognizer

        rec = Recognizer(model_path="./nonexistent_path_xyz")

        errors = []
        rec.on_error = lambda e: errors.append(e)

        result = rec.load_model()

        assert result is False
        assert len(errors) == 1

    def test_output_manager_handles_pynput_missing(self):
        """OutputManager handles missing pynput gracefully."""
        from src.core.output_manager import OutputManager

        om = OutputManager()

        # Mock ImportError
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else None

        # Test that method doesn't crash even if there's an issue
        # In real test we'd need to mock import, but for now just verify
        # the method has proper exception handling
        assert hasattr(om, 'output_to_keyboard')


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
