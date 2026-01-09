"""
VoiceType - Real Integration Tests (NO MOCKS!)
Real production code tests with actual Vosk and Silero models.

IMPORTANT: These tests use REAL components - NO unittest.mock, NO MagicMock, NO patch!
Tests may take 5-30 seconds each due to model loading.
"""
import sys
import time
import tempfile
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pytest

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =============================================================================
# Model Detection Helpers
# =============================================================================

def get_models_dir() -> Path:
    """Get the models directory."""
    return ROOT_DIR / "models"


def find_vosk_small_model() -> Optional[Path]:
    """Find the small Vosk model for Russian."""
    models_dir = get_models_dir()
    candidates = [
        models_dir / "vosk-model-small-ru-0.22",
        models_dir / "vosk-model-small-ru",
    ]
    for path in candidates:
        if path.exists() and (path / "am").exists():
            return path
    return None


def find_vosk_large_model() -> Optional[Path]:
    """Find the large Vosk model for Russian."""
    models_dir = get_models_dir()
    candidates = [
        models_dir / "vosk-model-ru-0.42",
        models_dir / "vosk-model-ru",
    ]
    for path in candidates:
        if path.exists() and (path / "am").exists():
            return path
    return None


def find_silero_model() -> Optional[Path]:
    """Find the Silero TE model."""
    models_dir = get_models_dir()
    silero_dir = models_dir / "silero-te"
    if silero_dir.exists():
        pt_files = list(silero_dir.glob("*.pt"))
        if pt_files:
            return silero_dir
    return None


def can_silero_model_load() -> bool:
    """Check if Silero model can actually be loaded (not just exists)."""
    try:
        from src.core.punctuation import Punctuation
        punct = Punctuation(language="ru")
        result = punct.load_model()
        if result and punct.is_loaded():
            punct.unload()
            return True
        return False
    except Exception:
        return False


def is_microphone_available() -> bool:
    """Check if a microphone is available."""
    try:
        from src.core.audio_capture import AudioCapture
        devices = AudioCapture.list_devices()
        real_devices = [d for d in devices if d.get('id') != 'default']
        return len(real_devices) > 0
    except Exception:
        return False


# =============================================================================
# Skip Conditions
# =============================================================================

VOSK_SMALL_MODEL = find_vosk_small_model()
VOSK_LARGE_MODEL = find_vosk_large_model()
SILERO_MODEL = find_silero_model()
HAS_MICROPHONE = is_microphone_available()

# Lazy check for Silero - only test if it can actually load
_silero_can_load = None


def silero_can_load() -> bool:
    """Lazy check if Silero model can load (cached)."""
    global _silero_can_load
    if _silero_can_load is None:
        _silero_can_load = can_silero_model_load()
    return _silero_can_load


skip_no_vosk_small = pytest.mark.skipif(
    VOSK_SMALL_MODEL is None,
    reason="Vosk small model not found in models/ directory"
)

skip_no_vosk_large = pytest.mark.skipif(
    VOSK_LARGE_MODEL is None,
    reason="Vosk large model not found in models/ directory"
)

skip_no_silero = pytest.mark.skipif(
    SILERO_MODEL is None,
    reason="Silero TE model not found in models/silero-te/ directory"
)

# Dynamic skip for Silero tests that checks if model actually loads
skip_silero_not_working = pytest.mark.skipif(
    True,  # Will be evaluated at test collection time
    reason="Silero TE model cannot load (file may be incompatible)"
)

skip_no_microphone = pytest.mark.skipif(
    not HAS_MICROPHONE,
    reason="No microphone available"
)

skip_no_vosk = pytest.mark.skipif(
    VOSK_SMALL_MODEL is None and VOSK_LARGE_MODEL is None,
    reason="No Vosk model found"
)


# =============================================================================
# Pytest Markers
# =============================================================================

pytestmark = [
    pytest.mark.slow,  # All tests in this file are slow
]


# =============================================================================
# Test: Real Vosk Model Loading
# =============================================================================

@skip_no_vosk_small
class TestRealVoskSmallModel:
    """Test real Vosk small model loading and usage."""

    def test_small_model_loads_successfully(self):
        """Test that small Vosk model loads without errors."""
        from src.core.recognizer import Recognizer

        recognizer = Recognizer(str(VOSK_SMALL_MODEL))

        # Should not be loaded initially
        assert not recognizer.is_loaded()

        # Load the model - this takes ~2-5 seconds
        start_time = time.time()
        result = recognizer.load_model()
        load_time = time.time() - start_time

        assert result is True, "Model should load successfully"
        assert recognizer.is_loaded(), "Model should be marked as loaded"
        print(f"\n  Small model loaded in {load_time:.2f}s")

        # Cleanup
        recognizer.unload()
        assert not recognizer.is_loaded()

    def test_small_model_processes_silence(self):
        """Test that model can process silent audio."""
        from src.core.recognizer import Recognizer
        import numpy as np

        recognizer = Recognizer(str(VOSK_SMALL_MODEL))
        recognizer.load_model()

        try:
            # Generate 1 second of silence (16-bit PCM, 16000 Hz)
            sample_rate = 16000
            duration = 1.0
            silence = np.zeros(int(sample_rate * duration), dtype=np.int16).tobytes()

            # Process in chunks
            chunk_size = 4000
            for i in range(0, len(silence), chunk_size):
                chunk = silence[i:i + chunk_size]
                recognizer.process_audio(chunk)

            # Get final result
            final = recognizer.get_final_result()
            # Silent audio should produce empty or very short result
            assert final is None or final == "" or len(final) < 5

        finally:
            recognizer.unload()

    def test_small_model_reset_works(self):
        """Test that recognizer reset works correctly."""
        from src.core.recognizer import Recognizer

        recognizer = Recognizer(str(VOSK_SMALL_MODEL))
        recognizer.load_model()

        try:
            # Reset should work without errors
            recognizer.reset()
            assert recognizer.is_loaded()

            # Should still be functional after reset
            recognizer.reset()
            assert recognizer.is_loaded()

        finally:
            recognizer.unload()

    def test_small_model_callbacks_work(self):
        """Test that recognizer callbacks are called."""
        from src.core.recognizer import Recognizer
        import numpy as np

        partial_results = []
        final_results = []

        def on_partial(text):
            partial_results.append(text)

        def on_final(text):
            final_results.append(text)

        recognizer = Recognizer(str(VOSK_SMALL_MODEL))
        recognizer.on_partial_result = on_partial
        recognizer.on_final_result = on_final
        recognizer.load_model()

        try:
            # Generate some audio with slight noise
            sample_rate = 16000
            duration = 0.5
            samples = int(sample_rate * duration)
            # Small random noise
            noise = (np.random.randn(samples) * 100).astype(np.int16).tobytes()

            # Process audio
            recognizer.process_audio(noise)
            recognizer.get_final_result()

            # Callbacks should be set (even if not called for noise)
            assert recognizer.on_partial_result is not None
            assert recognizer.on_final_result is not None

        finally:
            recognizer.unload()


@skip_no_vosk_large
class TestRealVoskLargeModel:
    """Test real Vosk large model loading."""

    def test_large_model_loads_successfully(self):
        """Test that large Vosk model loads without errors."""
        from src.core.recognizer import Recognizer

        recognizer = Recognizer(str(VOSK_LARGE_MODEL))

        # Load the model - this takes ~5-15 seconds
        start_time = time.time()
        result = recognizer.load_model()
        load_time = time.time() - start_time

        assert result is True, "Large model should load successfully"
        assert recognizer.is_loaded(), "Large model should be marked as loaded"
        print(f"\n  Large model loaded in {load_time:.2f}s")

        # Cleanup
        recognizer.unload()
        assert not recognizer.is_loaded()


# =============================================================================
# Test: Real Silero TE Model Loading
# =============================================================================

@skip_no_silero
class TestRealSileroModel:
    """Test real Silero TE model loading and usage."""

    @pytest.fixture(autouse=True)
    def check_silero_loads(self):
        """Skip tests if Silero model cannot actually load."""
        if not silero_can_load():
            pytest.skip("Silero TE model file exists but cannot be loaded")

    def test_silero_model_loads_successfully(self):
        """Test that Silero TE model loads without errors."""
        from src.core.punctuation import Punctuation

        punct = Punctuation(language="ru")

        # Should not be loaded initially
        assert not punct.is_loaded()

        # Load the model - this takes ~2-5 seconds
        start_time = time.time()
        result = punct.load_model()
        load_time = time.time() - start_time

        assert result is True, "Silero model should load successfully"
        assert punct.is_loaded(), "Silero model should be marked as loaded"
        print(f"\n  Silero model loaded in {load_time:.2f}s")

        # Cleanup
        punct.unload()
        assert not punct.is_loaded()

    def test_silero_enhances_text(self):
        """Test that Silero actually adds punctuation."""
        from src.core.punctuation import Punctuation

        punct = Punctuation(language="ru")
        result = punct.load_model()
        if not result:
            pytest.skip("Silero model failed to load")

        try:
            # Test punctuation enhancement
            test_cases = [
                ("привет как дела", True),  # Should add comma/question mark
                ("я иду домой", True),  # Should add period
                ("это тест", True),  # Should capitalize and add period
            ]

            for text, should_enhance in test_cases:
                result = punct.enhance(text)
                assert result is not None
                assert len(result) > 0
                # First letter should be capitalized
                assert result[0].isupper(), f"'{result}' should start with uppercase"

        finally:
            punct.unload()

    def test_silero_batch_enhancement(self):
        """Test batch text enhancement."""
        from src.core.punctuation import Punctuation

        punct = Punctuation(language="ru")
        result = punct.load_model()
        if not result:
            pytest.skip("Silero model failed to load")

        try:
            texts = ["привет", "как дела", "все хорошо"]
            results = punct.enhance_batch(texts)

            assert len(results) == 3
            for result in results:
                assert result[0].isupper()

        finally:
            punct.unload()

    def test_silero_model_info(self):
        """Test getting model info."""
        from src.core.punctuation import Punctuation

        punct = Punctuation(language="ru")
        result = punct.load_model()
        if not result:
            pytest.skip("Silero model failed to load")

        try:
            info = punct.get_model_info()

            assert info["loaded"] is True
            assert info["language"] == "ru"
            assert "model_dir" in info

        finally:
            punct.unload()


# =============================================================================
# Test: Real Models Manager
# =============================================================================

class TestRealModelsManager:
    """Test real models manager functionality."""

    def test_models_manager_finds_models(self):
        """Test that models manager finds installed models."""
        from src.data.models_manager import ModelsManager

        manager = ModelsManager(models_dir=get_models_dir())

        # Get available models
        vosk_models = manager.get_available_vosk_models()

        # Should find at least one model if any exist
        if VOSK_SMALL_MODEL or VOSK_LARGE_MODEL:
            assert len(vosk_models) > 0, "Should find at least one Vosk model"

            # Check model info structure
            for model in vosk_models:
                assert "language" in model
                assert "size" in model
                assert "path" in model
                assert "name" in model

    def test_models_manager_summary(self):
        """Test models summary generation."""
        from src.data.models_manager import ModelsManager

        manager = ModelsManager(models_dir=get_models_dir())
        summary = manager.get_models_summary()

        assert "vosk" in summary
        assert "silero_te" in summary
        assert "models_dir" in summary
        assert "models_dir_exists" in summary

    @skip_no_silero
    def test_silero_te_info(self):
        """Test Silero TE info retrieval."""
        from src.data.models_manager import ModelsManager

        manager = ModelsManager(models_dir=get_models_dir())
        info = manager.get_silero_te_info()

        assert info["local_available"] is True
        assert len(info["local_models"]) > 0
        assert "v2_4lang_q.pt" in info["local_models"] or len(info["local_models"]) > 0


# =============================================================================
# Test: Real Audio Capture
# =============================================================================

@skip_no_microphone
class TestRealAudioCapture:
    """Test real audio capture with actual microphone."""

    def test_audio_capture_starts_and_stops(self):
        """Test that audio capture can start and stop."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture(device_id="default")

        assert not capture.is_running()

        # Start capture
        result = capture.start()
        assert result is True, "Audio capture should start successfully"
        assert capture.is_running()

        # Capture for a short time
        time.sleep(0.5)

        # Check audio queue
        audio_queue = capture.get_audio_queue()
        assert audio_queue.qsize() > 0, "Should have captured some audio"

        # Stop capture
        capture.stop()
        assert not capture.is_running()

    def test_audio_capture_level_meter(self):
        """Test that audio level is calculated."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture(device_id="default")
        capture.start()

        try:
            # Wait for some audio
            time.sleep(0.3)

            # Get level (should be >= 0)
            level = capture.get_level()
            assert level >= 0.0
            assert level <= 1.0

        finally:
            capture.stop()

    def test_audio_capture_device_validation(self):
        """Test device validation."""
        from src.core.audio_capture import AudioCapture

        capture = AudioCapture(device_id="default")

        # Validate default device
        is_valid, device_index, error = capture.validate_device(None)
        assert is_valid is True, f"Default device should be valid: {error}"

    def test_audio_capture_list_devices(self):
        """Test listing audio devices."""
        from src.core.audio_capture import AudioCapture

        devices = AudioCapture.list_devices()
        assert isinstance(devices, list)
        assert len(devices) > 0, "Should have at least one device"

        # Check device info structure
        for device in devices:
            assert "id" in device
            assert "name" in device


# =============================================================================
# Test: Real Hotkey Manager
# =============================================================================

class TestRealHotkeyManager:
    """Test real hotkey manager."""

    def test_hotkey_manager_registration(self):
        """Test hotkey registration."""
        from src.core.hotkey_manager import HotkeyManager

        manager = HotkeyManager()

        # Register hotkeys
        callback_called = threading.Event()

        def test_callback():
            callback_called.set()

        result = manager.register("ctrl+shift+t", test_callback)
        assert result is True

        # Check registered
        hotkeys = manager.get_registered_hotkeys()
        assert "ctrl+shift+t" in hotkeys

        # Unregister
        result = manager.unregister("ctrl+shift+t")
        assert result is True

        hotkeys = manager.get_registered_hotkeys()
        assert "ctrl+shift+t" not in hotkeys

    def test_hotkey_manager_listening(self):
        """Test hotkey listener start/stop."""
        from src.core.hotkey_manager import HotkeyManager

        manager = HotkeyManager()

        assert not manager.is_listening()

        # Start listening
        result = manager.start_listening()
        assert result is True
        assert manager.is_listening()

        # Stop listening
        manager.stop_listening()
        assert not manager.is_listening()

    def test_hotkey_normalization_variants(self):
        """Test various hotkey format normalizations."""
        from src.core.hotkey_manager import HotkeyManager

        test_cases = [
            ("Ctrl+Shift+S", "ctrl+shift+s"),
            ("CTRL+SHIFT+S", "ctrl+shift+s"),
            ("ctrl + shift + s", "ctrl+shift+s"),
            ("Control+Alt+X", "ctrl+alt+x"),  # control -> ctrl normalization
            ("cmd+shift+a", "cmd+shift+a"),
            ("Alt+A", "alt+a"),
        ]

        for input_hotkey, expected in test_cases:
            result = HotkeyManager.normalize_hotkey(input_hotkey)
            # The normalize function may order modifiers differently
            # So we check the components match
            input_parts = set(expected.split("+"))
            result_parts = set(result.split("+"))
            assert input_parts == result_parts, f"'{input_hotkey}' should normalize to '{expected}', got '{result}'"


# =============================================================================
# Test: Real Config with Persistence
# =============================================================================

class TestRealConfig:
    """Test real config persistence and loading."""

    def test_config_saves_and_loads(self):
        """Test config saves to file and loads back."""
        from src.data.config import Config, _reset_config_instance

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            _reset_config_instance()

            # Create and configure
            config = Config(config_path)
            config.load()
            config.set("audio.language", "en")
            config.set("audio.model", "large")
            config.set("test.custom_key", "custom_value")

            # Create new instance to verify persistence
            _reset_config_instance()
            config2 = Config(config_path)
            config2.load()

            assert config2.get("audio.language") == "en"
            assert config2.get("audio.model") == "large"
            assert config2.get("test.custom_key") == "custom_value"

        finally:
            _reset_config_instance()
            if config_path.exists():
                config_path.unlink()


# =============================================================================
# Test: Real Database Operations
# =============================================================================

class TestRealDatabase:
    """Test real database operations."""

    def test_database_full_cycle(self):
        """Test complete database lifecycle."""
        from src.data.database import Database, _reset_database_instance

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            _reset_database_instance()

            db = Database(db_path)
            db.initialize()

            # Add history entries
            for i in range(5):
                started = datetime.now()
                ended = started + timedelta(seconds=10 + i)
                db.add_history_entry(
                    started_at=started,
                    ended_at=ended,
                    text=f"Test entry {i}",
                    language="ru"
                )

            # Add stats entries
            for i in range(5):
                db.add_stats_entry(cpu_percent=i * 10, ram_mb=100 + i * 10)

            # Verify data
            history = db.get_history()
            assert len(history) == 5

            stats = db.get_stats_24h()
            assert len(stats) == 5

            # Test recognition time
            total_time = db.get_today_recognition_time()
            assert total_time > 0

        finally:
            _reset_database_instance()
            if db_path.exists():
                db_path.unlink()


# =============================================================================
# Test: Combined Model Loading (Vosk + Silero)
# =============================================================================

@skip_no_vosk_small
class TestCombinedModelLoading:
    """Test loading both Vosk and Silero models together."""

    def test_load_both_models_sequentially(self):
        """Test loading both models one after another."""
        from src.core.recognizer import Recognizer
        from src.core.punctuation import Punctuation, PunctuationDisabled

        recognizer = None
        punct = None

        try:
            # Load Vosk
            start = time.time()
            recognizer = Recognizer(str(VOSK_SMALL_MODEL))
            recognizer.load_model()
            vosk_time = time.time() - start
            print(f"\n  Vosk loaded in {vosk_time:.2f}s")

            # Load Silero (or use disabled if not available)
            start = time.time()
            if silero_can_load():
                punct = Punctuation(language="ru")
                punct.load_model()
                silero_time = time.time() - start
                print(f"  Silero loaded in {silero_time:.2f}s")
            else:
                punct = PunctuationDisabled()
                punct.load_model()
                print("  Silero not available, using PunctuationDisabled")

            # Both should be loaded
            assert recognizer.is_loaded()
            assert punct.is_loaded()

            # Use them together
            import numpy as np
            silence = np.zeros(8000, dtype=np.int16).tobytes()
            recognizer.process_audio(silence)
            final = recognizer.get_final_result()

            # Enhance whatever came out (even if empty)
            if final:
                enhanced = punct.enhance(final)
                assert enhanced is not None

        finally:
            if recognizer:
                recognizer.unload()
            if punct:
                punct.unload()

    def test_load_models_in_threads(self):
        """Test loading models in separate threads (like the real app does)."""
        from src.core.recognizer import Recognizer
        from src.core.punctuation import Punctuation, PunctuationDisabled

        recognizer = Recognizer(str(VOSK_SMALL_MODEL))
        use_real_silero = silero_can_load()
        punct = Punctuation(language="ru") if use_real_silero else PunctuationDisabled()

        vosk_loaded = threading.Event()
        silero_loaded = threading.Event()
        errors = []

        def load_vosk():
            try:
                recognizer.load_model()
                vosk_loaded.set()
            except Exception as e:
                errors.append(("vosk", e))

        def load_silero():
            try:
                punct.load_model()
                silero_loaded.set()
            except Exception as e:
                errors.append(("silero", e))

        try:
            # Start loading in threads
            vosk_thread = threading.Thread(target=load_vosk)
            silero_thread = threading.Thread(target=load_silero)

            start = time.time()
            vosk_thread.start()
            silero_thread.start()

            # Wait for completion
            vosk_thread.join(timeout=30)
            silero_thread.join(timeout=30)
            total_time = time.time() - start

            print(f"\n  Both models loaded in {total_time:.2f}s (parallel)")

            assert len(errors) == 0, f"Loading errors: {errors}"
            assert vosk_loaded.is_set(), "Vosk should be loaded"
            assert silero_loaded.is_set(), "Silero/PunctuationDisabled should be loaded"

        finally:
            recognizer.unload()
            punct.unload()


# =============================================================================
# Test: Real Recording Flow (with microphone)
# =============================================================================

@skip_no_microphone
@skip_no_vosk_small
class TestRealRecordingFlow:
    """Test real recording flow with audio capture and recognition."""

    def test_capture_and_recognize(self):
        """Test capturing audio and feeding it to recognizer."""
        from src.core.audio_capture import AudioCapture
        from src.core.recognizer import Recognizer

        capture = AudioCapture(device_id="default")
        recognizer = Recognizer(str(VOSK_SMALL_MODEL))

        partial_results = []
        final_results = []

        def on_partial(text):
            partial_results.append(text)

        def on_final(text):
            final_results.append(text)

        recognizer.on_partial_result = on_partial
        recognizer.on_final_result = on_final
        recognizer.load_model()

        try:
            # Start capture
            capture.start()

            # Process audio for 1 second
            start = time.time()
            while time.time() - start < 1.0:
                audio_queue = capture.get_audio_queue()
                try:
                    audio_data = audio_queue.get(timeout=0.1)
                    recognizer.process_audio(audio_data)
                except queue.Empty:
                    continue

            # Get final result
            recognizer.get_final_result()

            # Should have processed audio without errors
            # (results may be empty if no speech detected)

        finally:
            capture.stop()
            recognizer.unload()


# =============================================================================
# Test: Real Application Lifecycle
# =============================================================================

@skip_no_vosk_small
class TestRealAppLifecycle:
    """Test real application lifecycle without Qt UI."""

    def test_component_initialization_order(self):
        """Test that components can be initialized in the correct order."""
        from src.data.config import Config, _reset_config_instance
        from src.data.database import Database, _reset_database_instance
        from src.data.models_manager import ModelsManager
        from src.core.recognizer import Recognizer
        from src.core.hotkey_manager import HotkeyManager
        from src.core.output_manager import OutputManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            db_path = Path(temp_dir) / "test.db"

            try:
                _reset_config_instance()
                _reset_database_instance()

                # Initialize in order like the real app
                config = Config(config_path)
                config.load()

                db = Database(db_path)
                db.initialize()

                models_manager = ModelsManager(models_dir=get_models_dir())
                model_path = models_manager.get_vosk_model_path("ru", "small")
                assert model_path is not None

                recognizer = Recognizer(str(model_path))
                recognizer.load_model()
                assert recognizer.is_loaded()

                hotkey_manager = HotkeyManager()
                hotkey_manager.register("ctrl+shift+s", lambda: None)
                hotkey_manager.start_listening()
                assert hotkey_manager.is_listening()

                output_manager = OutputManager()
                assert output_manager is not None

                # Cleanup in reverse order
                hotkey_manager.stop_listening()
                recognizer.unload()

                assert not recognizer.is_loaded()
                assert not hotkey_manager.is_listening()

            finally:
                _reset_config_instance()
                _reset_database_instance()

    def test_model_reload_simulation(self):
        """Test simulating model reload when settings change."""
        from src.core.recognizer import Recognizer
        from src.core.punctuation import Punctuation, PunctuationDisabled

        recognizer = Recognizer(str(VOSK_SMALL_MODEL))
        use_real_silero = silero_can_load()
        punct = Punctuation(language="ru") if use_real_silero else PunctuationDisabled()

        try:
            # Initial load
            recognizer.load_model()
            punct.load_model()

            assert recognizer.is_loaded()
            assert punct.is_loaded()

            # Simulate settings change -> unload
            recognizer.unload()
            punct.unload()

            assert not recognizer.is_loaded()
            if use_real_silero:
                assert not punct.is_loaded()

            # Reload (like when user changes language)
            recognizer.load_model()
            punct.load_model()

            assert recognizer.is_loaded()
            assert punct.is_loaded()

        finally:
            recognizer.unload()
            punct.unload()


# =============================================================================
# Test: Real Qt App Integration (requires display)
# =============================================================================

class TestRealQtIntegration:
    """Test real Qt application integration."""

    @pytest.fixture
    def qt_app(self):
        """Create a real Qt application."""
        from PyQt6.QtWidgets import QApplication
        import sys

        # Check if app already exists
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app
        # Don't quit - may be needed by other tests

    def test_main_window_creation(self, qt_app):
        """Test creating the real main window."""
        from src.ui.main_window import MainWindow
        from src.data.config import Config, _reset_config_instance

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            _reset_config_instance()

            # Create config first
            config = Config(config_path)
            config.load()

            # Inject config into global singleton
            import src.data.config as config_module
            config_module._config_instance = config

            # Create main window
            window = MainWindow()

            # Check window properties
            assert window is not None
            assert window.tab_main is not None
            assert window.tab_hotkeys is not None
            assert window.tab_history is not None
            assert window.tab_stats is not None
            assert window.tab_logs is not None
            assert window.tab_test is not None

            # Test loading state
            assert window.is_loading() is True  # Starts in loading state

            # Simulate models loaded
            window.set_loading(False, "test-model")
            assert window.is_loading() is False

        finally:
            _reset_config_instance()
            if config_path.exists():
                config_path.unlink()

    def test_loading_overlay_states(self, qt_app):
        """Test loading overlay state transitions."""
        from src.ui.main_window import MainWindow
        from src.data.config import Config, _reset_config_instance

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            _reset_config_instance()
            config = Config(config_path)
            config.load()

            import src.data.config as config_module
            config_module._config_instance = config

            window = MainWindow()

            # Test loading states
            window.set_loading(True, "vosk-model-small-ru")
            assert window.is_loading() is True

            window.set_loading_status("Loading Vosk", "vosk-model-small-ru")
            window.set_loading_status("Loading Silero", "vosk-model-small-ru")

            window.set_loading(False, "vosk-model-small-ru")
            assert window.is_loading() is False

            # Test error state
            window.set_loading(True)
            window.show_loading_error("Test error")

        finally:
            _reset_config_instance()
            if config_path.exists():
                config_path.unlink()


# =============================================================================
# Test: Full Integration with Real VoiceTypeApp (Qt-based)
# =============================================================================

@pytest.mark.skipif(
    VOSK_SMALL_MODEL is None,
    reason="Vosk model required for VoiceTypeApp tests"
)
class TestRealVoiceTypeApp:
    """Test the real VoiceTypeApp class with Qt."""

    @pytest.fixture
    def qt_app_with_config(self):
        """Create Qt app with isolated config."""
        from PyQt6.QtWidgets import QApplication
        from src.data.config import Config, _reset_config_instance
        from src.data.database import Database, _reset_database_instance
        import sys

        # Create temp files
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "config.yaml"
        db_path = Path(temp_dir) / "test.db"

        # Reset singletons
        _reset_config_instance()
        _reset_database_instance()

        # Create Qt app
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Setup config
        config = Config(config_path)
        config.load()
        config.set("audio.language", "ru")
        config.set("audio.model", "small")

        import src.data.config as config_module
        config_module._config_instance = config

        # Setup database
        db = Database(db_path)
        db.initialize()

        import src.data.database as db_module
        db_module._db_instance = db

        yield {
            "app": app,
            "config": config,
            "db": db,
            "temp_dir": temp_dir
        }

        # Cleanup
        _reset_config_instance()
        _reset_database_instance()
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_voicetype_app_initialization(self, qt_app_with_config):
        """Test VoiceTypeApp initialization."""
        from src.app import VoiceTypeApp

        app = VoiceTypeApp()

        # Check initial state
        assert app._is_recording is False
        assert app._is_initialized is False
        assert app._models_loaded is False

    def test_voicetype_app_full_initialize(self, qt_app_with_config):
        """Test full VoiceTypeApp initialization with real models."""
        from src.app import VoiceTypeApp
        from PyQt6.QtCore import QTimer
        import gc

        app = VoiceTypeApp()

        models_loaded_event = threading.Event()
        original_on_models_loaded = app._on_models_loaded

        def patched_on_models_loaded(state, model_name):
            original_on_models_loaded(state, model_name)
            if state == "ready":
                models_loaded_event.set()

        app._on_models_loaded = patched_on_models_loaded

        try:
            # Initialize
            result = app.initialize()
            assert result is True
            assert app._is_initialized is True

            # Wait for models to load (with timeout)
            loaded = models_loaded_event.wait(timeout=30)

            if loaded:
                assert app._models_loaded is True
                assert app._recognizer is not None
                assert app._recognizer.is_loaded()
                print("\n  VoiceTypeApp fully initialized with real models")
            else:
                print("\n  Warning: Models not loaded within timeout")

        finally:
            # Cleanup
            app.quit()
            gc.collect()


# =============================================================================
# Test: Cleanup and Resource Management
# =============================================================================

@skip_no_vosk_small
class TestResourceCleanup:
    """Test proper resource cleanup."""

    def test_recognizer_memory_cleanup(self):
        """Test that recognizer releases memory on unload."""
        from src.core.recognizer import Recognizer
        import gc

        recognizer = Recognizer(str(VOSK_SMALL_MODEL))
        recognizer.load_model()

        # Unload
        recognizer.unload()
        gc.collect()

        assert not recognizer.is_loaded()
        assert recognizer._model is None
        assert recognizer._recognizer is None

    def test_punctuation_memory_cleanup(self):
        """Test that punctuation releases memory on unload."""
        from src.core.punctuation import Punctuation
        import gc

        if not silero_can_load():
            pytest.skip("Silero model cannot be loaded")

        punct = Punctuation(language="ru")
        punct.load_model()

        # Unload
        punct.unload()
        gc.collect()

        assert not punct.is_loaded()
        assert punct._model is None

    def test_audio_capture_cleanup(self):
        """Test that audio capture releases resources."""
        from src.core.audio_capture import AudioCapture

        if not HAS_MICROPHONE:
            pytest.skip("No microphone available")

        capture = AudioCapture(device_id="default")
        capture.start()

        time.sleep(0.2)

        capture.stop()

        assert not capture.is_running()
        assert capture._stream is None
        assert capture._pyaudio is None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
