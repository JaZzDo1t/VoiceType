"""
VoiceType - App Controller Integration Tests
Integration tests for VoiceTypeApp (src/app.py) which coordinates all components.

Tests cover:
- VoiceTypeApp initialization with mocked models
- Signal connections and emissions
- Loading state management (_models_loaded flag)
- start_recording() behavior when models not loaded
- quit() cleanup behavior
- Model loading callbacks

Requires: pytest-qt
"""
import pytest
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer

# Import constants
from src.utils.constants import (
    TRAY_STATE_READY, TRAY_STATE_RECORDING,
    TRAY_STATE_LOADING, TRAY_STATE_ERROR,
    OUTPUT_MODE_KEYBOARD, OUTPUT_MODE_CLIPBOARD,
    DEFAULT_HOTKEY_TOGGLE
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config_values = {
        "audio.microphone_id": "default",
        "audio.language": "ru",
        "audio.model": "small",
        "recognition.punctuation_enabled": False,  # Disable to skip Silero loading
        "output.mode": OUTPUT_MODE_KEYBOARD,
        "system.autostart": False,
        "system.theme": "dark",
        "hotkeys.toggle_recording": DEFAULT_HOTKEY_TOGGLE,
        "internal.window_geometry": None,
    }

    mock = Mock()
    mock.get = lambda key, default=None: config_values.get(key, default)
    mock.set = Mock()
    mock.load = Mock()
    mock.save = Mock()

    return mock


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    mock = Mock()
    mock.get_history = Mock(return_value=[])
    mock.get_stats_24h = Mock(return_value=[])
    mock.get_today_recognition_time = Mock(return_value=0)
    mock.add_history_entry = Mock(return_value=1)
    mock.delete_history_entry = Mock()
    mock.clear_history = Mock()
    mock.add_stats_entry = Mock()
    mock.cleanup_old_stats = Mock()
    mock.initialize = Mock()
    return mock


@pytest.fixture
def mock_models_manager():
    """Mock models manager for testing."""
    mock = Mock()
    mock.get_vosk_model_path = Mock(return_value=Path("/fake/model/path"))
    mock.get_silero_te_path = Mock(return_value=None)
    mock.ensure_models_dir = Mock(return_value=True)
    return mock


@pytest.fixture
def mock_recognizer():
    """Mock recognizer for testing."""
    mock = Mock()
    mock.is_loaded = Mock(return_value=True)
    mock.load_model = Mock(return_value=True)
    mock.reset = Mock()
    mock.process_audio = Mock()
    mock.get_final_result = Mock(return_value="")
    mock.unload = Mock()
    mock.on_partial_result = None
    mock.on_final_result = None
    return mock


@pytest.fixture
def mock_audio_capture():
    """Mock audio capture for testing."""
    import queue

    mock = Mock()
    mock.start = Mock(return_value=True)
    mock.stop = Mock()
    mock.is_running = Mock(return_value=False)
    mock.get_audio_queue = Mock(return_value=queue.Queue())
    mock.get_level = Mock(return_value=0.0)
    return mock


@pytest.fixture
def mock_hotkey_manager():
    """Mock hotkey manager for testing."""
    mock = Mock()
    mock.register = Mock()
    mock.unregister = Mock()
    mock.start_listening = Mock()
    mock.stop_listening = Mock()
    mock.is_listening = Mock(return_value=False)
    return mock


@pytest.fixture
def mock_output_manager():
    """Mock output manager for testing."""
    mock = Mock()
    mock.output = Mock()
    mock.set_mode = Mock()
    mock.mode = OUTPUT_MODE_KEYBOARD
    return mock


@pytest.fixture
def mock_tray_icon(qapp):
    """Mock tray icon for testing."""
    mock = Mock()
    mock.set_state = Mock()
    mock.show = Mock()
    mock.hide = Mock()
    mock.show_notification = Mock()
    mock.get_state = Mock(return_value=TRAY_STATE_LOADING)
    mock.is_recording = Mock(return_value=False)

    # Signals
    mock.start_recording_clicked = Mock()
    mock.start_recording_clicked.connect = Mock()
    mock.stop_recording_clicked = Mock()
    mock.stop_recording_clicked.connect = Mock()
    mock.settings_clicked = Mock()
    mock.settings_clicked.connect = Mock()
    mock.exit_clicked = Mock()
    mock.exit_clicked.connect = Mock()

    return mock


@pytest.fixture
def mock_main_window(qapp):
    """Mock main window for testing."""
    mock = Mock()
    mock.show_and_activate = Mock()
    mock.set_loading = Mock()
    mock.set_loading_status = Mock()
    mock.show_loading_error = Mock()
    mock.is_loading = Mock(return_value=True)

    # Tab Main
    mock.tab_main = Mock()
    mock.tab_main.microphone_changed = Mock()
    mock.tab_main.microphone_changed.connect = Mock()
    mock.tab_main.language_changed = Mock()
    mock.tab_main.language_changed.connect = Mock()
    mock.tab_main.model_changed = Mock()
    mock.tab_main.model_changed.connect = Mock()
    mock.tab_main.output_mode_changed = Mock()
    mock.tab_main.output_mode_changed.connect = Mock()
    mock.tab_main.autostart_changed = Mock()
    mock.tab_main.autostart_changed.connect = Mock()

    # Tab Hotkeys - single toggle hotkey
    mock.tab_hotkeys = Mock()
    mock.tab_hotkeys.toggle_hotkey_changed = Mock()
    mock.tab_hotkeys.toggle_hotkey_changed.connect = Mock()

    # Tab Test
    mock.tab_test = Mock()
    mock.tab_test.test_started = Mock()
    mock.tab_test.test_started.connect = Mock()
    mock.tab_test.test_stopped = Mock()
    mock.tab_test.test_stopped.connect = Mock()
    mock.tab_test.is_testing = Mock(return_value=False)
    mock.tab_test.update_level = Mock()
    mock.tab_test.append_partial_result = Mock()
    mock.tab_test.append_final_result = Mock()
    mock.tab_test.set_models_ready = Mock()

    # Tab History
    mock.tab_history = Mock()
    mock.tab_history.refresh = Mock()

    return mock


# ============================================================================
# VOICETYPEAPP INITIALIZATION TESTS
# ============================================================================

class TestVoiceTypeAppInitialization:
    """Tests for VoiceTypeApp initialization."""

    def test_app_creation(self, qapp):
        """Test VoiceTypeApp can be instantiated."""
        with patch('src.app.get_config') as mock_get_config, \
             patch('src.app.get_database') as mock_get_db, \
             patch('src.app.get_models_manager') as mock_get_models:

            mock_get_config.return_value = Mock()
            mock_get_db.return_value = Mock()
            mock_get_models.return_value = Mock()

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()

            assert app is not None
            assert app._is_recording is False
            assert app._is_initialized is False
            assert app._models_loaded is False

    def test_app_initial_state(self, qapp, mock_config, mock_database, mock_models_manager):
        """Test VoiceTypeApp initial state after creation."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager):

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()

            # Check initial state
            assert app._audio_capture is None
            assert app._recognizer is None
            assert app._punctuation is None
            assert app._output_manager is None
            assert app._hotkey_manager is None
            assert app._tray_icon is None
            assert app._main_window is None
            assert app._session_text == ""
            assert app._session_start is None

    def test_app_signals_defined(self, qapp, mock_config, mock_database, mock_models_manager):
        """Test VoiceTypeApp signals are properly defined."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager):

            from src.app import VoiceTypeApp
            from PyQt6.QtCore import pyqtBoundSignal

            app = VoiceTypeApp()

            # Check all signals exist and are pyqtBoundSignal
            assert hasattr(app, '_update_level_signal')
            assert hasattr(app, '_partial_result_signal')
            assert hasattr(app, '_final_result_signal')
            assert hasattr(app, '_recognition_finished_signal')
            assert hasattr(app, '_models_loaded_signal')
            assert hasattr(app, '_loading_status_signal')
            assert hasattr(app, '_hotkey_triggered_signal')


# ============================================================================
# APP INITIALIZATION PROCESS TESTS
# ============================================================================

class TestVoiceTypeAppInitialize:
    """Tests for VoiceTypeApp.initialize() method."""

    def test_initialize_creates_ui(self, qapp, mock_config, mock_database,
                                   mock_models_manager, mock_tray_icon, mock_main_window):
        """Test initialize() creates UI components."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager') as mock_output_cls, \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_output_cls.return_value = Mock()
            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            result = app.initialize()

            assert result is True
            assert app._is_initialized is True
            mock_config.load.assert_called_once()
            mock_database.initialize.assert_called_once()

    def test_initialize_sets_loading_state(self, qapp, mock_config, mock_database,
                                           mock_models_manager, mock_tray_icon, mock_main_window):
        """Test initialize() sets tray icon to loading state."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Check tray icon was set to loading state
            mock_tray_icon.set_state.assert_called_with(TRAY_STATE_LOADING)
            mock_tray_icon.show.assert_called_once()

    def test_initialize_starts_stats_timer(self, qapp, mock_config, mock_database,
                                           mock_models_manager, mock_tray_icon, mock_main_window):
        """Test initialize() starts stats collection timer."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Stats timer should be created and started
            assert app._stats_timer is not None
            assert app._stats_timer.isActive()

            # Cleanup
            app._stats_timer.stop()

    def test_initialize_handles_exception(self, qapp, mock_config, mock_database,
                                          mock_models_manager, mock_tray_icon):
        """Test initialize() handles exceptions gracefully."""
        mock_config.load.side_effect = Exception("Config load error")

        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon):

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            result = app.initialize()

            assert result is False
            assert app._is_initialized is False


# ============================================================================
# MODELS LOADING TESTS
# ============================================================================

class TestModelsLoading:
    """Tests for model loading behavior."""

    def test_models_loaded_flag_initially_false(self, qapp, mock_config, mock_database,
                                                 mock_models_manager):
        """Test _models_loaded flag is initially False."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager):

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()

            assert app._models_loaded is False

    def test_on_models_loaded_success(self, qapp, mock_config, mock_database,
                                      mock_models_manager, mock_tray_icon, mock_main_window):
        """Test _on_models_loaded handler sets correct state on success."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Simulate models loaded successfully
            app._on_models_loaded(TRAY_STATE_READY, "vosk-model-ru-0.22")

            assert app._models_loaded is True
            mock_tray_icon.set_state.assert_called_with(TRAY_STATE_READY)
            mock_main_window.set_loading.assert_called_with(False, "vosk-model-ru-0.22")

            # Cleanup
            app._stats_timer.stop()

    def test_on_models_loaded_error(self, qapp, mock_config, mock_database,
                                    mock_models_manager, mock_tray_icon, mock_main_window):
        """Test _on_models_loaded handler sets correct state on error."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Simulate models loading failed
            app._on_models_loaded(TRAY_STATE_ERROR, "")

            assert app._models_loaded is False
            mock_tray_icon.set_state.assert_called_with(TRAY_STATE_ERROR)
            mock_main_window.show_loading_error.assert_called()

            # Cleanup
            app._stats_timer.stop()

    def test_on_loading_status_updates_ui(self, qapp, mock_config, mock_database,
                                          mock_models_manager, mock_tray_icon, mock_main_window):
        """Test _on_loading_status updates main window."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Simulate loading status update
            app._on_loading_status("Loading Vosk", "vosk-model-small-ru")

            mock_main_window.set_loading_status.assert_called_with(
                "Loading Vosk", "vosk-model-small-ru"
            )

            # Cleanup
            app._stats_timer.stop()


# ============================================================================
# START RECORDING TESTS
# ============================================================================

class TestStartRecording:
    """Tests for start_recording() behavior."""

    def test_start_recording_when_models_not_loaded(self, qapp, mock_config, mock_database,
                                                     mock_models_manager, mock_tray_icon,
                                                     mock_main_window):
        """Test start_recording() silently returns when models not loaded."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'), \
             patch('src.app.AudioCapture') as mock_audio_cls:

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Models not loaded yet
            assert app._models_loaded is False

            # Try to start recording
            app.start_recording()

            # Should not have started recording
            assert app._is_recording is False
            mock_audio_cls.assert_not_called()

            # Cleanup
            app._stats_timer.stop()

    def test_start_recording_when_already_recording(self, qapp, mock_config, mock_database,
                                                     mock_models_manager, mock_tray_icon,
                                                     mock_main_window, mock_recognizer):
        """Test start_recording() does nothing when already recording."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'), \
             patch('src.app.AudioCapture') as mock_audio_cls:

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )
            mock_audio_cls.return_value = Mock(
                start=Mock(return_value=True),
                get_audio_queue=Mock(return_value=Mock())
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Manually set up state for testing
            app._models_loaded = True
            app._recognizer = mock_recognizer
            app._is_recording = True  # Already recording

            # Reset mock to track new calls
            mock_audio_cls.reset_mock()

            # Try to start recording again
            app.start_recording()

            # Should not create new audio capture
            mock_audio_cls.assert_not_called()

            # Cleanup
            app._stats_timer.stop()

    def test_start_recording_success(self, qapp, mock_config, mock_database,
                                     mock_models_manager, mock_tray_icon,
                                     mock_main_window, mock_recognizer, mock_audio_capture):
        """Test start_recording() works when models are loaded."""
        import queue

        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'), \
             patch('src.app.AudioCapture', return_value=mock_audio_capture):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Set up models as loaded
            app._models_loaded = True
            app._recognizer = mock_recognizer

            # Start recording
            app.start_recording()

            # Should have started recording
            assert app._is_recording is True
            assert app._session_text == ""
            assert app._session_start is not None
            mock_audio_capture.start.assert_called_once()
            mock_recognizer.reset.assert_called_once()
            mock_tray_icon.set_state.assert_called_with(TRAY_STATE_RECORDING)

            # Cleanup
            app._is_recording = False  # Stop recognition loop
            if app._level_timer:
                app._level_timer.stop()
            app._stats_timer.stop()

    def test_start_recording_audio_capture_fails(self, qapp, mock_config, mock_database,
                                                  mock_models_manager, mock_tray_icon,
                                                  mock_main_window, mock_recognizer):
        """Test start_recording() handles audio capture failure."""
        mock_audio = Mock()
        mock_audio.start = Mock(return_value=False)  # Fails to start

        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'), \
             patch('src.app.AudioCapture', return_value=mock_audio):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Set up models as loaded
            app._models_loaded = True
            app._recognizer = mock_recognizer

            # Start recording
            app.start_recording()

            # Should have failed and reset state
            assert app._is_recording is False
            mock_tray_icon.show_notification.assert_called()

            # Cleanup
            app._stats_timer.stop()


# ============================================================================
# STOP RECORDING TESTS
# ============================================================================

class TestStopRecording:
    """Tests for stop_recording() behavior."""

    def test_stop_recording_when_not_recording(self, qapp, mock_config, mock_database,
                                                mock_models_manager, mock_tray_icon,
                                                mock_main_window):
        """Test stop_recording() does nothing when not recording."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Not recording
            assert app._is_recording is False

            # Try to stop recording
            app.stop_recording()

            # Should not have changed anything
            assert app._is_recording is False

            # Cleanup
            app._stats_timer.stop()

    def test_stop_recording_saves_to_history(self, qapp, mock_config, mock_database,
                                             mock_models_manager, mock_tray_icon,
                                             mock_main_window, mock_recognizer):
        """Test stop_recording() saves session to history."""
        mock_audio = Mock()
        mock_audio.stop = Mock()

        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Simulate recording state
            app._is_recording = True
            app._session_text = "Test recognition text"
            app._session_start = datetime.now()
            app._audio_capture = mock_audio
            app._recognizer = mock_recognizer
            app._level_timer = QTimer()
            app._level_timer.start(50)

            # Stop recording
            app.stop_recording()

            # Should have saved to history
            assert app._is_recording is False
            mock_database.add_history_entry.assert_called_once()
            mock_tray_icon.set_state.assert_called_with(TRAY_STATE_READY)

            # Cleanup
            app._stats_timer.stop()


# ============================================================================
# QUIT BEHAVIOR TESTS
# ============================================================================

class TestQuitBehavior:
    """Tests for quit() cleanup behavior."""

    def test_quit_stops_recording_if_active(self, qapp, mock_config, mock_database,
                                            mock_models_manager, mock_tray_icon,
                                            mock_main_window, mock_recognizer):
        """Test quit() stops recording if it's active."""
        mock_audio = Mock()
        mock_audio.stop = Mock()

        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'), \
             patch('src.app.QApplication') as mock_qapp:

            mock_hotkey_manager = Mock(
                register=Mock(),
                start_listening=Mock(),
                stop_listening=Mock()
            )
            mock_hotkey_cls.return_value = mock_hotkey_manager

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Set up recording state
            app._is_recording = True
            app._session_text = ""
            app._session_start = datetime.now()
            app._audio_capture = mock_audio
            app._recognizer = mock_recognizer
            app._level_timer = QTimer()

            # Quit
            app.quit()

            # Should have stopped recording
            assert app._is_recording is False
            mock_audio.stop.assert_called()

            # Cleanup timer should be stopped
            assert app._stats_timer is None or not app._stats_timer.isActive()

    def test_quit_stops_stats_timer(self, qapp, mock_config, mock_database,
                                    mock_models_manager, mock_tray_icon, mock_main_window):
        """Test quit() stops stats timer."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'), \
             patch('src.app.QApplication') as mock_qapp:

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock(),
                stop_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Verify timer is running
            assert app._stats_timer.isActive()

            # Quit
            app.quit()

            # Timer should be stopped
            assert not app._stats_timer.isActive()

    def test_quit_stops_hotkey_manager(self, qapp, mock_config, mock_database,
                                       mock_models_manager, mock_tray_icon, mock_main_window):
        """Test quit() stops hotkey manager."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'), \
             patch('src.app.QApplication') as mock_qapp:

            mock_hotkey_manager = Mock(
                register=Mock(),
                start_listening=Mock(),
                stop_listening=Mock()
            )
            mock_hotkey_cls.return_value = mock_hotkey_manager

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Quit
            app.quit()

            # Hotkey manager should be stopped
            mock_hotkey_manager.stop_listening.assert_called()

            # Cleanup
            app._stats_timer.stop()

    def test_quit_hides_tray_icon(self, qapp, mock_config, mock_database,
                                   mock_models_manager, mock_tray_icon, mock_main_window):
        """Test quit() hides tray icon."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'), \
             patch('src.app.QApplication') as mock_qapp:

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock(),
                stop_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Quit
            app.quit()

            # Tray icon should be hidden
            mock_tray_icon.hide.assert_called()

            # Cleanup
            app._stats_timer.stop()

    def test_quit_unloads_models(self, qapp, mock_config, mock_database,
                                  mock_models_manager, mock_tray_icon, mock_main_window,
                                  mock_recognizer):
        """Test quit() unloads models to free memory."""
        mock_punctuation = Mock()
        mock_punctuation.unload = Mock()

        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'), \
             patch('src.app.QApplication') as mock_qapp:

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock(),
                stop_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Set up models
            app._recognizer = mock_recognizer
            app._punctuation = mock_punctuation

            # Quit
            app.quit()

            # Models should be unloaded
            mock_recognizer.unload.assert_called()
            mock_punctuation.unload.assert_called()

            # Cleanup
            app._stats_timer.stop()


# ============================================================================
# SIGNAL CONNECTION TESTS
# ============================================================================

class TestSignalConnections:
    """Tests for signal connections."""

    def test_hotkey_triggered_signal_toggle(self, qapp, mock_config, mock_database,
                                            mock_models_manager, mock_tray_icon, mock_main_window):
        """Test _hotkey_triggered_signal with 'toggle' action."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Patch toggle_recording to verify it's called
            app.toggle_recording = Mock()

            # Emit signal
            app._on_hotkey_triggered("toggle")

            # Should call toggle_recording
            app.toggle_recording.assert_called_once()

            # Cleanup
            app._stats_timer.stop()


# ============================================================================
# SETTINGS CHANGE TESTS
# ============================================================================

class TestSettingsChanges:
    """Tests for settings change handlers."""

    def test_on_language_changed_reloads_models(self, qapp, mock_config, mock_database,
                                                 mock_models_manager, mock_tray_icon,
                                                 mock_main_window):
        """Test _on_language_changed triggers model reload."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Set models as loaded initially
            app._models_loaded = True

            # Change language
            app._on_language_changed("en")

            # Should have set models_loaded to False and triggered reload
            assert app._models_loaded is False
            mock_tray_icon.set_state.assert_called_with(TRAY_STATE_LOADING)
            mock_main_window.set_loading.assert_called_with(True)

            # Cleanup
            app._stats_timer.stop()

    def test_on_model_changed_reloads_models(self, qapp, mock_config, mock_database,
                                              mock_models_manager, mock_tray_icon,
                                              mock_main_window):
        """Test _on_model_changed triggers model reload."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Set models as loaded initially
            app._models_loaded = True

            # Change model
            app._on_model_changed("large")

            # Should have set models_loaded to False and triggered reload
            assert app._models_loaded is False
            mock_tray_icon.set_state.assert_called_with(TRAY_STATE_LOADING)
            mock_main_window.set_loading.assert_called_with(True)

            # Cleanup
            app._stats_timer.stop()

    def test_on_output_mode_changed(self, qapp, mock_config, mock_database,
                                    mock_models_manager, mock_tray_icon, mock_main_window):
        """Test _on_output_mode_changed updates output manager."""
        mock_output = Mock()
        mock_output.set_mode = Mock()

        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager', return_value=mock_output), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Change output mode
            app._on_output_mode_changed(OUTPUT_MODE_CLIPBOARD)

            # Output manager should be updated
            mock_output.set_mode.assert_called_with(OUTPUT_MODE_CLIPBOARD)

            # Cleanup
            app._stats_timer.stop()


# ============================================================================
# TEST TAB CALLBACKS
# ============================================================================

class TestTabCallbacks:
    """Tests for tab callback handlers."""

    def test_on_test_started_calls_start_recording(self, qapp, mock_config, mock_database,
                                                    mock_models_manager, mock_tray_icon,
                                                    mock_main_window):
        """Test _on_test_started calls start_recording."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Mock start_recording
            app.start_recording = Mock()

            # Call test started handler
            app._on_test_started()

            # Should call start_recording
            app.start_recording.assert_called_once()

            # Cleanup
            app._stats_timer.stop()

    def test_on_test_stopped_calls_stop_recording(self, qapp, mock_config, mock_database,
                                                   mock_models_manager, mock_tray_icon,
                                                   mock_main_window):
        """Test _on_test_stopped calls stop_recording."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Mock stop_recording
            app.stop_recording = Mock()

            # Call test stopped handler
            app._on_test_stopped()

            # Should call stop_recording
            app.stop_recording.assert_called_once()

            # Cleanup
            app._stats_timer.stop()


# ============================================================================
# RECOGNITION FINISHED TESTS
# ============================================================================

class TestRecognitionFinished:
    """Tests for recognition finished handler."""

    def test_on_recognition_finished_refreshes_history(self, qapp, mock_config, mock_database,
                                                        mock_models_manager, mock_tray_icon,
                                                        mock_main_window):
        """Test _on_recognition_finished refreshes history tab."""
        with patch('src.app.get_config', return_value=mock_config), \
             patch('src.app.get_database', return_value=mock_database), \
             patch('src.app.get_models_manager', return_value=mock_models_manager), \
             patch('src.app.TrayIcon', return_value=mock_tray_icon), \
             patch('src.app.MainWindow', return_value=mock_main_window), \
             patch('src.app.OutputManager'), \
             patch('src.app.HotkeyManager') as mock_hotkey_cls, \
             patch('src.app.Autostart'):

            mock_hotkey_cls.return_value = Mock(
                register=Mock(),
                start_listening=Mock()
            )

            from src.app import VoiceTypeApp

            app = VoiceTypeApp()
            app.initialize()

            # Call recognition finished handler
            app._on_recognition_finished()

            # Should refresh history tab
            mock_main_window.tab_history.refresh.assert_called_once()

            # Cleanup
            app._stats_timer.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
