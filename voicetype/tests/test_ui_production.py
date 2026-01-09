"""
VoiceType - Production UI Tests
Comprehensive UI testing for PyQt6 components.

Covers:
- Widget creation and initialization
- Signal/slot connections
- Theme application
- State management
- Thread-safety patterns
- Memory management

Requires: pytest-qt
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# PyQt6 imports
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtTest import QTest

# Import components under test
from src.ui.themes import get_theme, apply_theme, DARK_THEME, LIGHT_THEME, THEMES
from src.utils.constants import (
    THEME_DARK, THEME_LIGHT, APP_NAME,
    TRAY_STATE_READY, TRAY_STATE_RECORDING, TRAY_STATE_LOADING, TRAY_STATE_ERROR,
    OUTPUT_MODE_KEYBOARD, OUTPUT_MODE_CLIPBOARD,
    DEFAULT_HOTKEY_START, DEFAULT_HOTKEY_STOP
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
    # Don't quit - let pytest-qt handle cleanup


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config_values = {
        "audio.microphone_id": "default",
        "audio.language": "ru",
        "audio.model": "small",
        "recognition.punctuation_enabled": True,
        "output.mode": OUTPUT_MODE_KEYBOARD,
        "system.autostart": False,
        "system.theme": THEME_DARK,
        "hotkeys.start_recording": DEFAULT_HOTKEY_START,
        "hotkeys.stop_recording": DEFAULT_HOTKEY_STOP,
        "internal.window_geometry": None,
    }

    mock = Mock()
    mock.get = lambda key, default=None: config_values.get(key, default)
    mock.set = Mock()

    return mock


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    mock = Mock()
    mock.get_history = Mock(return_value=[])
    mock.get_stats_24h = Mock(return_value=[])
    mock.get_today_recognition_time = Mock(return_value=0)
    mock.add_history_entry = Mock()
    mock.delete_history_entry = Mock()
    mock.clear_history = Mock()
    mock.add_stats_entry = Mock()
    return mock


# ============================================================================
# THEME TESTS
# ============================================================================

class TestThemes:
    """Tests for theme system."""

    def test_get_dark_theme(self):
        """Test dark theme retrieval."""
        theme = get_theme(THEME_DARK)
        assert theme == DARK_THEME
        assert "background-color: #1F2937" in theme
        assert "color: #F9FAFB" in theme

    def test_get_light_theme(self):
        """Test light theme retrieval."""
        theme = get_theme(THEME_LIGHT)
        assert theme == LIGHT_THEME
        assert "background-color: #F9FAFB" in theme
        assert "color: #1F2937" in theme

    def test_get_invalid_theme_returns_dark(self):
        """Test fallback to dark theme for invalid theme name."""
        theme = get_theme("invalid_theme")
        assert theme == DARK_THEME

    def test_themes_dict_contains_both_themes(self):
        """Test THEMES dictionary has both themes."""
        assert THEME_DARK in THEMES
        assert THEME_LIGHT in THEMES
        assert len(THEMES) == 2

    def test_apply_theme_to_widget(self, qapp):
        """Test applying theme to a widget."""
        widget = QWidget()
        apply_theme(widget, THEME_DARK)

        stylesheet = widget.styleSheet()
        assert stylesheet == DARK_THEME

        widget.deleteLater()

    def test_theme_contains_all_widget_styles(self):
        """Test that themes cover all required widget types."""
        required_selectors = [
            "QMainWindow", "QWidget", "QTabWidget", "QTabBar",
            "QPushButton", "QComboBox", "QLineEdit", "QTextEdit",
            "QSlider", "QCheckBox", "QRadioButton", "QGroupBox",
            "QLabel", "QScrollBar", "QListWidget", "QProgressBar",
            "QToolTip", "QMenu"
        ]

        for selector in required_selectors:
            assert selector in DARK_THEME, f"Dark theme missing {selector}"
            assert selector in LIGHT_THEME, f"Light theme missing {selector}"

    def test_theme_button_states(self):
        """Test button state styles in themes."""
        for theme in [DARK_THEME, LIGHT_THEME]:
            assert "QPushButton:hover" in theme
            assert "QPushButton:pressed" in theme
            assert "QPushButton:disabled" in theme

    def test_theme_special_buttons(self):
        """Test special button styles (danger, secondary)."""
        for theme in [DARK_THEME, LIGHT_THEME]:
            assert "QPushButton#dangerButton" in theme
            assert "QPushButton#secondaryButton" in theme


# ============================================================================
# LEVEL METER WIDGET TESTS
# ============================================================================

class TestLevelMeter:
    """Tests for LevelMeter widget."""

    def test_create_level_meter(self, qapp):
        """Test LevelMeter creation."""
        from src.ui.widgets.level_meter import LevelMeter

        meter = LevelMeter()
        assert meter is not None
        assert meter.get_level() == 0.0
        meter.deleteLater()

    def test_set_level_clamped(self, qapp):
        """Test level value clamping."""
        from src.ui.widgets.level_meter import LevelMeter

        meter = LevelMeter()

        # Test clamping at upper bound
        meter.set_level(1.5)
        assert meter.get_level() == 1.0

        # Test clamping at lower bound
        meter.set_level(-0.5)
        assert meter.get_level() == 0.0

        # Test normal value
        meter.set_level(0.5)
        assert meter.get_level() == 0.5

        meter.deleteLater()

    def test_reset_level_meter(self, qapp):
        """Test LevelMeter reset."""
        from src.ui.widgets.level_meter import LevelMeter

        meter = LevelMeter()
        meter.set_level(0.8)
        assert meter.get_level() == 0.8

        meter.reset()
        assert meter.get_level() == 0.0

        meter.deleteLater()

    def test_level_meter_with_label(self, qapp):
        """Test LevelMeterWithLabel widget."""
        from src.ui.widgets.level_meter import LevelMeterWithLabel

        meter = LevelMeterWithLabel("Test Label")
        assert meter is not None

        meter.set_level(0.75)
        assert meter.get_level() == 0.75

        meter.reset()
        assert meter.get_level() == 0.0

        meter.deleteLater()

    def test_peak_timer_running(self, qapp):
        """Test that peak decay timer is active."""
        from src.ui.widgets.level_meter import LevelMeter

        meter = LevelMeter()
        # Timer should be active for peak animation
        assert meter._peak_timer.isActive()

        meter.deleteLater()


# ============================================================================
# HOTKEY EDIT WIDGET TESTS
# ============================================================================

class TestHotkeyEdit:
    """Tests for HotkeyEdit widget."""

    def test_create_hotkey_edit(self, qapp):
        """Test HotkeyEdit creation."""
        from src.ui.widgets.hotkey_edit import HotkeyEdit

        widget = HotkeyEdit()
        assert widget is not None
        assert widget.get_hotkey() == ""
        assert not widget.is_recording()

        widget.deleteLater()

    def test_set_hotkey(self, qapp):
        """Test setting hotkey."""
        from src.ui.widgets.hotkey_edit import HotkeyEdit

        widget = HotkeyEdit()
        widget.set_hotkey("ctrl+shift+s")

        assert widget.get_hotkey() == "ctrl+shift+s"

        widget.deleteLater()

    def test_hotkey_lowercase_conversion(self, qapp):
        """Test hotkey is converted to lowercase."""
        from src.ui.widgets.hotkey_edit import HotkeyEdit

        widget = HotkeyEdit()
        widget.set_hotkey("CTRL+SHIFT+S")

        assert widget.get_hotkey() == "ctrl+shift+s"

        widget.deleteLater()

    def test_clear_hotkey(self, qapp):
        """Test clearing hotkey."""
        from src.ui.widgets.hotkey_edit import HotkeyEdit

        widget = HotkeyEdit()
        widget.set_hotkey("ctrl+a")
        widget._clear_hotkey()

        assert widget.get_hotkey() == ""

        widget.deleteLater()

    def test_hotkey_changed_signal(self, qapp):
        """Test hotkey_changed signal emission."""
        from src.ui.widgets.hotkey_edit import HotkeyEdit

        widget = HotkeyEdit()
        signal_received = []

        widget.hotkey_changed.connect(lambda h: signal_received.append(h))
        widget._clear_hotkey()

        assert "" in signal_received

        widget.deleteLater()

    def test_format_hotkey_modifier_order(self, qapp):
        """Test hotkey formatting maintains correct modifier order."""
        from src.ui.widgets.hotkey_edit import HotkeyEdit

        widget = HotkeyEdit()

        # Test modifier ordering: ctrl+alt+shift+win
        keys = {"shift", "ctrl", "a"}
        result = widget._format_hotkey(keys)

        # Should be ctrl before shift
        assert result.index("ctrl") < result.index("shift")

        widget.deleteLater()


# ============================================================================
# TRAY ICON TESTS
# ============================================================================

class TestTrayIcon:
    """Tests for TrayIcon component."""

    def test_create_tray_icon(self, qapp):
        """Test TrayIcon creation."""
        from src.ui.tray_icon import TrayIcon

        tray = TrayIcon()
        assert tray is not None
        assert tray.get_state() == TRAY_STATE_LOADING
        assert not tray.is_recording()

        tray.deleteLater()

    def test_set_state_ready(self, qapp):
        """Test setting ready state."""
        from src.ui.tray_icon import TrayIcon

        tray = TrayIcon()
        tray.set_state(TRAY_STATE_READY)

        assert tray.get_state() == TRAY_STATE_READY
        assert not tray.is_recording()

        tray.deleteLater()

    def test_set_state_recording(self, qapp):
        """Test setting recording state."""
        from src.ui.tray_icon import TrayIcon

        tray = TrayIcon()
        tray.set_state(TRAY_STATE_RECORDING)

        assert tray.get_state() == TRAY_STATE_RECORDING
        assert tray.is_recording()

        tray.deleteLater()

    def test_set_invalid_state_fallback(self, qapp):
        """Test invalid state falls back to error."""
        from src.ui.tray_icon import TrayIcon

        tray = TrayIcon()
        tray.set_state("invalid_state")

        assert tray.get_state() == TRAY_STATE_ERROR

        tray.deleteLater()

    def test_state_colors_defined(self, qapp):
        """Test all states have colors defined."""
        from src.ui.tray_icon import TrayIcon

        tray = TrayIcon()

        for state in [TRAY_STATE_READY, TRAY_STATE_RECORDING,
                      TRAY_STATE_LOADING, TRAY_STATE_ERROR]:
            assert state in tray.STATE_COLORS

        tray.deleteLater()

    def test_tray_signals_defined(self, qapp):
        """Test tray signals are defined."""
        from src.ui.tray_icon import TrayIcon

        tray = TrayIcon()

        # Check signals exist
        assert hasattr(tray, 'start_recording_clicked')
        assert hasattr(tray, 'stop_recording_clicked')
        assert hasattr(tray, 'settings_clicked')
        assert hasattr(tray, 'exit_clicked')

        tray.deleteLater()

    def test_icon_caching(self, qapp):
        """Test icon caching mechanism."""
        from src.ui.tray_icon import TrayIcon

        tray = TrayIcon()

        # Generate icon first time
        icon1 = tray._create_icon(TRAY_STATE_READY)

        # Should be cached now
        assert TRAY_STATE_READY in tray._icons_cache

        # Second call should return cached
        icon2 = tray._create_icon(TRAY_STATE_READY)

        # Same object from cache
        assert icon1 is icon2

        tray.deleteLater()


# ============================================================================
# TAB MAIN TESTS
# ============================================================================

class TestTabMain:
    """Tests for TabMain component."""

    def test_create_tab_main(self, qapp, mock_config):
        """Test TabMain creation."""
        with patch('src.ui.tabs.tab_main.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_main.get_microphones', return_value=[
                {"id": "default", "name": "Default Microphone"}
            ]):
                from src.ui.tabs.tab_main import TabMain

                tab = TabMain()
                assert tab is not None

                tab.deleteLater()

    def test_tab_main_signals_defined(self, qapp, mock_config):
        """Test TabMain signals are defined."""
        with patch('src.ui.tabs.tab_main.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_main.get_microphones', return_value=[]):
                from src.ui.tabs.tab_main import TabMain

                tab = TabMain()

                # Check all signals exist
                signals = [
                    'microphone_changed', 'language_changed', 'model_changed',
                    'punctuation_changed', 'output_mode_changed',
                    'theme_changed', 'autostart_changed'
                ]

                for signal_name in signals:
                    assert hasattr(tab, signal_name), f"Missing signal: {signal_name}"

                tab.deleteLater()

    def test_tab_main_refresh(self, qapp, mock_config):
        """Test TabMain refresh method."""
        with patch('src.ui.tabs.tab_main.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_main.get_microphones', return_value=[]):
                from src.ui.tabs.tab_main import TabMain

                tab = TabMain()

                # Should not raise
                tab.refresh()

                tab.deleteLater()


# ============================================================================
# TAB HOTKEYS TESTS
# ============================================================================

class TestTabHotkeys:
    """Tests for TabHotkeys component."""

    def test_create_tab_hotkeys(self, qapp, mock_config):
        """Test TabHotkeys creation."""
        with patch('src.ui.tabs.tab_hotkeys.get_config', return_value=mock_config):
            from src.ui.tabs.tab_hotkeys import TabHotkeys

            tab = TabHotkeys()
            assert tab is not None

            tab.deleteLater()

    def test_get_hotkeys(self, qapp, mock_config):
        """Test getting hotkey values."""
        with patch('src.ui.tabs.tab_hotkeys.get_config', return_value=mock_config):
            from src.ui.tabs.tab_hotkeys import TabHotkeys

            tab = TabHotkeys()

            # Should return default hotkeys
            assert tab.get_start_hotkey() == DEFAULT_HOTKEY_START
            assert tab.get_stop_hotkey() == DEFAULT_HOTKEY_STOP

            tab.deleteLater()

    def test_hotkeys_reset_signal(self, qapp, mock_config):
        """Test hotkeys_reset signal."""
        with patch('src.ui.tabs.tab_hotkeys.get_config', return_value=mock_config):
            from src.ui.tabs.tab_hotkeys import TabHotkeys

            tab = TabHotkeys()

            signal_received = []
            tab.hotkeys_reset.connect(lambda: signal_received.append(True))

            tab._on_reset_clicked()

            assert len(signal_received) == 1

            tab.deleteLater()


# ============================================================================
# TAB HISTORY TESTS
# ============================================================================

class TestTabHistory:
    """Tests for TabHistory component."""

    def test_create_tab_history(self, qapp, mock_database):
        """Test TabHistory creation."""
        with patch('src.ui.tabs.tab_history.get_database', return_value=mock_database):
            from src.ui.tabs.tab_history import TabHistory

            tab = TabHistory()
            assert tab is not None

            tab.deleteLater()

    def test_refresh_history(self, qapp, mock_database):
        """Test refreshing history."""
        with patch('src.ui.tabs.tab_history.get_database', return_value=mock_database):
            from src.ui.tabs.tab_history import TabHistory

            tab = TabHistory()

            # Should not raise
            tab.refresh()

            # Database should be called
            mock_database.get_history.assert_called()

            tab.deleteLater()

    def test_history_item_creation(self, qapp):
        """Test HistoryItem widget creation."""
        from src.ui.tabs.tab_history import HistoryItem

        entry = {
            "id": 1,
            "started_at": "2024-01-01 10:00:00",
            "duration_seconds": 65,
            "text": "Test transcription text",
            "language": "ru"
        }

        item = HistoryItem(entry)
        assert item is not None

        item.deleteLater()

    def test_history_item_deleted_signal(self, qapp):
        """Test HistoryItem deleted signal."""
        from src.ui.tabs.tab_history import HistoryItem

        entry = {"id": 42, "started_at": "", "duration_seconds": 0,
                 "text": "Test", "language": "ru"}

        item = HistoryItem(entry)

        signal_received = []
        item.deleted.connect(lambda id: signal_received.append(id))

        item._on_delete()

        assert 42 in signal_received

        item.deleteLater()


# ============================================================================
# TAB STATS TESTS
# ============================================================================

class TestTabStats:
    """Tests for TabStats component."""

    def test_create_tab_stats(self, qapp, mock_database):
        """Test TabStats creation."""
        with patch('src.ui.tabs.tab_stats.get_database', return_value=mock_database):
            with patch('src.ui.tabs.tab_stats.get_process_cpu', return_value=10.0):
                with patch('src.ui.tabs.tab_stats.get_process_memory', return_value=100.0):
                    from src.ui.tabs.tab_stats import TabStats

                    tab = TabStats()
                    assert tab is not None

                    # Stop timer to prevent issues in tests
                    tab._update_timer.stop()

                    tab.deleteLater()

    def test_simple_graph_creation(self, qapp):
        """Test SimpleGraph widget creation."""
        from src.ui.tabs.tab_stats import SimpleGraph

        graph = SimpleGraph(color="#FF0000")
        assert graph is not None

        graph.deleteLater()

    def test_simple_graph_data(self, qapp):
        """Test SimpleGraph data handling."""
        from src.ui.tabs.tab_stats import SimpleGraph

        graph = SimpleGraph()

        # Set data
        graph.set_data([10, 20, 30, 40, 50])
        assert len(graph._data) == 5

        # Add point
        graph.add_point(60)
        assert len(graph._data) == 6

        # Clear
        graph.clear()
        assert len(graph._data) == 0

        graph.deleteLater()

    def test_simple_graph_max_points(self, qapp):
        """Test SimpleGraph max points limit."""
        from src.ui.tabs.tab_stats import SimpleGraph

        graph = SimpleGraph()

        # Add more than max points
        data = list(range(100))
        graph.set_data(data)

        # Should be limited to max_points
        assert len(graph._data) == graph._max_points

        graph.deleteLater()


# ============================================================================
# TAB LOGS TESTS
# ============================================================================

class TestTabLogs:
    """Tests for TabLogs component."""

    def test_create_tab_logs(self, qapp):
        """Test TabLogs creation."""
        from src.ui.tabs.tab_logs import TabLogs

        tab = TabLogs()
        assert tab is not None

        # Stop auto-refresh timer
        tab._refresh_timer.stop()

        tab.deleteLater()

    def test_log_levels_defined(self, qapp):
        """Test log levels are defined."""
        from src.ui.tabs.tab_logs import TabLogs

        tab = TabLogs()

        # Check log levels
        level_names = [name for name, _ in tab.LOG_LEVELS]
        assert "Все" in level_names
        assert "DEBUG" in level_names
        assert "INFO" in level_names
        assert "WARNING" in level_names
        assert "ERROR" in level_names

        tab._refresh_timer.stop()
        tab.deleteLater()

    def test_append_log(self, qapp):
        """Test appending log message."""
        from src.ui.tabs.tab_logs import TabLogs

        tab = TabLogs()
        tab._refresh_timer.stop()

        # Append message
        tab.append_log("Test log message")

        # Should contain the message
        text = tab._log_view.toPlainText()
        assert "Test log message" in text

        tab.deleteLater()

    def test_clear_display(self, qapp):
        """Test clearing log display."""
        from src.ui.tabs.tab_logs import TabLogs

        tab = TabLogs()
        tab._refresh_timer.stop()

        tab.append_log("Some message")
        tab._clear_display()

        assert tab._log_view.toPlainText() == ""

        tab.deleteLater()


# ============================================================================
# TAB TEST TESTS
# ============================================================================

class TestTabTest:
    """Tests for TabTest component."""

    def test_create_tab_test(self, qapp, mock_config):
        """Test TabTest creation."""
        with patch('src.ui.tabs.tab_test.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_test.get_microphone_by_id', return_value=None):
                from src.ui.tabs.tab_test import TabTest

                tab = TabTest()
                assert tab is not None
                assert not tab.is_testing()

                tab.deleteLater()

    def test_start_stop_test(self, qapp, mock_config):
        """Test start/stop testing."""
        with patch('src.ui.tabs.tab_test.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_test.get_microphone_by_id', return_value=None):
                from src.ui.tabs.tab_test import TabTest

                tab = TabTest()

                # Start test
                tab._start_test()
                assert tab.is_testing()

                # Stop test
                tab._stop_test()
                assert not tab.is_testing()

                tab.deleteLater()

    def test_test_signals(self, qapp, mock_config):
        """Test TabTest signals."""
        with patch('src.ui.tabs.tab_test.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_test.get_microphone_by_id', return_value=None):
                from src.ui.tabs.tab_test import TabTest

                tab = TabTest()

                started_signal = []
                stopped_signal = []

                tab.test_started.connect(lambda: started_signal.append(True))
                tab.test_stopped.connect(lambda: stopped_signal.append(True))

                tab._start_test()
                assert len(started_signal) == 1

                tab._stop_test()
                assert len(stopped_signal) == 1

                tab.deleteLater()

    def test_update_level(self, qapp, mock_config):
        """Test level meter update."""
        with patch('src.ui.tabs.tab_test.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_test.get_microphone_by_id', return_value=None):
                from src.ui.tabs.tab_test import TabTest

                tab = TabTest()

                # Should not raise
                tab.update_level(0.5)

                tab.deleteLater()

    def test_append_results(self, qapp, mock_config):
        """Test appending recognition results."""
        with patch('src.ui.tabs.tab_test.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_test.get_microphone_by_id', return_value=None):
                from src.ui.tabs.tab_test import TabTest

                tab = TabTest()

                tab.append_partial_result("Partial text")
                assert "Partial text" in tab._result_text.toPlainText()

                tab.append_final_result("Final text")
                assert "Final text" in tab._result_text.toPlainText()

                tab.deleteLater()


# ============================================================================
# MAIN WINDOW TESTS
# ============================================================================

class TestMainWindow:
    """Tests for MainWindow component."""

    def test_create_main_window(self, qapp, mock_config):
        """Test MainWindow creation."""
        with patch('src.ui.main_window.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_main.get_config', return_value=mock_config):
                with patch('src.ui.tabs.tab_main.get_microphones', return_value=[]):
                    with patch('src.ui.tabs.tab_hotkeys.get_config', return_value=mock_config):
                        with patch('src.ui.tabs.tab_history.get_database') as mock_db:
                            mock_db.return_value.get_history.return_value = []
                            with patch('src.ui.tabs.tab_stats.get_database') as mock_stats_db:
                                mock_stats_db.return_value.get_stats_24h.return_value = []
                                mock_stats_db.return_value.get_today_recognition_time.return_value = 0
                                with patch('src.ui.tabs.tab_stats.get_process_cpu', return_value=0):
                                    with patch('src.ui.tabs.tab_stats.get_process_memory', return_value=0):
                                        with patch('src.ui.tabs.tab_test.get_config', return_value=mock_config):
                                            with patch('src.ui.tabs.tab_test.get_microphone_by_id', return_value=None):
                                                from src.ui.main_window import MainWindow

                                                window = MainWindow()
                                                assert window is not None
                                                assert window.windowTitle().startswith(APP_NAME)

                                                # Stop timers
                                                window.tab_stats._update_timer.stop()
                                                window.tab_logs._refresh_timer.stop()

                                                window.deleteLater()

    def test_main_window_signals(self, qapp, mock_config):
        """Test MainWindow signals are defined."""
        with patch('src.ui.main_window.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_main.get_config', return_value=mock_config):
                with patch('src.ui.tabs.tab_main.get_microphones', return_value=[]):
                    with patch('src.ui.tabs.tab_hotkeys.get_config', return_value=mock_config):
                        with patch('src.ui.tabs.tab_history.get_database') as mock_db:
                            mock_db.return_value.get_history.return_value = []
                            with patch('src.ui.tabs.tab_stats.get_database') as mock_stats_db:
                                mock_stats_db.return_value.get_stats_24h.return_value = []
                                mock_stats_db.return_value.get_today_recognition_time.return_value = 0
                                with patch('src.ui.tabs.tab_stats.get_process_cpu', return_value=0):
                                    with patch('src.ui.tabs.tab_stats.get_process_memory', return_value=0):
                                        with patch('src.ui.tabs.tab_test.get_config', return_value=mock_config):
                                            with patch('src.ui.tabs.tab_test.get_microphone_by_id', return_value=None):
                                                from src.ui.main_window import MainWindow

                                                window = MainWindow()

                                                assert hasattr(window, 'window_closed')
                                                assert hasattr(window, 'theme_changed')

                                                window.tab_stats._update_timer.stop()
                                                window.tab_logs._refresh_timer.stop()
                                                window.deleteLater()


# ============================================================================
# THREAD-SAFETY TESTS
# ============================================================================

class TestThreadSafety:
    """Tests for thread-safety patterns."""

    def test_timer_in_main_thread(self, qapp):
        """Test that QTimer is created in main thread."""
        from src.ui.widgets.level_meter import LevelMeter

        meter = LevelMeter()

        # Timer should be associated with main thread
        assert meter._peak_timer.thread() == qapp.thread()

        meter.deleteLater()

    def test_signal_slot_connection_type(self, qapp, mock_config):
        """Test signal/slot connections use proper types."""
        # In PyQt6, connections are auto-queued across threads
        # This test verifies signals are properly defined

        with patch('src.ui.tabs.tab_main.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_main.get_microphones', return_value=[]):
                from src.ui.tabs.tab_main import TabMain

                tab = TabMain()

                # Check signal is pyqtSignal
                from PyQt6.QtCore import pyqtBoundSignal
                assert isinstance(tab.theme_changed, pyqtBoundSignal)

                tab.deleteLater()


# ============================================================================
# MEMORY MANAGEMENT TESTS
# ============================================================================

class TestMemoryManagement:
    """Tests for proper memory management."""

    def test_widget_parent_chain(self, qapp):
        """Test widgets maintain proper parent chain."""
        from src.ui.widgets.level_meter import LevelMeterWithLabel

        widget = LevelMeterWithLabel()

        # Child meter should have parent
        assert widget._meter.parent() is not None

        widget.deleteLater()

    def test_timer_cleanup(self, qapp, mock_database):
        """Test timers are associated with their widgets."""
        with patch('src.ui.tabs.tab_stats.get_database', return_value=mock_database):
            with patch('src.ui.tabs.tab_stats.get_process_cpu', return_value=0):
                with patch('src.ui.tabs.tab_stats.get_process_memory', return_value=0):
                    from src.ui.tabs.tab_stats import TabStats

                    tab = TabStats()

                    # Timer should have parent (the tab)
                    assert tab._update_timer.parent() == tab

                    tab._update_timer.stop()
                    tab.deleteLater()


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_hotkey_handling(self, qapp):
        """Test handling of empty hotkey."""
        from src.ui.widgets.hotkey_edit import HotkeyEdit

        widget = HotkeyEdit()
        widget.set_hotkey("")

        assert widget.get_hotkey() == ""

        widget.deleteLater()

    def test_none_hotkey_handling(self, qapp):
        """Test handling of None hotkey."""
        from src.ui.widgets.hotkey_edit import HotkeyEdit

        widget = HotkeyEdit()
        widget.set_hotkey(None)

        assert widget.get_hotkey() == ""

        widget.deleteLater()

    def test_history_item_long_text(self, qapp):
        """Test HistoryItem with very long text."""
        from src.ui.tabs.tab_history import HistoryItem

        # Text longer than 200 chars
        long_text = "A" * 500

        entry = {
            "id": 1,
            "started_at": "2024-01-01 10:00:00",
            "duration_seconds": 60,
            "text": long_text,
            "language": "ru"
        }

        item = HistoryItem(entry)
        # Should not crash
        assert item is not None

        item.deleteLater()

    def test_graph_empty_data_paint(self, qapp):
        """Test graph painting with no data."""
        from src.ui.tabs.tab_stats import SimpleGraph

        graph = SimpleGraph()
        graph.show()

        # Force paint - should not crash
        graph.repaint()

        graph.deleteLater()

    def test_tray_icon_generated_fallback(self, qapp):
        """Test tray icon generation when file not found."""
        from src.ui.tray_icon import TrayIcon

        tray = TrayIcon()

        # Clear cache to force regeneration
        tray._icons_cache.clear()

        # Generate icon - should not crash
        icon = tray._generate_icon(TRAY_STATE_READY)

        assert icon is not None
        assert not icon.isNull()

        tray.deleteLater()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for UI components."""

    def test_theme_change_propagation(self, qapp, mock_config):
        """Test theme change propagates correctly."""
        with patch('src.ui.tabs.tab_main.get_config', return_value=mock_config):
            with patch('src.ui.tabs.tab_main.get_microphones', return_value=[]):
                from src.ui.tabs.tab_main import TabMain
                from src.ui.themes import get_theme

                tab = TabMain()

                theme_received = []
                tab.theme_changed.connect(lambda t: theme_received.append(t))

                # Simulate theme change - find theme combo and change it
                # Note: This simulates the internal signal emission
                tab.theme_changed.emit(THEME_LIGHT)

                assert THEME_LIGHT in theme_received

                tab.deleteLater()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
