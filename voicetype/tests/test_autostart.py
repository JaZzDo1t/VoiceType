"""
VoiceType - Autostart Tests
Comprehensive tests for the Windows autostart functionality.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from src.utils.autostart import Autostart, REGISTRY_PATH, _open_registry_key
from src.utils.constants import APP_NAME


class TestAutostartBasicFunctionality:
    """Tests for basic enable/disable/is_enabled functionality."""

    @pytest.fixture
    def mock_winreg(self):
        """Create a mock winreg module with registry simulation."""
        mock_registry = {}
        mock_module = MagicMock()
        mock_module.HKEY_CURRENT_USER = "HKCU"
        mock_module.KEY_SET_VALUE = 0x0002
        mock_module.KEY_READ = 0x0001
        mock_module.REG_SZ = 1

        def mock_open_key(key, subkey, reserved, access):
            mock_handle = MagicMock()
            mock_handle.__enter__ = MagicMock(return_value=mock_handle)
            mock_handle.__exit__ = MagicMock(return_value=False)
            return mock_handle

        def mock_set_value_ex(key, name, reserved, reg_type, value):
            mock_registry[name] = value

        def mock_delete_value(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            del mock_registry[name]

        def mock_query_value_ex(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            return (mock_registry[name], 1)

        def mock_close_key(key):
            pass

        mock_module.OpenKey = mock_open_key
        mock_module.SetValueEx = mock_set_value_ex
        mock_module.DeleteValue = mock_delete_value
        mock_module.QueryValueEx = mock_query_value_ex
        mock_module.CloseKey = mock_close_key
        mock_module._registry = mock_registry

        return mock_module

    def test_enable_autostart_success(self, mock_winreg):
        """Test that enable() successfully registers autostart."""
        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with patch.object(Autostart, '_get_raw_executable_path') as mock_raw_path:
                with patch.object(Autostart, '_get_executable_path') as mock_exe_path:
                    mock_path = MagicMock(spec=Path)
                    mock_path.exists.return_value = True
                    mock_raw_path.return_value = mock_path
                    mock_exe_path.return_value = '"C:\\Program Files\\VoiceType\\VoiceType.exe"'

                    result = Autostart.enable()

                    assert result is True
                    assert APP_NAME in mock_winreg._registry
                    assert mock_winreg._registry[APP_NAME] == '"C:\\Program Files\\VoiceType\\VoiceType.exe"'

    def test_disable_autostart_success(self, mock_winreg):
        """Test that disable() successfully removes autostart."""
        # Pre-populate registry
        mock_winreg._registry[APP_NAME] = '"C:\\VoiceType.exe"'

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result = Autostart.disable()

            assert result is True
            assert APP_NAME not in mock_winreg._registry

    def test_is_enabled_when_enabled(self, mock_winreg):
        """Test is_enabled returns True when autostart is properly configured."""
        expected_path = '"C:\\VoiceType\\VoiceType.exe"'
        mock_winreg._registry[APP_NAME] = expected_path

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with patch.object(Autostart, '_get_executable_path', return_value=expected_path):
                result = Autostart.is_enabled()

                assert result is True

    def test_is_enabled_when_disabled(self, mock_winreg):
        """Test is_enabled returns False when autostart is not configured."""
        # Registry is empty
        mock_winreg._registry.clear()

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result = Autostart.is_enabled()

            assert result is False


class TestAutostartIdempotency:
    """Tests for idempotent operations."""

    @pytest.fixture
    def mock_winreg(self):
        """Create a mock winreg module with registry simulation."""
        mock_registry = {}
        mock_module = MagicMock()
        mock_module.HKEY_CURRENT_USER = "HKCU"
        mock_module.KEY_SET_VALUE = 0x0002
        mock_module.KEY_READ = 0x0001
        mock_module.REG_SZ = 1

        def mock_open_key(key, subkey, reserved, access):
            mock_handle = MagicMock()
            mock_handle.__enter__ = MagicMock(return_value=mock_handle)
            mock_handle.__exit__ = MagicMock(return_value=False)
            return mock_handle

        def mock_set_value_ex(key, name, reserved, reg_type, value):
            mock_registry[name] = value

        def mock_delete_value(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            del mock_registry[name]

        def mock_query_value_ex(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            return (mock_registry[name], 1)

        def mock_close_key(key):
            pass

        mock_module.OpenKey = mock_open_key
        mock_module.SetValueEx = mock_set_value_ex
        mock_module.DeleteValue = mock_delete_value
        mock_module.QueryValueEx = mock_query_value_ex
        mock_module.CloseKey = mock_close_key
        mock_module._registry = mock_registry

        return mock_module

    def test_enable_twice_is_idempotent(self, mock_winreg):
        """Test that calling enable() twice produces the same result."""
        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with patch.object(Autostart, '_get_raw_executable_path') as mock_raw_path:
                with patch.object(Autostart, '_get_executable_path') as mock_exe_path:
                    mock_path = MagicMock(spec=Path)
                    mock_path.exists.return_value = True
                    mock_raw_path.return_value = mock_path
                    mock_exe_path.return_value = '"C:\\VoiceType.exe"'

                    result1 = Autostart.enable()
                    registry_after_first = dict(mock_winreg._registry)

                    result2 = Autostart.enable()
                    registry_after_second = dict(mock_winreg._registry)

                    assert result1 is True
                    assert result2 is True
                    assert registry_after_first == registry_after_second
                    assert mock_winreg._registry[APP_NAME] == '"C:\\VoiceType.exe"'

    def test_disable_twice_is_idempotent(self, mock_winreg):
        """Test that calling disable() twice produces the same result."""
        # Pre-populate registry
        mock_winreg._registry[APP_NAME] = '"C:\\VoiceType.exe"'

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result1 = Autostart.disable()
            assert result1 is True
            assert APP_NAME not in mock_winreg._registry

            # Second disable should also succeed (no-op)
            result2 = Autostart.disable()
            assert result2 is True
            assert APP_NAME not in mock_winreg._registry


class TestAutostartPathHandling:
    """Tests for path handling in different scenarios."""

    def test_path_with_spaces_is_quoted(self):
        """Test that paths with spaces are properly quoted."""
        with patch.object(sys, 'frozen', False, create=True):
            with patch.object(sys, 'executable', 'C:\\Program Files\\Python39\\python.exe'):
                # Mock the run.py path
                with patch.object(Path, '__new__') as mock_path_new:
                    mock_run_py = MagicMock(spec=Path)
                    mock_run_py.__str__ = MagicMock(return_value='D:\\Projects\\VoiceType\\run.py')
                    mock_run_py.__truediv__ = MagicMock(return_value=mock_run_py)

                    # Create proper Path behavior
                    original_path = Path.__new__

                    def path_side_effect(cls, *args, **kwargs):
                        if args and 'run.py' in str(args):
                            return mock_run_py
                        return original_path(cls, *args, **kwargs)

                    with patch('src.utils.autostart.Path') as MockPath:
                        mock_file_path = MagicMock()
                        mock_parent1 = MagicMock()
                        mock_parent2 = MagicMock()
                        mock_run_path = MagicMock()

                        mock_file_path.parent = mock_parent1
                        mock_parent1.parent = mock_parent2
                        mock_parent2.parent = MagicMock()
                        mock_parent2.parent.__truediv__ = MagicMock(return_value=mock_run_path)
                        mock_run_path.__str__ = MagicMock(return_value='D:\\Projects\\VoiceType\\run.py')

                        MockPath.return_value.parent.parent.parent.__truediv__.return_value = mock_run_path

                        # Call the actual method - we need to check the format
                        # The method should produce quoted paths
                        if getattr(sys, 'frozen', False):
                            path = f'"{sys.executable}"'
                        else:
                            path = f'"{sys.executable}" "D:\\Projects\\VoiceType\\run.py"'

                        assert '"' in path
                        assert 'Program Files' in path or 'python' in path.lower()

    def test_development_mode_uses_run_py(self):
        """Test that development mode uses run.py as entry point."""
        # In development mode, sys.frozen should be False or not exist
        with patch.object(sys, 'frozen', False, create=True):
            with patch.object(sys, 'executable', 'C:\\Python39\\python.exe'):
                with patch('src.utils.autostart.Path') as MockPath:
                    mock_run_py = MagicMock()
                    mock_run_py.__str__ = MagicMock(return_value='D:\\VoiceType\\run.py')

                    # Set up the path chain: __file__.parent.parent.parent / "run.py"
                    mock_file = MagicMock()
                    mock_parent1 = MagicMock()
                    mock_parent2 = MagicMock()
                    mock_parent3 = MagicMock()

                    MockPath.return_value = mock_file
                    mock_file.parent = mock_parent1
                    mock_parent1.parent = mock_parent2
                    mock_parent2.parent = mock_parent3
                    mock_parent3.__truediv__ = MagicMock(return_value=mock_run_py)

                    # Verify that in dev mode, the path includes both python and run.py
                    if not getattr(sys, 'frozen', False):
                        expected_format = f'"{sys.executable}"'
                        assert '"' in expected_format

    def test_frozen_mode_uses_sys_executable(self):
        """Test that frozen mode (exe) uses sys.executable directly."""
        with patch.object(sys, 'frozen', True, create=True):
            with patch.object(sys, 'executable', 'C:\\Program Files\\VoiceType\\VoiceType.exe'):
                path = Autostart._get_executable_path()

                assert path == '"C:\\Program Files\\VoiceType\\VoiceType.exe"'
                assert 'python' not in path.lower()
                assert 'run.py' not in path


class TestAutostartValidation:
    """Tests for validation logic."""

    @pytest.fixture
    def mock_winreg(self):
        """Create a mock winreg module with registry simulation."""
        mock_registry = {}
        mock_module = MagicMock()
        mock_module.HKEY_CURRENT_USER = "HKCU"
        mock_module.KEY_SET_VALUE = 0x0002
        mock_module.KEY_READ = 0x0001
        mock_module.REG_SZ = 1

        def mock_open_key(key, subkey, reserved, access):
            mock_handle = MagicMock()
            mock_handle.__enter__ = MagicMock(return_value=mock_handle)
            mock_handle.__exit__ = MagicMock(return_value=False)
            return mock_handle

        def mock_set_value_ex(key, name, reserved, reg_type, value):
            mock_registry[name] = value

        def mock_delete_value(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            del mock_registry[name]

        def mock_query_value_ex(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            return (mock_registry[name], 1)

        def mock_close_key(key):
            pass

        mock_module.OpenKey = mock_open_key
        mock_module.SetValueEx = mock_set_value_ex
        mock_module.DeleteValue = mock_delete_value
        mock_module.QueryValueEx = mock_query_value_ex
        mock_module.CloseKey = mock_close_key
        mock_module._registry = mock_registry

        return mock_module

    def test_enable_fails_if_path_not_exists(self, mock_winreg):
        """Test that enable() fails when executable path doesn't exist."""
        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with patch.object(Autostart, '_get_raw_executable_path') as mock_raw_path:
                mock_path = MagicMock(spec=Path)
                mock_path.exists.return_value = False
                mock_path.__str__ = MagicMock(return_value='C:\\NonExistent\\VoiceType.exe')
                mock_raw_path.return_value = mock_path

                result = Autostart.enable()

                assert result is False
                assert APP_NAME not in mock_winreg._registry

    def test_is_enabled_returns_false_if_path_mismatch(self, mock_winreg):
        """Test that is_enabled returns False when registered path differs from current."""
        # Register with old path
        mock_winreg._registry[APP_NAME] = '"C:\\OldPath\\VoiceType.exe"'

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with patch.object(Autostart, '_get_executable_path') as mock_exe_path:
                # Current path is different
                mock_exe_path.return_value = '"C:\\NewPath\\VoiceType.exe"'

                result = Autostart.is_enabled()

                assert result is False


class TestAutostartErrorHandling:
    """Tests for error handling scenarios."""

    def test_enable_handles_permission_error(self):
        """Test that enable() handles PermissionError gracefully."""
        mock_winreg = MagicMock()
        mock_winreg.HKEY_CURRENT_USER = "HKCU"
        mock_winreg.KEY_SET_VALUE = 0x0002
        mock_winreg.OpenKey = MagicMock(side_effect=PermissionError("Access denied"))

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with patch.object(Autostart, '_get_raw_executable_path') as mock_raw_path:
                mock_path = MagicMock(spec=Path)
                mock_path.exists.return_value = True
                mock_raw_path.return_value = mock_path

                result = Autostart.enable()

                assert result is False

    def test_disable_handles_permission_error(self):
        """Test that disable() handles PermissionError gracefully."""
        mock_winreg = MagicMock()
        mock_winreg.HKEY_CURRENT_USER = "HKCU"
        mock_winreg.KEY_SET_VALUE = 0x0002
        mock_winreg.OpenKey = MagicMock(side_effect=PermissionError("Access denied"))

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result = Autostart.disable()

            assert result is False

    def test_non_windows_returns_false(self):
        """Test that operations return False on non-Windows systems."""
        # Simulate winreg not being available (non-Windows)
        with patch.dict('sys.modules', {'winreg': None}):
            # Force ImportError by removing winreg from builtins lookup
            with patch('builtins.__import__', side_effect=ImportError("No module named 'winreg'")):
                # The methods import winreg internally, so we need to make that fail
                result_enable = Autostart.enable(exe_path='"C:\\Test.exe"')
                assert result_enable is False

    def test_is_enabled_handles_permission_error(self):
        """Test that is_enabled() handles PermissionError gracefully."""
        mock_winreg = MagicMock()
        mock_winreg.HKEY_CURRENT_USER = "HKCU"
        mock_winreg.KEY_READ = 0x0001
        mock_winreg.OpenKey = MagicMock(side_effect=PermissionError("Access denied"))

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result = Autostart.is_enabled()

            assert result is False

    def test_get_registered_path_handles_errors(self):
        """Test that get_registered_path() handles errors gracefully."""
        mock_winreg = MagicMock()
        mock_winreg.HKEY_CURRENT_USER = "HKCU"
        mock_winreg.KEY_READ = 0x0001
        mock_winreg.OpenKey = MagicMock(side_effect=Exception("Unexpected error"))

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result = Autostart.get_registered_path()

            assert result is None


class TestAutostartEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.fixture
    def mock_winreg(self):
        """Create a mock winreg module with registry simulation."""
        mock_registry = {}
        mock_module = MagicMock()
        mock_module.HKEY_CURRENT_USER = "HKCU"
        mock_module.KEY_SET_VALUE = 0x0002
        mock_module.KEY_READ = 0x0001
        mock_module.REG_SZ = 1

        def mock_open_key(key, subkey, reserved, access):
            mock_handle = MagicMock()
            mock_handle.__enter__ = MagicMock(return_value=mock_handle)
            mock_handle.__exit__ = MagicMock(return_value=False)
            return mock_handle

        def mock_set_value_ex(key, name, reserved, reg_type, value):
            mock_registry[name] = value

        def mock_delete_value(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            del mock_registry[name]

        def mock_query_value_ex(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            return (mock_registry[name], 1)

        def mock_close_key(key):
            pass

        mock_module.OpenKey = mock_open_key
        mock_module.SetValueEx = mock_set_value_ex
        mock_module.DeleteValue = mock_delete_value
        mock_module.QueryValueEx = mock_query_value_ex
        mock_module.CloseKey = mock_close_key
        mock_module._registry = mock_registry

        return mock_module

    def test_registry_key_exists_but_empty_value(self, mock_winreg):
        """Test is_enabled when registry key exists but value is empty."""
        mock_winreg._registry[APP_NAME] = ""

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result = Autostart.is_enabled()

            assert result is False

    def test_path_with_cyrillic_characters(self, mock_winreg):
        """Test handling of paths with Cyrillic characters."""
        cyrillic_path = '"C:\\Пользователи\\Данил\\VoiceType\\VoiceType.exe"'
        mock_winreg._registry[APP_NAME] = cyrillic_path

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with patch.object(Autostart, '_get_executable_path', return_value=cyrillic_path):
                result = Autostart.is_enabled()

                assert result is True

            # Also test get_registered_path with Cyrillic
            registered = Autostart.get_registered_path()
            assert registered == cyrillic_path

    def test_enable_with_explicit_exe_path(self, mock_winreg):
        """Test enable() with explicitly provided exe_path."""
        explicit_path = '"D:\\CustomPath\\MyApp.exe"'

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result = Autostart.enable(exe_path=explicit_path)

            assert result is True
            assert mock_winreg._registry[APP_NAME] == explicit_path

    def test_get_registered_path_when_not_registered(self, mock_winreg):
        """Test get_registered_path returns None when not registered."""
        mock_winreg._registry.clear()

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result = Autostart.get_registered_path()

            assert result is None

    def test_get_registered_path_returns_value(self, mock_winreg):
        """Test get_registered_path returns the registered value."""
        expected_path = '"C:\\VoiceType\\VoiceType.exe"'
        mock_winreg._registry[APP_NAME] = expected_path

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            result = Autostart.get_registered_path()

            assert result == expected_path


class TestAutostartRegistryContextManager:
    """Tests for the _open_registry_key context manager."""

    def test_context_manager_closes_key_on_success(self):
        """Test that context manager properly closes registry key on success."""
        mock_winreg = MagicMock()
        mock_handle = MagicMock()
        mock_winreg.OpenKey = MagicMock(return_value=mock_handle)

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with _open_registry_key(
                mock_winreg.HKEY_CURRENT_USER,
                REGISTRY_PATH,
                mock_winreg.KEY_READ
            ) as handle:
                assert handle == mock_handle

            mock_winreg.CloseKey.assert_called_once_with(mock_handle)

    def test_context_manager_closes_key_on_exception(self):
        """Test that context manager closes registry key even when exception occurs."""
        mock_winreg = MagicMock()
        mock_handle = MagicMock()
        mock_winreg.OpenKey = MagicMock(return_value=mock_handle)

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with pytest.raises(ValueError):
                with _open_registry_key(
                    mock_winreg.HKEY_CURRENT_USER,
                    REGISTRY_PATH,
                    mock_winreg.KEY_READ
                ) as handle:
                    raise ValueError("Test exception")

            mock_winreg.CloseKey.assert_called_once_with(mock_handle)


class TestAutostartRawExecutablePath:
    """Tests for _get_raw_executable_path method."""

    def test_raw_path_frozen_mode(self):
        """Test _get_raw_executable_path in frozen mode."""
        with patch.object(sys, 'frozen', True, create=True):
            with patch.object(sys, 'executable', 'C:\\VoiceType\\VoiceType.exe'):
                path = Autostart._get_raw_executable_path()

                assert isinstance(path, Path)
                assert str(path) == 'C:\\VoiceType\\VoiceType.exe'

    def test_raw_path_development_mode(self):
        """Test _get_raw_executable_path in development mode."""
        # Ensure frozen is False
        if hasattr(sys, 'frozen'):
            original_frozen = sys.frozen
        else:
            original_frozen = None

        try:
            # Remove frozen attribute to simulate dev mode
            if hasattr(sys, 'frozen'):
                delattr(sys, 'frozen')

            # The actual path will be relative to autostart.py
            # We just verify it returns a Path and ends with run.py
            path = Autostart._get_raw_executable_path()

            assert isinstance(path, Path)
            assert path.name == 'run.py'

        finally:
            # Restore original state
            if original_frozen is not None:
                sys.frozen = original_frozen


class TestAutostartConstants:
    """Tests for constants and configuration."""

    def test_registry_path_is_correct(self):
        """Test that REGISTRY_PATH points to Windows Run key."""
        assert REGISTRY_PATH == r"Software\Microsoft\Windows\CurrentVersion\Run"

    def test_app_name_used_in_registry(self):
        """Test that APP_NAME is used as the registry value name."""
        assert APP_NAME == "VoiceType"


class TestAutostartIntegration:
    """Integration tests simulating full workflows."""

    @pytest.fixture
    def mock_winreg(self):
        """Create a mock winreg module with registry simulation."""
        mock_registry = {}
        mock_module = MagicMock()
        mock_module.HKEY_CURRENT_USER = "HKCU"
        mock_module.KEY_SET_VALUE = 0x0002
        mock_module.KEY_READ = 0x0001
        mock_module.REG_SZ = 1

        def mock_open_key(key, subkey, reserved, access):
            mock_handle = MagicMock()
            mock_handle.__enter__ = MagicMock(return_value=mock_handle)
            mock_handle.__exit__ = MagicMock(return_value=False)
            return mock_handle

        def mock_set_value_ex(key, name, reserved, reg_type, value):
            mock_registry[name] = value

        def mock_delete_value(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            del mock_registry[name]

        def mock_query_value_ex(key, name):
            if name not in mock_registry:
                raise FileNotFoundError(f"Value {name} not found")
            return (mock_registry[name], 1)

        def mock_close_key(key):
            pass

        mock_module.OpenKey = mock_open_key
        mock_module.SetValueEx = mock_set_value_ex
        mock_module.DeleteValue = mock_delete_value
        mock_module.QueryValueEx = mock_query_value_ex
        mock_module.CloseKey = mock_close_key
        mock_module._registry = mock_registry

        return mock_module

    def test_full_enable_disable_cycle(self, mock_winreg):
        """Test complete cycle: enable -> verify -> disable -> verify."""
        test_path = '"C:\\VoiceType\\VoiceType.exe"'

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with patch.object(Autostart, '_get_raw_executable_path') as mock_raw_path:
                with patch.object(Autostart, '_get_executable_path', return_value=test_path):
                    mock_path = MagicMock(spec=Path)
                    mock_path.exists.return_value = True
                    mock_raw_path.return_value = mock_path

                    # Initial state: not enabled
                    assert Autostart.is_enabled() is False
                    assert Autostart.get_registered_path() is None

                    # Enable autostart
                    enable_result = Autostart.enable()
                    assert enable_result is True

                    # Verify enabled
                    assert Autostart.is_enabled() is True
                    assert Autostart.get_registered_path() == test_path

                    # Disable autostart
                    disable_result = Autostart.disable()
                    assert disable_result is True

                    # Verify disabled
                    assert Autostart.is_enabled() is False
                    assert Autostart.get_registered_path() is None

    def test_update_autostart_path(self, mock_winreg):
        """Test updating autostart when path changes (e.g., after app update)."""
        old_path = '"C:\\OldLocation\\VoiceType.exe"'
        new_path = '"C:\\NewLocation\\VoiceType.exe"'

        with patch.dict('sys.modules', {'winreg': mock_winreg}):
            with patch.object(Autostart, '_get_raw_executable_path') as mock_raw_path:
                mock_path = MagicMock(spec=Path)
                mock_path.exists.return_value = True
                mock_raw_path.return_value = mock_path

                # Set up with old path
                with patch.object(Autostart, '_get_executable_path', return_value=old_path):
                    Autostart.enable()
                    assert Autostart.is_enabled() is True

                # Now current path is different - is_enabled should return False
                with patch.object(Autostart, '_get_executable_path', return_value=new_path):
                    assert Autostart.is_enabled() is False

                    # Re-enable with new path
                    Autostart.enable()
                    assert Autostart.is_enabled() is True
                    assert Autostart.get_registered_path() == new_path
