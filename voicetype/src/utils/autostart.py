"""
VoiceType - Windows Autostart Manager
Управление автозапуском приложения при старте Windows через реестр.
"""
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from loguru import logger

from src.utils.constants import APP_NAME


# Windows Registry path for autostart
REGISTRY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"


@contextmanager
def _open_registry_key(key, subkey, access):
    """
    Контекстный менеджер для безопасной работы с ключами реестра.
    Гарантирует закрытие ключа даже при возникновении исключения.
    """
    import winreg
    handle = winreg.OpenKey(key, subkey, 0, access)
    try:
        yield handle
    finally:
        winreg.CloseKey(handle)


class Autostart:
    """
    Управление автозапуском при старте Windows.
    Использует реестр Windows (HKEY_CURRENT_USER).
    """

    @staticmethod
    def _get_executable_path() -> str:
        """Получить путь к исполняемому файлу."""
        if getattr(sys, 'frozen', False):
            # Запущено как скомпилированный exe
            # Wrap in quotes for paths with spaces (e.g., "Program Files")
            return f'"{sys.executable}"'
        else:
            # Запущено как Python скрипт - use run.py as entry point
            run_py_path = Path(__file__).parent.parent.parent / "run.py"
            return f'"{sys.executable}" "{run_py_path}"'

    @staticmethod
    def _get_raw_executable_path() -> Path:
        """Получить путь к исполняемому файлу без кавычек для проверки существования."""
        if getattr(sys, 'frozen', False):
            return Path(sys.executable)
        else:
            return Path(__file__).parent.parent.parent / "run.py"

    @staticmethod
    def enable(exe_path: str = None) -> bool:
        """
        Включить автозапуск.

        Args:
            exe_path: Путь к exe. Если не указан, определяется автоматически.

        Returns:
            True если успешно
        """
        try:
            import winreg

            if exe_path is None:
                # Validate that executable path exists before registering
                raw_path = Autostart._get_raw_executable_path()
                if not raw_path.exists():
                    logger.error(f"Executable path does not exist: {raw_path}")
                    return False
                exe_path = Autostart._get_executable_path()

            # Открываем ключ реестра с использованием контекстного менеджера
            with _open_registry_key(
                winreg.HKEY_CURRENT_USER,
                REGISTRY_PATH,
                winreg.KEY_SET_VALUE
            ) as key:
                # Устанавливаем значение
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, exe_path)

            logger.info(f"Autostart enabled: {exe_path}")
            return True

        except ImportError:
            logger.error("winreg module not available (not Windows?)")
            return False
        except PermissionError:
            logger.error("Permission denied to modify registry")
            return False
        except Exception as e:
            logger.error(f"Failed to enable autostart: {e}")
            return False

    @staticmethod
    def disable() -> bool:
        """
        Выключить автозапуск.

        Returns:
            True если успешно
        """
        try:
            import winreg

            with _open_registry_key(
                winreg.HKEY_CURRENT_USER,
                REGISTRY_PATH,
                winreg.KEY_SET_VALUE
            ) as key:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                    logger.info("Autostart disabled")
                except FileNotFoundError:
                    # Значение уже не существует
                    logger.debug("Autostart was not enabled")

            return True

        except ImportError:
            logger.error("winreg module not available (not Windows?)")
            return False
        except PermissionError:
            logger.error("Permission denied to modify registry")
            return False
        except Exception as e:
            logger.error(f"Failed to disable autostart: {e}")
            return False

    @staticmethod
    def is_enabled() -> bool:
        """
        Проверить, включен ли автозапуск.

        Returns:
            True если автозапуск включен и путь соответствует текущему исполняемому файлу
        """
        try:
            import winreg

            with _open_registry_key(
                winreg.HKEY_CURRENT_USER,
                REGISTRY_PATH,
                winreg.KEY_READ
            ) as key:
                try:
                    value, _ = winreg.QueryValueEx(key, APP_NAME)
                    if not value:
                        return False

                    # Validate that registered path matches current executable
                    current_path = Autostart._get_executable_path()
                    if value != current_path:
                        logger.warning(
                            f"Autostart path mismatch. Registered: {value}, Current: {current_path}"
                        )
                        return False

                    return True
                except FileNotFoundError:
                    return False

        except ImportError:
            return False
        except Exception as e:
            logger.error(f"Failed to check autostart status: {e}")
            return False

    @staticmethod
    def get_registered_path() -> Optional[str]:
        """
        Получить путь, зарегистрированный в автозапуске.

        Returns:
            Путь или None если не зарегистрирован
        """
        try:
            import winreg

            with _open_registry_key(
                winreg.HKEY_CURRENT_USER,
                REGISTRY_PATH,
                winreg.KEY_READ
            ) as key:
                try:
                    value, _ = winreg.QueryValueEx(key, APP_NAME)
                    return value
                except FileNotFoundError:
                    return None

        except Exception:
            return None
