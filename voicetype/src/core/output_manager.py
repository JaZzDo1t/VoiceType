"""
VoiceType - Output Manager
Вывод распознанного текста в активное окно или буфер обмена.
"""
import time
from typing import Optional
from loguru import logger

from src.utils.constants import (
    OUTPUT_MODE_KEYBOARD, OUTPUT_MODE_CLIPBOARD,
    DEFAULT_TYPING_DELAY, LAYOUT_SWITCH_DELAY
)


class OutputManager:
    """
    Выводит текст в активное окно или буфер обмена.
    Поддерживает два режима: keyboard (эмуляция набора) и clipboard (копирование).
    """

    def __init__(self, mode: str = OUTPUT_MODE_KEYBOARD, language: str = "ru"):
        """
        Args:
            mode: Режим вывода - "keyboard" или "clipboard"
            language: Язык для переключения раскладки ("ru" или "en")
        """
        self._mode = mode
        self._language = language
        self._keyboard_controller = None
        self._typing_delay = DEFAULT_TYPING_DELAY

    @property
    def mode(self) -> str:
        """Текущий режим вывода."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """
        Изменить режим вывода.

        Args:
            mode: "keyboard" или "clipboard"
        """
        if mode not in (OUTPUT_MODE_KEYBOARD, OUTPUT_MODE_CLIPBOARD):
            logger.warning(f"Invalid output mode: {mode}, using keyboard")
            mode = OUTPUT_MODE_KEYBOARD

        self._mode = mode
        logger.debug(f"Output mode set to: {mode}")

    def set_language(self, language: str) -> None:
        """
        Установить язык для переключения раскладки.

        Args:
            language: "ru" или "en"
        """
        self._language = language
        logger.debug(f"Output language set to: {language}")

    def output(self, text: str) -> bool:
        """
        Вывести текст согласно текущему режиму.

        Args:
            text: Текст для вывода

        Returns:
            True если успешно
        """
        if not text:
            return True

        if self._mode == OUTPUT_MODE_KEYBOARD:
            return self.output_to_keyboard(text)
        else:
            return self.output_to_clipboard(text)

    def output_to_keyboard(self, text: str) -> bool:
        """
        Эмуляция набора текста через pynput.
        Перед набором переключает раскладку на нужный язык.

        Args:
            text: Текст для набора

        Returns:
            True если успешно
        """
        try:
            from pynput.keyboard import Controller

            if self._keyboard_controller is None:
                self._keyboard_controller = Controller()

            # Переключаем раскладку на нужный язык
            self._switch_keyboard_layout(self._language)

            # Задержка после смены раскладки для надёжности
            time.sleep(LAYOUT_SWITCH_DELAY)

            # Печатаем текст
            self._keyboard_controller.type(text)

            logger.debug(f"Typed to keyboard: {text[:50]}...")
            return True

        except ImportError:
            logger.error("pynput library not installed")
            return False

        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return False

    def _switch_keyboard_layout(self, language: str) -> bool:
        """
        Переключить раскладку клавиатуры на нужный язык.

        Args:
            language: "ru" или "en"

        Returns:
            True если успешно
        """
        try:
            import ctypes
            from ctypes import wintypes

            # Коды раскладок Windows (KLID - Keyboard Layout ID)
            # Формат: строка из 8 hex-цифр
            layout_klid = {
                "ru": "00000419",
                "en": "00000409",
            }

            klid = layout_klid.get(language, "00000409")

            user32 = ctypes.WinDLL('user32', use_last_error=True)

            # Настраиваем типы для LoadKeyboardLayoutW
            user32.LoadKeyboardLayoutW.argtypes = [wintypes.LPCWSTR, wintypes.UINT]
            user32.LoadKeyboardLayoutW.restype = wintypes.HKL

            # KLF_ACTIVATE = 0x00000001 - активировать раскладку
            KLF_ACTIVATE = 0x00000001

            # LoadKeyboardLayoutW загружает и активирует раскладку, возвращает HKL
            hkl = user32.LoadKeyboardLayoutW(klid, KLF_ACTIVATE)

            if not hkl:
                logger.warning(f"Failed to load keyboard layout: {klid}")
                return False

            # Дополнительно отправляем WM_INPUTLANGCHANGEREQUEST активному окну
            # lParam должен быть HKL (результат LoadKeyboardLayoutW)
            hwnd = user32.GetForegroundWindow()
            if hwnd:
                user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                                wintypes.WPARAM, wintypes.LPARAM]
                # WM_INPUTLANGCHANGEREQUEST = 0x0050
                user32.PostMessageW(hwnd, 0x0050, 0, hkl)

            logger.debug(f"Switched keyboard layout to: {language} (HKL=0x{hkl:08X})")
            return True

        except Exception as e:
            logger.warning(f"Failed to switch keyboard layout: {e}")
            return False

    def output_to_clipboard(self, text: str) -> bool:
        """
        Копировать текст в буфер обмена.

        Args:
            text: Текст для копирования

        Returns:
            True если успешно
        """
        try:
            import pyperclip

            pyperclip.copy(text)

            logger.debug(f"Copied to clipboard: {text[:50]}...")
            return True

        except ImportError:
            logger.error("pyperclip library not installed")
            return False

        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            return False
