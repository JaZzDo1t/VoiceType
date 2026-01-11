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
        Если не удаётся - использует clipboard + Ctrl+V как fallback.

        Args:
            text: Текст для набора

        Returns:
            True если успешно
        """
        try:
            from pynput.keyboard import Controller, Key

            if self._keyboard_controller is None:
                self._keyboard_controller = Controller()

            # Переключаем раскладку на нужный язык
            layout_switched = self._switch_keyboard_layout(self._language)

            # Задержка после смены раскладки для надёжности
            time.sleep(LAYOUT_SWITCH_DELAY)

            # Проверяем текущую раскладку
            current_layout = self._get_current_layout_language()

            # Если раскладка не переключилась и текст содержит кириллицу -
            # используем clipboard + Ctrl+V как более надёжный метод
            has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in text)

            if has_cyrillic and current_layout != "ru":
                logger.warning("Layout not switched to Russian, using clipboard fallback")
                return self._output_via_clipboard_paste(text)

            # Печатаем текст
            self._keyboard_controller.type(text)

            logger.debug(f"Typed to keyboard: {text[:50]}...")
            return True

        except ImportError:
            logger.error("pynput library not installed")
            return False

        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            # Fallback на clipboard
            logger.info("Falling back to clipboard paste")
            return self._output_via_clipboard_paste(text)

    def _output_via_clipboard_paste(self, text: str) -> bool:
        """
        Вывод текста через буфер обмена и Ctrl+V.
        Более надёжный метод, не зависит от раскладки.

        Args:
            text: Текст для вывода

        Returns:
            True если успешно
        """
        try:
            import pyperclip
            from pynput.keyboard import Controller, Key

            if self._keyboard_controller is None:
                self._keyboard_controller = Controller()

            # Сохраняем текущее содержимое буфера
            try:
                old_clipboard = pyperclip.paste()
            except Exception:
                old_clipboard = None

            # Копируем текст в буфер
            pyperclip.copy(text)
            time.sleep(0.05)  # Небольшая задержка

            # Вставляем через Ctrl+V
            with self._keyboard_controller.pressed(Key.ctrl):
                self._keyboard_controller.tap('v')

            time.sleep(0.05)

            # Восстанавливаем буфер (опционально)
            # Закомментировано, т.к. может мешать пользователю
            # if old_clipboard is not None:
            #     time.sleep(0.1)
            #     pyperclip.copy(old_clipboard)

            logger.debug(f"Pasted via Ctrl+V: {text[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to paste via clipboard: {e}")
            return False

    def _get_current_layout_language(self) -> Optional[str]:
        """
        Получить текущий язык раскладки клавиатуры.

        Returns:
            "ru", "en" или None если не удалось определить
        """
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL('user32', use_last_error=True)

            # Получаем HKL текущей раскладки для активного окна
            hwnd = user32.GetForegroundWindow()
            thread_id = user32.GetWindowThreadProcessId(hwnd, None)

            user32.GetKeyboardLayout.argtypes = [wintypes.DWORD]
            user32.GetKeyboardLayout.restype = wintypes.HKL

            hkl = user32.GetKeyboardLayout(thread_id)

            # Младшие 16 бит HKL - это Language ID (LANGID)
            lang_id = hkl & 0xFFFF

            # Russian = 0x0419, English US = 0x0409
            if lang_id == 0x0419:
                return "ru"
            elif lang_id == 0x0409:
                return "en"
            else:
                logger.debug(f"Unknown layout language ID: 0x{lang_id:04X}")
                return None

        except Exception as e:
            logger.warning(f"Failed to get current layout: {e}")
            return None

    def _switch_keyboard_layout(self, language: str) -> bool:
        """
        Переключить раскладку клавиатуры на нужный язык.
        Использует несколько методов для надёжности.

        Args:
            language: "ru" или "en"

        Returns:
            True если успешно
        """
        try:
            import ctypes
            from ctypes import wintypes

            # Проверяем текущую раскладку
            current = self._get_current_layout_language()
            if current == language:
                logger.debug(f"Layout already set to: {language}")
                return True

            user32 = ctypes.WinDLL('user32', use_last_error=True)

            # Коды раскладок Windows (KLID - Keyboard Layout ID)
            layout_klid = {
                "ru": "00000419",
                "en": "00000409",
            }
            klid = layout_klid.get(language, "00000409")

            # Метод 1: LoadKeyboardLayoutW с KLF_ACTIVATE
            user32.LoadKeyboardLayoutW.argtypes = [wintypes.LPCWSTR, wintypes.UINT]
            user32.LoadKeyboardLayoutW.restype = wintypes.HKL

            KLF_ACTIVATE = 0x00000001
            hkl = user32.LoadKeyboardLayoutW(klid, KLF_ACTIVATE)

            if not hkl:
                logger.warning(f"Failed to load keyboard layout: {klid}")
                return False

            # Метод 2: ActivateKeyboardLayout для текущего потока
            user32.ActivateKeyboardLayout.argtypes = [wintypes.HKL, wintypes.UINT]
            user32.ActivateKeyboardLayout.restype = wintypes.HKL
            user32.ActivateKeyboardLayout(hkl, 0)

            # Метод 3: WM_INPUTLANGCHANGEREQUEST для активного окна
            hwnd = user32.GetForegroundWindow()
            if hwnd:
                user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                                wintypes.WPARAM, wintypes.LPARAM]
                # WM_INPUTLANGCHANGEREQUEST = 0x0050
                user32.PostMessageW(hwnd, 0x0050, 0, hkl)

                # Также пробуем SendMessage для синхронного переключения
                user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                                wintypes.WPARAM, wintypes.LPARAM]
                user32.SendMessageW(hwnd, 0x0050, 0, hkl)

            # Даём время на переключение
            time.sleep(0.05)

            # Проверяем результат
            new_layout = self._get_current_layout_language()
            if new_layout == language:
                logger.debug(f"Successfully switched layout to: {language}")
                return True
            else:
                logger.warning(f"Layout switch may have failed. Target: {language}, Current: {new_layout}")
                # Всё равно возвращаем True - попытка была сделана
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
