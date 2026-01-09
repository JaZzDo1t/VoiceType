"""
VoiceType - Hotkey Manager
Управление глобальными горячими клавишами.
"""
import sys
import threading
import time
from typing import Dict, Callable, Optional, Set, Tuple
from loguru import logger

from src.utils.constants import DEFAULT_HOTKEY_START, DEFAULT_HOTKEY_STOP

# Debounce interval in seconds
HOTKEY_DEBOUNCE_INTERVAL = 0.3  # 300ms


class HotkeyManager:
    """
    Управление глобальными хоткеями.
    Работает даже когда приложение не в фокусе.
    Использует pynput для прослушивания клавиатуры.
    """

    def __init__(self):
        self._hotkeys: Dict[str, Callable] = {}
        self._listener = None
        self._is_listening = False
        self._pressed_keys: Set[str] = set()
        self._lock = threading.Lock()
        # Track triggered hotkeys to prevent repeated firing while keys held
        self._triggered_hotkeys: Set[str] = set()
        # Track last trigger time for debouncing
        self._last_trigger_time: Dict[str, float] = {}

    @staticmethod
    def parse_hotkey(hotkey_str: str) -> Tuple[Set[str], str]:
        """
        Распарсить строку хоткея в компоненты.

        Args:
            hotkey_str: Строка вида "ctrl+shift+s"

        Returns:
            (modifiers, key) - множество модификаторов и основная клавиша
        """
        parts = hotkey_str.lower().replace(" ", "").split("+")

        # Модификаторы
        modifier_names = {"ctrl", "control", "alt", "shift", "cmd", "win", "super"}
        modifiers = {p for p in parts[:-1] if p in modifier_names}

        # Нормализуем названия модификаторов
        normalized_modifiers = set()
        for mod in modifiers:
            if mod in ("ctrl", "control"):
                normalized_modifiers.add("ctrl")
            elif mod in ("cmd", "win", "super"):
                normalized_modifiers.add("cmd")
            else:
                normalized_modifiers.add(mod)

        # Основная клавиша (последняя)
        key = parts[-1] if parts else ""

        return normalized_modifiers, key

    @staticmethod
    def normalize_hotkey(hotkey_str: str) -> str:
        """
        Нормализовать строку хоткея.

        Args:
            hotkey_str: "Ctrl + Shift + S" или "ctrl+shift+s"

        Returns:
            Нормализованная строка "ctrl+shift+s"
        """
        modifiers, key = HotkeyManager.parse_hotkey(hotkey_str)

        # Сортируем модификаторы для консистентности
        mod_order = ["ctrl", "alt", "shift", "cmd"]
        sorted_mods = sorted(modifiers, key=lambda x: mod_order.index(x) if x in mod_order else 99)

        return "+".join(sorted_mods + [key])

    def register(self, hotkey: str, callback: Callable) -> bool:
        """
        Зарегистрировать хоткей.

        Args:
            hotkey: Строка хоткея, например "ctrl+shift+s"
            callback: Функция, вызываемая при нажатии

        Returns:
            True если успешно
        """
        normalized = self.normalize_hotkey(hotkey)

        with self._lock:
            self._hotkeys[normalized] = callback
            logger.info(f"Hotkey registered: {normalized}")

        return True

    def unregister(self, hotkey: str) -> bool:
        """
        Снять регистрацию хоткея.

        Args:
            hotkey: Строка хоткея

        Returns:
            True если был зарегистрирован и снят
        """
        normalized = self.normalize_hotkey(hotkey)

        with self._lock:
            if normalized in self._hotkeys:
                del self._hotkeys[normalized]
                logger.info(f"Hotkey unregistered: {normalized}")
                return True

        return False

    def unregister_all(self) -> None:
        """Снять все хоткеи."""
        with self._lock:
            self._hotkeys.clear()
            logger.info("All hotkeys unregistered")

    def start_listening(self) -> bool:
        """
        Начать слушать хоткеи.

        Returns:
            True если успешно запущен
        """
        if self._is_listening:
            logger.warning("Hotkey listener already running")
            return True

        is_frozen = getattr(sys, 'frozen', False)
        if is_frozen:
            logger.info("Starting hotkey listener in frozen build...")
            # Проверяем флаг runtime hook
            pynput_hook = getattr(sys, '_pynput_rthook_success', None)
            if pynput_hook is True:
                logger.debug("pynput runtime hook: SUCCESS")
            elif pynput_hook is False:
                logger.warning("pynput runtime hook: FAILED - hotkeys may not work")
            else:
                logger.debug("pynput runtime hook: NOT RUN (not frozen or hook missing)")

        try:
            from pynput import keyboard

            if is_frozen:
                logger.debug(f"pynput.keyboard module loaded: {keyboard}")
                logger.debug(f"pynput.keyboard.Listener: {keyboard.Listener}")

            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self._listener.start()
            self._is_listening = True

            # Логируем зарегистрированные хоткеи
            registered = self.get_registered_hotkeys()
            logger.info(f"Hotkey listener started, registered hotkeys: {list(registered.keys())}")

            return True

        except ImportError as e:
            logger.error(f"pynput library not installed or import failed: {e}")
            if is_frozen:
                logger.error("In frozen build - check that pynput is included in hiddenimports")
            return False

        except Exception as e:
            logger.error(f"Failed to start hotkey listener: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False

    def stop_listening(self) -> None:
        """Остановить прослушивание."""
        if self._listener:
            try:
                self._listener.stop()
            except Exception as e:
                logger.debug(f"Error stopping listener: {e}")
            self._listener = None

        self._is_listening = False
        with self._lock:
            self._pressed_keys.clear()
            self._triggered_hotkeys.clear()
            self._last_trigger_time.clear()
        logger.info("Hotkey listener stopped")

    def validate_listener(self) -> bool:
        """
        Проверить, что listener действительно работает.

        Returns:
            True если listener активен и работает
        """
        if not self._is_listening:
            logger.warning("validate_listener: _is_listening is False")
            return False

        if self._listener is None:
            logger.warning("validate_listener: _listener is None")
            return False

        # Проверяем, что поток listener жив
        try:
            is_alive = self._listener.is_alive()
            if not is_alive:
                logger.error("validate_listener: listener thread is NOT alive")
                self._is_listening = False
                return False

            logger.debug(f"validate_listener: listener thread is alive, daemon={self._listener.daemon}")
            return True

        except Exception as e:
            logger.error(f"validate_listener: error checking listener: {e}")
            return False

    def _key_to_string(self, key) -> Optional[str]:
        """Преобразовать объект клавиши pynput в строку."""
        from pynput.keyboard import Key

        try:
            # Специальные клавиши (модификаторы)
            if key == Key.ctrl_l or key == Key.ctrl_r:
                return "ctrl"
            elif key == Key.alt_l or key == Key.alt_r or key == Key.alt_gr:
                return "alt"
            elif key == Key.shift_l or key == Key.shift_r:
                return "shift"
            elif key == Key.cmd_l or key == Key.cmd_r:
                return "cmd"

            # Для обычных клавиш - СНАЧАЛА проверяем vk (virtual key code)
            # Это важно! Когда нажат Ctrl+буква, key.char возвращает control code
            # (например Ctrl+T -> '\x14'), но vk содержит правильный код клавиши
            if hasattr(key, 'vk') and key.vk:
                vk = key.vk
                # A-Z: vk 65-90
                if 65 <= vk <= 90:
                    return chr(vk).lower()
                # 0-9: vk 48-57
                elif 48 <= vk <= 57:
                    return chr(vk)
                # Другие печатные символы
                elif 32 <= vk <= 126:
                    return chr(vk).lower()

            # Fallback на char (для случаев без модификаторов)
            if hasattr(key, 'char') and key.char:
                # Проверяем что это НЕ control character
                if ord(key.char) >= 32:
                    return key.char.lower()

            # Специальные клавиши по имени
            if hasattr(key, 'name') and key.name:
                return key.name.lower()

            return str(key).replace("Key.", "").lower()

        except Exception as e:
            logger.debug(f"_key_to_string error: {e} for key {key}")
            return None

    def _on_press(self, key) -> None:
        """Обработчик нажатия клавиши."""
        key_str = self._key_to_string(key)
        if key_str:
            with self._lock:
                self._pressed_keys.add(key_str)
            self._check_hotkeys()

    def _on_release(self, key) -> None:
        """Обработчик отпускания клавиши."""
        key_str = self._key_to_string(key)
        if key_str:
            with self._lock:
                self._pressed_keys.discard(key_str)
                # Reset triggered hotkeys that used this key so they can fire again
                hotkeys_to_reset = set()
                for hotkey_str in self._triggered_hotkeys:
                    modifiers, hotkey_key = self.parse_hotkey(hotkey_str)
                    all_keys = modifiers | {hotkey_key}
                    if key_str in all_keys:
                        hotkeys_to_reset.add(hotkey_str)
                self._triggered_hotkeys -= hotkeys_to_reset

    def _check_hotkeys(self) -> None:
        """Проверить, нажат ли зарегистрированный хоткей."""
        callback_to_run = None
        hotkey_triggered = None

        with self._lock:
            current_time = time.time()

            for hotkey_str, callback in self._hotkeys.items():
                # Skip already triggered hotkeys (prevents repeated firing while held)
                if hotkey_str in self._triggered_hotkeys:
                    continue

                modifiers, key = self.parse_hotkey(hotkey_str)

                # Проверяем, что все клавиши хоткея нажаты
                all_keys = modifiers | {key}

                if all_keys <= self._pressed_keys:
                    # Check debounce - ensure 300ms passed since last trigger
                    last_time = self._last_trigger_time.get(hotkey_str, 0)
                    if current_time - last_time < HOTKEY_DEBOUNCE_INTERVAL:
                        logger.debug(f"Hotkey {hotkey_str} debounced (too soon)")
                        continue

                    logger.debug(f"Hotkey triggered: {hotkey_str}")
                    # Mark as triggered and update last trigger time
                    self._triggered_hotkeys.add(hotkey_str)
                    self._last_trigger_time[hotkey_str] = current_time
                    # Save callback to run outside lock
                    callback_to_run = callback
                    hotkey_triggered = hotkey_str
                    break

        # Spawn thread outside lock to avoid holding lock during thread creation
        if callback_to_run is not None:
            def safe_callback_wrapper():
                try:
                    callback_to_run()
                except Exception as e:
                    logger.error(f"Error in hotkey callback for {hotkey_triggered}: {e}")

            threading.Thread(target=safe_callback_wrapper, daemon=True).start()

    def is_listening(self) -> bool:
        """Проверить, запущено ли прослушивание."""
        return self._is_listening

    def get_registered_hotkeys(self) -> Dict[str, str]:
        """
        Получить список зарегистрированных хоткеев.

        Returns:
            {hotkey_str: callback_name}
        """
        with self._lock:
            return {k: v.__name__ if hasattr(v, '__name__') else str(v)
                    for k, v in self._hotkeys.items()}

    def __enter__(self):
        """Контекстный менеджер - вход."""
        self.start_listening()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер - выход."""
        self.stop_listening()
        return False
