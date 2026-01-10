"""
VoiceType - Lazy Model Manager
Менеджер отложенной загрузки/выгрузки моделей для экономии RAM.

Saves ~400 MB RAM when not recording by unloading models after inactivity.
Uses ONNX Runtime for punctuation (no PyTorch dependency for inference).
"""
import gc
import sys
import threading
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from loguru import logger

from src.utils.constants import SAMPLE_RATE


class LazyModelManager(QObject):
    """
    Менеджер отложенной загрузки моделей.

    Загружает модели только когда нужно (при начале записи)
    и выгружает после периода бездействия для экономии RAM.

    Signals:
        vosk_status_changed(bool, str): (loaded, model_name) - статус Vosk модели
        punctuation_status_changed(bool, str): (loaded, model_name) - статус пунктуации
    """

    # Signals for UI updates
    vosk_status_changed = pyqtSignal(bool, str)  # (loaded, model_name)
    punctuation_status_changed = pyqtSignal(bool, str)  # (loaded, model_name)

    # Default auto-unload timeout (30 seconds)
    DEFAULT_UNLOAD_TIMEOUT_SEC = 30

    def __init__(self, parent=None):
        super().__init__(parent)

        # Model instances
        self._vosk_model = None
        self._vosk_recognizer = None
        self._punctuation = None

        # Model paths
        self._vosk_model_path: Optional[Path] = None
        self._punctuation_model_path: Optional[Path] = None

        # State tracking
        self._vosk_loaded = False
        self._punctuation_loaded = False
        self._vosk_model_name = ""
        self._punctuation_model_name = ""

        # Thread safety
        self._lock = threading.Lock()

        # Auto-unload timer
        self._unload_timer: Optional[QTimer] = None
        self._unload_timeout_sec = self.DEFAULT_UNLOAD_TIMEOUT_SEC

        # Callbacks
        self.on_vosk_loading_progress: Optional[Callable[[int], None]] = None
        self.on_punctuation_loading_progress: Optional[Callable[[int], None]] = None

    # === Vosk Model Management ===

    def load_vosk(self, model_path: str) -> bool:
        """
        Загрузить модель Vosk.

        Args:
            model_path: Путь к папке с моделью Vosk

        Returns:
            True если успешно загружена
        """
        with self._lock:
            if self._vosk_loaded and str(self._vosk_model_path) == model_path:
                logger.debug("Vosk model already loaded")
                return True

            # Unload previous model if different path
            if self._vosk_loaded:
                self._unload_vosk_internal()

        try:
            from vosk import Model, KaldiRecognizer, SetLogLevel

            # Disable Vosk logs (too verbose)
            SetLogLevel(-1)

            path = Path(model_path)
            if not path.exists():
                logger.error(f"Vosk model not found: {path}")
                return False

            if self.on_vosk_loading_progress:
                self.on_vosk_loading_progress(10)

            logger.info(f"Loading Vosk model from {path}...")

            # Load model
            self._vosk_model = Model(str(path))

            if self.on_vosk_loading_progress:
                self.on_vosk_loading_progress(80)

            # Create recognizer
            self._vosk_recognizer = KaldiRecognizer(self._vosk_model, SAMPLE_RATE)
            self._vosk_recognizer.SetWords(True)

            if self.on_vosk_loading_progress:
                self.on_vosk_loading_progress(100)

            self._vosk_model_path = path
            self._vosk_model_name = path.name
            self._vosk_loaded = True

            logger.info(f"Vosk model loaded: {self._vosk_model_name}")
            self.vosk_status_changed.emit(True, self._vosk_model_name)

            return True

        except ImportError:
            logger.error("Vosk library not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            return False

    def _unload_vosk_internal(self) -> None:
        """Internal: выгрузить Vosk без lock (для вызова из locked context)."""
        if self._vosk_recognizer:
            self._vosk_recognizer = None
        if self._vosk_model:
            self._vosk_model = None

        self._vosk_loaded = False
        old_name = self._vosk_model_name
        self._vosk_model_name = ""

        gc.collect()
        logger.info(f"Vosk model unloaded: {old_name}")

    def unload_vosk(self) -> None:
        """Выгрузить модель Vosk для освобождения памяти."""
        with self._lock:
            if not self._vosk_loaded:
                return
            self._unload_vosk_internal()

        self.vosk_status_changed.emit(False, "")

    def get_vosk_recognizer(self):
        """Получить экземпляр KaldiRecognizer для обработки аудио."""
        return self._vosk_recognizer

    def reset_vosk_recognizer(self) -> None:
        """Сбросить recognizer для новой сессии."""
        with self._lock:
            if self._vosk_loaded and self._vosk_model:
                try:
                    from vosk import KaldiRecognizer
                    self._vosk_recognizer = KaldiRecognizer(self._vosk_model, SAMPLE_RATE)
                    self._vosk_recognizer.SetWords(True)
                    logger.debug("Vosk recognizer reset")
                except Exception as e:
                    logger.error(f"Error resetting recognizer: {e}")

    def is_vosk_loaded(self) -> bool:
        """Проверить, загружена ли Vosk модель."""
        return self._vosk_loaded

    # === Punctuation Model Management (ONNX) ===

    def load_punctuation(self, model_path: Optional[str] = None) -> bool:
        """
        Загрузить модель пунктуации (ONNX RUPunct).

        Args:
            model_path: Путь к папке с ONNX моделью (если None - используется default)

        Returns:
            True если успешно загружена
        """
        with self._lock:
            if self._punctuation_loaded:
                logger.debug("Punctuation model already loaded")
                return True

        try:
            # Import RUPunctONNX (lightweight ONNX-based punctuation)
            from src.core.punctuation import RUPunctONNX

            if self.on_punctuation_loading_progress:
                self.on_punctuation_loading_progress(10)

            logger.info("Loading RUPunct ONNX model...")

            # Create and load model
            if model_path:
                self._punctuation = RUPunctONNX(model_path=model_path)
            else:
                self._punctuation = RUPunctONNX()

            if self.on_punctuation_loading_progress:
                self.on_punctuation_loading_progress(50)

            success = self._punctuation.load_model()

            if self.on_punctuation_loading_progress:
                self.on_punctuation_loading_progress(100)

            if success:
                with self._lock:
                    self._punctuation_loaded = True
                    self._punctuation_model_name = "RUPunct ONNX"

                logger.info("RUPunct ONNX model loaded")
                self.punctuation_status_changed.emit(True, self._punctuation_model_name)
                return True
            else:
                logger.error("Failed to load RUPunct ONNX model")
                self._punctuation = None
                return False

        except ImportError as e:
            logger.error(f"Failed to import RUPunctONNX: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load punctuation model: {e}")
            return False

    def unload_punctuation(self) -> None:
        """Выгрузить модель пунктуации для освобождения памяти."""
        with self._lock:
            if not self._punctuation_loaded:
                return

            if self._punctuation:
                self._punctuation.unload()
                self._punctuation = None

            self._punctuation_loaded = False
            self._punctuation_model_name = ""

        gc.collect()
        logger.info("Punctuation model unloaded")
        self.punctuation_status_changed.emit(False, "")

    def is_punctuation_loaded(self) -> bool:
        """Проверить, загружена ли модель пунктуации."""
        return self._punctuation_loaded

    def enhance_text(self, text: str) -> str:
        """
        Применить пунктуацию к тексту.

        Args:
            text: Текст без пунктуации

        Returns:
            Текст с пунктуацией
        """
        if not self._punctuation_loaded or not self._punctuation:
            return text

        return self._punctuation.enhance(text)

    # === Auto-Unload Timer ===

    def start_auto_unload_timer(self, timeout_sec: int = None) -> None:
        """
        Запустить таймер автоматической выгрузки моделей.

        Args:
            timeout_sec: Таймаут в секундах (default: 30)
        """
        if timeout_sec is not None:
            self._unload_timeout_sec = timeout_sec

        # Cancel existing timer
        self.cancel_auto_unload_timer()

        # Create new timer
        self._unload_timer = QTimer(self)
        self._unload_timer.setSingleShot(True)
        self._unload_timer.timeout.connect(self._on_auto_unload_timeout)
        self._unload_timer.start(self._unload_timeout_sec * 1000)

        logger.debug(f"Auto-unload timer started: {self._unload_timeout_sec}s")

    def cancel_auto_unload_timer(self) -> None:
        """Отменить таймер автоматической выгрузки."""
        if self._unload_timer:
            self._unload_timer.stop()
            self._unload_timer.deleteLater()
            self._unload_timer = None
            logger.debug("Auto-unload timer cancelled")

    def _on_auto_unload_timeout(self) -> None:
        """Callback таймера автовыгрузки - выгружаем все модели для экономии RAM."""
        logger.info("Auto-unload timeout reached, unloading all models...")
        # Сигналим app.py чтобы выгрузил свои экземпляры моделей
        self.vosk_status_changed.emit(False, "")
        self.unload_punctuation()
        gc.collect()
        logger.info("All models unloaded, RAM freed (~400 MB)")

    # === Utility Methods ===

    def unload_all(self) -> None:
        """Выгрузить все модели для освобождения памяти."""
        self.cancel_auto_unload_timer()
        self.unload_vosk()
        self.unload_punctuation()

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if torch is loaded (frozen builds)
        if getattr(sys, '_torch_rthook_success', False) and 'torch' in sys.modules:
            try:
                torch_module = sys.modules['torch']
                if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
            except Exception:
                pass

        logger.info("All models unloaded, memory freed")

    def load_all(self, vosk_model_path: str, punctuation_enabled: bool = True) -> bool:
        """
        Загрузить все необходимые модели.

        Args:
            vosk_model_path: Путь к Vosk модели
            punctuation_enabled: Загружать ли модель пунктуации

        Returns:
            True если Vosk загружен успешно
        """
        # Cancel auto-unload if was scheduled
        self.cancel_auto_unload_timer()

        # Load Vosk (required)
        vosk_success = self.load_vosk(vosk_model_path)

        if not vosk_success:
            return False

        # Load punctuation (optional)
        if punctuation_enabled:
            punct_success = self.load_punctuation()
            if not punct_success:
                logger.warning("Punctuation model failed to load, continuing without it")

        return True

    def get_status(self) -> dict:
        """
        Получить статус всех моделей.

        Returns:
            Словарь со статусом
        """
        return {
            "vosk_loaded": self._vosk_loaded,
            "vosk_model_name": self._vosk_model_name,
            "punctuation_loaded": self._punctuation_loaded,
            "punctuation_model_name": self._punctuation_model_name,
        }
