"""
VoiceType - Punctuation Enhancement
Обёртка над Silero Text Enhancement для добавления пунктуации и заглавных букв.

В frozen builds (PyInstaller) используется TorchScript формат вместо torch.package,
так как torch.package использует pickle и динамическую загрузку модулей,
несовместимую с PyInstaller.
"""
import sys
import threading
from pathlib import Path
from typing import Optional, Callable, List, Set
from loguru import logger


# ============================================================================
# Helper functions for TeModelJit (TorchScript-based Silero TE)
# These are copied from the original release_module.py of Silero TE
# ============================================================================

def _is_transformed_char(char: str) -> bool:
    """Check if char is a transformed unicode character like {1234}."""
    return char[0] == '{' and char[-1] == '}' and char[1:-1].isdigit()


def _split_into_chars(text: str) -> List[str]:
    """Split text into chars, handling transformed unicode chars."""
    if _is_transformed_char(text):
        return [text]

    splitted = []
    char = ""
    uni_start = False

    for c in text:
        if not uni_start:
            if c != '{':
                splitted.append(c)
            else:
                char += c
                uni_start = True
        elif uni_start:
            if c.isdigit():
                char += c
            elif c == '}':
                char += c
                splitted.append(char)
                char = ""
                uni_start = False
            else:
                assert '{' in text, "Service character { in text"

    return splitted


def _unitoken_into_token(unitoken: str) -> str:
    """Convert unitoken to regular token."""
    chars = _split_into_chars(unitoken)
    token = ""

    for c in chars:
        if _is_transformed_char(c):
            token += chr(int(c[1:-1]))
        else:
            token += c

    return token


def _process_unicode(text: str, uni_symbols: Set[str]) -> str:
    """Process text to handle unicode characters."""
    processed = ""

    for c in text:
        if ord(c) < 127:
            processed += c
        elif c not in uni_symbols:
            processed += '&'
        else:
            processed += '{' + str(ord(c)) + '}'

    return processed


def _enhance_tokens(tokens: List[str], punct, capital, index2punct: dict) -> List[str]:
    """Apply punctuation and capitalization to tokens."""
    output = []
    punct_np = punct.cpu().numpy()
    capital_np = capital.cpu().numpy()

    sentence_end = False

    for token, p, c in zip(tokens, punct_np, capital_np):
        if sentence_end:
            if c == 0:
                c = 1
            sentence_end = False

        if c == 1:
            if token[0].isalnum():
                token = token[0].upper() + token[1:]
            else:
                if len(token) < 2:
                    pass  # Short token, skip capitalization
                else:
                    token = '##' + token[2].upper() + token[3:]
        if c == 2:
            token = token.upper()
        output.append(token)

        if p:
            symbol = index2punct[p]
            output.append(symbol)

            if symbol in '.!?':
                sentence_end = True

    return output


# ============================================================================
# TeModelJit - TorchScript-based Silero TE wrapper
# This loads te_model_jit.pt and uses pure Python tokenizer from te_vocabs.pkl
# ============================================================================

class SimpleTokenizer:
    """
    Pure Python tokenizer for Silero TE.
    Uses vocabulary data from te_vocabs.pkl since TorchScript doesn't preserve Python methods.
    """

    def __init__(self, vocab: dict, ids_to_tokens: dict, uni_vocab: list):
        self.vocab = vocab  # token -> id
        self.ids_to_tokens = ids_to_tokens  # id -> token
        self.uni_vocab = uni_vocab
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'

    def convert_string_to_ids(self, text: str) -> List[int]:
        """Convert text to token IDs using character-level tokenization."""
        ids = [self.vocab.get(self.cls_token, 0)]  # Start with CLS

        # Character-level tokenization
        for char in text:
            if char in self.vocab:
                ids.append(self.vocab[char])
            else:
                ids.append(self.vocab.get(self.unk_token, 1))

        ids.append(self.vocab.get(self.sep_token, 0))  # End with SEP
        return ids

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs back to tokens."""
        return [self.ids_to_tokens.get(int(id), self.unk_token) for id in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Join tokens back to string."""
        result = []
        for token in tokens:
            if token.startswith('##'):
                result.append(token[2:])
            elif token in [self.pad_token, self.cls_token, self.sep_token, self.unk_token]:
                continue
            else:
                result.append(token)
        return ''.join(result)


class TeModelJit:
    """
    TorchScript-based Silero Text Enhancement model.
    Uses torch.jit.load() for BOTH the neural network AND tokenizer.
    This works correctly in PyInstaller frozen builds because both components
    are TorchScript modules loaded the same way.
    """

    # JIT tokenizer file name
    SILERO_JIT_TOKENIZER = "te_tokenizer_jit.pt"

    def __init__(self, model_dir: Path, torch_module):
        """
        Args:
            model_dir: Path to directory with te_model_jit.pt and te_tokenizer_jit.pt
            torch_module: The torch module (passed to avoid import issues)
        """
        self.model_dir = model_dir
        self.torch = torch_module
        self.device = torch_module.device('cpu')

        # File paths
        model_path = model_dir / "te_model_jit.pt"
        tokenizer_path = model_dir / self.SILERO_JIT_TOKENIZER

        if not model_path.exists():
            raise FileNotFoundError(f"JIT model not found: {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"JIT tokenizer not found: {tokenizer_path}")

        # Load TorchScript model (neural network)
        logger.info(f"Loading TorchScript model from {model_dir}...")
        torch_module.set_grad_enabled(False)

        try:
            self.model = torch_module.jit.load(str(model_path), map_location='cpu')
            self.model.eval()
            logger.info(f"Model type: {type(self.model).__name__}, has forward: {hasattr(self.model, 'forward')}")
        except Exception as e:
            logger.error(f"Failed to load JIT model: {e}")
            raise

        # Load TorchScript tokenizer (also a JIT module)
        logger.info(f"Loading TorchScript tokenizer from {tokenizer_path}...")
        try:
            self.tokenizer = torch_module.jit.load(str(tokenizer_path), map_location='cpu')
            logger.info(f"Tokenizer type: {type(self.tokenizer).__name__}")
        except Exception as e:
            logger.error(f"Failed to load JIT tokenizer: {e}")
            raise

        # Build uni_symbols from tokenizer's uni_vocab attribute
        self.uni_symbols: Set[str] = set()
        if hasattr(self.tokenizer, 'uni_vocab'):
            for unitoken in self.tokenizer.uni_vocab:
                self.uni_symbols.update(set(_unitoken_into_token(unitoken)))
            logger.info(f"Loaded {len(self.uni_symbols)} uni_symbols from JIT tokenizer")
        else:
            logger.warning("JIT tokenizer has no uni_vocab attribute, using empty set")

        self.index2punct = {1: '.', 2: ',', 3: '-', 4: '!', 5: '?', 6: '_'}
        self.lan2index = {'en': 0, 'de': 1, 'es': 2, 'ru': 3}

        # Verify model can be called (forward check)
        self._verify_model()

        logger.info("TeModelJit loaded successfully!")

    def _verify_model(self):
        """Verify that model forward() works."""
        torch = self.torch
        try:
            # Create minimal test input
            x = torch.zeros(1, 5, dtype=torch.long)
            att = torch.ones(1, 5, dtype=torch.long)
            lan = torch.tensor([[[0]]])

            # Try calling forward
            with torch.no_grad():
                result = self.model(x, att, lan)

            logger.info(f"Model forward() verified: outputs {len(result)} tensors")
        except Exception as e:
            logger.error(f"Model forward() verification failed: {e}")
            # Log detailed info for debugging
            logger.error(f"Model type: {type(self.model)}")
            logger.error(f"Model has _c: {hasattr(self.model, '_c')}")
            if hasattr(self.model, '_c'):
                logger.error(f"Model._c type: {type(self.model._c)}")
            raise RuntimeError(f"Model forward() failed: {e}")

    def _pad_ids(self, ids, limit=18):
        """Pad token IDs for batch processing."""
        torch = self.torch
        if ids.shape[1] < limit:
            ids_padded = torch.LongTensor(1, limit)
        else:
            ids_padded = torch.LongTensor(1, min(ids.shape[1] + limit, 512))

        ids_padded.zero_()
        ids_padded[0, :ids.shape[1] - 1] = ids[0, :-1]
        ids_padded[0, -1] = ids[0, -1]

        att_mask = torch.ones_like(ids_padded)
        att_mask[0, ids.shape[1] - 1:-1].zero_()

        return ids_padded, True, att_mask

    def _enhance_textblock(self, text: str, lan_id):
        """Enhance a single text block."""
        torch = self.torch
        device = self.device
        lan_id = lan_id.to(device)

        with torch.no_grad():
            x = _process_unicode(text, self.uni_symbols)
            x = torch.tensor([self.tokenizer.convert_string_to_ids(x)])
            x, pad, att_mask = self._pad_ids(x)
            punct, capital = self.model(x.to(device), att_mask.to(device), lan_id)
            punct_argmax = torch.argmax(punct, dim=-1)
            capital_argmax = torch.argmax(capital, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens([item for item in x[0]])
        tokens = list(map(_unitoken_into_token, tokens))

        if pad:
            pad_token = self.tokenizer.pad_token
            tokens = tokens[:tokens.index(pad_token)] + tokens[-1:]
            punct_argmax = punct_argmax[:, att_mask[0].bool()]
            capital_argmax = capital_argmax[:, att_mask[0].bool()]

        enhanced_tokens = _enhance_tokens(
            tokens[1:-1],
            punct_argmax[0][1:-1],
            capital_argmax[0][1:-1],
            self.index2punct
        )

        return self.tokenizer.convert_tokens_to_string(enhanced_tokens)

    def _count_occurrences(self, text: str, char: str) -> int:
        if char in text:
            counter = 0
            ind = -1
            while True:
                ind = text.find(char, ind + 1)
                if ind == -1:
                    break
                counter += 1
            return counter
        return 0

    def _enhance_long_textblock(self, text: str, lan_id, len_limit: int):
        """Enhance a long text block by splitting into chunks."""
        result = ''
        words = text.split()

        _from, _to = 0, 0
        while _to < len(words):
            _to = _from + len_limit
            block = ' '.join(words[_from:_to])

            enhanced = self._enhance_textblock(block, lan_id)
            enhanced_words = enhanced.split()
            symbols = ''.join([word[-1] for word in enhanced_words])

            ind = max(symbols.rfind('.'), symbols.rfind('!'), symbols.rfind('?')) + 1
            ind += self._count_occurrences(enhanced, '-')

            result += ' '.join(enhanced_words[:ind]) + ' '
            _from = _from + ind

        return result

    def enhance_text(self, text: str, lan: str = 'en', len_limit: int = 150) -> str:
        """
        Enhance text by adding punctuation and capitalization.

        Args:
            text: Input text without punctuation
            lan: Language code ('en', 'de', 'es', 'ru')
            len_limit: Maximum words per block

        Returns:
            Enhanced text with punctuation and capitalization
        """
        torch = self.torch

        if lan not in self.lan2index:
            lan = 'en'
        lan_id = torch.tensor([[[self.lan2index[lan]]]])

        if len(text.split()) < len_limit:
            enhanced = self._enhance_textblock(text, lan_id)
        else:
            enhanced = self._enhance_long_textblock(text, lan_id, len_limit)

        # Spanish inverted punctuation
        if lan == 'es' and ('?' in enhanced or '!' in enhanced):
            for m, rm in zip('?!', '¿¡'):
                ind = 0
                prev = 0
                while ind < len(enhanced) and m in enhanced[ind:]:
                    ind = enhanced.find(m, ind)
                    if ind == -1:
                        break
                    part = enhanced[prev:ind]
                    end = max(part.rfind('.'), part.rfind('!'), part.rfind('?'))
                    if end != -1:
                        prev += end + 1
                        enhanced = enhanced[:prev + 1] + rm + enhanced[prev + 1:]
                    else:
                        enhanced = rm + enhanced
                    ind += 2

        enhanced = enhanced.replace('_', ' —').strip()
        return enhanced


# Pre-initialize PyTorch at module level to avoid circular import in frozen builds
# This must happen BEFORE any function tries to import torch
_torch_available = False
_torch_module = None
_torch_package_module = None


def _clean_torch_modules():
    """Remove all torch-related modules from sys.modules to recover from partial init."""
    torch_modules = [key for key in sys.modules if key == 'torch' or key.startswith('torch.')]
    for mod in torch_modules:
        try:
            del sys.modules[mod]
        except KeyError:
            pass
    return len(torch_modules)


def _try_import_torch():
    """Try to import torch - call only when needed, not at module load."""
    global _torch_available, _torch_module, _torch_package_module

    # Already initialized successfully
    if _torch_available and _torch_module is not None:
        return True

    # In frozen build, check if runtime hook succeeded
    if getattr(sys, 'frozen', False):
        # Check marker set by rthook_torch.py
        if getattr(sys, '_torch_rthook_success', False):
            # Runtime hook succeeded - use modules from sys.modules
            if 'torch' in sys.modules and 'torch.package' in sys.modules:
                _torch_module = sys.modules['torch']
                _torch_package_module = sys.modules['torch.package']
                _torch_available = True
                logger.info(f"PyTorch from runtime hook: {_torch_module.__version__}")
                return True
        else:
            # Runtime hook failed
            rthook_error = getattr(sys, '_torch_rthook_error', None)
            logger.error(f"Runtime hook failed: {rthook_error}")
            _torch_available = False
            return False

    # Non-frozen build - try fresh import
    try:
        import torch as _t
        import torch.package as _tp
        _torch_module = _t
        _torch_package_module = _tp
        _torch_available = True
        logger.debug(f"PyTorch imported: {_t.__version__}")
        return True
    except Exception as e:
        logger.error(f"PyTorch import failed: {e}")

    _torch_available = False
    return False


# DON'T call at module level - defer until needed


def get_models_base_dir() -> Path:
    """Получить базовую директорию с моделями."""
    if getattr(sys, 'frozen', False):
        # Запущено как exe (PyInstaller)
        return Path(sys.executable).parent / "_internal" / "models"
    else:
        # Запущено как скрипт
        return Path(__file__).parent.parent.parent / "models"


class Punctuation:
    """
    Обёртка над Silero Text Enhancement.
    Добавляет точки, запятые, заглавные буквы к распознанному тексту.

    Поддерживает два режима загрузки:
    1. Из локального файла models/silero-te/ (для production)
    2. Из torch.hub (fallback для разработки)
    """

    # Названия файлов моделей Silero TE
    SILERO_MODEL_FILES = {
        "ru": "v4_ru.pt",
        "en": "v4_en.pt",
    }

    # Универсальная многоязычная модель (torch.package format)
    SILERO_MULTI_LANG_MODEL = "v2_4lang_q.pt"

    # TorchScript JIT model files (работает в PyInstaller)
    SILERO_JIT_MODEL = "te_model_jit.pt"
    SILERO_JIT_TOKENIZER = "te_tokenizer_jit.pt"

    def __init__(self, model_path: str = None, language: str = "ru"):
        """
        Args:
            model_path: Путь к папке с моделью Silero TE
            language: Язык ("ru" или "en")
        """
        self.language = language

        # Определяем путь к модели
        if model_path:
            self.model_dir = Path(model_path)
        else:
            self.model_dir = get_models_base_dir() / "silero-te"

        self._model = None
        self._apply_te = None  # Функция apply_te из torch.hub
        self._is_loaded = False
        self._lock = threading.Lock()
        self._use_local = False  # Используется ли локальная модель
        self._use_jit = False  # Используется ли JIT модель (для frozen builds)

        # Callbacks
        self.on_loading_progress: Optional[Callable[[int], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    def _get_local_model_path(self) -> Optional[Path]:
        """
        Получить путь к локальному файлу модели.

        Returns:
            Path к файлу модели или None если не найден
        """
        if not self.model_dir.exists():
            return None

        # Сначала ищем многоязычную модель
        multi_lang_path = self.model_dir / self.SILERO_MULTI_LANG_MODEL
        if multi_lang_path.exists():
            logger.info(f"Found multi-language Silero model: {multi_lang_path}")
            return multi_lang_path

        # Затем ищем модель для конкретного языка
        lang_model = self.SILERO_MODEL_FILES.get(self.language)
        if lang_model:
            lang_path = self.model_dir / lang_model
            if lang_path.exists():
                logger.info(f"Found language-specific Silero model: {lang_path}")
                return lang_path

        # Ищем любой .pt файл в папке
        pt_files = list(self.model_dir.glob("*.pt"))
        if pt_files:
            logger.info(f"Found Silero model file: {pt_files[0]}")
            return pt_files[0]

        return None

    def _has_jit_models(self) -> bool:
        """
        Проверить, есть ли TorchScript JIT модели (модель + токенизатор).
        JIT модели работают в PyInstaller frozen builds.

        Returns:
            True если модель и токенизатор существуют
        """
        if not self.model_dir.exists():
            return False

        model_path = self.model_dir / self.SILERO_JIT_MODEL
        tokenizer_path = self.model_dir / self.SILERO_JIT_TOKENIZER

        exists = model_path.exists() and tokenizer_path.exists()
        if exists:
            logger.info(f"Found JIT model files in {self.model_dir}")
        return exists

    def _load_jit_model(self) -> 'TeModelJit':
        """
        Загрузить TorchScript JIT модель.
        Использует torch.jit.load() вместо torch.package.

        Returns:
            TeModelJit instance

        Raises:
            FileNotFoundError: Если файлы не найдены
            ImportError: Если PyTorch недоступен
        """
        if not _torch_available or _torch_module is None:
            raise ImportError("PyTorch not available for JIT loading")

        logger.info(f"Loading TorchScript JIT model from {self.model_dir}...")
        return TeModelJit(self.model_dir, _torch_module)

    def _load_local_model(self, model_path: Path):
        """
        Загрузить модель из локального файла.

        Silero TE модели (v2_4lang_q.pt и др.) используют torch.package формат,
        а не стандартный JIT формат. Загружаем через PackageImporter.

        Args:
            model_path: Путь к файлу .pt

        Returns:
            Загруженная модель
        """
        # Используем pre-initialized модули из начала файла
        if not _torch_available or _torch_package_module is None:
            raise ImportError("PyTorch not available")

        logger.info(f"Loading Silero TE model from local file: {model_path}")

        # Silero TE модели упакованы через torch.package, не torch.jit
        # Используем PackageImporter для загрузки
        imp = _torch_package_module.PackageImporter(str(model_path))
        model = imp.load_pickle("te_model", "model")

        logger.info(f"Model type: {type(model)}, has enhance_text: {hasattr(model, 'enhance_text')}")

        return model

    def load_model(self) -> bool:
        """
        Загрузить модель Silero TE.

        Порядок загрузки:
        1. В frozen builds: TorchScript JIT модель (работает с PyInstaller)
        2. В обычном режиме: torch.package модель
        3. Fallback: torch.hub (требует интернет)

        Returns:
            True если успешно загружена
        """
        if self._is_loaded:
            logger.warning("Punctuation model already loaded")
            return True

        try:
            # Инициализируем torch (отложенная инициализация)
            _try_import_torch()

            if not _torch_available:
                raise ImportError("PyTorch not available")

            if self.on_loading_progress:
                self.on_loading_progress(10)

            is_frozen = getattr(sys, 'frozen', False)
            logger.info(f"Loading Silero model (frozen={is_frozen})...")
            logger.info(f"Model dir: {self.model_dir}, exists: {self.model_dir.exists()}")

            # В frozen builds (PyInstaller) используем TorchScript JIT модель
            # так как torch.package несовместим с PyInstaller
            if is_frozen and self._has_jit_models():
                try:
                    logger.info("Frozen build detected, loading TorchScript JIT model...")
                    self._model = self._load_jit_model()
                    self._use_local = True
                    self._use_jit = True

                    if self.on_loading_progress:
                        self.on_loading_progress(100)

                    self._is_loaded = True
                    logger.info("Silero TE JIT model loaded successfully!")
                    return True

                except Exception as e:
                    import traceback
                    logger.error(f"Failed to load JIT model: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # В frozen build нет смысла пробовать torch.package - он не работает
                    if self.on_error:
                        self.on_error(e)
                    return False

            # Обычный режим (не frozen) - пробуем torch.package
            local_path = self._get_local_model_path()
            logger.info(f"Local model path: {local_path}")

            if local_path:
                try:
                    logger.info(f"Attempting to load model from: {local_path}")
                    self._model = self._load_local_model(local_path)
                    self._use_local = True
                    self._use_jit = False

                    if self.on_loading_progress:
                        self.on_loading_progress(100)

                    self._is_loaded = True
                    logger.info("Silero TE model loaded from local file successfully")
                    return True

                except Exception as e:
                    import traceback
                    logger.error(f"Failed to load local Silero model: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    logger.warning("Trying torch.hub as fallback...")

            # Fallback: загрузка из torch.hub (требует интернет)
            logger.info(f"Loading Silero TE model from torch.hub for language: {self.language}...")

            # ВАЖНО: torch.hub.load возвращает 5 значений!
            # (model, example_texts, languages, punct, apply_te)
            # Нам нужна функция apply_te для правильной работы
            self._model, _, _, _, self._apply_te = _torch_module.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_te',
                language=self.language,
                trust_repo=True
            )
            self._use_local = False
            logger.info(f"Got apply_te function: {self._apply_te is not None}")

            if self.on_loading_progress:
                self.on_loading_progress(100)

            self._is_loaded = True
            logger.info("Silero TE model loaded from torch.hub successfully")
            return True

        except ImportError as e:
            import traceback
            logger.error(f"PyTorch import failed: {e}")
            logger.error(f"Import traceback: {traceback.format_exc()}")
            if self.on_error:
                self.on_error(ImportError(f"PyTorch import failed: {e}"))
            return False

        except Exception as e:
            import traceback
            logger.error(f"Failed to load Silero TE model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if self.on_error:
                self.on_error(e)
            return False

    def enhance(self, text: str) -> str:
        """
        Добавить пунктуацию к тексту.

        Args:
            text: Текст без пунктуации

        Returns:
            Текст с пунктуацией и заглавными буквами

        Examples:
            'привет как дела' -> 'Привет, как дела?'
            'я иду домой сегодня вечером' -> 'Я иду домой сегодня вечером.'
        """
        if not text or not text.strip():
            return text

        if not self._is_loaded:
            logger.warning("Model not loaded, returning original text")
            return text

        with self._lock:
            try:
                input_text = text.strip()

                # Применяем модель
                if self._use_local:
                    # Для локальной модели (JIT или package) вызываем метод enhance_text
                    enhanced = self._model.enhance_text(input_text, self.language)
                    mode = "JIT" if self._use_jit else "package"
                    logger.info(f"Local model ({mode}) enhance: '{input_text}' -> '{enhanced}'")
                elif self._apply_te:
                    # Для torch.hub используем функцию apply_te с параметром языка
                    enhanced = self._apply_te(input_text, lan=self.language)
                    logger.info(f"apply_te enhance: '{input_text}' -> '{enhanced}'")
                else:
                    logger.warning("No apply_te function available, returning original")
                    return text

                # Убеждаемся, что первая буква заглавная
                if enhanced and len(enhanced) > 0 and enhanced[0].islower():
                    enhanced = enhanced[0].upper() + enhanced[1:]

                logger.debug(f"Enhanced: '{text}' -> '{enhanced}'")
                return enhanced

            except Exception as e:
                logger.error(f"Error enhancing text: {e}", exc_info=True)
                return text

    def enhance_batch(self, texts: list) -> list:
        """
        Добавить пунктуацию к списку текстов.

        Args:
            texts: Список текстов

        Returns:
            Список текстов с пунктуацией
        """
        if not self._is_loaded:
            return texts

        return [self.enhance(t) for t in texts]

    def is_loaded(self) -> bool:
        """Проверить, загружена ли модель."""
        return self._is_loaded

    def is_using_local_model(self) -> bool:
        """Проверить, используется ли локальная модель."""
        return self._use_local

    def set_language(self, language: str) -> bool:
        """
        Изменить язык модели.
        Требует перезагрузки модели.

        Args:
            language: Новый язык ("ru" или "en")

        Returns:
            True если успешно
        """
        if language == self.language and self._is_loaded:
            return True

        self.language = language
        self.unload()
        return self.load_model()

    def unload(self) -> None:
        """Выгрузить модель для освобождения памяти."""
        with self._lock:
            self._model = None
            self._apply_te = None
            self._is_loaded = False
            self._use_local = False
            self._use_jit = False
            logger.info("Silero TE model unloaded")

    def is_using_jit_model(self) -> bool:
        """Проверить, используется ли TorchScript JIT модель."""
        return self._use_jit

    def get_model_info(self) -> dict:
        """
        Получить информацию о загруженной модели.

        Returns:
            Словарь с информацией
        """
        return {
            "loaded": self._is_loaded,
            "local": self._use_local,
            "jit": self._use_jit,
            "has_apply_te": self._apply_te is not None,
            "language": self.language,
            "model_dir": str(self.model_dir),
            "model_dir_exists": self.model_dir.exists(),
            "has_jit_models": self._has_jit_models(),
        }

    def __del__(self):
        """Деструктор."""
        self.unload()


class RUPunctMedium:
    """
    Пунктуация на базе RUPunct/RUPunct_medium (ELECTRA-based).
    Легче чем Silero TE, использует transformers pipeline.

    Поддерживаемые знаки: . , ? ! : ; — - ...
    Также восстанавливает регистр (LOWER/UPPER/UPPER_TOTAL).

    Hugging Face: https://huggingface.co/RUPunct/RUPunct_medium
    """

    MODEL_NAME = "RUPunct/RUPunct_medium"

    # Маппинг меток на знаки препинания
    PUNCT_MAP = {
        'PERIOD': '.',
        'COMMA': ',',
        'QUESTION': '?',
        'EXCLAMATION': '!',
        'COLON': ':',
        'SEMICOLON': ';',
        'DASH': '—',
        'HYPHEN': '-',
        'ELLIPSIS': '...',
        'QUESTIONEXCLAMATION': '?!',
    }

    def __init__(self, model_path: str = None, language: str = "ru"):
        """
        Args:
            model_path: Путь к локальной модели (опционально)
            language: Язык (для совместимости, RUPunct только для русского)
        """
        self.language = language
        self.model_path = model_path
        self._classifier = None
        self._tokenizer = None
        self._is_loaded = False
        self._lock = threading.Lock()

        # Callbacks
        self.on_loading_progress: Optional[Callable[[int], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    def load_model(self) -> bool:
        """
        Загрузить модель RUPunct_medium.

        Returns:
            True если успешно загружена
        """
        if self._is_loaded:
            logger.warning("RUPunct model already loaded")
            return True

        try:
            if self.on_loading_progress:
                self.on_loading_progress(10)

            logger.info(f"Loading RUPunct_medium model...")

            # Импортируем transformers
            from transformers import pipeline, AutoTokenizer

            if self.on_loading_progress:
                self.on_loading_progress(30)

            # Определяем путь к модели
            model_name = self.model_path if self.model_path else self.MODEL_NAME

            # Загружаем токенизатор с нужными параметрами
            logger.info(f"Loading tokenizer from {model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                strip_accents=False,
                add_prefix_space=True
            )

            if self.on_loading_progress:
                self.on_loading_progress(60)

            # Создаём pipeline для NER
            logger.info(f"Creating NER pipeline...")
            self._classifier = pipeline(
                "ner",
                model=model_name,
                tokenizer=self._tokenizer,
                aggregation_strategy="first"
            )

            if self.on_loading_progress:
                self.on_loading_progress(100)

            self._is_loaded = True
            logger.info("RUPunct_medium model loaded successfully!")
            return True

        except ImportError as e:
            logger.error(f"transformers not installed: {e}")
            if self.on_error:
                self.on_error(ImportError("transformers library required for RUPunct"))
            return False

        except Exception as e:
            import traceback
            logger.error(f"Failed to load RUPunct model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if self.on_error:
                self.on_error(e)
            return False

    def _parse_label(self, label: str) -> tuple:
        """
        Парсит метку RUPunct.

        Метки имеют формат: CASE_PUNCT
        Например: LOWER_PERIOD, UPPER_COMMA, UPPER_TOTAL_QUESTION

        Returns:
            (case, punct) где case: 'lower'/'upper'/'upper_total', punct: символ или None
        """
        parts = label.split('_')

        # Определяем регистр
        if parts[0] == 'UPPER':
            if len(parts) > 1 and parts[1] == 'TOTAL':
                case = 'upper_total'
                punct_parts = parts[2:]
            else:
                case = 'upper'
                punct_parts = parts[1:]
        else:
            case = 'lower'
            punct_parts = parts[1:] if len(parts) > 1 else []

        # Определяем пунктуацию
        punct = None
        if punct_parts:
            punct_key = '_'.join(punct_parts)
            punct = self.PUNCT_MAP.get(punct_key)

        return case, punct

    def enhance(self, text: str) -> str:
        """
        Добавить пунктуацию к тексту.

        Args:
            text: Текст без пунктуации

        Returns:
            Текст с пунктуацией и заглавными буквами
        """
        if not text or not text.strip():
            return text

        if not self._is_loaded:
            logger.warning("RUPunct model not loaded, returning original text")
            return text

        with self._lock:
            try:
                input_text = text.strip().lower()  # RUPunct ожидает lowercase

                # Получаем предсказания
                predictions = self._classifier(input_text)

                if not predictions:
                    # Если нет предсказаний, возвращаем базовую обработку
                    result = input_text[0].upper() + input_text[1:] if input_text else input_text
                    if result and result[-1] not in '.!?':
                        result += '.'
                    return result

                # Собираем результат
                result = []
                prev_end = 0
                sentence_start = True

                for pred in predictions:
                    word = pred['word']
                    label = pred['entity_group']
                    start = pred.get('start', 0)
                    end = pred.get('end', len(word))

                    # Парсим метку
                    case, punct = self._parse_label(label)

                    # Применяем регистр
                    if case == 'upper_total':
                        word = word.upper()
                    elif case == 'upper' or sentence_start:
                        if word:
                            word = word[0].upper() + word[1:]

                    result.append(word)

                    # Добавляем пунктуацию
                    if punct:
                        result.append(punct)
                        sentence_start = punct in '.!?'
                    else:
                        sentence_start = False

                # Собираем строку
                enhanced = ' '.join(result)

                # Убираем пробелы перед знаками препинания
                for p in '.,!?:;—':
                    enhanced = enhanced.replace(f' {p}', p)

                # Убираем двойные пробелы
                while '  ' in enhanced:
                    enhanced = enhanced.replace('  ', ' ')

                # Убеждаемся, что первая буква заглавная
                if enhanced and enhanced[0].islower():
                    enhanced = enhanced[0].upper() + enhanced[1:]

                # Добавляем точку в конце если нет
                if enhanced and enhanced[-1] not in '.!?':
                    enhanced += '.'

                logger.debug(f"RUPunct enhanced: '{text}' -> '{enhanced}'")
                return enhanced.strip()

            except Exception as e:
                logger.error(f"Error enhancing text with RUPunct: {e}", exc_info=True)
                return text

    def enhance_batch(self, texts: list) -> list:
        """Добавить пунктуацию к списку текстов."""
        if not self._is_loaded:
            return texts
        return [self.enhance(t) for t in texts]

    def is_loaded(self) -> bool:
        """Проверить, загружена ли модель."""
        return self._is_loaded

    def is_using_local_model(self) -> bool:
        """Проверить, используется ли локальная модель."""
        return self.model_path is not None

    def set_language(self, language: str) -> bool:
        """
        Изменить язык (для совместимости).
        RUPunct поддерживает только русский.
        """
        self.language = language
        return True

    def unload(self) -> None:
        """Выгрузить модель для освобождения памяти."""
        with self._lock:
            self._classifier = None
            self._tokenizer = None
            self._is_loaded = False
            logger.info("RUPunct model unloaded")

    def is_using_jit_model(self) -> bool:
        """Для совместимости с Silero API."""
        return False

    def get_model_info(self) -> dict:
        """Получить информацию о модели."""
        return {
            "loaded": self._is_loaded,
            "local": self.model_path is not None,
            "jit": False,
            "has_apply_te": False,
            "language": self.language,
            "model_name": self.MODEL_NAME,
            "type": "RUPunct_medium (ELECTRA-based)",
        }

    def __del__(self):
        """Деструктор."""
        self.unload()


class PunctuationDisabled:
    """
    Базовая пунктуация без ML.
    Добавляет точку в конце и делает первую букву заглавной.
    """

    def __init__(self, *args, **kwargs):
        pass

    def load_model(self) -> bool:
        return True

    def enhance(self, text: str) -> str:
        if not text or not text.strip():
            return text

        text = text.strip()

        # Убираем многоточие от Vosk если есть
        if text.endswith('...'):
            text = text[:-3].strip()

        # Делаем первую букву заглавной
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Добавляем точку в конце если нет знака препинания
        if text and text[-1] not in '.!?,:;':
            text = text + '.'

        return text

    def enhance_batch(self, texts: list) -> list:
        return [self.enhance(t) for t in texts]

    def is_loaded(self) -> bool:
        return True

    def is_using_local_model(self) -> bool:
        return False

    def set_language(self, language: str) -> bool:
        return True

    def unload(self) -> None:
        pass

    def get_model_info(self) -> dict:
        return {
            "loaded": True,
            "local": False,
            "language": "basic",
            "model_dir": "",
            "model_dir_exists": False,
        }
