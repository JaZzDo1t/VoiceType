# Lazy Loading Implementation Review

**Date:** 2026-01-10
**Branch:** feature/lazy-loading-onnx
**Reviewer:** Code Review Agent

## Executive Summary

This review analyzes the VoiceType codebase in preparation for implementing lazy loading of ML models to save ~400 MB RAM when not recording. The codebase is currently on the `feature/lazy-loading-onnx` branch with uncommitted changes related to hotkey consolidation (two hotkeys -> one toggle hotkey).

**Key Findings:**
1. ONNX model files exist and are valid (`models/rupunct-onnx/`)
2. Test scripts for ONNX lazy loading exist and work with `tokenizers` library
3. No LazyModelManager or RUPunctONNX classes have been created yet
4. Current architecture supports lazy loading with minor modifications
5. Several potential issues need attention during implementation

---

## 1. Current State Analysis

### 1.1 Files Modified (Uncommitted)
The following files have uncommitted changes related to hotkey consolidation:
- `voicetype/src/app.py` - Changed from start/stop to toggle hotkey
- `voicetype/src/ui/tabs/tab_hotkeys.py` - Single toggle hotkey UI
- `voicetype/src/data/config.py` - Updated default config
- `voicetype/src/utils/constants.py` - `DEFAULT_HOTKEY_TOGGLE` instead of separate
- `voicetype/tests/test_app_integration.py` - Updated tests
- Plus other test files

**Recommendation:** Commit hotkey changes before implementing lazy loading to avoid merge conflicts.

### 1.2 ONNX Model Directory
Location: `voicetype/models/rupunct-onnx/`

Files present:
- `model.onnx` - 340 MB (RUPunct medium ELECTRA model)
- `tokenizer.json` - 1.9 MB (tokenizers format)
- `config.json` - Model configuration with label mappings
- `vocab.txt` - 908 KB vocabulary
- `tokenizer_config.json` - ElectraTokenizer config
- `special_tokens_map.json` - Special tokens

**Status:** All required files present for ONNX inference with `tokenizers` library.

### 1.3 Test Scripts
Two test scripts exist:
- `voicetype/test_lazy_loading.py` - Tests Vosk + Silero TE load/unload
- `voicetype/test_onnx_lazy_loading.py` - Tests ONNX + tokenizers (no PyTorch!)

The ONNX test demonstrates using `tokenizers.Tokenizer` instead of `transformers.AutoTokenizer`, which avoids PyTorch dependency.

---

## 2. Potential Issues Found

### 2.1 CRITICAL: Thread Safety Issues

**Location:** `voicetype/src/app.py`, lines 239-276 (`_load_models_async`)

**Problem:** No protection against concurrent loading attempts.

```python
def _load_models_async(self):
    """Загрузить модели в фоновом потоке."""
    def load():
        # ... loading code ...

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
```

**Issues:**
1. If user rapidly changes settings, multiple load threads could run simultaneously
2. No `_is_loading` flag to prevent duplicate loads
3. Signal emissions from background thread are correct (PyQt handles cross-thread signals), but model assignment is not thread-safe

**Suggested Fix:**
```python
_is_loading: bool = False
_loading_lock = threading.Lock()

def _load_models_async(self):
    with self._loading_lock:
        if self._is_loading:
            logger.warning("Models already loading, ignoring request")
            return
        self._is_loading = True

    def load():
        try:
            # ... existing load code ...
        finally:
            with self._loading_lock:
                self._is_loading = False

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
```

### 2.2 HIGH: Memory Leak in Model Unloading

**Location:** `voicetype/src/app.py`, lines 183-207 (`_unload_models`)

**Problem:** `gc.collect()` is called but ONNX Runtime sessions may not be released properly.

```python
def _unload_models(self) -> None:
    if self._recognizer is not None:
        if hasattr(self._recognizer, 'unload'):
            self._recognizer.unload()
        self._recognizer = None

    if self._punctuation is not None:
        if hasattr(self._punctuation, 'unload'):
            self._punctuation.unload()
        self._punctuation = None

    gc.collect()
```

**Issues:**
1. ONNX Runtime `InferenceSession` needs explicit cleanup in some cases
2. PyTorch models keep references through `sys.modules`
3. `gc.collect()` may not release all GPU memory (if GPU is used)

**Suggested Fix for RUPunctONNX:**
```python
def unload(self) -> None:
    with self._lock:
        if self._session is not None:
            # ONNX Runtime cleanup
            del self._session
            self._session = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_loaded = False

    # Force garbage collection
    gc.collect()
    logger.info("RUPunct ONNX model unloaded")
```

### 2.3 HIGH: Race Condition in Recording Start

**Location:** `voicetype/src/app.py`, lines 424-465 (`start_recording`)

**Problem:** User could press hotkey while models are loading.

```python
def start_recording(self):
    if self._is_recording:
        logger.warning("Already recording")
        return

    if not self._models_loaded or not self._recognizer or not self._recognizer.is_loaded():
        logger.warning("Models not loaded yet, ignoring start_recording")
        return
```

**Issue:** If LazyModelManager needs to load models on-demand, there's no way to queue the recording start.

**Suggested Fix:**
```python
def start_recording(self):
    if self._is_recording:
        return

    if not self._models_loaded:
        # Queue recording to start after models load
        self._pending_recording = True
        if not self._is_loading:
            self._load_models_async()
        return

    # ... existing code ...
```

### 2.4 MEDIUM: Error Handling Inconsistency

**Location:** `voicetype/src/core/punctuation.py`, `RUPunctMedium` class

**Problem:** Inconsistent error handling patterns between Silero and RUPunct.

```python
# Silero uses on_error callback
if self.on_error:
    self.on_error(e)

# RUPunctMedium also uses on_error but has different behavior
if self.on_error:
    self.on_error(ImportError("transformers library required for RUPunct"))
```

**Issue:** New RUPunctONNX class needs consistent error handling pattern.

**Suggested Standard:**
```python
# All punctuation classes should:
# 1. Log the error with loguru
# 2. Call on_error callback if set
# 3. Return False from load_model()
# 4. enhance() should return original text on error
```

### 2.5 MEDIUM: Signal Naming Inconsistency

**Location:** Various files

**Current patterns:**
- `_models_loaded_signal` - past tense
- `_loading_status_signal` - present continuous
- `microphone_changed` - past tense
- `toggle_hotkey_changed` - past tense

**Recommendation:** For new signals, use past tense for events that happened:
- `models_loaded` (not `models_load`)
- `model_unloaded`
- `loading_started`
- `loading_failed`

### 2.6 MEDIUM: Missing Timeout for Model Loading

**Location:** `voicetype/src/app.py`, `_load_models_async`

**Problem:** No timeout mechanism if model loading hangs.

**Suggested Fix:**
```python
LOADING_TIMEOUT_SECONDS = 60

def _load_models_async(self):
    def load():
        try:
            # ... loading code ...
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self._models_loaded_signal.emit(TRAY_STATE_ERROR, "")

    thread = threading.Thread(target=load, daemon=True)
    thread.start()

    # Optional: Add timeout monitoring
    QTimer.singleShot(
        LOADING_TIMEOUT_SECONDS * 1000,
        lambda: self._check_loading_timeout(thread)
    )
```

### 2.7 LOW: Config Key Naming

**Current:** `recognition.punctuation_enabled`
**New needed:** Config keys for lazy loading settings

**Suggested:**
```yaml
models:
  lazy_loading: true  # Enable/disable lazy loading
  unload_timeout_minutes: 10  # Unload after N minutes of inactivity
  punctuation_backend: "onnx"  # "silero", "onnx", "disabled"
```

### 2.8 LOW: UI Status Display

**Location:** `voicetype/src/ui/tabs/tab_main.py`, line 271-283

**Current:** Only shows Silero TE status
```python
def set_silero_status(self, loaded: bool):
```

**Needed:** Support multiple punctuation backends
```python
def set_punctuation_status(self, backend: str, loaded: bool):
    """
    Args:
        backend: "silero", "onnx", or "disabled"
        loaded: Whether model is loaded
    """
    if backend == "onnx":
        self._punctuation_status.setText("RUPunct ONNX")
    elif backend == "silero":
        self._punctuation_status.setText("Silero TE")
    else:
        self._punctuation_status.setText("Basic")

    # Color coding
    color = "#10B981" if loaded else "#9CA3AF"
    self._punctuation_status.setStyleSheet(f"color: {color};")
```

---

## 3. ONNX Integration Verification

### 3.1 Label Mapping Analysis

From `config.json`, RUPunct uses these labels:
- `UPPER_PERIOD`, `LOWER_PERIOD`, `UPPER_TOTAL_PERIOD` (0, 1, 2)
- `UPPER_COMMA`, `LOWER_COMMA`, `UPPER_TOTAL_COMMA` (3, 4, 5)
- `UPPER_QUESTION`, etc.
- Special Russian: `TIRE` (dash), `VOSKL` (exclamation), `DVOETOCHIE` (colon)

**Issue in existing `RUPunctMedium`:** Label mapping uses different names:
```python
PUNCT_MAP = {
    'PERIOD': '.',
    'COMMA': ',',
    # ...
}
```

But actual labels have CASE prefix: `UPPER_PERIOD`, `LOWER_PERIOD`, etc.

The `_parse_label` method correctly handles this, but ONNX output needs same parsing.

### 3.2 Tokenizer Compatibility

**Test from `test_onnx_lazy_loading.py`:**
```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("models/rupunct-onnx/tokenizer.json")
encoded = tokenizer.encode(text)
input_ids = np.array([encoded.ids], dtype=np.int64)
attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
```

**Verified:** `tokenizers` library works without `transformers` or PyTorch.

### 3.3 ONNX Model Inputs

From test script:
- `input_ids`: int64, shape [batch, seq_len]
- `attention_mask`: int64, shape [batch, seq_len]
- `token_type_ids`: int64, shape [batch, seq_len] (zeros)

**Note:** ONNX model expects `token_type_ids` which tokenizers library doesn't produce. Must create zeros manually.

---

## 4. Suggested Architecture

### 4.1 New Files to Create

1. **`voicetype/src/core/lazy_model_manager.py`**
   - Manages model lifecycle
   - Handles load/unload timing
   - Provides unified interface

2. **`voicetype/src/core/punctuation_onnx.py`** (or extend punctuation.py)
   - RUPunctONNX class
   - Uses onnxruntime + tokenizers
   - Same interface as Punctuation class

3. **`voicetype/tests/test_lazy_loading_unit.py`**
   - Unit tests for LazyModelManager
   - Mock-based, no real models needed

### 4.2 LazyModelManager Interface

```python
class LazyModelManager(QObject):
    """
    Manages lazy loading/unloading of ML models.
    Thread-safe, emits Qt signals for UI updates.
    """

    # Signals
    loading_started = pyqtSignal(str)  # model_name
    loading_progress = pyqtSignal(str, int)  # model_name, percent
    loading_finished = pyqtSignal(str, bool)  # model_name, success
    model_unloaded = pyqtSignal(str)  # model_name

    def __init__(self, config: Config):
        ...

    def load_models(self, models: List[str] = None) -> None:
        """Load specified models (or all) asynchronously."""
        ...

    def unload_models(self, models: List[str] = None) -> None:
        """Unload specified models (or all)."""
        ...

    def get_recognizer(self) -> Optional[Recognizer]:
        """Get Vosk recognizer (loads if needed)."""
        ...

    def get_punctuation(self) -> Optional[BasePunctuation]:
        """Get punctuation model (loads if needed)."""
        ...

    def is_loaded(self, model: str) -> bool:
        """Check if specific model is loaded."""
        ...

    def start_idle_timer(self) -> None:
        """Start timer to unload after inactivity."""
        ...

    def reset_idle_timer(self) -> None:
        """Reset timer on user activity."""
        ...
```

### 4.3 RUPunctONNX Interface

```python
class RUPunctONNX:
    """
    ONNX Runtime-based punctuation using RUPunct model.
    Does NOT require PyTorch or transformers.
    """

    MODEL_DIR = "models/rupunct-onnx"

    def __init__(self, model_path: str = None, language: str = "ru"):
        ...

    def load_model(self) -> bool:
        """Load ONNX model and tokenizer."""
        ...

    def enhance(self, text: str) -> str:
        """Add punctuation to text."""
        ...

    def is_loaded(self) -> bool:
        ...

    def unload(self) -> None:
        ...

    # Callbacks (for compatibility)
    on_loading_progress: Optional[Callable[[int], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
```

---

## 5. Test Cases to Write

### 5.1 Unit Tests for LazyModelManager

```python
class TestLazyModelManager:

    def test_load_models_async_emits_signals(self):
        """Loading should emit loading_started and loading_finished signals."""

    def test_concurrent_load_prevented(self):
        """Second load request while loading should be ignored."""

    def test_unload_releases_memory(self):
        """Unloading should reduce process memory."""

    def test_idle_timer_triggers_unload(self):
        """After timeout, models should be unloaded."""

    def test_activity_resets_idle_timer(self):
        """User activity should reset the unload timer."""

    def test_load_on_demand(self):
        """get_recognizer() should load if not loaded."""

    def test_error_handling_on_load_failure(self):
        """Failed load should emit signal and not crash."""
```

### 5.2 Unit Tests for RUPunctONNX

```python
class TestRUPunctONNX:

    def test_load_model_success(self, rupunct_onnx_path):
        """Model should load successfully."""

    def test_enhance_adds_punctuation(self, loaded_rupunct):
        """enhance() should add periods and commas."""

    def test_enhance_handles_empty_text(self, loaded_rupunct):
        """Empty text should return empty."""

    def test_enhance_preserves_unicode(self, loaded_rupunct):
        """Russian text should be preserved correctly."""

    def test_unload_clears_session(self, loaded_rupunct):
        """After unload, session should be None."""

    def test_inference_after_reload(self, rupunct_onnx_path):
        """Model should work after unload + reload cycle."""

    def test_no_pytorch_import(self):
        """Loading should not import torch module."""
```

### 5.3 Integration Tests

```python
class TestLazyLoadingIntegration:

    @pytest.mark.slow
    def test_full_load_unload_cycle(self):
        """Complete cycle: load -> use -> unload -> reload."""

    @pytest.mark.slow
    def test_memory_freed_after_unload(self):
        """Process memory should decrease after unload."""

    @pytest.mark.hardware
    def test_recording_with_lazy_loading(self):
        """Recording should work with lazy-loaded models."""
```

### 5.4 Fixtures to Add to conftest.py

```python
@pytest.fixture
def rupunct_onnx_path() -> Optional[Path]:
    """Get path to ONNX model, skip if not present."""
    path = ROOT_DIR / "models" / "rupunct-onnx"
    if not path.exists() or not (path / "model.onnx").exists():
        pytest.skip("RUPunct ONNX model not found")
    return path

@pytest.fixture
def mock_lazy_manager():
    """Create LazyModelManager with mocked models."""
    ...
```

---

## 6. Implementation Checklist

### Phase 1: Preparation
- [ ] Commit current hotkey changes
- [ ] Add `onnxruntime` and `tokenizers` to requirements.txt
- [ ] Create test fixtures for ONNX model

### Phase 2: RUPunctONNX Class
- [ ] Create `punctuation_onnx.py` with RUPunctONNX class
- [ ] Implement load_model() with onnxruntime
- [ ] Implement enhance() with proper label parsing
- [ ] Implement unload() with proper cleanup
- [ ] Write unit tests

### Phase 3: LazyModelManager
- [ ] Create `lazy_model_manager.py`
- [ ] Implement thread-safe loading with signals
- [ ] Implement idle timeout functionality
- [ ] Add loading state protection
- [ ] Write unit tests

### Phase 4: Integration
- [ ] Integrate LazyModelManager into app.py
- [ ] Update tab_main.py for new status display
- [ ] Add config keys for lazy loading settings
- [ ] Update existing tests

### Phase 5: Testing
- [ ] Run all unit tests
- [ ] Run integration tests with real models
- [ ] Measure memory before/after unload
- [ ] Test rapid toggle/settings changes

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Race condition during load | Medium | High | Add loading lock |
| Memory not fully released | Medium | Medium | Test with memory profiler |
| ONNX output format mismatch | Low | High | Verify with test cases |
| PyTorch accidentally imported | Low | Medium | Add import check in tests |
| UI freezes during load | Low | Medium | All loading in background threads |

---

## 8. References

- Existing test: `voicetype/test_onnx_lazy_loading.py`
- Memory analysis: `voicetype/MEMORY_ANALYSIS_REPORT.md`
- RAM optimization guide: `RAM_OPTIMIZATION_GUIDE.md`
- RUPunct config: `voicetype/models/rupunct-onnx/config.json`

---

## Appendix A: Current Signal Patterns

```python
# app.py signals (for reference when adding new ones)
_update_level_signal = pyqtSignal(float)
_partial_result_signal = pyqtSignal(str)
_final_result_signal = pyqtSignal(str)
_recognition_finished_signal = pyqtSignal()
_models_loaded_signal = pyqtSignal(str, str)  # (state, model_name)
_loading_status_signal = pyqtSignal(str, str)  # (status_text, model_name)
_hotkey_triggered_signal = pyqtSignal(str)  # action
_stats_collected_signal = pyqtSignal(float, float)  # (cpu, ram)
```

## Appendix B: ONNX Model Label Mapping

```python
# From config.json - use for RUPunctONNX
LABEL_TO_PUNCT = {
    # Period
    'UPPER_PERIOD': ('upper', '.'),
    'LOWER_PERIOD': ('lower', '.'),
    'UPPER_TOTAL_PERIOD': ('upper_total', '.'),
    # Comma
    'UPPER_COMMA': ('upper', ','),
    'LOWER_COMMA': ('lower', ','),
    'UPPER_TOTAL_COMMA': ('upper_total', ','),
    # Question
    'UPPER_QUESTION': ('upper', '?'),
    'LOWER_QUESTION': ('lower', '?'),
    'UPPER_TOTAL_QUESTION': ('upper_total', '?'),
    # Tire (em dash)
    'UPPER_TIRE': ('upper', ' -'),
    'LOWER_TIRE': ('lower', ' -'),
    'UPPER_TOTAL_TIRE': ('upper_total', ' -'),
    # Voskl (exclamation)
    'UPPER_VOSKL': ('upper', '!'),
    'LOWER_VOSKL': ('lower', '!'),
    'UPPER_TOTAL_VOSKL': ('upper_total', '!'),
    # Dvoetochie (colon)
    'UPPER_DVOETOCHIE': ('upper', ':'),
    'LOWER_DVOETOCHIE': ('lower', ':'),
    'UPPER_TOTAL_DVOETOCHIE': ('upper_total', ':'),
    # Periodcomma (semicolon)
    'UPPER_PERIODCOMMA': ('upper', ';'),
    'LOWER_PERIODCOMMA': ('lower', ';'),
    'UPPER_TOTAL_PERIODCOMMA': ('upper_total', ';'),
    # Defis (hyphen)
    'UPPER_DEFIS': ('upper', '-'),
    'LOWER_DEFIS': ('lower', '-'),
    'UPPER_TOTAL_DEFIS': ('upper_total', '-'),
    # Questionvoskl (?!)
    'UPPER_QUESTIONVOSKL': ('upper', '?!'),
    'LOWER_QUESTIONVOSKL': ('lower', '?!'),
    'UPPER_TOTAL_QUESTIONVOSKL': ('upper_total', '?!'),
    # Mnogotochie (ellipsis)
    'UPPER_MNOGOTOCHIE': ('upper', '...'),
    'LOWER_MNOGOTOCHIE': ('lower', '...'),
    'UPPER_TOTAL_MNOGOTOCHIE': ('upper_total', '...'),
    # O (no punctuation)
    'UPPER_O': ('upper', None),
    'LOWER_O': ('lower', None),
    'UPPER_TOTAL_O': ('upper_total', None),
}
```
