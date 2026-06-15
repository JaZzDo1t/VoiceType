# Error-Handling Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Залатать 4 дыры в обработке ошибок VoiceType, чтобы приложение честно сообщало о повреждённой/удалённой модели и отсутствующих CUDA-библиотеках вместо тихого зависания в трее.

**Architecture:** Новый модуль `src/core/diagnostics.py` с чистыми функциями проактивной проверки (целостность модели + наличие cuDNN/nvJitLink), интегрированными в поток загрузки в `app.py` перед `load_model()`. Молчаливый CUDA→CPU fallback убирается; `recognizer.on_error` подключается к трею через thread-safe сигнал. Диагностические функции принимают пути и порог через параметры — для тестирования без GPU и реальных моделей.

**Tech Stack:** Python 3, faster-whisper/ctranslate2, PyQt6, pytest, loguru.

---

## File Structure

| Файл | Ответственность |
|------|------------------|
| `voicetype/src/core/diagnostics.py` **(новый)** | Чистые функции `check_model`, `check_cuda`, `diagnose`; типы `Issue`, `IssueCode`; тексты диагноза. Не зависит от Qt/GPU. |
| `voicetype/src/utils/constants.py` *(правка)* | `MIN_MODEL_BIN_SIZE`, `CUDA_REQUIRED_DLLS`. |
| `voicetype/src/data/models_manager.py` *(правка)* | `is_whisper_model_cached` переиспользует `diagnostics.check_model`. |
| `voicetype/src/core/whisper_recognizer.py` *(правка)* | Убрать молчаливый CUDA→CPU fallback; сохранять `_last_load_error`. |
| `voicetype/src/ui/main_window.py` *(правка)* | `show_loading_error(text, details="")` прокидывает детали в overlay. |
| `voicetype/src/app.py` *(правка)* | Вызов `diagnose()` перед загрузкой; сигнал `_error_signal`; подключение `on_error`; показ диагноза. |
| `voicetype/tests/conftest.py` **(новый)** | Добавляет `voicetype/` в `sys.path` для импорта `src.*`. |
| `voicetype/tests/test_diagnostics.py` **(новый)** | Юнит-тесты диагностики. |
| `voicetype/tests/test_models_manager.py` **(новый)** | Тест усиленного `is_whisper_model_cached`. |
| `voicetype/tests/test_whisper_recognizer.py` **(новый)** | Тест отсутствия молчаливого CPU-fallback. |

**Все команды pytest запускаются из каталога `voicetype/`.**

---

## Task 1: Константы

**Files:**
- Modify: `voicetype/src/utils/constants.py` (после строки 32, блок WHISPER)

- [ ] **Step 1: Добавить константы**

В `voicetype/src/utils/constants.py` сразу после строки `WHISPER_DEFAULT_MODEL = "small"` (строка 32) вставить:

```python

# Диагностика окружения
# Минимальный размер model.bin (байты). small ≈ 0.48 ГБ, medium ≈ 1.53 ГБ.
# Порог 50 МБ надёжно отсекает повреждённые/пустые (.incomplete) файлы.
MIN_MODEL_BIN_SIZE = 50 * 1024 * 1024

# Требуемые CUDA DLL для device=cuda.
# Кортеж: (имя DLL, подпапка пакета nvidia, человекочитаемое имя, pip-пакет)
CUDA_REQUIRED_DLLS = [
    ("cudnn64_9.dll", "cudnn", "cuDNN", "nvidia-cudnn-cu12"),
    ("nvJitLink64_12.dll", "nvjitlink", "nvJitLink", "nvidia-nvjitlink-cu12"),
]
```

- [ ] **Step 2: Проверить импорт**

Run: `voicetype\venv\Scripts\python -c "from src.utils.constants import MIN_MODEL_BIN_SIZE, CUDA_REQUIRED_DLLS; print(MIN_MODEL_BIN_SIZE, len(CUDA_REQUIRED_DLLS))"` (из каталога `voicetype/`)
Expected: `52428800 2`

- [ ] **Step 3: Commit**

```bash
git add voicetype/src/utils/constants.py
git commit -m "feat: add diagnostics constants (model size threshold, CUDA DLLs)"
```

---

## Task 2: Тестовая инфраструктура + проверка модели

**Files:**
- Create: `voicetype/tests/conftest.py`
- Create: `voicetype/tests/test_diagnostics.py`
- Create: `voicetype/src/core/diagnostics.py`

- [ ] **Step 1: Создать conftest.py**

`voicetype/tests/conftest.py`:

```python
"""Pytest config: добавляем корень voicetype/ в sys.path для импорта src.*"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent  # voicetype/
sys.path.insert(0, str(ROOT))
```

- [ ] **Step 2: Написать падающие тесты на проверку модели**

`voicetype/tests/test_diagnostics.py`:

```python
"""Тесты диагностики окружения."""
from pathlib import Path

from src.core.diagnostics import check_model, IssueCode


def _make_model(tmp_path: Path, size: int, *, incomplete: bool = False) -> Path:
    """Создать структуру HF-кэша для модели medium с model.bin заданного размера."""
    cache_dir = tmp_path / "hub"
    model_dir = cache_dir / "models--Systran--faster-whisper-medium"
    snap = model_dir / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (model_dir / "blobs").mkdir()
    (snap / "model.bin").write_bytes(b"\0" * size)
    if incomplete:
        (model_dir / "blobs" / "deadbeef.incomplete").write_bytes(b"")
    return cache_dir


def test_check_model_ok(tmp_path):
    cache_dir = _make_model(tmp_path, size=2000)
    issues = check_model("medium", cache_dir, min_size=1000)
    assert issues == []


def test_check_model_missing(tmp_path):
    cache_dir = tmp_path / "hub"  # пусто, папки модели нет
    cache_dir.mkdir()
    issues = check_model("medium", cache_dir, min_size=1000)
    assert len(issues) == 1
    assert issues[0].code == IssueCode.MODEL_MISSING


def test_check_model_corrupt_small(tmp_path):
    cache_dir = _make_model(tmp_path, size=100)  # меньше порога
    issues = check_model("medium", cache_dir, min_size=1000)
    assert len(issues) == 1
    assert issues[0].code == IssueCode.MODEL_CORRUPT


def test_check_model_corrupt_incomplete(tmp_path):
    cache_dir = _make_model(tmp_path, size=2000, incomplete=True)
    issues = check_model("medium", cache_dir, min_size=1000)
    assert len(issues) == 1
    assert issues[0].code == IssueCode.MODEL_CORRUPT
```

- [ ] **Step 3: Запустить — убедиться, что падает**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_diagnostics.py -v` (из `voicetype/`)
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.diagnostics'`

- [ ] **Step 4: Реализовать diagnostics.py (типы + check_model)**

`voicetype/src/core/diagnostics.py`:

```python
"""
Диагностика окружения VoiceType.

Чистые функции проверки целостности модели и наличия CUDA-библиотек.
Не зависят от Qt/GPU — пути и порог принимаются параметрами для тестируемости.
"""
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from src.utils.constants import MIN_MODEL_BIN_SIZE, CUDA_REQUIRED_DLLS

# Приблизительные размеры моделей для текста диагноза
_MODEL_SIZE_GB = {"small": 0.5, "medium": 1.5}


class IssueCode(str, Enum):
    MODEL_MISSING = "model_missing"
    MODEL_CORRUPT = "model_corrupt"
    CUDA_CUDNN_MISSING = "cuda_cudnn_missing"
    CUDA_NVJITLINK_MISSING = "cuda_nvjitlink_missing"


@dataclass
class Issue:
    code: IssueCode
    title: str    # короткий заголовок для overlay/трея
    detail: str   # понятное объяснение + что делать


def check_model(model_size: str, cache_dir: Path,
                min_size: int = MIN_MODEL_BIN_SIZE) -> List[Issue]:
    """Проверить целостность модели. Пустой список = модель в порядке."""
    model_dir = cache_dir / f"models--Systran--faster-whisper-{model_size}"
    snapshots = model_dir / "snapshots"

    size_gb = _MODEL_SIZE_GB.get(model_size)
    size_hint = f" (≈{size_gb} ГБ)" if size_gb else ""

    if not snapshots.exists() or not any(snapshots.iterdir()):
        return [Issue(
            IssueCode.MODEL_MISSING,
            f"Модель {model_size} не найдена",
            f"Модель {model_size} не найдена{size_hint}. "
            f"Скачайте её и перезапустите приложение.",
        )]

    blobs = model_dir / "blobs"
    if blobs.exists() and any(blobs.glob("*.incomplete")):
        return [Issue(
            IssueCode.MODEL_CORRUPT,
            f"Модель {model_size} повреждена",
            f"Модель {model_size} повреждена (незавершённая загрузка). "
            f"Удалите кэш модели и скачайте заново.",
        )]

    for snap in snapshots.iterdir():
        if snap.is_dir():
            model_bin = snap / "model.bin"
            if model_bin.exists() and model_bin.stat().st_size >= min_size:
                return []  # модель в порядке

    return [Issue(
        IssueCode.MODEL_CORRUPT,
        f"Модель {model_size} повреждена",
        f"Файл model.bin отсутствует или повреждён. "
        f"Удалите кэш модели и скачайте заново.",
    )]
```

- [ ] **Step 5: Запустить — убедиться, что проходит**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_diagnostics.py -v` (из `voicetype/`)
Expected: PASS — 4 passed

- [ ] **Step 6: Commit**

```bash
git add voicetype/tests/conftest.py voicetype/tests/test_diagnostics.py voicetype/src/core/diagnostics.py
git commit -m "feat: add diagnostics module with model integrity check"
```

---

## Task 3: Проверка CUDA-библиотек

**Files:**
- Modify: `voicetype/tests/test_diagnostics.py`
- Modify: `voicetype/src/core/diagnostics.py`

- [ ] **Step 1: Написать падающие тесты на check_cuda**

Добавить в конец `voicetype/tests/test_diagnostics.py`:

```python
from src.core.diagnostics import check_cuda


def _make_nvidia(tmp_path: Path, dlls: list) -> Path:
    """Создать структуру site-packages/nvidia с заданными DLL (в подпапке bin)."""
    nvidia_base = tmp_path / "nvidia"
    layout = {
        "cudnn64_9.dll": "cudnn",
        "nvJitLink64_12.dll": "nvjitlink",
    }
    for dll in dlls:
        d = nvidia_base / layout[dll] / "bin"
        d.mkdir(parents=True, exist_ok=True)
        (d / dll).write_bytes(b"\0")
    return nvidia_base


def test_check_cuda_all_present(tmp_path):
    nvidia_base = _make_nvidia(tmp_path, ["cudnn64_9.dll", "nvJitLink64_12.dll"])
    assert check_cuda(nvidia_base) == []


def test_check_cuda_cudnn_missing(tmp_path):
    nvidia_base = _make_nvidia(tmp_path, ["nvJitLink64_12.dll"])  # нет cudnn
    issues = check_cuda(nvidia_base)
    codes = [i.code for i in issues]
    assert IssueCode.CUDA_CUDNN_MISSING in codes


def test_check_cuda_nvjitlink_missing(tmp_path):
    nvidia_base = _make_nvidia(tmp_path, ["cudnn64_9.dll"])  # нет nvjitlink
    issues = check_cuda(nvidia_base)
    codes = [i.code for i in issues]
    assert IssueCode.CUDA_NVJITLINK_MISSING in codes


def test_check_cuda_both_missing(tmp_path):
    nvidia_base = tmp_path / "nvidia"  # пусто
    nvidia_base.mkdir()
    issues = check_cuda(nvidia_base)
    assert len(issues) == 2
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_diagnostics.py -v` (из `voicetype/`)
Expected: FAIL — `ImportError: cannot import name 'check_cuda'`

- [ ] **Step 3: Реализовать check_cuda**

Добавить в `voicetype/src/core/diagnostics.py` (после `check_model`):

```python
_DLL_TO_CODE = {
    "cudnn64_9.dll": IssueCode.CUDA_CUDNN_MISSING,
    "nvJitLink64_12.dll": IssueCode.CUDA_NVJITLINK_MISSING,
}


def check_cuda(nvidia_base: Path) -> List[Issue]:
    """Проверить наличие CUDA DLL. Пустой список = всё на месте."""
    issues: List[Issue] = []
    for dll_name, pkg_subdir, human, pip_pkg in CUDA_REQUIRED_DLLS:
        found = False
        for sub in ("bin", os.path.join("lib", "x64"), "lib"):
            if (nvidia_base / pkg_subdir / sub / dll_name).exists():
                found = True
                break
        if not found:
            code = _DLL_TO_CODE[dll_name]
            issues.append(Issue(
                code,
                f"GPU недоступен: не найден {human}",
                f"GPU недоступен: не найден {human}. "
                f"Установите {pip_pkg} или переключитесь на CPU в настройках.",
            ))
    return issues
```

- [ ] **Step 4: Запустить — убедиться, что проходит**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_diagnostics.py -v` (из `voicetype/`)
Expected: PASS — 8 passed

- [ ] **Step 5: Commit**

```bash
git add voicetype/tests/test_diagnostics.py voicetype/src/core/diagnostics.py
git commit -m "feat: add CUDA DLL presence check to diagnostics"
```

---

## Task 4: Оркестрация diagnose()

**Files:**
- Modify: `voicetype/tests/test_diagnostics.py`
- Modify: `voicetype/src/core/diagnostics.py`

- [ ] **Step 1: Написать падающие тесты на diagnose**

Добавить в конец `voicetype/tests/test_diagnostics.py`:

```python
from src.core.diagnostics import diagnose


def test_diagnose_ok_cuda(tmp_path):
    cache_dir = _make_model(tmp_path, size=2000)
    nvidia_base = _make_nvidia(tmp_path, ["cudnn64_9.dll", "nvJitLink64_12.dll"])
    issues = diagnose("medium", "cuda", cache_dir=cache_dir,
                      nvidia_base=nvidia_base, min_size=1000)
    assert issues == []


def test_diagnose_cpu_skips_cuda(tmp_path):
    """device=cpu — CUDA не проверяется, даже если DLL отсутствуют."""
    cache_dir = _make_model(tmp_path, size=2000)
    nvidia_base = tmp_path / "nvidia"  # пусто
    nvidia_base.mkdir()
    issues = diagnose("medium", "cpu", cache_dir=cache_dir,
                      nvidia_base=nvidia_base, min_size=1000)
    assert issues == []


def test_diagnose_reports_model_and_cuda(tmp_path):
    """Битая модель + нет cuDNN/nvJitLink — обе категории проблем."""
    cache_dir = _make_model(tmp_path, size=100)  # повреждена
    nvidia_base = tmp_path / "nvidia"  # пусто
    nvidia_base.mkdir()
    issues = diagnose("medium", "cuda", cache_dir=cache_dir,
                      nvidia_base=nvidia_base, min_size=1000)
    codes = {i.code for i in issues}
    assert IssueCode.MODEL_CORRUPT in codes
    assert IssueCode.CUDA_CUDNN_MISSING in codes
    assert IssueCode.CUDA_NVJITLINK_MISSING in codes
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_diagnostics.py -v` (из `voicetype/`)
Expected: FAIL — `ImportError: cannot import name 'diagnose'`

- [ ] **Step 3: Реализовать diagnose**

Добавить в `voicetype/src/core/diagnostics.py` (после `check_cuda`):

```python
def diagnose(model_size: str, device: str, *,
             cache_dir: Optional[Path] = None,
             nvidia_base: Optional[Path] = None,
             min_size: int = MIN_MODEL_BIN_SIZE) -> List[Issue]:
    """
    Проверить окружение перед загрузкой моделей.

    Пустой список = всё в порядке, можно грузить. Иначе — список проблем.
    cache_dir / nvidia_base вычисляются по умолчанию (для прода);
    параметры нужны для тестов.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if nvidia_base is None:
        nvidia_base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"

    issues = check_model(model_size, cache_dir, min_size)
    if device == "cuda":
        issues = issues + check_cuda(nvidia_base)
    return issues
```

- [ ] **Step 4: Запустить — убедиться, что проходит**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_diagnostics.py -v` (из `voicetype/`)
Expected: PASS — 11 passed

- [ ] **Step 5: Commit**

```bash
git add voicetype/tests/test_diagnostics.py voicetype/src/core/diagnostics.py
git commit -m "feat: add diagnose() orchestration"
```

---

## Task 5: Усилить is_whisper_model_cached

**Files:**
- Create: `voicetype/tests/test_models_manager.py`
- Modify: `voicetype/src/data/models_manager.py:46-75`

- [ ] **Step 1: Написать падающий тест**

`voicetype/tests/test_models_manager.py`:

```python
"""Тест честности is_whisper_model_cached (видит повреждённую модель)."""
from pathlib import Path

from src.data.models_manager import get_models_manager


def _make_corrupt(cache_dir: Path):
    model_dir = cache_dir / "models--Systran--faster-whisper-medium"
    snap = model_dir / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (model_dir / "blobs").mkdir()
    (model_dir / "blobs" / "x.incomplete").write_bytes(b"")  # повреждена


def test_corrupt_model_not_cached(tmp_path, monkeypatch):
    cache_dir = tmp_path / "hub"
    cache_dir.mkdir()
    _make_corrupt(cache_dir)

    mgr = get_models_manager()
    monkeypatch.setattr(mgr, "get_whisper_cache_dir", lambda: cache_dir)

    # Раньше возвращал True (видел только папку snapshots); теперь должен видеть .incomplete
    assert mgr.is_whisper_model_cached("medium") is False
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_models_manager.py -v` (из `voicetype/`)
Expected: FAIL — `assert True is False` (текущая реализация считает повреждённую модель закэшированной)

- [ ] **Step 3: Переписать is_whisper_model_cached**

В `voicetype/src/data/models_manager.py` заменить тело метода `is_whisper_model_cached` (строки 46-75) на:

```python
    def is_whisper_model_cached(self, model_size: str) -> bool:
        """
        Проверить, скачана ли модель Whisper (и не повреждена).

        Args:
            model_size: Размер модели (small, medium)

        Returns:
            True если модель скачана и цела
        """
        if model_size not in WHISPER_MODEL_SIZES:
            logger.warning(f"Unknown Whisper model size: {model_size}")
            return False

        from src.core.diagnostics import check_model
        cache_dir = self.get_whisper_cache_dir()
        return len(check_model(model_size, cache_dir)) == 0
```

(Импорт `check_model` локальный — внутри метода — чтобы исключить любые проблемы порядка импорта при старте.)

- [ ] **Step 4: Запустить — убедиться, что проходит**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_models_manager.py -v` (из `voicetype/`)
Expected: PASS — 1 passed

- [ ] **Step 5: Commit**

```bash
git add voicetype/tests/test_models_manager.py voicetype/src/data/models_manager.py
git commit -m "fix: is_whisper_model_cached now detects corrupt models"
```

---

## Task 6: Убрать молчаливый CUDA→CPU fallback

**Files:**
- Create: `voicetype/tests/test_whisper_recognizer.py`
- Modify: `voicetype/src/core/whisper_recognizer.py:116` (новый атрибут), `:225-242` (fallback), `:275-279` (except)

- [ ] **Step 1: Написать падающий тест**

`voicetype/tests/test_whisper_recognizer.py`:

```python
"""Тест: при device=cuda и сбое НЕТ молчаливого перехода на CPU."""
from unittest.mock import patch, MagicMock

from src.core.whisper_recognizer import WhisperRecognizer


def test_cuda_failure_does_not_fallback_to_cpu():
    rec = WhisperRecognizer(model_size="small", device="cuda", language="ru")
    errors = []
    rec.on_error = lambda e: errors.append(e)

    # faster_whisper.WhisperModel всегда кидает ошибку
    with patch("faster_whisper.WhisperModel",
               side_effect=RuntimeError("cudnn missing")) as mock_model:
        ok = rec.load_model()

    assert ok is False                 # загрузка провалилась
    assert rec.device == "cuda"        # device НЕ сменился на cpu
    assert mock_model.call_count == 1  # CPU-попытки не было (был бы 2-й вызов)
    assert len(errors) == 1            # on_error вызван
    assert rec._last_load_error        # ошибка сохранена для UI
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_whisper_recognizer.py -v` (из `voicetype/`)
Expected: FAIL — `assert rec.device == "cuda"` падает (текущий код меняет device на "cpu" и делает 2-й вызов), либо `AttributeError: _last_load_error`

- [ ] **Step 3: Добавить атрибут _last_load_error**

В `voicetype/src/core/whisper_recognizer.py` после строки 118 (`self.on_model_unloaded = ...`) добавить:

```python
        self._last_load_error: Optional[str] = None  # текст последней ошибки загрузки (для UI)
```

- [ ] **Step 4: Убрать fallback**

В `voicetype/src/core/whisper_recognizer.py` заменить блок (текущие строки 225-242):

```python
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=compute_type,
                )
            except Exception as cuda_error:
                if self.device == "cuda":
                    logger.warning(f"CUDA загрузка не удалась: {cuda_error}, пробуем CPU...")
                    self.device = "cpu"
                    compute_type = "int8"
                    self._model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=compute_type,
                    )
                else:
                    raise
```

на:

```python
            # Без молчаливого CUDA→CPU fallback: при device=cuda и сбое ошибка
            # пробрасывается наверх, чтобы пользователь увидел честный диагноз.
            # Переключение на CPU делается осознанно через настройки.
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
```

- [ ] **Step 5: Сохранять текст ошибки в общем except**

В `voicetype/src/core/whisper_recognizer.py` в блоке `except Exception as e:` метода `load_model` (текущая строка 275) после `logger.error(...)` добавить строку сохранения. Блок станет:

```python
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self._last_load_error = str(e)
            if self.on_error:
                self.on_error(e)
            return False
```

- [ ] **Step 6: Запустить — убедиться, что проходит**

Run: `voicetype\venv\Scripts\python -m pytest tests/test_whisper_recognizer.py -v` (из `voicetype/`)
Expected: PASS — 1 passed

- [ ] **Step 7: Commit**

```bash
git add voicetype/tests/test_whisper_recognizer.py voicetype/src/core/whisper_recognizer.py
git commit -m "fix: remove silent CUDA->CPU fallback, store load error"
```

---

## Task 7: main_window.show_loading_error прокидывает детали

**Files:**
- Modify: `voicetype/src/ui/main_window.py:288-296`

- [ ] **Step 1: Расширить сигнатуру**

В `voicetype/src/ui/main_window.py` заменить метод `show_loading_error` (строки 288-296) на:

```python
    def show_loading_error(self, error_text: str, details: str = ""):
        """
        Показать ошибку загрузки.

        Args:
            error_text: Текст ошибки (заголовок)
            details: Подробности/инструкция (необязательно)
        """
        self._loading_overlay.show_error(error_text, details)
        self.setWindowTitle(f"{APP_NAME} - Ошибка загрузки")
```

- [ ] **Step 2: Проверить, что модуль импортируется без ошибок**

Run: `voicetype\venv\Scripts\python -c "import ast; ast.parse(open('src/ui/main_window.py', encoding='utf-8').read()); print('OK')"` (из `voicetype/`)
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add voicetype/src/ui/main_window.py
git commit -m "feat: show_loading_error forwards details to overlay"
```

---

## Task 8: Интеграция в app.py

**Files:**
- Modify: `voicetype/src/app.py` — сигнал (после :54), connect (после :163), `_on_recognition_error` (новый метод), on_error assignment (после :257), `_do_load_models` (:272-302), `_do_reload_models_then_start` (:458-488), `_on_models_loaded` (:669-679)

- [ ] **Step 1: Добавить сигнал _error_signal**

В `voicetype/src/app.py` после строки 54 (`_vad_status_signal = ...`) добавить:

```python
    _error_signal = pyqtSignal(str, str)  # (title, detail) — ошибка распознавания во время работы
```

- [ ] **Step 2: Подключить сигнал**

В `voicetype/src/app.py` в `_create_ui` после строки 163 (`self._vad_status_signal.connect(...)`) добавить:

```python
        self._error_signal.connect(self._on_recognition_error)
```

- [ ] **Step 3: Добавить обработчик ошибок распознавания**

В `voicetype/src/app.py` после метода `_on_whisper_status_changed` (после строки 694) добавить новый метод:

```python
    def _on_recognition_error(self, title: str, detail: str):
        """Обработчик ошибки распознавания во время работы (в UI потоке)."""
        logger.error(f"{title}: {detail}")
        self._tray_icon.show_notification(f"VoiceType - {title}", detail)
```

- [ ] **Step 4: Назначить recognizer.on_error**

В `voicetype/src/app.py` в `_create_recognizer` после строки 257 (`self._recognizer.on_model_unloaded = ...`) добавить:

```python
        # Ошибки во время работы (вызывается из recognition-потока) → UI через сигнал
        self._recognizer.on_error = lambda e: self._error_signal.emit("Ошибка распознавания", str(e))
```

- [ ] **Step 5: Добавить проактивную проверку в _do_load_models**

В `voicetype/src/app.py` в методе `_do_load_models`, после строки 287 (`QCoreApplication.processEvents()` сразу за получением `model_name`) и ДО блока `if self._recognizer and self._recognizer.load_model():`, вставить:

```python
            # Проактивная диагностика окружения перед загрузкой
            from src.core.diagnostics import diagnose
            device = self._config.get("audio.whisper.device", "cuda")
            issues = diagnose(model_name, device)
            if issues:
                title = issues[0].title
                detail = "\n".join(i.detail for i in issues)
                logger.error(f"Диагностика выявила проблемы: {detail}")
                self._main_window.show_loading_error(title, detail)
                self._tray_icon.show_notification("VoiceType - Ошибка", title)
                self._models_loaded_signal.emit(TRAY_STATE_ERROR, "")
                self._models_loaded = False
                return
```

- [ ] **Step 6: Добавить проактивную проверку в _do_reload_models_then_start**

В `voicetype/src/app.py` в методе `_do_reload_models_then_start`, после строки 464 (`logger.info(f"Перезагрузка моделей: {model_name}")`) вставить:

```python
            from src.core.diagnostics import diagnose
            device = self._config.get("audio.whisper.device", "cuda")
            issues = diagnose(model_name, device)
            if issues:
                title = issues[0].title
                detail = "\n".join(i.detail for i in issues)
                logger.error(f"Диагностика выявила проблемы при reload: {detail}")
                self._main_window.show_loading_error(title, detail)
                self._tray_icon.show_notification("VoiceType - Ошибка", title)
                self._tray_icon.set_state(TRAY_STATE_ERROR)
                self._recognizer.set_processing(False)
                return
```

- [ ] **Step 7: Передать детали в _on_models_loaded (реактивный путь)**

В `voicetype/src/app.py` в методе `_on_models_loaded`, в ветке `elif state == TRAY_STATE_ERROR:` (строки 669-679) заменить строку:

```python
            self._main_window.show_loading_error("Ошибка загрузки модели")
```

на:

```python
            detail = ""
            if self._recognizer is not None:
                detail = getattr(self._recognizer, "_last_load_error", "") or ""
            self._main_window.show_loading_error(
                "Ошибка загрузки модели",
                detail or "Подробности в логах.",
            )
```

- [ ] **Step 8: Проверить синтаксис и импорт app.py**

Run: `voicetype\venv\Scripts\python -c "import ast; ast.parse(open('src/app.py', encoding='utf-8').read()); print('OK')"` (из `voicetype/`)
Expected: `OK`

- [ ] **Step 9: Прогнать весь тестовый набор (регрессия)**

Run: `voicetype\venv\Scripts\python -m pytest tests/ -v` (из `voicetype/`)
Expected: PASS — 13 passed

- [ ] **Step 10: Commit**

```bash
git add voicetype/src/app.py
git commit -m "feat: wire diagnostics and on_error into app startup/reload"
```

---

## Task 9: Ручная верификация на реальном состоянии

**Files:** нет (проверка)

- [ ] **Step 1: Прогнать diagnose() на текущем (битом) состоянии машины**

Run (из `voicetype/`):
`voicetype\venv\Scripts\python -c "from src.core.diagnostics import diagnose; [print(i.code.value, '-', i.detail) for i in diagnose('medium','cuda')]"`

Expected: две строки —
```
model_corrupt - Модель medium повреждена ...
cuda_cudnn_missing - GPU недоступен: не найден cuDNN ...
```
(плюс, возможно, `cuda_nvjitlink_missing`). Это подтверждает, что диагностика видит реальное повреждение модели и отсутствие cuDNN.

- [ ] **Step 2: Запустить приложение и убедиться, что overlay показывает диагноз**

Run (из `voicetype/`): `voicetype\venv\Scripts\python run.py`
Expected: окно загрузки показывает заголовок «Модель medium повреждена» и детали с упоминанием cuDNN; приложение НЕ висит на бесконечной загрузке и НЕ уходит молча на CPU. Закрыть приложение после проверки.

- [ ] **Step 3: Финальный прогон тестов**

Run: `voicetype\venv\Scripts\python -m pytest tests/ -v` (из `voicetype/`)
Expected: PASS — 13 passed

---

## Замечания по реализации

- **Кодировка файлов:** проект на Windows; при чтении исходников в проверках синтаксиса использовать `encoding="utf-8"` (см. команды выше).
- **Запуск тестов всегда из каталога `voicetype/`** — иначе `conftest.py` не добавит корректный путь и `import src.*` упадёт.
- **Task 8 не покрыт юнит-тестами** намеренно: это Qt/event-loop интеграция, проверяется вручную в Task 9. Вся тестируемая логика вынесена в `diagnostics` (Tasks 2-4).
- **Восстановление GPU** (докачка cuDNN + nvJitLink + модель medium, ≈2.26 ГБ) — отдельный трек после этого плана; в объём не входит.
```