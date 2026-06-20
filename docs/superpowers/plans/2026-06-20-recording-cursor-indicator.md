# Recording Cursor Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Во время записи подменять системный курсор мыши (стрелка + I-beam) на красный, чтобы факт идущей записи был очевиден, и возвращать обычный курсор на стопе/выходе.

**Architecture:** Новый синглтон `RecordingCursor` (`src/utils/recording_cursor.py`) через WinAPI `SetSystemCursor` ставит красные курсоры на старте записи и `SystemParametersInfo(SPI_SETCURSORS)` возвращает их на стопе. Подключается к существующим точкам переключения состояния (`recording_session`), плюс безусловный сброс на старте/выходе приложения (`app.py`). Ноль фоновой нагрузки, без новых зависимостей.

**Tech Stack:** Python, `ctypes` (WinAPI: user32/gdi32), PyQt6 (`QPainter`/`QImage` для отрисовки), pytest + unittest.mock для тестов.

**Спека:** `docs/superpowers/specs/2026-06-20-recording-cursor-indicator-design.md`

---

## File Structure

- **Create** `voicetype/src/utils/recording_cursor.py` — модуль с классом `RecordingCursor`, синглтоном `get_recording_cursor()` и платформенными хелперами (`_create_red_cursor`, `_set_system_cursor`, `_restore_default_cursors`, `_cursor_size`).
- **Create** `voicetype/tests/test_recording_cursor.py` — unit-тесты логики (WinAPI/Qt замоканы).
- **Modify** `voicetype/src/core/recording_session.py` — `activate()` на старте, `deactivate()` на стопе и при сбое загрузки.
- **Modify** `voicetype/src/app.py` — `restore()` на старте приложения (`initialize`) и на выходе (`quit`).
- **Modify** `CLAUDE.md` — строка `ui.recording_cursor` в таблице конфигурации.

Запуск тестов из каталога `voicetype/` (там лежит `conftest.py`, добавляющий корень в `sys.path`).

---

## Task 1: Модуль RecordingCursor

**Files:**
- Create: `voicetype/src/utils/recording_cursor.py`
- Test: `voicetype/tests/test_recording_cursor.py`

- [ ] **Step 1: Написать падающий тест**

Создать `voicetype/tests/test_recording_cursor.py`:

```python
"""Тесты RecordingCursor: подмена системного курсора на время записи.

WinAPI/Qt-отрисовка замоканы — проверяем только логику activate/deactivate/
restore: какие OCR-курсоры ставятся, уважение конфиг-рубильника,
идемпотентность и то, что сбой WinAPI НЕ пробрасывается (запись не должна падать).
"""
from unittest.mock import patch, MagicMock

import src.utils.recording_cursor as rc
from src.utils.recording_cursor import (
    RecordingCursor, get_recording_cursor, OCR_NORMAL, OCR_IBEAM,
)


def _config(enabled=True):
    cfg = MagicMock()
    cfg.get.return_value = enabled
    return cfg


def test_activate_sets_both_system_cursors():
    cur = RecordingCursor()
    with patch.object(rc, "get_config", return_value=_config(True)), \
         patch.object(rc, "_create_red_cursor", side_effect=[111, 222]) as mk, \
         patch.object(rc, "_set_system_cursor") as setc, \
         patch.object(rc, "_restore_default_cursors"):
        cur.activate()
    assert cur._active is True
    assert mk.call_count == 2
    setc.assert_any_call(111, OCR_NORMAL)
    setc.assert_any_call(222, OCR_IBEAM)
    assert setc.call_count == 2


def test_activate_respects_config_switch_off():
    cur = RecordingCursor()
    with patch.object(rc, "get_config", return_value=_config(False)), \
         patch.object(rc, "_create_red_cursor") as mk, \
         patch.object(rc, "_set_system_cursor") as setc:
        cur.activate()
    assert cur._active is False
    mk.assert_not_called()
    setc.assert_not_called()


def test_activate_is_idempotent():
    cur = RecordingCursor()
    with patch.object(rc, "get_config", return_value=_config(True)), \
         patch.object(rc, "_create_red_cursor", side_effect=[1, 2, 3, 4]), \
         patch.object(rc, "_set_system_cursor") as setc, \
         patch.object(rc, "_restore_default_cursors"):
        cur.activate()
        cur.activate()  # второй раз — no-op
    assert setc.call_count == 2  # не 4


def test_deactivate_restores_and_clears_active():
    cur = RecordingCursor()
    cur._active = True
    with patch.object(rc, "_restore_default_cursors") as restore:
        cur.deactivate()
    restore.assert_called_once()
    assert cur._active is False


def test_deactivate_when_inactive_is_noop():
    cur = RecordingCursor()
    with patch.object(rc, "_restore_default_cursors") as restore:
        cur.deactivate()
    restore.assert_not_called()


def test_restore_always_calls_winapi_and_clears_active():
    cur = RecordingCursor()
    cur._active = True
    with patch.object(rc, "_restore_default_cursors") as restore:
        cur.restore()
    restore.assert_called_once()
    assert cur._active is False


def test_activate_swallows_winapi_errors():
    cur = RecordingCursor()
    with patch.object(rc, "get_config", return_value=_config(True)), \
         patch.object(rc, "_create_red_cursor", side_effect=RuntimeError("winapi boom")), \
         patch.object(rc, "_set_system_cursor"), \
         patch.object(rc, "_restore_default_cursors"):
        cur.activate()  # не должно бросить исключение
    assert cur._active is False


def test_get_recording_cursor_is_singleton():
    assert get_recording_cursor() is get_recording_cursor()
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `cd voicetype && python -m pytest tests/test_recording_cursor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.recording_cursor'`

- [ ] **Step 3: Реализовать модуль**

Создать `voicetype/src/utils/recording_cursor.py`:

```python
"""
VoiceType — подмена системного курсора на время записи.

Во время записи курсор (стрелка + I-beam) становится красным, чтобы факт идущей
записи был очевиден. Реализация — через WinAPI SetSystemCursor: один вызов на
старте, возврат через SystemParametersInfo(SPI_SETCURSORS) на стопе/выходе.
Ноль фоновой нагрузки. Любой сбой WinAPI логируется и не ломает запись.
"""
from loguru import logger

from src.data.config import get_config

# Идентификаторы системных курсоров (winuser.h)
OCR_NORMAL = 32512   # обычная стрелка
OCR_IBEAM = 32513    # текстовый курсор

# Параметры WinAPI
_SPI_SETCURSORS = 0x0057   # перезагрузка курсоров из реестра
_SM_CXCURSOR = 13          # ширина системного курсора


def _cursor_size() -> int:
    """Размер системного курсора в пикселях (учитывает масштаб/доступность)."""
    try:
        import ctypes
        size = ctypes.windll.user32.GetSystemMetrics(_SM_CXCURSOR)
        return int(size) if size and size > 0 else 32
    except Exception:
        return 32


def _create_red_cursor(shape: str, size: int) -> int:
    """Нарисовать красный курсор заданной формы и вернуть HCURSOR.

    shape: "arrow" (стрелка, hotspot в острие 0,0) или "ibeam" (I-beam, hotspot
    по центру). Рисуем через QPainter (как tray_icon._generate_icon), затем
    конвертируем QImage -> HCURSOR через CreateDIBSection + CreateIconIndirect.
    """
    import ctypes
    from ctypes import wintypes
    from PyQt6.QtGui import QImage, QPainter, QColor, QPolygon, QPen, QBrush
    from PyQt6.QtCore import QPoint

    # 1) Рисуем в QImage (ARGB32, прозрачный фон)
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    red = QColor(0xE0, 0x10, 0x10)
    white = QColor(255, 255, 255)
    s = size / 32.0

    if shape == "arrow":
        pts = [(0, 0), (0, 22), (6, 16), (10, 25), (14, 23), (10, 15), (17, 15)]
        poly = QPolygon([QPoint(int(x * s), int(y * s)) for x, y in pts])
        painter.setPen(QPen(white, max(1, int(round(1.5 * s)))))
        painter.setBrush(QBrush(red))
        painter.drawPolygon(poly)
        hotspot = (0, 0)
    else:  # ibeam
        cx = size // 2
        top = int(round(4 * s))
        bot = int(round(28 * s))
        serif = int(round(3 * s))
        painter.setPen(QPen(white, max(3, int(round(4 * s)))))
        painter.drawLine(cx, top, cx, bot)            # белая подложка-контур
        painter.setPen(QPen(red, max(1, int(round(2 * s)))))
        painter.drawLine(cx, top, cx, bot)            # красная черта
        painter.drawLine(cx - serif, top, cx + serif, top)
        painter.drawLine(cx - serif, bot, cx + serif, bot)
        hotspot = (cx, size // 2)

    painter.end()

    width = img.width()
    height = img.height()

    gdi32 = ctypes.windll.gdi32
    user32 = ctypes.windll.user32

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class ICONINFO(ctypes.Structure):
        _fields_ = [
            ("fIcon", wintypes.BOOL),
            ("xHotspot", wintypes.DWORD),
            ("yHotspot", wintypes.DWORD),
            ("hbmMask", wintypes.HBITMAP),
            ("hbmColor", wintypes.HBITMAP),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = width
    bmi.biHeight = -height          # top-down, как QImage
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0           # BI_RGB

    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.CreateDIBSection.argtypes = [
        wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
        ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
    ]
    user32.CreateIconIndirect.restype = wintypes.HICON
    user32.CreateIconIndirect.argtypes = [ctypes.POINTER(ICONINFO)]

    ppv_bits = ctypes.c_void_p()
    hdc = user32.GetDC(0)
    color_bmp = gdi32.CreateDIBSection(hdc, ctypes.byref(bmi), 0,
                                       ctypes.byref(ppv_bits), None, 0)
    user32.ReleaseDC(0, hdc)

    # Копируем пиксели QImage (ARGB32 == BGRA в памяти) в DIB
    n_bytes = width * height * 4
    bits = img.constBits()
    bits.setsize(n_bytes)
    ctypes.memmove(ppv_bits, bytes(bits), n_bytes)

    # Монохромная AND-маска из нулей (прозрачность берётся из альфы цветного DIB)
    mask_bmp = gdi32.CreateBitmap(width, height, 1, 1, None)

    info = ICONINFO()
    info.fIcon = False               # FALSE => курсор (а не иконка)
    info.xHotspot = hotspot[0]
    info.yHotspot = hotspot[1]
    info.hbmMask = mask_bmp
    info.hbmColor = color_bmp

    hcursor = user32.CreateIconIndirect(ctypes.byref(info))

    gdi32.DeleteObject(color_bmp)
    gdi32.DeleteObject(mask_bmp)
    return hcursor


def _set_system_cursor(hcursor: int, ocr_id: int) -> None:
    """SetSystemCursor: ставит hcursor как системный курсор ocr_id.

    Внимание: система забирает владение hcursor и уничтожает его, поэтому на
    каждый вызов передаётся свежесозданный хендл (не переиспользуется).
    """
    import ctypes
    ctypes.windll.user32.SetSystemCursor(hcursor, ocr_id)


def _restore_default_cursors() -> None:
    """Вернуть настоящие курсоры пользователя (перезагрузка из реестра)."""
    import ctypes
    ctypes.windll.user32.SystemParametersInfoW(_SPI_SETCURSORS, 0, None, 0)


class RecordingCursor:
    """Управление подменой системного курсора на время записи."""

    def __init__(self):
        self._active = False

    def activate(self) -> None:
        """Поставить красные курсоры (если включено в конфиге)."""
        if self._active:
            return
        try:
            if not get_config().get("ui.recording_cursor", True):
                return
            size = _cursor_size()
            arrow = _create_red_cursor("arrow", size)
            _set_system_cursor(arrow, OCR_NORMAL)
            ibeam = _create_red_cursor("ibeam", size)
            _set_system_cursor(ibeam, OCR_IBEAM)
            self._active = True
            logger.info("RecordingCursor: красные курсоры установлены")
        except Exception as e:
            logger.warning(f"RecordingCursor.activate failed: {e}")

    def deactivate(self) -> None:
        """Вернуть обычные курсоры (если были подменены)."""
        if not self._active:
            return
        try:
            _restore_default_cursors()
            logger.info("RecordingCursor: курсоры возвращены")
        except Exception as e:
            logger.warning(f"RecordingCursor.deactivate failed: {e}")
        finally:
            self._active = False

    def restore(self) -> None:
        """Безусловный сброс к системным курсорам (страховка на старте/выходе)."""
        try:
            _restore_default_cursors()
        except Exception as e:
            logger.warning(f"RecordingCursor.restore failed: {e}")
        finally:
            self._active = False


_instance = None


def get_recording_cursor() -> RecordingCursor:
    """Получить синглтон RecordingCursor."""
    global _instance
    if _instance is None:
        _instance = RecordingCursor()
    return _instance
```

- [ ] **Step 4: Запустить тест — убедиться, что проходит**

Run: `cd voicetype && python -m pytest tests/test_recording_cursor.py -v`
Expected: PASS — все 8 тестов зелёные.

- [ ] **Step 5: Коммит**

```bash
git add voicetype/src/utils/recording_cursor.py voicetype/tests/test_recording_cursor.py
git commit -m "feat: add RecordingCursor (red system cursor during recording)"
```

---

## Task 2: Привязка к жизненному циклу записи

**Files:**
- Modify: `voicetype/src/core/recording_session.py` (импорт; `_do_start_recording` ~line 126; `stop` ~line 171; `abort_load_failure` ~line 247)

- [ ] **Step 1: Добавить импорт**

В `voicetype/src/core/recording_session.py` после строки
`from src.utils.constants import TRAY_STATE_READY, TRAY_STATE_RECORDING` добавить:

```python
from src.utils.recording_cursor import get_recording_cursor
```

- [ ] **Step 2: Активировать курсор на старте записи**

В методе `_do_start_recording`, в блоке обновления UI, заменить:

```python
        # Обновляем UI
        self._app._tray_icon.set_state(TRAY_STATE_RECORDING)
```

на:

```python
        # Обновляем UI
        self._app._tray_icon.set_state(TRAY_STATE_RECORDING)
        get_recording_cursor().activate()
```

- [ ] **Step 3: Вернуть курсор на стопе**

В методе `stop`, в блоке обновления UI, заменить:

```python
        # Обновляем UI
        self._app._tray_icon.set_state(TRAY_STATE_READY)
        self._app._recognition_finished_signal.emit()
```

на:

```python
        # Обновляем UI
        self._app._tray_icon.set_state(TRAY_STATE_READY)
        get_recording_cursor().deactivate()
        self._app._recognition_finished_signal.emit()
```

- [ ] **Step 4: Вернуть курсор при сбое фоновой загрузки**

В методе `abort_load_failure`, в самом конце, заменить:

```python
        if restore_tray:
            self._app._tray_icon.set_state(TRAY_STATE_READY)

        self._app._recognition_finished_signal.emit()
```

на:

```python
        if restore_tray:
            self._app._tray_icon.set_state(TRAY_STATE_READY)

        get_recording_cursor().deactivate()
        self._app._recognition_finished_signal.emit()
```

- [ ] **Step 5: Прогнать весь тест-сьют (регрессий нет) и проверить импорт**

Run: `cd voicetype && python -m pytest -q && python -c "import src.core.recording_session"`
Expected: тесты зелёные, импорт без ошибок (нет циклического импорта).

- [ ] **Step 6: Коммит**

```bash
git add voicetype/src/core/recording_session.py
git commit -m "feat: drive recording cursor from recording lifecycle"
```

---

## Task 3: Сброс курсора на старте и выходе приложения

**Files:**
- Modify: `voicetype/src/app.py` (импорт после line 27; `initialize` ~line 93; `quit` ~line 544)

- [ ] **Step 1: Добавить импорт**

В `voicetype/src/app.py` после строки
`from src.utils.system_info import get_process_cpu, get_process_memory, get_vram_usage, reset_vram_baseline`
добавить:

```python
from src.utils.recording_cursor import get_recording_cursor
```

- [ ] **Step 2: Безусловный сброс курсора на старте приложения**

В методе `initialize`, заменить:

```python
            # Конфигурация
            self._config.load()
```

на:

```python
            # Конфигурация
            self._config.load()

            # Страховка: если прошлый запуск упал во время записи и оставил
            # красный курсор — сбрасываем его на старте.
            get_recording_cursor().restore()
```

- [ ] **Step 3: Сброс курсора на выходе**

В методе `quit`, заменить:

```python
        # Останавливаем запись если идёт
        if self._recording.is_recording():
            self.stop_recording()
```

на:

```python
        # Останавливаем запись если идёт
        if self._recording.is_recording():
            self.stop_recording()

        # Гарантируем возврат обычного курсора
        get_recording_cursor().restore()
```

- [ ] **Step 4: Прогнать тест-сьют и проверить импорт**

Run: `cd voicetype && python -m pytest -q && python -c "import src.app"`
Expected: тесты зелёные, импорт без ошибок.

- [ ] **Step 5: Коммит**

```bash
git add voicetype/src/app.py
git commit -m "feat: reset cursor on app start/exit (crash safety)"
```

---

## Task 4: Документация и ручная проверка

**Files:**
- Modify: `CLAUDE.md` (таблица конфигурации в разделе "Configuration")

- [ ] **Step 1: Добавить ключ в таблицу конфигурации**

В `CLAUDE.md`, в таблице настроек раздела "Configuration", после строки
`| output.mode | "keyboard", "clipboard" | "keyboard" |` добавить:

```markdown
| `ui.recording_cursor` | true/false (красный курсор при записи) | true |
```

- [ ] **Step 2: Коммит документации**

```bash
git add CLAUDE.md
git commit -m "docs: document ui.recording_cursor config key"
```

- [ ] **Step 3: Ручная проверка (запуск приложения)**

Run: `cd voicetype && python run.py`

Проверить по чек-листу:
1. Старт записи (хоткей `Ctrl+Shift+S` или двойной клик по трею) → курсор-стрелка стала красной; наведение на поле ввода → I-beam тоже красный.
2. Стоп записи → курсоры вернулись к обычным.
3. Во время записи убить процесс (Диспетчер задач) → курсор остаётся красным; снова запустить `python run.py` → на старте курсор сбросился к обычному.
4. В `%APPDATA%/VoiceType/config.yaml` выставить `ui: { recording_cursor: false }`, перезапустить, начать запись → курсор НЕ меняется (рубильник работает).

Expected: все 4 пункта выполняются. Если какой-то пункт не проходит — это баг реализации, чинить до закрытия задачи.

---

## Self-Review

**Spec coverage:**
- Подмена стрелки + I-beam на красный → Task 1 (`_create_red_cursor` для "arrow"/"ibeam"), Task 2 (activate).
- WinAPI-подход, ноль нагрузки, без зависимостей → Task 1 (ctypes, без новых пакетов).
- Возврат на стопе → Task 2 Step 3; при сбое загрузки → Task 2 Step 4.
- Безусловный сброс на старте → Task 3 Step 2; на выходе → Task 3 Step 3.
- Рубильник `ui.recording_cursor` (default true) → Task 1 (`activate` читает конфиг), Task 4 (docs), Task 4 Step 3.4 (ручная проверка).
- Размер из `GetSystemMetrics(SM_CXCURSOR)` → Task 1 (`_cursor_size`).
- Сбой WinAPI не ломает запись → Task 1 (try/except в activate/deactivate/restore) + тест `test_activate_swallows_winapi_errors`.
- Тесты → Task 1 (8 тестов); визуальная проверка → Task 4 Step 3.

**Placeholder scan:** плейсхолдеров нет — все шаги содержат полный код/команды.

**Type/name consistency:** `RecordingCursor`, `get_recording_cursor`, `OCR_NORMAL`, `OCR_IBEAM`, `_create_red_cursor`, `_set_system_cursor`, `_restore_default_cursors`, `_cursor_size`, `_active` — имена согласованы между Task 1 (определение), тестами и Task 2/3 (вызовы `activate`/`deactivate`/`restore`).
