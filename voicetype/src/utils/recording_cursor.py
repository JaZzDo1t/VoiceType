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
