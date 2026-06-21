"""
VoiceType — подмена системного курсора на время записи.

Во время записи к курсору (стрелка + I-beam) добавляется красная точка REC,
чтобы факт идущей записи был очевиден. Берём НАСТОЯЩИЙ системный курсор
(пиксель-в-пиксель, родной силуэт и хотспот) через WinAPI, рисуем поверх него
красную точку и ставим как системный курсор через SetSystemCursor. Возврат —
через SystemParametersInfo(SPI_SETCURSORS) на стопе/выходе. Ноль фоновой
нагрузки. Любой сбой WinAPI логируется и не ломает запись.
"""
import ctypes
from ctypes import wintypes

from loguru import logger

from src.data.config import get_config

# Идентификаторы системных курсоров. Значения OCR_* (SetSystemCursor) и IDC_*
# (LoadCursor) для стрелки/текста совпадают численно — используем как одни и те же.
OCR_NORMAL = 32512   # обычная стрелка
OCR_IBEAM = 32513    # текстовый курсор

# Параметры WinAPI
_SPI_SETCURSORS = 0x0057   # перезагрузка курсоров из реестра
_SM_CXCURSOR = 13          # ширина системного курсора
_DI_NORMAL = 0x0003        # DrawIconEx: рисовать изображение + маску


class _BITMAPINFOHEADER(ctypes.Structure):
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


class _ICONINFO(ctypes.Structure):
    _fields_ = [
        ("fIcon", wintypes.BOOL),
        ("xHotspot", wintypes.DWORD),
        ("yHotspot", wintypes.DWORD),
        ("hbmMask", wintypes.HBITMAP),
        ("hbmColor", wintypes.HBITMAP),
    ]


def _winapi():
    """Вернуть (user32, gdi32) с правильными сигнатурами.

    На 64-битной Windows хендлы — указатели. Без argtypes ctypes пытается уложить
    64-битный хендл в C int → OverflowError. Поэтому задаём сигнатуры всем
    используемым функциям (идемпотентно, дёшево).
    """
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    user32.GetDC.restype = wintypes.HDC
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.LoadCursorW.restype = wintypes.HANDLE
    user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
    user32.GetIconInfo.restype = wintypes.BOOL
    user32.GetIconInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ICONINFO)]
    user32.DrawIconEx.restype = wintypes.BOOL
    user32.DrawIconEx.argtypes = [
        wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.HANDLE,
        ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.HANDLE, wintypes.UINT,
    ]
    user32.CreateIconIndirect.restype = wintypes.HICON
    user32.CreateIconIndirect.argtypes = [ctypes.POINTER(_ICONINFO)]

    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.CreateDIBSection.argtypes = [
        wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
        ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
    ]
    gdi32.CreateBitmap.restype = wintypes.HBITMAP
    gdi32.CreateBitmap.argtypes = [
        ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
    ]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
    gdi32.GdiFlush.restype = wintypes.BOOL
    gdi32.GdiFlush.argtypes = []

    return user32, gdi32


def _cursor_size() -> int:
    """Размер системного курсора в пикселях (учитывает масштаб/доступность)."""
    try:
        size = ctypes.windll.user32.GetSystemMetrics(_SM_CXCURSOR)
        return int(size) if size and size > 0 else 32
    except Exception:
        return 32


def _native_cursor_image(idc: int, size: int):
    """Отрисовать родной системный курсор idc в QImage.

    Возвращает (QImage ARGB32, hotspot_x, hotspot_y). Хотспот считывается у
    настоящего курсора, чтобы клик попадал точно.
    """
    from PyQt6.QtGui import QImage

    user32, gdi32 = _winapi()

    hcur = user32.LoadCursorW(None, idc)   # общий системный курсор, НЕ уничтожать
    if not hcur:
        raise RuntimeError(f"LoadCursorW({idc}) returned NULL")

    # Хотспот настоящего курсора (GetIconInfo создаёт битмапы — их надо удалить)
    hx, hy = 0, 0
    ii = _ICONINFO()
    if user32.GetIconInfo(hcur, ctypes.byref(ii)):
        hx, hy = int(ii.xHotspot), int(ii.yHotspot)
        if ii.hbmMask:
            gdi32.DeleteObject(ii.hbmMask)
        if ii.hbmColor:
            gdi32.DeleteObject(ii.hbmColor)

    # DIB (top-down 32bpp), куда DrawIconEx отрисует курсор с альфой
    bmi = _BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.biWidth = size
    bmi.biHeight = -size            # top-down, как QImage
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0           # BI_RGB

    ppv_bits = ctypes.c_void_p()
    screen = user32.GetDC(0)
    color_bmp = gdi32.CreateDIBSection(screen, ctypes.byref(bmi), 0,
                                       ctypes.byref(ppv_bits), None, 0)
    user32.ReleaseDC(0, screen)

    memdc = gdi32.CreateCompatibleDC(0)
    old = gdi32.SelectObject(memdc, color_bmp)
    user32.DrawIconEx(memdc, 0, 0, hcur, size, size, 0, None, _DI_NORMAL)
    gdi32.GdiFlush()
    gdi32.SelectObject(memdc, old)
    gdi32.DeleteDC(memdc)

    n_bytes = size * size * 4
    raw = ctypes.string_at(ppv_bits, n_bytes)
    gdi32.DeleteObject(color_bmp)

    img = QImage(raw, size, size, QImage.Format.Format_ARGB32).copy()
    return img, hx, hy


def _is_blank(img) -> bool:
    """True, если у изображения нет ни одного непрозрачного пикселя.

    Так детектируем монохромные курсоры (например, классический I-beam), для
    которых DrawIconEx не проставляет альфу и картинка выходит прозрачной.
    """
    n = img.width() * img.height() * 4
    bits = img.constBits()
    bits.setsize(n)
    data = bytes(bits)
    return not any(data[3::4])   # все альфа-байты == 0


def _fallback_image(idc: int, size: int):
    """Нарисовать курсор самостоятельно, если родной не отрисовался.

    Возвращает (QImage, hotspot_x, hotspot_y). Для I-beam — чистый симметричный
    глиф (тёмная «I» с белым ореолом, читается на любом фоне).
    """
    from PyQt6.QtGui import QImage, QPainter, QColor, QPen, QPolygon, QBrush
    from PyQt6.QtCore import QPoint

    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 32.0
    dark = QColor(0x2A, 0x2A, 0x2A)
    white = QColor(255, 255, 255)

    if idc == OCR_IBEAM:
        cx = size // 2
        top = int(round(6 * s))
        bot = int(round(26 * s))
        serif = int(round(4 * s))
        painter.setPen(QPen(white, max(3.0, 3.4 * s)))   # белый ореол
        painter.drawLine(cx, top, cx, bot)
        painter.drawLine(cx - serif, top, cx + serif, top)
        painter.drawLine(cx - serif, bot, cx + serif, bot)
        painter.setPen(QPen(dark, max(1.4, 1.6 * s)))    # тёмный глиф
        painter.drawLine(cx, top, cx, bot)
        painter.drawLine(cx - serif, top, cx + serif, top)
        painter.drawLine(cx - serif, bot, cx + serif, bot)
        hotspot = (cx, size // 2)
    else:  # стрелка (запасной силуэт)
        pts = [(0, 0), (0, 18), (4, 14), (7, 21), (9, 20), (6, 13), (12, 13)]
        poly = QPolygon([QPoint(int(round(x * s)), int(round(y * s))) for x, y in pts])
        painter.setPen(QPen(white, max(1.0, 1.4 * s)))
        painter.setBrush(QBrush(dark))
        painter.drawPolygon(poly)
        hotspot = (0, 0)

    painter.end()
    return img, hotspot[0], hotspot[1]


def _add_rec_dot(img, size: int, idc: int) -> None:
    """Дорисовать красную точку REC рядом с курсором (in place).

    Положение зависит от формы: у стрелки — в «локте» снизу-справа, у I-beam —
    правее вертикальной черты, чтобы её не перекрывать.
    """
    from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
    from PyQt6.QtCore import QRectF

    s = size / 32.0
    r = 5.0 * s
    if idc == OCR_IBEAM:
        cx, cy = 23.0 * s, 21.0 * s
    else:
        cx, cy = 16.0 * s, 18.0 * s

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(255, 255, 255), max(1.0, 1.4 * s)))
    painter.setBrush(QBrush(QColor(0xDC, 0x26, 0x26)))
    painter.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
    painter.end()


def _qimage_to_hcursor(img, hotspot_x: int, hotspot_y: int) -> int:
    """Конвертировать QImage (ARGB32) в HCURSOR с заданным хотспотом."""
    user32, gdi32 = _winapi()

    width = img.width()
    height = img.height()

    bmi = _BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.biWidth = width
    bmi.biHeight = -height          # top-down, как QImage
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0           # BI_RGB

    ppv_bits = ctypes.c_void_p()
    screen = user32.GetDC(0)
    color_bmp = gdi32.CreateDIBSection(screen, ctypes.byref(bmi), 0,
                                       ctypes.byref(ppv_bits), None, 0)
    user32.ReleaseDC(0, screen)

    # Копируем пиксели QImage (ARGB32 == BGRA в памяти) в DIB
    n_bytes = width * height * 4
    bits = img.constBits()
    bits.setsize(n_bytes)
    ctypes.memmove(ppv_bits, bytes(bits), n_bytes)

    # Монохромная AND-маска из нулей (прозрачность берётся из альфы цветного DIB)
    mask_bmp = gdi32.CreateBitmap(width, height, 1, 1, None)

    info = _ICONINFO()
    info.fIcon = False               # FALSE => курсор (а не иконка)
    info.xHotspot = hotspot_x
    info.yHotspot = hotspot_y
    info.hbmMask = mask_bmp
    info.hbmColor = color_bmp

    hcursor = user32.CreateIconIndirect(ctypes.byref(info))

    gdi32.DeleteObject(color_bmp)
    gdi32.DeleteObject(mask_bmp)
    return hcursor


def _create_cursor_with_dot(idc: int, size: int) -> int:
    """Собрать курсор: родной системный курсор idc + красная точка REC."""
    img, hx, hy = _native_cursor_image(idc, size)
    if _is_blank(img):
        img, hx, hy = _fallback_image(idc, size)
    _add_rec_dot(img, size, idc)
    return _qimage_to_hcursor(img, hx, hy)


def _set_system_cursor(hcursor: int, ocr_id: int) -> None:
    """SetSystemCursor: ставит hcursor как системный курсор ocr_id.

    Внимание: система забирает владение hcursor и уничтожает его, поэтому на
    каждый вызов передаётся свежесозданный хендл (не переиспользуется).
    """
    user32 = ctypes.windll.user32
    user32.SetSystemCursor.restype = wintypes.BOOL
    user32.SetSystemCursor.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    user32.SetSystemCursor(hcursor, ocr_id)


def _restore_default_cursors() -> None:
    """Вернуть настоящие курсоры пользователя (перезагрузка из реестра)."""
    user32 = ctypes.windll.user32
    user32.SystemParametersInfoW.restype = wintypes.BOOL
    user32.SystemParametersInfoW.argtypes = [
        wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT,
    ]
    user32.SystemParametersInfoW(_SPI_SETCURSORS, 0, None, 0)


class RecordingCursor:
    """Управление подменой системного курсора на время записи."""

    def __init__(self):
        self._active = False

    def activate(self) -> None:
        """Поставить курсоры с точкой REC (если включено в конфиге)."""
        if self._active:
            return
        try:
            if not get_config().get("ui.recording_cursor", True):
                return
            size = _cursor_size()
            for ocr in (OCR_NORMAL, OCR_IBEAM):
                hcursor = _create_cursor_with_dot(ocr, size)
                _set_system_cursor(hcursor, ocr)
            self._active = True
            logger.info("RecordingCursor: курсоры с точкой REC установлены")
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
