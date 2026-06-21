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
         patch.object(rc, "_create_cursor_with_dot", side_effect=[111, 222]) as mk, \
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
         patch.object(rc, "_create_cursor_with_dot") as mk, \
         patch.object(rc, "_set_system_cursor") as setc:
        cur.activate()
    assert cur._active is False
    mk.assert_not_called()
    setc.assert_not_called()


def test_activate_is_idempotent():
    cur = RecordingCursor()
    with patch.object(rc, "get_config", return_value=_config(True)), \
         patch.object(rc, "_create_cursor_with_dot", side_effect=[1, 2, 3, 4]), \
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
         patch.object(rc, "_create_cursor_with_dot", side_effect=RuntimeError("winapi boom")), \
         patch.object(rc, "_set_system_cursor"), \
         patch.object(rc, "_restore_default_cursors"):
        cur.activate()  # не должно бросить исключение
    assert cur._active is False


def test_get_recording_cursor_is_singleton():
    assert get_recording_cursor() is get_recording_cursor()
