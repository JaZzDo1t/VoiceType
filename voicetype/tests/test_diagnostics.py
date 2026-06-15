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
