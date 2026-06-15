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

    assert mgr.is_whisper_model_cached("medium") is False
