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
