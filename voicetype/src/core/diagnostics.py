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
