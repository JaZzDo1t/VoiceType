"""
Скачивание модели Whisper medium и проверка загрузки на GPU.

Запускать ПОСЛЕ установки cuDNN/nvJitLink. Для скорости — с отключённым VPN
(HuggingFace доступен напрямую). Показывает прогресс скачивания (~1.53 ГБ).

Удобнее всего запускать через download_model.bat (двойной клик).
"""
import os
import sys
import shutil
from pathlib import Path

# Добавляем voicetype/ в sys.path чтобы импортировать src.* (скрипт запускается из voicetype/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.cuda_paths import add_cuda_dll_paths  # noqa: E402

MODEL = "medium"


def main():
    print("=" * 60)
    print(f"  Скачивание модели Whisper '{MODEL}' + проверка GPU")
    print("=" * 60)

    # 1. Удаляем повреждённый кэш модели (пустышки .incomplete), чтобы
    #    скачать начисто, а не спотыкаться об обрывки прошлой загрузки.
    cache = (Path.home() / ".cache" / "huggingface" / "hub"
             / f"models--Systran--faster-whisper-{MODEL}")
    if cache.exists():
        print(f"\n[1/3] Удаляю повреждённый кэш модели:\n      {cache}")
        shutil.rmtree(cache, ignore_errors=True)
    else:
        print("\n[1/3] Старого кэша модели нет — качаем с нуля.")

    # 2. Добавляем CUDA DLL пути (нужно для GPU-загрузки на Windows).
    print("[2/3] Готовлю пути к CUDA-библиотекам...")
    add_cuda_dll_paths()

    # 3. WhisperModel сам скачает модель через huggingface_hub (с прогресс-баром)
    #    и загрузит её на GPU — это заодно проверяет весь CUDA-стек.
    print("[3/3] Качаю модель и загружаю на GPU (float16)...")
    print("      Размер ~1.53 ГБ — ниже будет прогресс HuggingFace:\n")

    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cuda", compute_type="float16")

    print("\n" + "=" * 60)
    print("  ✅ ГОТОВО")
    print("=" * 60)
    print(f"  Модель '{MODEL}' скачана и загружена на GPU (CUDA, float16).")
    print("  Работает весь стек: модель + cuDNN + nvJitLink + cuBLAS.")
    print("  Можно запускать приложение: python run.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  ❌ ОШИБКА: {type(e).__name__}: {e}")
        print("=" * 60)
        print("  Если ошибка про CUDA/cuDNN — проверь, что библиотеки на месте.")
        print("  Если про сеть/таймаут — проверь интернет (VPN?) и запусти заново.")
        sys.exit(1)
