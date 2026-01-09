#!/usr/bin/env python
"""
VoiceType - Silero Model Downloader
Скачивает модель Silero Text Enhancement для локального использования.

Использование:
    python download_silero.py

Модель будет сохранена в: models/silero-te/v2_4lang_q.pt
"""
import sys
import urllib.request
from pathlib import Path

# Корневая директория проекта
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models" / "silero-te"

# URL модели Silero TE (многоязычная, квантизированная)
# Эта модель поддерживает: ru, en, de, es
SILERO_MODEL_URL = "https://models.silero.ai/te_models/v2_4lang_q.pt"
SILERO_MODEL_FILE = "v2_4lang_q.pt"


def download_with_progress(url: str, dest_path: Path) -> bool:
    """
    Скачать файл с отображением прогресса.

    Args:
        url: URL для скачивания
        dest_path: Путь для сохранения

    Returns:
        True если успешно
    """
    print(f"Downloading: {url}")
    print(f"Destination: {dest_path}")

    try:
        # Создаем директорию
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Получаем информацию о файле
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))

            # Скачиваем с отображением прогресса
            downloaded = 0
            block_size = 8192

            with open(dest_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break

                    f.write(buffer)
                    downloaded += len(buffer)

                    # Отображаем прогресс
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r  Progress: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="")
                    else:
                        mb_downloaded = downloaded / (1024 * 1024)
                        print(f"\r  Downloaded: {mb_downloaded:.1f} MB", end="")

        print()  # Новая строка после прогресса
        return True

    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def verify_model(model_path: Path) -> bool:
    """
    Проверить что модель загружается корректно.

    Args:
        model_path: Путь к файлу модели

    Returns:
        True если модель валидна
    """
    print("Verifying model...")

    try:
        import torch

        model = torch.jit.load(str(model_path), map_location=torch.device('cpu'))
        model.eval()

        # Тестируем на простом тексте
        test_text = "привет как дела"
        result = model(test_text)

        print(f"  Test input:  '{test_text}'")
        print(f"  Test output: '{result}'")
        print("  Model verification: OK")

        return True

    except ImportError:
        print("  Warning: PyTorch not installed, skipping verification")
        return True  # Не критично

    except Exception as e:
        print(f"  Error verifying model: {e}")
        return False


def main():
    """Основная функция."""
    print("=" * 50)
    print("Silero Text Enhancement Model Downloader")
    print("=" * 50)

    model_path = MODELS_DIR / SILERO_MODEL_FILE

    # Проверяем, есть ли уже модель
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\nModel already exists: {model_path}")
        print(f"Size: {size_mb:.1f} MB")

        response = input("\nRe-download? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping download.")

            # Проверяем существующую модель
            if verify_model(model_path):
                print("\nModel is ready to use!")
            return 0

    print(f"\nDownloading Silero TE model...")
    print(f"This model supports: Russian, English, German, Spanish")

    # Скачиваем
    if not download_with_progress(SILERO_MODEL_URL, model_path):
        print("\nDownload failed!")
        return 1

    # Проверяем размер
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\nDownload complete!")
        print(f"File: {model_path}")
        print(f"Size: {size_mb:.1f} MB")
    else:
        print("\nError: File not found after download")
        return 1

    # Верифицируем
    if not verify_model(model_path):
        print("\nWarning: Model verification failed")
        print("The model may still work, but please check.")

    print("\n" + "=" * 50)
    print("Silero model is ready!")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
