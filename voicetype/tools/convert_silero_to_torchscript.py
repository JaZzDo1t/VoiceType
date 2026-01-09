"""
Конвертация Silero TE модели из torch.package в TorchScript формат.
TorchScript работает с PyInstaller без проблем.

Запуск: python tools/convert_silero_to_torchscript.py
"""
import sys
from pathlib import Path

# Добавляем корень проекта в path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.package import PackageImporter

def convert_model():
    """Конвертировать Silero TE из torch.package в TorchScript."""

    # Пути
    source_model = ROOT / "models" / "silero-te" / "v2_4lang_q.pt"
    output_dir = ROOT / "models" / "silero-te-jit"

    if not source_model.exists():
        print(f"Модель не найдена: {source_model}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загружаем модель из: {source_model}")

    try:
        # Загружаем через torch.package
        importer = PackageImporter(str(source_model))
        model = importer.load_pickle("te_model", "model")
        model.eval()

        print(f"Модель загружена: {type(model)}")

        # Пробуем сохранить как TorchScript
        output_path = output_dir / "silero_te.pt"

        # Модель уже является ScriptModule, просто сохраняем
        torch.jit.save(model, str(output_path))

        print(f"Модель сохранена в TorchScript: {output_path}")

        # Проверяем что можно загрузить
        test_model = torch.jit.load(str(output_path))
        print("Проверка загрузки: OK")

        # Тест работы
        test_text = "привет как дела"
        result = test_model.enhance(test_text, "ru")
        print(f"Тест: '{test_text}' -> '{result}'")

        return True

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_model()
    sys.exit(0 if success else 1)
