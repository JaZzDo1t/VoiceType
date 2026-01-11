"""
Скрипт для конвертации RUPunct_big в ONNX формат.

Использование:
    python scripts/convert_rupunct_big_to_onnx.py

Результат будет в models/rupunct-big-onnx/
"""
import os
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def convert_rupunct_big_to_onnx():
    """Скачать RUPunct_big и конвертировать в ONNX."""

    print("=" * 60)
    print("Конвертация RUPunct_big в ONNX")
    print("=" * 60)

    # Output directory
    output_dir = project_root / "models" / "rupunct-big-onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nВыходная директория: {output_dir}")

    # Step 1: Download model
    print("\n[1/4] Скачивание RUPunct_big...")

    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        import torch
    except ImportError as e:
        print(f"ОШИБКА: Не установлены зависимости: {e}")
        print("Установите: pip install transformers torch")
        return False

    model_name = "RUPunct/RUPunct_big"

    print(f"Загрузка модели: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Модель загружена! Параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Step 2: Save tokenizer and config
    print("\n[2/4] Сохранение токенизатора и конфига...")

    tokenizer.save_pretrained(str(output_dir))

    # Copy config.json with id2label
    config_path = output_dir / "config.json"
    model.config.to_json_file(str(config_path))
    print(f"Конфиг сохранён: {config_path}")

    # Step 3: Export to ONNX
    print("\n[3/4] Конвертация в ONNX...")

    try:
        import onnx
        from torch.onnx import export as torch_onnx_export
    except ImportError:
        print("ОШИБКА: pip install onnx")
        return False

    # Prepare dummy input
    model.eval()

    # Use tokenizer to create sample input
    sample_text = "привет как дела"
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))

    # ONNX export path
    onnx_path = output_dir / "model.onnx"

    print(f"Экспорт в: {onnx_path}")

    # Dynamic axes for variable length sequences
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "token_type_ids": {0: "batch_size", 1: "sequence"},
        "output": {0: "batch_size", 1: "sequence"},
    }

    with torch.no_grad():
        torch_onnx_export(
            model,
            (input_ids, attention_mask, token_type_ids),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )

    print(f"ONNX модель сохранена!")

    # Step 4: Verify ONNX model
    print("\n[4/4] Проверка ONNX модели...")

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX модель валидна!")

    # Check file sizes
    print("\n" + "=" * 60)
    print("Результат:")
    print("=" * 60)

    total_size = 0
    for f in output_dir.iterdir():
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name}: {size / 1024 / 1024:.2f} MB")

    print(f"\nОбщий размер: {total_size / 1024 / 1024:.2f} MB")
    print(f"\nМодель сохранена в: {output_dir}")

    # Test inference
    print("\n" + "=" * 60)
    print("Тест inference с ONNX Runtime:")
    print("=" * 60)

    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        test_text = "привет как дела что делаешь сегодня вечером"
        test_inputs = tokenizer(test_text, return_tensors="np", padding=True)

        outputs = session.run(
            None,
            {
                "input_ids": test_inputs["input_ids"].astype(np.int64),
                "attention_mask": test_inputs["attention_mask"].astype(np.int64),
                "token_type_ids": test_inputs.get("token_type_ids", np.zeros_like(test_inputs["input_ids"])).astype(np.int64),
            }
        )

        print(f"Вход: '{test_text}'")
        print(f"Выход shape: {outputs[0].shape}")
        print("ONNX Runtime inference работает!")

    except Exception as e:
        print(f"Ошибка теста: {e}")

    return True


if __name__ == "__main__":
    success = convert_rupunct_big_to_onnx()
    sys.exit(0 if success else 1)
