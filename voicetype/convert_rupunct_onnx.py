#!/usr/bin/env python3
"""
Script to convert RUPunct/RUPunct_medium model from PyTorch to ONNX format.

This script uses Hugging Face's optimum library for proper transformers model conversion.
The model is an ELECTRA-based token classification model for Russian punctuation restoration.

Usage:
    python convert_rupunct_onnx.py

Output:
    models/rupunct-onnx/
    ├── model.onnx
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.txt
"""

import os
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("RUPunct/RUPunct_medium PyTorch to ONNX Conversion")
    print("=" * 60)

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / "models" / "rupunct-onnx"

    print(f"\nOutput directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Import required libraries
    print("\nImporting libraries...")
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer
        print("  - optimum.onnxruntime: OK")
        print("  - transformers: OK")
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Please install: pip install optimum[onnxruntime] transformers")
        sys.exit(1)

    model_name = "RUPunct/RUPunct_medium"

    # Step 1: Load tokenizer and save it
    print(f"\n[1/3] Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    print(f"  Tokenizer saved to {output_dir}")

    # Step 2: Export model to ONNX
    print(f"\n[2/3] Exporting model to ONNX format...")
    print("  This may take a few minutes on first run (downloading model)...")

    # Use optimum to export the model to ONNX
    # This handles all the complexity of proper ONNX export for transformers
    model = ORTModelForTokenClassification.from_pretrained(
        model_name,
        export=True,  # This triggers the ONNX export
    )

    # Save the ONNX model
    model.save_pretrained(output_dir)
    print(f"  Model exported and saved to {output_dir}")

    # Step 3: Verify the output
    print(f"\n[3/3] Verifying output files...")
    expected_files = ["model.onnx", "config.json", "tokenizer_config.json"]

    all_files = list(output_dir.iterdir())
    print(f"  Files in output directory:")
    for f in sorted(all_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    - {f.name}: {size_mb:.2f} MB")

    # Check for ONNX model file
    onnx_files = list(output_dir.glob("*.onnx"))
    if not onnx_files:
        print("\nError: No ONNX file found in output directory!")
        sys.exit(1)

    onnx_file = onnx_files[0]
    onnx_size = onnx_file.stat().st_size / (1024 * 1024)
    print(f"\n  ONNX model: {onnx_file.name} ({onnx_size:.2f} MB)")

    # Step 4: Quick validation - load the ONNX model
    print("\n[4/4] Validating ONNX model can be loaded...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        print("  ONNX Runtime session created successfully!")

        # Show model inputs/outputs
        print("\n  Model inputs:")
        for inp in session.get_inputs():
            print(f"    - {inp.name}: {inp.shape} ({inp.type})")

        print("\n  Model outputs:")
        for out in session.get_outputs():
            print(f"    - {out.name}: {out.shape} ({out.type})")

    except Exception as e:
        print(f"  Warning: Could not validate with ONNX Runtime: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)
    print(f"\nONNX model saved to: {output_dir}")
    print(f"Total size: {sum(f.stat().st_size for f in output_dir.iterdir()) / (1024*1024):.2f} MB")
    print("\nYou can now use the ONNX model with:")
    print('  from optimum.onnxruntime import ORTModelForTokenClassification')
    print(f'  model = ORTModelForTokenClassification.from_pretrained("{output_dir}")')

    return 0


if __name__ == "__main__":
    sys.exit(main())
