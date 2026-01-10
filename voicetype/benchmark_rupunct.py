#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark script for comparing PyTorch vs ONNX versions of RUPunct_medium model.

Compares:
- RAM usage (before/after model load)
- Inference speed
- Output quality (punctuation results)

Usage:
    ./venv_cpu/Scripts/python.exe benchmark_rupunct.py
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import psutil


# ==============================================================================
# Test Phrases
# ==============================================================================

TEST_PHRASES = [
    "привет как дела",
    "я иду домой сегодня вечером",
    "что ты делаешь",
    "это очень интересно",
    "москва столица россии",
    "как тебя зовут меня зовут иван",
    "завтра будет хорошая погода",
    "мне нужно купить продукты в магазине",
    "ты любишь читать книги",
    "сколько стоит эта вещь",
    "я работаю программистом уже пять лет",
    "давай встретимся в кафе в субботу",
    "он сказал что придёт позже",
    "мы поедем на море летом",
    "почему ты опоздал на работу",
    "она очень красивая девушка",
    "мне нравится слушать музыку",
    "где находится ближайшая аптека",
    "я хочу заказать пиццу с доставкой",
    "спасибо за помощь до свидания",
]


# ==============================================================================
# Memory Utilities
# ==============================================================================

def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def force_gc():
    """Force garbage collection."""
    gc.collect()
    # Try to clean CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ==============================================================================
# PyTorch RUPunct Wrapper (from punctuation.py)
# ==============================================================================

class PyTorchRUPunct:
    """
    PyTorch version using transformers pipeline.
    """

    MODEL_NAME = "RUPunct/RUPunct_medium"

    # Mapping from RUPunct label parts to punctuation marks
    # Based on config.json: TIRE=dash, VOSKL=exclamation, DVOETOCHIE=colon,
    # PERIODCOMMA=semicolon, DEFIS=hyphen, MNOGOTOCHIE=ellipsis, O=none
    PUNCT_MAP = {
        'PERIOD': '.',
        'COMMA': ',',
        'QUESTION': '?',
        'TIRE': '—',           # dash (tire in Russian)
        'VOSKL': '!',          # exclamation (vosklitsatel'nyj znak)
        'DVOETOCHIE': ':',     # colon (dvoetochie)
        'PERIODCOMMA': ';',    # semicolon (tochka s zapyatoj)
        'DEFIS': '-',          # hyphen (defis)
        'QUESTIONVOSKL': '?!', # question + exclamation
        'MNOGOTOCHIE': '...',  # ellipsis (mnogotochie)
        'O': '',               # no punctuation
    }

    def __init__(self):
        self._classifier = None
        self._tokenizer = None
        self._is_loaded = False

    def load_model(self) -> bool:
        """Load PyTorch model."""
        from transformers import pipeline, AutoTokenizer

        print(f"  Loading PyTorch model from {self.MODEL_NAME}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            strip_accents=False,
            add_prefix_space=True
        )

        self._classifier = pipeline(
            "ner",
            model=self.MODEL_NAME,
            tokenizer=self._tokenizer,
            aggregation_strategy="first"
        )

        self._is_loaded = True
        return True

    def _parse_label(self, label: str) -> tuple:
        """
        Parse RUPunct label.

        Labels have format: CASE_PUNCT
        Examples: LOWER_PERIOD, UPPER_COMMA, UPPER_TOTAL_QUESTION, LOWER_O

        Returns:
            (case, punct) where case: 'lower'/'upper'/'upper_total', punct: symbol or None
        """
        parts = label.split('_')

        if parts[0] == 'UPPER':
            if len(parts) > 1 and parts[1] == 'TOTAL':
                case = 'upper_total'
                punct_parts = parts[2:]
            else:
                case = 'upper'
                punct_parts = parts[1:]
        else:
            case = 'lower'
            punct_parts = parts[1:] if len(parts) > 1 else []

        punct = None
        if punct_parts:
            punct_key = '_'.join(punct_parts)
            punct = self.PUNCT_MAP.get(punct_key)
            # Empty string means no punctuation (like 'O')
            if punct == '':
                punct = None

        return case, punct

    def enhance(self, text: str) -> str:
        """Add punctuation to text."""
        if not text or not text.strip():
            return text

        if not self._is_loaded:
            return text

        input_text = text.strip().lower()
        predictions = self._classifier(input_text)

        if not predictions:
            result = input_text[0].upper() + input_text[1:] if input_text else input_text
            if result and result[-1] not in '.!?':
                result += '.'
            return result

        result = []
        sentence_start = True

        for pred in predictions:
            word = pred['word']
            label = pred['entity_group']
            case, punct = self._parse_label(label)

            if case == 'upper_total':
                word = word.upper()
            elif case == 'upper' or sentence_start:
                if word:
                    word = word[0].upper() + word[1:]

            result.append(word)

            if punct:
                result.append(punct)
                sentence_start = punct in '.!?'
            else:
                sentence_start = False

        enhanced = ' '.join(result)

        for p in '.,!?:;—':
            enhanced = enhanced.replace(f' {p}', p)

        while '  ' in enhanced:
            enhanced = enhanced.replace('  ', ' ')

        if enhanced and enhanced[0].islower():
            enhanced = enhanced[0].upper() + enhanced[1:]

        if enhanced and enhanced[-1] not in '.!?':
            enhanced += '.'

        return enhanced.strip()

    def unload(self):
        """Unload model."""
        self._classifier = None
        self._tokenizer = None
        self._is_loaded = False


# ==============================================================================
# ONNX RUPunct Wrapper
# ==============================================================================

class ONNXRUPunct:
    """
    ONNX version using optimum.onnxruntime.
    """

    # Mapping from RUPunct label parts to punctuation marks
    # Based on config.json: TIRE=dash, VOSKL=exclamation, DVOETOCHIE=colon,
    # PERIODCOMMA=semicolon, DEFIS=hyphen, MNOGOTOCHIE=ellipsis, O=none
    PUNCT_MAP = {
        'PERIOD': '.',
        'COMMA': ',',
        'QUESTION': '?',
        'TIRE': '—',           # dash (tire in Russian)
        'VOSKL': '!',          # exclamation (vosklitsatel'nyj znak)
        'DVOETOCHIE': ':',     # colon (dvoetochie)
        'PERIODCOMMA': ';',    # semicolon (tochka s zapyatoj)
        'DEFIS': '-',          # hyphen (defis)
        'QUESTIONVOSKL': '?!', # question + exclamation
        'MNOGOTOCHIE': '...',  # ellipsis (mnogotochie)
        'O': '',               # no punctuation
    }

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._id2label = None

    def load_model(self) -> bool:
        """Load ONNX model."""
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer, AutoConfig

        print(f"  Loading ONNX model from {self.model_path}...")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            strip_accents=False,
            add_prefix_space=True
        )

        # Load config for label mapping
        config = AutoConfig.from_pretrained(str(self.model_path))
        self._id2label = config.id2label

        # Load ONNX model
        self._model = ORTModelForTokenClassification.from_pretrained(
            str(self.model_path),
            provider="CPUExecutionProvider"
        )

        self._is_loaded = True
        return True

    def _parse_label(self, label: str) -> tuple:
        """
        Parse RUPunct label.

        Labels have format: CASE_PUNCT
        Examples: LOWER_PERIOD, UPPER_COMMA, UPPER_TOTAL_QUESTION, LOWER_O

        Returns:
            (case, punct) where case: 'lower'/'upper'/'upper_total', punct: symbol or None
        """
        parts = label.split('_')

        if parts[0] == 'UPPER':
            if len(parts) > 1 and parts[1] == 'TOTAL':
                case = 'upper_total'
                punct_parts = parts[2:]
            else:
                case = 'upper'
                punct_parts = parts[1:]
        else:
            case = 'lower'
            punct_parts = parts[1:] if len(parts) > 1 else []

        punct = None
        if punct_parts:
            punct_key = '_'.join(punct_parts)
            punct = self.PUNCT_MAP.get(punct_key)
            # Empty string means no punctuation (like 'O')
            if punct == '':
                punct = None

        return case, punct

    def _aggregate_predictions(self, tokens: List[str], predictions: List[int],
                                offsets: List[Tuple], original_text: str) -> List[Dict]:
        """
        Aggregate subword predictions back to words.
        Uses 'first' aggregation strategy (like PyTorch version).
        Handles [UNK] tokens by extracting original text from offsets.
        """
        results = []
        current_word = ""
        current_label = None
        current_start = None
        current_end = None

        # Special tokens to skip (have offset (0, 0))
        skip_tokens = {'[CLS]', '[SEP]', '[PAD]', '[MASK]'}

        for i, (token, pred, offset) in enumerate(zip(tokens, predictions, offsets)):
            # Skip special tokens with (0, 0) offset
            if offset == (0, 0) or token in skip_tokens:
                continue

            label = self._id2label.get(pred, "LOWER_O")  # Default to no punctuation

            # Handle [UNK] tokens - extract original text from offsets
            if token == '[UNK]':
                # Get original text from the input using offsets
                start_idx, end_idx = offset
                token = original_text[start_idx:end_idx]

            # Check if this is a continuation (subword starting with ##)
            is_subword = token.startswith("##")

            if is_subword:
                # Continue current word
                current_word += token[2:]  # Remove ##
                current_end = offset[1]
            else:
                # Save previous word if exists
                if current_word and current_label:
                    results.append({
                        'word': current_word,
                        'entity_group': current_label,
                        'start': current_start,
                        'end': current_end
                    })

                # Start new word
                current_word = token
                current_label = label
                current_start = offset[0]
                current_end = offset[1]

        # Don't forget the last word
        if current_word and current_label:
            results.append({
                'word': current_word,
                'entity_group': current_label,
                'start': current_start,
                'end': current_end
            })

        return results

    def enhance(self, text: str) -> str:
        """Add punctuation to text."""
        if not text or not text.strip():
            return text

        if not self._is_loaded:
            return text

        import numpy as np

        input_text = text.strip().lower()

        # Tokenize with max_length to avoid warnings
        encoding = self._tokenizer(
            input_text,
            return_tensors="np",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=512
        )

        offsets = encoding.pop("offset_mapping")[0].tolist()

        # Run inference
        outputs = self._model(**{k: v for k, v in encoding.items()})
        logits = outputs.logits[0]
        predictions = np.argmax(logits, axis=-1).tolist()

        # Get tokens
        tokens = self._tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        # Aggregate predictions (skipping special tokens, handling [UNK])
        aggregated = self._aggregate_predictions(tokens, predictions, offsets, input_text)

        if not aggregated:
            result = input_text[0].upper() + input_text[1:] if input_text else input_text
            if result and result[-1] not in '.!?':
                result += '.'
            return result

        result = []
        sentence_start = True

        for pred in aggregated:
            word = pred['word']
            label = pred['entity_group']
            case, punct = self._parse_label(label)

            if case == 'upper_total':
                word = word.upper()
            elif case == 'upper' or sentence_start:
                if word:
                    word = word[0].upper() + word[1:]

            result.append(word)

            if punct:
                result.append(punct)
                sentence_start = punct in '.!?'
            else:
                sentence_start = False

        enhanced = ' '.join(result)

        # Clean up punctuation spacing
        for p in '.,!?:;—-':
            enhanced = enhanced.replace(f' {p}', p)

        while '  ' in enhanced:
            enhanced = enhanced.replace('  ', ' ')

        if enhanced and enhanced[0].islower():
            enhanced = enhanced[0].upper() + enhanced[1:]

        if enhanced and enhanced[-1] not in '.!?':
            enhanced += '.'

        return enhanced.strip()

    def unload(self):
        """Unload model."""
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._id2label = None


# ==============================================================================
# Benchmark Functions
# ==============================================================================

def benchmark_model(model, model_name: str, phrases: List[str]) -> Dict:
    """
    Run benchmark for a single model.

    Returns:
        Dict with results: ram_before, ram_after, ram_used, inference_time, outputs
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    # Force GC before measuring
    force_gc()
    time.sleep(0.5)

    # Measure RAM before load
    ram_before = get_process_memory_mb()
    print(f"  RAM before load: {ram_before:.1f} MB")

    # Load model
    load_start = time.time()
    model.load_model()
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.2f}s")

    # Force GC and measure RAM after load
    force_gc()
    time.sleep(0.5)
    ram_after = get_process_memory_mb()
    ram_used = ram_after - ram_before
    print(f"  RAM after load: {ram_after:.1f} MB")
    print(f"  RAM used by model: {ram_used:.1f} MB")

    # Warm-up run
    print(f"  Warm-up run...")
    _ = model.enhance(phrases[0])

    # Run inference on all phrases
    print(f"  Running inference on {len(phrases)} phrases...")
    outputs = []
    inference_times = []

    for phrase in phrases:
        start = time.time()
        result = model.enhance(phrase)
        elapsed = time.time() - start
        outputs.append(result)
        inference_times.append(elapsed)

    total_inference = sum(inference_times)
    avg_inference = total_inference / len(phrases)

    print(f"  Total inference time: {total_inference:.3f}s")
    print(f"  Average per phrase: {avg_inference*1000:.1f}ms")

    return {
        'model_name': model_name,
        'ram_before': ram_before,
        'ram_after': ram_after,
        'ram_used': ram_used,
        'load_time': load_time,
        'total_inference': total_inference,
        'avg_inference': avg_inference,
        'inference_times': inference_times,
        'outputs': outputs,
    }


def compare_outputs(pytorch_outputs: List[str], onnx_outputs: List[str], phrases: List[str]) -> Tuple[int, int, List[int]]:
    """
    Compare outputs between PyTorch and ONNX.

    Returns:
        (matches, total, diff_indices)
    """
    matches = 0
    diff_indices = []

    for i, (pt_out, onnx_out) in enumerate(zip(pytorch_outputs, onnx_outputs)):
        if pt_out == onnx_out:
            matches += 1
        else:
            diff_indices.append(i)

    return matches, len(phrases), diff_indices


def print_results_table(pytorch_results: Dict, onnx_results: Dict):
    """Print comparison table."""
    print("\n")
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # RAM comparison
    print("\n--- Memory Usage ---")
    print(f"{'Metric':<30} {'PyTorch':>20} {'ONNX':>20}")
    print("-" * 70)
    print(f"{'RAM before load (MB)':<30} {pytorch_results['ram_before']:>20.1f} {onnx_results['ram_before']:>20.1f}")
    print(f"{'RAM after load (MB)':<30} {pytorch_results['ram_after']:>20.1f} {onnx_results['ram_after']:>20.1f}")
    print(f"{'RAM used by model (MB)':<30} {pytorch_results['ram_used']:>20.1f} {onnx_results['ram_used']:>20.1f}")

    ram_diff = pytorch_results['ram_used'] - onnx_results['ram_used']
    ram_pct = (ram_diff / pytorch_results['ram_used'] * 100) if pytorch_results['ram_used'] > 0 else 0
    print(f"\n  ONNX saves {ram_diff:.1f} MB ({ram_pct:.1f}% less RAM)")

    # Speed comparison
    print("\n--- Speed ---")
    print(f"{'Metric':<30} {'PyTorch':>20} {'ONNX':>20}")
    print("-" * 70)
    print(f"{'Model load time (s)':<30} {pytorch_results['load_time']:>20.2f} {onnx_results['load_time']:>20.2f}")
    print(f"{'Total inference (s)':<30} {pytorch_results['total_inference']:>20.3f} {onnx_results['total_inference']:>20.3f}")
    print(f"{'Average per phrase (ms)':<30} {pytorch_results['avg_inference']*1000:>20.1f} {onnx_results['avg_inference']*1000:>20.1f}")

    speed_ratio = pytorch_results['total_inference'] / onnx_results['total_inference'] if onnx_results['total_inference'] > 0 else 0
    if speed_ratio > 1:
        print(f"\n  ONNX is {speed_ratio:.2f}x faster")
    else:
        print(f"\n  PyTorch is {1/speed_ratio:.2f}x faster")


def print_output_comparison(pytorch_results: Dict, onnx_results: Dict, phrases: List[str]):
    """Print output comparison."""
    matches, total, diff_indices = compare_outputs(
        pytorch_results['outputs'],
        onnx_results['outputs'],
        phrases
    )

    print("\n--- Output Quality ---")
    print(f"Matching outputs: {matches}/{total} ({matches/total*100:.1f}%)")

    if diff_indices:
        print(f"\nDifferences found ({len(diff_indices)}):")
        print("-" * 80)
        for idx in diff_indices[:10]:  # Show first 10 differences
            print(f"\n[{idx+1}] Input:   {phrases[idx]}")
            print(f"    PyTorch: {pytorch_results['outputs'][idx]}")
            print(f"    ONNX:    {onnx_results['outputs'][idx]}")
        if len(diff_indices) > 10:
            print(f"\n  ... and {len(diff_indices) - 10} more differences")

    # Show all outputs
    print("\n--- All Outputs ---")
    print(f"{'#':<3} {'Input':<40} {'PyTorch':<45} {'ONNX':<45} {'Match':<5}")
    print("-" * 140)
    for i, phrase in enumerate(phrases):
        pt_out = pytorch_results['outputs'][i]
        onnx_out = onnx_results['outputs'][i]
        match = "OK" if pt_out == onnx_out else "DIFF"
        # Truncate for display
        phrase_disp = phrase[:38] + ".." if len(phrase) > 40 else phrase
        pt_disp = pt_out[:43] + ".." if len(pt_out) > 45 else pt_out
        onnx_disp = onnx_out[:43] + ".." if len(onnx_out) > 45 else onnx_out
        print(f"{i+1:<3} {phrase_disp:<40} {pt_disp:<45} {onnx_disp:<45} {match:<5}")


# ==============================================================================
# Main
# ==============================================================================

def run_isolated_benchmark(model_type: str, onnx_path: str) -> Dict:
    """
    Run benchmark for a single model type.
    Used for subprocess isolation.
    """
    import json

    force_gc()
    time.sleep(0.5)
    ram_start = get_process_memory_mb()

    if model_type == "pytorch":
        model = PyTorchRUPunct()
        model_name = "PyTorch (transformers)"
    else:
        model = ONNXRUPunct(onnx_path)
        model_name = "ONNX (optimum)"

    # Load model
    load_start = time.time()
    model.load_model()
    load_time = time.time() - load_start

    force_gc()
    time.sleep(0.5)
    ram_after = get_process_memory_mb()
    ram_used = ram_after - ram_start

    # Warm-up
    _ = model.enhance(TEST_PHRASES[0])

    # Inference
    outputs = []
    inference_times = []

    for phrase in TEST_PHRASES:
        start = time.time()
        result = model.enhance(phrase)
        elapsed = time.time() - start
        outputs.append(result)
        inference_times.append(elapsed)

    total_inference = sum(inference_times)
    avg_inference = total_inference / len(TEST_PHRASES)

    model.unload()

    return {
        'model_name': model_name,
        'model_type': model_type,
        'ram_start': ram_start,
        'ram_after': ram_after,
        'ram_used': ram_used,
        'load_time': load_time,
        'total_inference': total_inference,
        'avg_inference': avg_inference,
        'inference_times': inference_times,
        'outputs': outputs,
    }


def main():
    print("=" * 80)
    print("RUPunct_medium Benchmark: PyTorch vs ONNX")
    print("=" * 80)
    print(f"Test phrases: {len(TEST_PHRASES)}")
    print(f"Current RAM: {get_process_memory_mb():.1f} MB")

    # ONNX model path
    script_dir = Path(__file__).parent
    onnx_model_path = script_dir / "models" / "rupunct-onnx"

    if not onnx_model_path.exists():
        print(f"\nERROR: ONNX model not found at {onnx_model_path}")
        print("Please convert the model first.")
        sys.exit(1)

    print(f"ONNX model path: {onnx_model_path}")

    # Check if running as subprocess
    if len(sys.argv) > 1 and sys.argv[1] in ('pytorch', 'onnx'):
        import json
        model_type = sys.argv[1]
        result = run_isolated_benchmark(model_type, str(onnx_model_path))
        print(json.dumps(result, ensure_ascii=False))
        return

    # =========================================================================
    # Run both tests in current process (simple mode)
    # Note: RAM measurements may be affected by library caching
    # =========================================================================
    print("\n--- Running benchmarks in single process ---")
    print("  Note: RAM measurements may be affected by shared library caching.")
    print("  For accurate RAM comparison, run each model in separate process.\n")

    # Pre-load libraries
    print("  Pre-loading shared libraries...")
    import transformers
    from optimum.onnxruntime import ORTModelForTokenClassification
    import numpy as np
    force_gc()
    time.sleep(0.5)
    base_ram = get_process_memory_mb()
    print(f"  Base RAM after library imports: {base_ram:.1f} MB")

    # =========================================================================
    # Test ONNX model FIRST
    # =========================================================================
    onnx_model = ONNXRUPunct(str(onnx_model_path))
    onnx_results = benchmark_model(onnx_model, "ONNX (optimum)", TEST_PHRASES)

    # Unload and clean up
    print("\n  Unloading ONNX model...")
    onnx_model.unload()
    del onnx_model
    force_gc()
    time.sleep(1)

    print(f"  RAM after unload: {get_process_memory_mb():.1f} MB")

    # =========================================================================
    # Test PyTorch model SECOND
    # =========================================================================
    pytorch_model = PyTorchRUPunct()
    pytorch_results = benchmark_model(pytorch_model, "PyTorch (transformers)", TEST_PHRASES)

    # Unload and clean up
    print("\n  Unloading PyTorch model...")
    pytorch_model.unload()
    del pytorch_model
    force_gc()
    time.sleep(1)

    print(f"  RAM after unload: {get_process_memory_mb():.1f} MB")

    # =========================================================================
    # Print comparison
    # =========================================================================
    print_results_table(pytorch_results, onnx_results)
    print_output_comparison(pytorch_results, onnx_results, TEST_PHRASES)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    ram_savings = pytorch_results['ram_used'] - onnx_results['ram_used']
    speed_ratio = pytorch_results['total_inference'] / onnx_results['total_inference'] if onnx_results['total_inference'] > 0 else 0
    matches, total, _ = compare_outputs(pytorch_results['outputs'], onnx_results['outputs'], TEST_PHRASES)

    print(f"  RAM savings:     {ram_savings:+.1f} MB (ONNX uses {'less' if ram_savings > 0 else 'more'})")
    print(f"  Speed:           ONNX is {speed_ratio:.2f}x {'faster' if speed_ratio > 1 else 'slower'}")
    print(f"  Output match:    {matches}/{total} ({matches/total*100:.1f}%)")

    if matches == total:
        print("\n  All outputs match - ONNX is a perfect replacement!")
    else:
        print(f"\n  {total - matches} outputs differ - review differences above")


if __name__ == "__main__":
    main()
