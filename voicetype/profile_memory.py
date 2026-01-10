"""
Memory Profiling Script for VoiceType Application
Measures RAM consumption of each component step by step.
"""

import psutil
import os
import sys
import gc

def get_mem_mb():
    """Get current process memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def format_delta(current, previous):
    """Format memory delta with sign."""
    delta = current - previous
    return f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

# Store measurements
measurements = []

def record(name, mem_before):
    """Record a measurement."""
    gc.collect()  # Force garbage collection for accurate measurement
    mem_after = get_mem_mb()
    delta = mem_after - mem_before
    measurements.append({
        'name': name,
        'memory': mem_after,
        'delta': delta
    })
    print(f"{name}: {mem_after:.1f} MB ({format_delta(mem_after, mem_before)} MB)")
    return mem_after

print("=" * 60)
print("VoiceType Memory Profiling")
print("=" * 60)
print()

# Baseline
baseline = get_mem_mb()
print(f"Baseline (Python + psutil): {baseline:.1f} MB")
measurements.append({'name': 'Baseline', 'memory': baseline, 'delta': 0})
prev_mem = baseline

print()
print("-" * 60)
print("Step 1: PyQt6 Import")
print("-" * 60)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
prev_mem = record("PyQt6.QtWidgets import", prev_mem)

# Need QApplication instance for some Qt features
app = QApplication(sys.argv)
prev_mem = record("QApplication instance", prev_mem)

print()
print("-" * 60)
print("Step 2: PyTorch Import")
print("-" * 60)

import torch
prev_mem = record("torch import", prev_mem)

print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

print()
print("-" * 60)
print("Step 3: Vosk Import")
print("-" * 60)

import vosk
vosk.SetLogLevel(-1)  # Suppress logs
prev_mem = record("vosk import", prev_mem)

print()
print("-" * 60)
print("Step 4: Other Imports (pynput, pyaudio, etc.)")
print("-" * 60)

import pynput
prev_mem = record("pynput import", prev_mem)

import pyaudio
prev_mem = record("pyaudio import", prev_mem)

import numpy
prev_mem = record("numpy import", prev_mem)

print()
print("-" * 60)
print("Step 5: Vosk Model Loading")
print("-" * 60)

# Check which models are available
models_dir = os.path.join(os.path.dirname(__file__), "models")
print(f"Models directory: {models_dir}")

small_ru_model_path = os.path.join(models_dir, "vosk-model-small-ru-0.22")
large_ru_model_path = os.path.join(models_dir, "vosk-model-ru-0.42")

if os.path.exists(small_ru_model_path):
    print(f"Loading small Russian model from: {small_ru_model_path}")
    vosk_model_small = vosk.Model(small_ru_model_path)
    prev_mem = record("Vosk small-ru model load", prev_mem)
else:
    print(f"Small Russian model not found at: {small_ru_model_path}")

# Don't load large model by default - it's huge
# if os.path.exists(large_ru_model_path):
#     print(f"Loading large Russian model from: {large_ru_model_path}")
#     vosk_model_large = vosk.Model(large_ru_model_path)
#     prev_mem = record("Vosk large-ru model load", prev_mem)

print()
print("-" * 60)
print("Step 6: Silero TE (Text Enhancement) Model Loading")
print("-" * 60)

# Load Silero TE model the same way the app does (using JIT models)
try:
    silero_te_path = os.path.join(models_dir, "silero-te")
    model_jit_path = os.path.join(silero_te_path, "te_model_jit.pt")
    tokenizer_jit_path = os.path.join(silero_te_path, "te_tokenizer_jit.pt")

    if os.path.exists(model_jit_path) and os.path.exists(tokenizer_jit_path):
        print(f"Loading Silero TE JIT model from: {model_jit_path}")

        # Load TorchScript model
        torch.set_grad_enabled(False)
        silero_model = torch.jit.load(model_jit_path, map_location='cpu')
        silero_model.eval()
        prev_mem = record("Silero TE model load", prev_mem)

        # Load TorchScript tokenizer
        print(f"Loading Silero TE tokenizer from: {tokenizer_jit_path}")
        silero_tokenizer = torch.jit.load(tokenizer_jit_path, map_location='cpu')
        prev_mem = record("Silero TE tokenizer load", prev_mem)

        # Try a test inference to see if that allocates more memory
        print("Running test inference...")
        test_text = "привет как дела"
        with torch.no_grad():
            # Tokenize
            x = torch.tensor([silero_tokenizer.convert_string_to_ids(test_text)])
            # Simple forward (without full enhance logic)
            att = torch.ones_like(x)
            lan = torch.tensor([[[3]]])  # Russian
            result = silero_model(x, att, lan)
        prev_mem = record("Silero TE first inference", prev_mem)
    else:
        # Try torch.package format
        package_model_path = os.path.join(silero_te_path, "v2_4lang_q.pt")
        if os.path.exists(package_model_path):
            print(f"Loading Silero TE package model from: {package_model_path}")
            import torch.package
            imp = torch.package.PackageImporter(package_model_path)
            silero_model = imp.load_pickle("te_model", "model")
            prev_mem = record("Silero TE model load (package)", prev_mem)

            # Test inference
            print("Running test inference...")
            result = silero_model.enhance_text("привет как дела", "ru")
            prev_mem = record("Silero TE first inference", prev_mem)
            print(f"  Result: '{result}'")
        else:
            print(f"Silero TE model not found at: {silero_te_path}")
            print(f"  Checked: {model_jit_path}")
            print(f"  Checked: {package_model_path}")

except Exception as e:
    import traceback
    print(f"Error loading Silero TE: {e}")
    print(traceback.format_exc())

print()
print("-" * 60)
print("Step 7: VoiceType Application Components")
print("-" * 60)

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_path)

try:
    from data.config import get_config
    prev_mem = record("Config module", prev_mem)
except Exception as e:
    print(f"Config import error: {e}")

try:
    from data.database import get_database
    prev_mem = record("Database module", prev_mem)
except Exception as e:
    print(f"Database import error: {e}")

try:
    from core.audio_capture import AudioCapture
    prev_mem = record("AudioCapture module", prev_mem)
except Exception as e:
    print(f"AudioCapture import error: {e}")

try:
    from core.recognizer import Recognizer
    prev_mem = record("Recognizer module", prev_mem)
except Exception as e:
    print(f"Recognizer import error: {e}")

print()
print("=" * 60)
print("MEMORY CONSUMPTION SUMMARY")
print("=" * 60)
print()

# Calculate total
total_mem = get_mem_mb()
total_delta = total_mem - baseline

print(f"{'Component':<35} {'Memory (MB)':>12} {'Delta (MB)':>12} {'% of Total':>10}")
print("-" * 70)

# Group measurements by category
categories = {
    'Python Baseline': ['Baseline'],
    'PyQt6 (UI Framework)': ['PyQt6.QtWidgets import', 'QApplication instance'],
    'PyTorch (ML Framework)': ['torch import'],
    'Vosk (Speech Recognition)': ['vosk import', 'Vosk small-ru model load'],
    'Silero TE (Punctuation)': ['Silero TE model load', 'Silero TE tokenizer load', 'Silero TE first inference', 'Silero TE model load (package)'],
    'Other Libraries': ['pynput import', 'pyaudio import', 'numpy import'],
    'VoiceType Modules': ['Config module', 'Database module', 'AudioCapture module', 'Recognizer module'],
}

category_totals = {}

for category, items in categories.items():
    cat_delta = 0
    for m in measurements:
        if m['name'] in items:
            cat_delta += m['delta']
    category_totals[category] = cat_delta
    if cat_delta > 0:
        pct = (cat_delta / total_delta) * 100 if total_delta > 0 else 0
        print(f"{category:<35} {'':<12} {cat_delta:>+12.1f} {pct:>9.1f}%")

print("-" * 70)
print(f"{'TOTAL':<35} {total_mem:>12.1f} {total_delta:>+12.1f} {'100.0%':>10}")

print()
print("=" * 60)
print("DETAILED BREAKDOWN")
print("=" * 60)
print()

print(f"{'Step':<40} {'Total (MB)':>12} {'Delta (MB)':>12}")
print("-" * 65)

for m in measurements:
    print(f"{m['name']:<40} {m['memory']:>12.1f} {m['delta']:>+12.1f}")

print("-" * 65)
print(f"{'FINAL TOTAL':<40} {total_mem:>12.1f}")

print()
print("=" * 60)
print("ANALYSIS")
print("=" * 60)
print()

# Find top consumers
sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
print("Top memory consumers:")
for i, (cat, mem) in enumerate(sorted_categories[:5], 1):
    if mem > 0:
        pct = (mem / total_delta) * 100 if total_delta > 0 else 0
        print(f"  {i}. {cat}: {mem:.1f} MB ({pct:.1f}%)")

print()
print("Recommendations based on analysis:")

# PyTorch analysis
pytorch_mem = category_totals.get('PyTorch (ML Framework)', 0)
silero_mem = category_totals.get('Silero TE (Punctuation)', 0)
if pytorch_mem + silero_mem > 100:
    print(f"  - PyTorch + Silero TE use {pytorch_mem + silero_mem:.0f} MB total")
    print(f"    Consider: Disable punctuation if not needed, or use alternative")

# Vosk analysis
vosk_mem = category_totals.get('Vosk (Speech Recognition)', 0)
if vosk_mem > 50:
    print(f"  - Vosk uses {vosk_mem:.0f} MB")
    print(f"    The small model is already minimal; large model would use more")

# PyQt6 analysis
pyqt_mem = category_totals.get('PyQt6 (UI Framework)', 0)
if pyqt_mem > 50:
    print(f"  - PyQt6 uses {pyqt_mem:.0f} MB")
    print(f"    This is baseline for Qt applications, hard to reduce")

print()
