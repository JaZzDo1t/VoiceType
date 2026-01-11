"""
Benchmark script for Whisper model and Silero VAD loading performance.

Measures:
1. Cold load times (first load)
2. Unload times and memory recovery
3. Hot load times (models cached by OS)
"""

import sys
import os
import gc
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import psutil


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def format_memory(mb: float) -> str:
    """Format memory in MB."""
    return f"{mb:.1f} MB"


def run_benchmark():
    """Run the complete benchmark suite."""
    print("=" * 70)
    print(" Whisper + Silero VAD Loading Performance Benchmark")
    print("=" * 70)
    print()

    # Initial memory
    gc.collect()
    initial_memory = get_memory_mb()
    print(f"Initial process memory: {format_memory(initial_memory)}")
    print()

    # Import the recognizer (measure import time separately)
    print("Importing WhisperRecognizer module...")
    t_import_start = time.perf_counter()
    from src.core.whisper_recognizer import WhisperRecognizer
    t_import = time.perf_counter() - t_import_start
    memory_after_import = get_memory_mb()
    print(f"  Import time: {format_time(t_import)}")
    print(f"  Memory after import: {format_memory(memory_after_import)}")
    print()

    # Results storage
    results = {
        'cold': {},
        'hot': {},
        'unload': {}
    }

    # =========================================================================
    # COLD LOAD (first time, models may need to download)
    # =========================================================================
    print("-" * 70)
    print(" COLD LOAD (first load, downloading if needed)")
    print("-" * 70)

    gc.collect()
    memory_before_cold = get_memory_mb()
    print(f"Memory before cold load: {format_memory(memory_before_cold)}")

    # Create recognizer
    recognizer = WhisperRecognizer(
        model_size="small",
        device="cpu",
        language="ru",
        unload_timeout_sec=0  # Disable auto-unload for benchmark
    )

    # Measure Whisper load time separately
    print("\nLoading Whisper model (small)...")
    t_whisper_start = time.perf_counter()

    # Direct load of Whisper model
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

    t_whisper_cold = time.perf_counter() - t_whisper_start
    gc.collect()
    memory_after_whisper = get_memory_mb()
    print(f"  Whisper load time: {format_time(t_whisper_cold)}")
    print(f"  Memory after Whisper: {format_memory(memory_after_whisper)}")
    print(f"  Whisper memory delta: {format_memory(memory_after_whisper - memory_before_cold)}")

    # Measure VAD load time separately
    print("\nLoading Silero VAD ONNX...")
    t_vad_start = time.perf_counter()

    import onnxruntime as ort
    import urllib.request
    from pathlib import Path
    import numpy as np

    SILERO_VAD_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    vad_cache_dir = Path.home() / ".cache" / "silero-vad"
    vad_cache_dir.mkdir(parents=True, exist_ok=True)
    vad_onnx_path = vad_cache_dir / "silero_vad.onnx"

    # Download if needed
    if not vad_onnx_path.exists():
        print("  Downloading VAD model...")
        urllib.request.urlretrieve(SILERO_VAD_ONNX_URL, str(vad_onnx_path))

    # Load ONNX session
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 1
    vad_session = ort.InferenceSession(
        str(vad_onnx_path),
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    vad_state = np.zeros((2, 1, 128), dtype=np.float32)

    t_vad_cold = time.perf_counter() - t_vad_start
    gc.collect()
    memory_after_vad = get_memory_mb()
    print(f"  VAD load time: {format_time(t_vad_cold)}")
    print(f"  Memory after VAD: {format_memory(memory_after_vad)}")
    print(f"  VAD memory delta: {format_memory(memory_after_vad - memory_after_whisper)}")

    t_total_cold = t_whisper_cold + t_vad_cold
    memory_delta_cold = memory_after_vad - memory_before_cold

    results['cold'] = {
        'whisper_time': t_whisper_cold,
        'vad_time': t_vad_cold,
        'total_time': t_total_cold,
        'memory_delta': memory_delta_cold,
        'memory_after': memory_after_vad
    }

    print(f"\nCOLD LOAD TOTAL: {format_time(t_total_cold)}")
    print(f"Total memory increase: {format_memory(memory_delta_cold)}")

    # =========================================================================
    # UNLOAD
    # =========================================================================
    print()
    print("-" * 70)
    print(" UNLOAD (releasing models)")
    print("-" * 70)

    memory_before_unload = get_memory_mb()
    print(f"Memory before unload: {format_memory(memory_before_unload)}")

    t_unload_start = time.perf_counter()

    # Delete models
    del whisper_model
    del vad_session
    del vad_state
    del recognizer

    # Force garbage collection
    gc.collect()
    gc.collect()
    gc.collect()

    t_unload = time.perf_counter() - t_unload_start

    # Wait a bit for memory to settle
    time.sleep(0.5)
    gc.collect()

    memory_after_unload = get_memory_mb()

    results['unload'] = {
        'time': t_unload,
        'memory_before': memory_before_unload,
        'memory_after': memory_after_unload,
        'memory_freed': memory_before_unload - memory_after_unload
    }

    print(f"  Unload time: {format_time(t_unload)}")
    print(f"  Memory after unload: {format_memory(memory_after_unload)}")
    print(f"  Memory freed: {format_memory(memory_before_unload - memory_after_unload)}")

    # =========================================================================
    # HOT LOAD (models cached by OS)
    # =========================================================================
    print()
    print("-" * 70)
    print(" HOT LOAD (models cached in OS memory)")
    print("-" * 70)

    gc.collect()
    memory_before_hot = get_memory_mb()
    print(f"Memory before hot load: {format_memory(memory_before_hot)}")

    # Measure Whisper hot load
    print("\nLoading Whisper model (from cache)...")
    t_whisper_hot_start = time.perf_counter()

    whisper_model_hot = WhisperModel("small", device="cpu", compute_type="int8")

    t_whisper_hot = time.perf_counter() - t_whisper_hot_start
    gc.collect()
    memory_after_whisper_hot = get_memory_mb()
    print(f"  Whisper load time: {format_time(t_whisper_hot)}")
    print(f"  Memory after Whisper: {format_memory(memory_after_whisper_hot)}")

    # Measure VAD hot load
    print("\nLoading Silero VAD ONNX (from cache)...")
    t_vad_hot_start = time.perf_counter()

    sess_options_hot = ort.SessionOptions()
    sess_options_hot.inter_op_num_threads = 1
    sess_options_hot.intra_op_num_threads = 1
    vad_session_hot = ort.InferenceSession(
        str(vad_onnx_path),
        sess_options=sess_options_hot,
        providers=['CPUExecutionProvider']
    )
    vad_state_hot = np.zeros((2, 1, 128), dtype=np.float32)

    t_vad_hot = time.perf_counter() - t_vad_hot_start
    gc.collect()
    memory_after_vad_hot = get_memory_mb()
    print(f"  VAD load time: {format_time(t_vad_hot)}")
    print(f"  Memory after VAD: {format_memory(memory_after_vad_hot)}")

    t_total_hot = t_whisper_hot + t_vad_hot
    memory_delta_hot = memory_after_vad_hot - memory_before_hot

    results['hot'] = {
        'whisper_time': t_whisper_hot,
        'vad_time': t_vad_hot,
        'total_time': t_total_hot,
        'memory_delta': memory_delta_hot,
        'memory_after': memory_after_vad_hot
    }

    print(f"\nHOT LOAD TOTAL: {format_time(t_total_hot)}")
    print(f"Total memory increase: {format_memory(memory_delta_hot)}")

    # Cleanup
    del whisper_model_hot
    del vad_session_hot
    del vad_state_hot
    gc.collect()

    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print()
    print("=" * 70)
    print(" RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Table header
    print(f"{'Metric':<35} {'Cold Load':<15} {'Hot Load':<15} {'Speedup':<10}")
    print("-" * 75)

    # Whisper load time
    cold_whisper = results['cold']['whisper_time']
    hot_whisper = results['hot']['whisper_time']
    speedup_whisper = cold_whisper / hot_whisper if hot_whisper > 0 else 0
    print(f"{'Whisper (small) load time':<35} {format_time(cold_whisper):<15} {format_time(hot_whisper):<15} {speedup_whisper:.1f}x")

    # VAD load time
    cold_vad = results['cold']['vad_time']
    hot_vad = results['hot']['vad_time']
    speedup_vad = cold_vad / hot_vad if hot_vad > 0 else 0
    print(f"{'Silero VAD ONNX load time':<35} {format_time(cold_vad):<15} {format_time(hot_vad):<15} {speedup_vad:.1f}x")

    # Total load time
    cold_total = results['cold']['total_time']
    hot_total = results['hot']['total_time']
    speedup_total = cold_total / hot_total if hot_total > 0 else 0
    print(f"{'TOTAL load time':<35} {format_time(cold_total):<15} {format_time(hot_total):<15} {speedup_total:.1f}x")

    print("-" * 75)

    # Memory
    print(f"{'RAM after load':<35} {format_memory(results['cold']['memory_after']):<15} {format_memory(results['hot']['memory_after']):<15}")
    print(f"{'RAM delta (load)':<35} {format_memory(results['cold']['memory_delta']):<15} {format_memory(results['hot']['memory_delta']):<15}")

    print()
    print("-" * 75)
    print(f"{'Unload time':<35} {format_time(results['unload']['time'])}")
    print(f"{'RAM freed after unload':<35} {format_memory(results['unload']['memory_freed'])}")
    print("-" * 75)

    print()
    print("Benchmark complete!")

    return results


if __name__ == "__main__":
    run_benchmark()
