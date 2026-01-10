"""
Test script to measure timing and RAM usage for large Vosk model.
Measures: load time, unload time, reload time, and RAM at each stage.
"""

import gc
import time
import psutil
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def get_process_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def format_memory(mb):
    """Format memory value."""
    if mb >= 1024:
        return f"{mb:.1f} MB ({mb/1024:.2f} GB)"
    return f"{mb:.1f} MB"


def format_time(seconds):
    """Format time value."""
    if seconds >= 60:
        return f"{seconds:.2f} sec ({seconds/60:.1f} min)"
    return f"{seconds:.2f} sec"


def run_vosk_model_benchmark(model_path: str):
    """Run benchmark for Vosk model loading/unloading."""

    print("=" * 70)
    print("VOSK LARGE MODEL BENCHMARK")
    print("=" * 70)
    print(f"\nModel path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")

    if not os.path.exists(model_path):
        print("\nERROR: Model path does not exist!")
        return

    # Get model size on disk
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    print(f"Model size on disk: {format_memory(total_size / (1024 * 1024))}")

    print("\n" + "-" * 70)

    # Import vosk here to measure baseline before import
    print("\n[1] BASELINE (before vosk import)")
    gc.collect()
    time.sleep(0.5)
    ram_baseline = get_process_memory_mb()
    print(f"    RAM: {format_memory(ram_baseline)}")

    print("\n[2] IMPORTING VOSK")
    t_start = time.perf_counter()
    import vosk
    vosk.SetLogLevel(-1)  # Suppress vosk logs
    t_import = time.perf_counter() - t_start
    ram_after_import = get_process_memory_mb()
    print(f"    Import time: {format_time(t_import)}")
    print(f"    RAM after import: {format_memory(ram_after_import)}")
    print(f"    RAM delta: +{format_memory(ram_after_import - ram_baseline)}")

    print("\n" + "-" * 70)

    # First load
    print("\n[3] FIRST MODEL LOAD")
    gc.collect()
    time.sleep(0.5)
    ram_before_load = get_process_memory_mb()
    print(f"    RAM before load: {format_memory(ram_before_load)}")

    t_start = time.perf_counter()
    model = vosk.Model(model_path)
    t_load1 = time.perf_counter() - t_start

    ram_after_load = get_process_memory_mb()
    ram_delta_load = ram_after_load - ram_before_load

    print(f"    Load time: {format_time(t_load1)}")
    print(f"    RAM after load: {format_memory(ram_after_load)}")
    print(f"    RAM delta: +{format_memory(ram_delta_load)}")

    print("\n" + "-" * 70)

    # Unload
    print("\n[4] MODEL UNLOAD (del + gc.collect)")
    t_start = time.perf_counter()
    del model
    gc.collect()
    t_unload = time.perf_counter() - t_start

    time.sleep(1)  # Give OS time to release memory
    gc.collect()

    ram_after_unload = get_process_memory_mb()
    ram_delta_unload = ram_after_load - ram_after_unload

    print(f"    Unload time: {format_time(t_unload)}")
    print(f"    RAM after unload: {format_memory(ram_after_unload)}")
    print(f"    RAM freed: -{format_memory(ram_delta_unload)}")
    print(f"    RAM retained: {format_memory(ram_after_unload - ram_before_load)}")

    print("\n" + "-" * 70)

    # Second load (warm cache)
    print("\n[5] SECOND MODEL LOAD (warm OS cache)")
    gc.collect()
    time.sleep(0.5)
    ram_before_reload = get_process_memory_mb()
    print(f"    RAM before reload: {format_memory(ram_before_reload)}")

    t_start = time.perf_counter()
    model = vosk.Model(model_path)
    t_load2 = time.perf_counter() - t_start

    ram_after_reload = get_process_memory_mb()
    ram_delta_reload = ram_after_reload - ram_before_reload

    print(f"    Reload time: {format_time(t_load2)}")
    print(f"    RAM after reload: {format_memory(ram_after_reload)}")
    print(f"    RAM delta: +{format_memory(ram_delta_reload)}")

    # Cleanup
    del model
    gc.collect()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Model: vosk-model-ru-0.42 (large Russian)
    Model size on disk: {format_memory(total_size / (1024 * 1024))}

    TIMING:
    - First load:  {format_time(t_load1)}
    - Unload:      {format_time(t_unload)}
    - Second load: {format_time(t_load2)} (warm cache)
    - Speedup:     {t_load1/t_load2:.1f}x faster on reload

    RAM USAGE:
    - Baseline:        {format_memory(ram_baseline)}
    - After import:    {format_memory(ram_after_import)} (+{format_memory(ram_after_import - ram_baseline)})
    - Model loaded:    {format_memory(ram_after_load)} (+{format_memory(ram_delta_load)})
    - After unload:    {format_memory(ram_after_unload)} (freed {format_memory(ram_delta_unload)})
    - Model reloaded:  {format_memory(ram_after_reload)} (+{format_memory(ram_delta_reload)})

    PEAK RAM for model: ~{format_memory(ram_delta_load)}
""")
    print("=" * 70)


if __name__ == "__main__":
    model_path = "d:/Projects/VibeCoding/VoiceType/voicetype/models/vosk-model-ru-0.42"
    run_vosk_model_benchmark(model_path)
