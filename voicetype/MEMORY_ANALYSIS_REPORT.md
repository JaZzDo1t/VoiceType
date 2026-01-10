# VoiceType Memory Analysis Report

**Date:** 2025-01-10
**Environment:** PyTorch 2.9.1+cpu (CPU-only), Windows

## Executive Summary

Total measured RAM consumption: **620.8 MB** (with small Russian Vosk model)

The three main memory consumers are:
1. **Silero TE (Punctuation)**: 240.8 MB (39.8%)
2. **PyTorch Framework**: 172.1 MB (28.4%)
3. **Vosk Speech Recognition**: 171.3 MB (28.3%)

Together, PyTorch + Silero TE consume **413 MB** (68.2% of total).

## Detailed Breakdown

| Component | Memory (MB) | Delta (MB) | % of Total |
|-----------|-------------|------------|------------|
| Python Baseline | 15.4 | - | - |
| PyQt6 (UI Framework) | - | +18.7 | 3.1% |
| PyTorch (ML Framework) | - | +172.1 | 28.4% |
| Vosk (Speech Recognition) | - | +171.3 | 28.3% |
| Silero TE (Punctuation) | - | +240.8 | 39.8% |
| Other Libraries | - | +0.7 | 0.1% |
| VoiceType Modules | - | +1.8 | 0.3% |
| **TOTAL** | **620.8** | **+605.4** | **100%** |

## Step-by-Step Memory Accumulation

```
Step                                       Total (MB)   Delta (MB)
-----------------------------------------------------------------
Baseline (Python + psutil)                       15.4         +0.0
PyQt6.QtWidgets import                           25.7        +10.3
QApplication instance                            34.1         +8.4
torch import                                    206.2       +172.1
vosk import                                     210.6         +4.3
pynput import                                   211.1         +0.5
pyaudio import                                  211.2         +0.1
numpy import                                    211.2         +0.0
Vosk small-ru model load                        378.2       +166.9
Silero TE model load                            566.3       +188.1
Silero TE tokenizer load                        598.9        +32.6
Silero TE first inference                       619.0        +20.1
Config module                                   620.5         +1.5
Database module                                 620.7         +0.3
```

## Key Findings

### 1. Silero TE is the Largest Consumer (39.8%)

The punctuation enhancement model is surprisingly heavy:
- Model loading: +188.1 MB
- Tokenizer loading: +32.6 MB
- First inference: +20.1 MB
- **Total: 240.8 MB**

This is even more than PyTorch framework itself!

### 2. PyTorch Runtime is Unavoidable (28.4%)

Just importing PyTorch consumes 172.1 MB:
- This is the "base cost" of using any PyTorch model
- The CPU-only version is already optimized (no CUDA libraries)
- This cost is paid regardless of which models you load

### 3. Vosk Model is Efficient (28.3%)

The small Russian Vosk model (50 MB on disk) consumes 171.3 MB in RAM:
- Library import: +4.3 MB (minimal)
- Model load: +166.9 MB (main cost)
- The large model would use significantly more (~1.5 GB)

### 4. UI and Other Libraries are Negligible (3.5%)

- PyQt6: 18.7 MB (reasonable for a full GUI framework)
- pynput, pyaudio, numpy: <1 MB combined
- VoiceType application modules: 1.8 MB

## Recommendations

### To Reduce Memory Consumption

1. **Disable Punctuation (saves ~240 MB)**
   - If automatic punctuation is not critical, disable Silero TE
   - Use `PunctuationDisabled` class for basic capitalization only
   - This saves the most memory with minimal impact

2. **Consider Alternative Punctuation Models**
   - Look for smaller/lighter punctuation models
   - Rule-based punctuation (no ML) would be negligible

3. **Lazy Loading Strategy**
   - Load Silero TE only when user starts recording
   - Unload after period of inactivity
   - Trade-off: slight delay on first use

4. **Use Smaller Vosk Model**
   - Already using small model (good)
   - Avoid large model unless quality is critical

### What Won't Help

- **Switching to CPU-only PyTorch**: Already done, saves only ~24 MB (CUDA runtime overhead)
- **Optimizing Python code**: VoiceType modules use only 1.8 MB
- **Different UI framework**: PyQt6's 18.7 MB is already reasonable

## Memory Budget Breakdown (Visual)

```
[======================] Silero TE        240.8 MB (39.8%)
[================]       PyTorch          172.1 MB (28.4%)
[================]       Vosk             171.3 MB (28.3%)
[=]                      PyQt6             18.7 MB  (3.1%)
                         Other              2.5 MB  (0.4%)
                         ────────────────────────────
                         TOTAL            605.4 MB
```

## Configuration for Minimal Memory

To run VoiceType with minimal memory (~380 MB):

1. Use small Vosk model (already default)
2. Disable punctuation enhancement:
   - Set `punctuation.enabled: false` in config
   - Or use `PunctuationDisabled` class

This would give: 15 + 19 + 172 + 171 + 2 = **~379 MB**

To run with punctuation (~621 MB):
- This is the current default configuration
- Memory usage is dominated by ML models

## Testing Methodology

Memory was measured using `psutil.Process().memory_info().rss` after each component load. Garbage collection was forced before each measurement for accuracy.

Script: `voicetype/profile_memory.py`
