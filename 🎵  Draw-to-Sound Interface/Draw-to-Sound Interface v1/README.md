# Geometric Synthesizer V1

**Version:** 1.8.2 (Continuous Instrument)  
**Date:** 2025  
**Language:** Python 3.8+

A polyphonic visual theremin that transforms hand-drawn gestures into continuous, evolving sound. Each stroke becomes a living voice that responds to your drawing movements in real-time.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Controls](#controls)
- [Sound Mapping](#sound-mapping)
- [Waveforms](#waveforms)
- [Technical Details](#technical-details)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

The Geometric Synthesizer V1 is a **continuous drawing instrument** where sound evolves as you draw. Unlike traditional step sequencers, this is a living, breathing synthesizer that responds to every movement of your hand.

### Design Philosophy

Each stroke is a **voice** in a polyphonic ensemble. The sound changes continuously based on:
- Where you draw (pitch and stereo position)
- How fast you draw (volume)
- How curved your strokes are (harmonic richness)
- The direction you move (modulation)

This creates an intimate, expressive instrument that feels like **painting with sound**.

---

## Key Features

✨ **Continuous Sound Generation**
- Real-time audio synthesis with sub-20ms latency
- Polyphonic (multiple simultaneous strokes)
- Band-limited waveforms (no aliasing)

🎨 **Visual Feedback**
- Draw with 7 different colored waveforms
- Active strokes glow while playing
- Clean, minimal interface

🎵 **Expressive Mapping**
- Y-axis → Pitch (130Hz - 1046Hz range)
- X-axis → Stereo panning
- Drawing speed → Volume
- Curvature → Harmonic content
- Direction → Amplitude modulation

🎹 **7 Waveform Types**
- Pure (sine) - Clean, fundamental tone
- Soft (triangle) - Mellow, warm sound
- Bright (square) - Hollow, reed-like
- Rich (sawtooth) - Full, string-like
- Hollow (pulse) - Breathy, airy
- Textured (noise) - Percussive, grainy
- Complex (mixed) - Chorused, detuned

---

## Installation

### Prerequisites

**System Requirements:**
- Python 3.8 or higher
- Audio output device
- Working soundcard drivers

**Python Dependencies:**

```bash
pip install pygame>=2.5.0
pip install numpy>=1.24.0
pip install sounddevice>=0.4.6
```

### Installation Steps

```bash
# 1. Clone or download the code
# Save as geometric_synth_v1.py

# 2. Install dependencies
pip install pygame numpy sounddevice

# 3. Run
python geometric_synth_v1.py
```

---

## Quick Start

1. **Launch the synthesizer**
   ```bash
   python geometric_synth_v1.py
   ```

2. **Start playing**
   - Click and drag to draw
   - Sound begins immediately as you draw
   - Each stroke is a separate voice

3. **Experiment with waveforms**
   - Press keys **1-7** to change colors/waveforms
   - Draw multiple strokes for harmony

4. **Save your work**
   - Press **S** to save the visual
   - Press **SPACE** to clear and start fresh

---

## Controls

### Drawing

| Action | Control |
|--------|---------|
| **Draw/Play** | Click and drag mouse |
| **Continue drawing** | Keep mouse button held |
| **End stroke** | Release mouse button |

### Waveforms

| Key | Waveform | Character |
|-----|----------|-----------|
| **1** | Pure (Sine) | Clean fundamental |
| **2** | Soft (Triangle) | Mellow warmth |
| **3** | Bright (Square) | Hollow brightness |
| **4** | Rich (Sawtooth) | Full spectrum |
| **5** | Hollow (Pulse) | Breathy air |
| **6** | Textured (Noise) | Grainy texture |
| **7** | Complex (Mixed) | Chorused richness |

### Commands

| Key | Action |
|-----|--------|
| **S** | Save current image (PNG) |
| **SPACE** | Clear all strokes |
| **ESC** | Quit application |

---

## Sound Mapping

### Pitch (Y-Axis)

```
Top of screen    → 1046 Hz (C6)  ─┐ High
                                   │
Middle           → 523 Hz  (C5)   ─┤ Mid
                                   │
Bottom of screen → 130 Hz  (C3)  ─┘ Low
```

**Logarithmic scaling** mimics musical perception - equal visual distances = equal musical intervals.

### Stereo Pan (X-Axis)

```
Left edge  → 100% Left speaker
Center     → Equal (50/50)
Right edge → 100% Right speaker
```

**Panning uses equal-power law** for smooth stereo imaging.

### Volume (Drawing Speed)

- **Slow drawing** → Quiet, gentle tones
- **Fast drawing** → Loud, energetic sound
- **Dynamic range:** Responds to speed in real-time

### Harmonics (Curvature)

Calculated from stroke geometry:
- **Straight lines** → Pure, simple tone
- **Curved strokes** → Additional harmonics (2nd & 3rd)
- **Sharp angles** → Maximum harmonic content

### Modulation (Direction)

- **Upward motion** → Positive modulation
- **Downward motion** → Negative modulation
- **Effect:** Subtle amplitude LFO (4.5-7.5 Hz)

---

## Waveforms

### 1. Pure (Sine Wave)

```python
waveform: 'sine'
```

**Characteristics:**
- Single fundamental frequency
- No harmonics
- Clean, pure tone
- Best for: Smooth melodies, sub-bass

**Mathematical:** `sin(2πft)`

### 2. Soft (Triangle Wave)

```python
waveform: 'triangle'
```

**Characteristics:**
- Odd harmonics only (1/n²)
- Gentle, mellow sound
- Less bright than square
- Best for: Pads, soft leads

**Band-limited:** Summed odd harmonics with 1/n² rolloff

### 3. Bright (Square Wave)

```python
waveform: 'square'
```

**Characteristics:**
- Odd harmonics (1/n)
- Hollow, clarinet-like
- Strong fundamental
- Best for: Leads, bass

**Band-limited:** Summed odd harmonics with 1/n rolloff

### 4. Rich (Sawtooth Wave)

```python
waveform: 'sawtooth'
```

**Characteristics:**
- All harmonics (1/n)
- Brightest waveform
- Full harmonic spectrum
- Best for: Strings, brass

**Band-limited:** Summed all harmonics with 1/n rolloff

### 5. Hollow (Pulse Wave)

```python
waveform: 'pulse'
```

**Characteristics:**
- 25% duty cycle
- Nasal, hollow timbre
- Unique character
- Best for: Basses, effects

**Band-limited:** Modified harmonic series

### 6. Textured (Noise)

```python
waveform: 'noise'
```

**Characteristics:**
- Smooth white noise
- Low-pass filtered
- Percussive texture
- Best for: Percussion, FX

**Processing:** One-pole filter (α = 0.06)

### 7. Complex (Mixed)

```python
waveform: 'mixed'
```

**Characteristics:**
- Detuned oscillators (1.997x, 2.003x)
- Chorused effect
- Rich, moving sound
- Best for: Pads, atmospheres

**Implementation:** 3 detuned sine waves

---

## Technical Details

### Audio Architecture

**Sample Rate:** 44.1 kHz (CD quality)  
**Buffer Size:** 512 samples (~11.6ms latency)  
**Bit Depth:** 32-bit float (internal)  
**Channels:** 2 (stereo)

### Band-Limited Synthesis

All waveforms (except sine and noise) use **additive synthesis** to prevent aliasing:

```python
# Maximum harmonic calculation (anti-aliasing)
nyquist = 0.5 * sample_rate  # 22050 Hz
max_harmonic = floor(nyquist / base_frequency)
max_harmonic = min(31, max_harmonic)  # Cap at 31
```

**Result:** Clean, artifact-free sound at all frequencies.

### Parameter Smoothing

All parameters use **exponential smoothing** to prevent clicks:

```python
frequency_smoothing = 0.02  # 2% per frame
volume_smoothing = 0.01     # 1% per frame
```

### Polyphony System

- **Unlimited voices** (software-limited)
- **Thread-safe** audio callback
- **Automatic fade-out** when stroke ends (0.5s)
- **Memory efficient** (immediate cleanup)

### Fade System

```python
fade_duration = 0.5  # seconds
fade_curve = exponential  # (1 - t)²
```

Smooth, natural-sounding note releases.

---

## Configuration

### Adjusting Audio Parameters

```python
# In __init__()
self.sample_rate = 44100  # Try 48000 for higher quality
self.buffer_size = 512    # Lower = less latency, more CPU
```

### Frequency Range

```python
# In pos_to_frequency()
min_freq = 130.0   # C3
max_freq = 1046.0  # C6
```

### Volume Levels

```python
# In update_oscillator_for_stroke()
base_volume = 0.02      # Minimum volume
max_additional = 0.06   # Speed-based boost
```

### Brush Size

```python
# In __init__()
self.brush_size = 3  # Pixel width of strokes
```

### Color Palette

```python
# Add new waveforms in __init__()
self.color_palette.append(
    ((R, G, B), 'waveform_name', 'Description')
)
```

---

## Troubleshooting

### No Sound Output

**Check:**
- System audio not muted
- Correct audio device selected
- Volume sufficient in system mixer

**Solution:**
```bash
# Test sounddevice
python -c "import sounddevice as sd; print(sd.query_devices())"

# List available devices
python -c "import sounddevice as sd; sd.default.device = 'your_device_name'"
```

### Crackling/Distortion

**Causes:**
- Buffer size too small
- CPU overload (too many voices)
- Sample rate mismatch

**Solutions:**
```python
# Increase buffer size
self.buffer_size = 1024  # Less latency-sensitive

# Reduce max voices
if len(self.active_strokes) > 8:
    # Limit polyphony
```

### High Latency

**Symptom:** Delay between drawing and sound

**Solutions:**
```python
# Decrease buffer size
self.buffer_size = 256  # ~5.8ms latency

# Use ASIO/CoreAudio instead of default
```

### Window Not Responding

**Cause:** Audio callback blocking

**Solution:**
- Restart application
- Check for exceptions in console
- Verify all dependencies installed

### Audio Pops/Clicks

**Causes:**
- Insufficient parameter smoothing
- Phase discontinuities

**Already implemented fixes:**
- Exponential parameter smoothing
- Phase continuity preservation
- Soft limiting output (-0.95 to 0.95)

---

## Performance Tips

### Optimization

1. **Limit active strokes:** Clear old strokes regularly (SPACE key)
2. **Reduce harmonic count:** Lower `max_harmonic` cap in code
3. **Increase buffer size:** Trade latency for stability
4. **Close other audio apps:** Prevent device conflicts

### Best Practices

- **Draw smoothly** for best sound quality
- **Release mouse** to end notes cleanly
- **Use SPACE** frequently to manage CPU load
- **Save work often** with S key

---

## License

This project is released under the **MIT License**:

```
MIT License

Copyright (c) 2025 Julie Le Rudulier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

**Made with sound and vision in Python**
