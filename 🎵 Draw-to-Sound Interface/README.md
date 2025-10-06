# 🎵 Draw-to-Sound Interface

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

An interactive musical interface that transforms hand-drawn gestures into expressive music using real instrument sounds. Draw vertically to control pitch, horizontally to control stereo positioning, and create melodic phrases through continuous gestural motion.

![Demo Screenshot](screenshots/interface.png)

## 🎶 Features

### Core Functionality
- **Continuous gestural control**: Y-axis controls pitch in real-time, X-axis controls stereo panning
- **Live audio feedback**: Hear notes continuously as you draw, with pitch following your gesture
- **Multi-note detection**: Long traces with pitch variation automatically split into melodic phrases
- **108 instruments**: Organized in 12 musical presets (Classical, Jazz, Rock, Electro, Latin, Country, Soul, World, Drum Kit, Percussions, Latin Drums, Miscellaneous)
- **Real instrument sounds**: Professional SoundFont-based synthesis via FluidSynth

### Musical Intelligence
- **Scale quantization**: Snap notes to 6 key signatures (C, G, D, F, Am, Em) for harmonic coherence
- **Tempo control**: Adjustable BPM (60-180) with visual metronome
- **Quantization system**: Snap to rhythmic grid (1/2, 1/4, 1/8, 1/16, 1/32 notes)
- **Visual grid**: Optional temporal grid overlay showing beat divisions
- **Dynamic velocity**: Drawing speed influences note loudness

### Production Tools
- **MIDI export**: Save compositions as standard MIDI files (.mid)
- **Polyphonic playback**: Multiple notes play simultaneously based on horizontal positioning
- **Loop mode**: Automatic repetition with measure-aligned loops in quantized mode
- **Undo/Redo**: Full edit history for trace management
- **Eraser tool**: Remove unwanted traces with 30px radius brush

### Visual Design
- **Warm professional palette**: Terra cotta, slate blue, sage green, and burnt orange tones
- **Real-time feedback**: Color-coded instruments, note highlighting during playback, animated playhead
- **Adaptive UI**: Context-sensitive help overlay with 5-column layout
- **Performance-focused**: 60 FPS rendering with responsive controls

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- macOS, Linux, or Windows
- Audio output device
- Mouse or touchpad (tablet/stylus supported)

### Python Dependencies
```bash
pip install pygame>=2.5.0
pip install numpy>=1.24.0
pip install pyfluidsynth>=1.3.0
pip install midiutil>=1.2.1
```

### System-Level Audio
**Linux:**
```bash
sudo apt-get install fluidsynth libfluidsynth-dev
```

**macOS:**
```bash
brew install fluid-synth
```

**Windows:**
Download from [FluidSynth Releases](https://github.com/FluidSynth/fluidsynth/releases)

### SoundFont File
Download a General MIDI SoundFont (.sf2):
- **FluidR3_GM.sf2** (recommended, 142MB)
  - Download: [FluidR3_GM](https://member.keymusician.com/Member/FluidR3_GM/index.html)
- Alternative: GeneralUser GS (28MB) or MuseScore_General.sf2 (35MB)

Place the SoundFont file in the same directory as the script.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/geometric-synthesizer-v2.git
cd geometric-synthesizer-v2

# Install Python dependencies
pip install -r requirements.txt

# Download SoundFont (example using wget)
wget https://example.com/FluidR3_GM.sf2 -O FluidR3_GM.sf2

# Run the synthesizer
python geometric_synth_v2.py
```

## 🎹 Quick Start

1. **Select an instrument**: Press keys `1-9` to choose from the current preset
2. **Change presets**: Use `Left/Right` arrow keys to cycle through instrument categories
3. **Draw to play**: Click and drag vertically - higher positions = higher notes
4. **Adjust settings**: Press `M` for scale lock, `K` to change key, `Up/Down` for velocity
5. **Enable quantization**: Press `Q` to snap to tempo, `T` to change BPM, `G` to show grid
6. **Playback**: Press `SPACE` to play back your composition from left to right
7. **Export**: Press `X` to save as MIDI file

## 🎮 Controls Reference

### Drawing & Editing
| Key | Action |
|-----|--------|
| **Mouse Drag** | Draw trace with continuous sound |
| **E** | Toggle eraser mode |
| **Z** | Undo last trace |
| **Y** | Redo last undone trace |
| **C** | Clear entire canvas |

### Instrument Selection
| Key | Action |
|-----|--------|
| **1-9** | Select instrument from current preset |
| **←** | Previous preset |
| **→** | Next preset |

### Musical Controls
| Key | Action |
|-----|--------|
| **M** | Toggle scale lock (quantize to key) |
| **K** | Cycle key signature (C, G, D, F, Am, Em) |
| **P** | Toggle stereo pan on/off |
| **+** or **=** | Increase base velocity (+10) |
| **-** or **_** | Decrease base velocity (-10) |

### Tempo & Quantization
| Key | Action |
|-----|--------|
| **T** | Change tempo (60, 80, 100, 120, 140, 160, 180 BPM) |
| **Q** | Toggle quantization on/off |
| **D** | Change division (1/2, 1/4, 1/8, 1/16, 1/32 notes) |
| **G** | Toggle visual grid display |

### Playback
| Key | Action |
|-----|--------|
| **SPACE** | Play/Pause |
| **BACKSPACE** | Stop |
| **L** | Toggle loop mode |

### Other
| Key | Action |
|-----|--------|
| **X** | Export to MIDI file |
| **H** | Toggle help overlay |
| **Q** or **ESC** | Quit application |

## 🎼 Musical Concepts

### Spatial Mapping
- **Y-axis (Vertical)**: Pitch control (C2 to C7, 5 octaves)
  - Top of screen = high notes
  - Bottom of screen = low notes
- **X-axis (Horizontal)**: Stereo positioning (toggleable with P)
  - Left side = left speaker
  - Right side = right speaker
- **Trace length**: Duration and velocity
  - Longer traces = longer sustain and louder notes

### Multi-Note Detection
Long traces with significant pitch variation (>35px Y-range) are automatically split into sequential notes at pitch peaks and valleys, creating melodic phrases from single gestures.

### Quantization System
When enabled (Q key):
- Notes snap to rhythmic grid based on selected division
- Playback speed controlled by tempo (BPM)
- Loop mode aligns to complete beats (visual grid "thick lines")
- Grid shows beat subdivisions with varying opacity

When disabled:
- Free timing based on horizontal spacing
- Adaptive playback speed (4-28 seconds for full canvas)
- Natural, expressive timing

### Scale Lock
Enable with M key to snap all notes to the current key signature:
- Ensures harmonic coherence
- Six supported keys: C, G, D, F, Am, Em
- Disable for full chromatic access

## 📦 Project Structure

```
geometric-synthesizer-v2/
├── geometric_synth_v2.py      # Main application
├── FluidR3_GM.sf2              # SoundFont file (not included)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
├── docs/
│   ├── technical_documentation.md
│   └── article.md
└── screenshots/
    ├── interface.png
    └── demo.gif
```

## 🏗️ Architecture

### Core Components

**AudioEngine** (`FluidSynth wrapper`)
- Non-blocking note playback with timestamp tracking
- Polyphony management (unlimited simultaneous voices)
- Instrument and drum program selection
- Stereo panning via MIDI CC

**ShapeAnalyzer** (`Spatial → Musical mapping`)
- Y-position → MIDI pitch conversion
- Trace length → duration & velocity
- Scale quantization (6 key signatures)
- Multi-note segment detection

**GeometricSynth** (`Main orchestrator`)
- 60 FPS event loop with Pygame
- Mouse/keyboard input handling
- Playback system with quantization
- MIDI export functionality
- UI rendering and feedback

## 🎨 Instrument Presets

1. **Classical**: Piano, Violin, Cello, Oboe, Flute, Clarinet, Trumpet, Harp, Strings
2. **Jazz**: Jazz Guitar, Slap Bass, Electric Piano, Brass Section, Trombone, Organ
3. **Rock**: Electric Guitar, Distortion Guitar, Bass, Rock Organ, Synth Lead
4. **Electro**: Synth Bass 1/2, Lead 1/2, Pad 1/2, Keys, FX, Pluck
5. **Latin**: Acoustic Guitars, Accordion, Trumpet, Tenor Sax, Flute
6. **Country**: Acoustic Guitar, Pedal Steel, Fiddle, Harmonica, Banjo
7. **Soul**: Piano, Organ, Bass, Alto/Tenor Sax, Trumpet, Choir, Strings
8. **World**: Sitar, Shamisen, Koto, Kalimba, Bagpipe, Pan Flute, Ocarina, Marimba
9. **Drum Kit**: Kick, Snare, Hi-Hats, Cymbals, Toms, Tambourine
10. **Percussions**: Claves, Woodblocks, Cowbell, Shaker, Triangle
11. **Latin Drums**: Congas, Timbales, Bongos, Cabasa, Maracas
12. **Miscellaneous**: Echoes, Helicopter, Seashore, Whistle, Applause

## 🔧 Configuration

Edit these values in `geometric_synth_v2.py`:

```python
# Audio
SOUNDFONT_PATH = "FluidR3_GM.sf2"  # Change to your SoundFont path

# MIDI range
self.min_midi = 36  # C2 (default)
self.max_midi = 96  # C7 (default)

# Display
self.width, self.height = 1200, 760  # Window size

# Eraser
self.eraser_radius = 30  # Pixels

# Multi-note sensitivity
y_range_threshold = 35  # Minimum Y variation for split (pixels)
min_y_change = 25       # Minimum change between segments (pixels)
```

## 🐛 Troubleshooting

### No sound output
- Verify SoundFont file exists and path is correct
- Check system audio drivers installed (ALSA/CoreAudio)
- Test FluidSynth: `python -c "import fluidsynth; print('OK')"`
- Increase base velocity with `+` key

### Crackling audio
- Close other audio applications
- Update audio drivers
- Use dedicated audio interface

### Multi-note detection not working
- Draw traces with >35px vertical variation
- Ensure trace has clear direction changes (peaks/valleys)
- Check console for "Multi-note shape" debug messages

### Quantization issues
- Verify quantization enabled (Q key, status shows "Quantize: 1/16")
- Grid must be visible (G key) to see alignment
- Tempo affects grid spacing (T key to adjust)

### MIDI export fails
- Install midiutil: `pip install midiutil`
- Check write permissions in current directory
- Console shows export confirmation or error details

## 📚 Documentation

- **Technical Documentation**: `docs/technical_documentation.md` - Complete system architecture, algorithms, and API reference
- **Development Article**: `docs/article.md` - Design decisions, challenges, and lessons learned

## 🎥 Demo

[Link to demo video on YouTube]

Watch a 3-minute demonstration showing:
- Continuous gesture control
- Multi-note phrase creation
- Quantization and tempo system
- MIDI export workflow

## 🤝 Contributing

This project is complete and maintained as a portfolio piece. However, if you find bugs or have suggestions:

1. Open an issue describing the problem/suggestion
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

## 📄 License

MIT License 
Copyright (c) 2025 [Julie Le Rudulier]

## 🙏 Acknowledgments

**Technologies:**
- [Pygame](https://www.pygame.org) - Graphics and input handling
- [FluidSynth](https://www.fluidsynth.org) - Real-time audio synthesis
- [NumPy](https://numpy.org) - Mathematical operations
- [MIDIUtil](https://github.com/MarkCWirt/MIDIUtil) - MIDI file generation

**SoundFont:**
- FluidR3_GM by Frank Wen (MIT License)

**Inspiration:**
- Theremin (continuous pitch control)
- Ableton Push (clip launching and quantization)
- Reactable (tangible interface for music)

## 📧 Contact

- **GitHub**: [julielerudulier](https://github.com/julielerudulier/Projects)
- **Portfolio**: [julielerudulier.github.io](https://julielerudulier.github.io/)

---

**Made with ❤️ for expressive musical performance**
