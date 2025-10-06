# Geometric Synthesizer V2 - Technical Documentation

**Version:** 2.0 (Musical Edition)  
**Date:** October 2025  
**Language:** Python 3.8+

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Dependencies & Installation](#3-dependencies--installation)
4. [Core Components](#4-core-components)
5. [Audio Engine](#5-audio-engine)
6. [Shape Analysis](#6-shape-analysis)
7. [Spatial Mapping](#7-spatial-mapping)
8. [Multi-Note Gesture System](#8-multi-note-gesture-system)
9. [Playback System](#9-playback-system)
10. [Quantization & Tempo System](#10-quantization--tempo-system)
11. [User Interface](#11-user-interface)
12. [Controls Reference](#12-controls-reference)
13. [Eraser Tool](#13-eraser-tool)
14. [Undo/Redo System](#14-undoredo-system)
15. [MIDI Export](#15-midi-export)
16. [Configuration](#16-configuration)
17. [Known Limitations](#17-known-limitations)
18. [Future Enhancements](#18-future-enhancements)
19. [Troubleshooting](#19-troubleshooting)
20. [Credits & License](#20-credits--license)
21. [Appendices](#21-appendices)

---

## 1. Overview

The Geometric Synthesizer V2 is an interactive musical instrument that transforms hand-drawn gestures into musical notes using real instrument samples via FluidSynth/SoundFont technology.

### Key Features

- **Real instrument sounds** via FluidSynth/SoundFont2 (.sf2)
- **Spatial audio mapping**: Y-axis → pitch, X-axis → stereo pan (toggleable)
- **12 musical presets** (Classical, Jazz, Rock, Electro, Latin, Country, Soul, World, Drum Kit, Percussions, Latin Drums, Miscellaneous) with 9 instruments each
- **Continuous live drawing mode** with real-time pitch-following
- **Musical intelligence**: Scale locking (C, G, D, F, Am, Em), velocity control
- **Multi-note gesture support**: Long traces with pitch variation automatically split into sequential notes
- **Quantization system**: Snap notes to tempo grid (1/2, 1/4, 1/8, 1/16, 1/32 notes)
- **Tempo control**: Adjustable BPM (60-180) with optional visual grid
- **Visual playback**: Playhead line shows playback position, notes highlight when playing
- **Loop mode**: Automatic playback repetition
- **Eraser tool**: Remove unwanted traces with preview sound
- **Undo/Redo system** for comprehensive trace management
- **MIDI export**: Save compositions as standard MIDI files
- **Color-coded instruments** with warm professional aesthetic

### Design Philosophy

Version 2 prioritizes gestural musicality and direct spatial control. The Y-axis controls pitch continuously during drawing, creating an intuitive vertical keyboard. The X-axis provides optional stereo positioning. The playback system reads the canvas left-to-right like a musical score, with temporal spacing determined by horizontal distances between traces. Multi-note gestures enable expressive melodic phrases within single strokes, transforming gestural improvisation into composed musical sequences.

---

## 2. System Architecture

### Three-Layer Architecture

```
┌─────────────────────────────────────────┐
│      GeometricSynth (Main App)          │
│  - Event handling                       │
│  - Visual rendering                     │
│  - Playback orchestration               │
│  - MIDI export                          │
└──────────┬──────────────────────────────┘
           │
           ├──────────────────┬─────────────────┐
           │                  │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌─────▼─────┐
    │ AudioEngine │   │ShapeAnalyzer│   │  Pygame   │
    │             │   │             │   │  Display  │
    │ FluidSynth  │   │ Geometry    │   │           │
    │ MIDI        │   │ Multi-note  │   │           │
    │ Polyphony   │   │ Musical     │   │           │
    └─────────────┘   └─────────────┘   └───────────┘
```

### Component Responsibilities

**AudioEngine**
- FluidSynth initialization and management
- Note triggering and tracking (non-blocking)
- Instrument/drum program changes
- Stereo panning (MIDI CC 10)
- Velocity control
- Active note lifecycle management

**ShapeAnalyzer**
- Trace analysis and segment detection
- Multi-note gesture splitting
- Spatial coordinate mapping to musical parameters
- Scale quantization (6 key signatures)
- MIDI note calculation (range: C2-C7)
- Velocity and duration computation

**GeometricSynth**
- Main event loop (60 FPS)
- Mouse/keyboard input handling
- Trace drawing and storage
- Spatial-temporal playback system
- Quantization and tempo management
- Visual rendering with playhead
- UI overlay and help system
- Eraser tool management
- Undo/redo stack management
- MIDI file export

---

## 3. Dependencies & Installation

### Required Dependencies

```bash
# Core dependencies
pip install pygame>=2.5.0
pip install numpy>=1.24.0
pip install pyfluidsynth>=1.3.0

# MIDI export (optional but recommended)
pip install midiutil

# System-level FluidSynth
# Ubuntu/Debian:
sudo apt-get install fluidsynth libfluidsynth-dev

# macOS:
brew install fluid-synth

# Windows:
# Download from: https://github.com/FluidSynth/fluidsynth/releases
```

### SoundFont Requirement

Download a General MIDI SoundFont (.sf2 file):

- **FluidR3_GM.sf2** (recommended, 142MB)
  - Download from: https://member.keymusician.com/Member/FluidR3_GM/index.html
  - Place in same directory as script

Alternative SoundFonts:
- GeneralUser GS (28MB)
- MuseScore_General.sf2 (35MB)

### Installation Steps

```bash
# 1. Clone repository
git clone [your-repo-url]
cd geometric-synthesizer-v2

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download SoundFont
wget [soundfont-url] -O FluidR3_GM.sf2

# 4. Run
python geometric_synth_v2.py
```

---

## 4. Core Components

### AudioEngine Class

**Purpose:** Manages all audio output through FluidSynth

**Key Methods:**

```python
def __init__(self, soundfont_path: str, parent_synth=None)
    """Initialize FluidSynth with SoundFont and drum channel"""
    
def play_note(self, midi_note: int, velocity: int, 
              duration: float, pan: float, 
              instrument: int, is_drum: bool) -> int
    """Play a note and return note_id for tracking"""
    
def stop_note_by_id(self, note_id: int)
    """Stop a specific note by its ID"""
    
def update()
    """Called every frame to stop expired notes"""
    
def set_instrument(self, program_id: int = None, drum_note: int = None)
    """Override instrument selection (melodic or drum)"""
    
def stop_all_notes()
    """Emergency stop for all active notes"""
```

### Instrument Organization

The engine supports 108 instruments organized into 12 musical presets. Users select instruments via number keys (1-9) or cycle through presets with arrow keys (←/→).

**Preset List:**

1. **Classical** - Piano, Violin, Cello, Oboe, Flute, Clarinet, Trumpet, Harp, String Ensemble
2. **Jazz** - Jazz Guitar, Slap Bass, Acoustic Bass, Electric Piano, Clarinet, Brass Section, Muted Trumpet, Trombone, Organ
3. **Rock** - Electric Guitar, Distortion Guitar, Rhythmic Guitar, Acoustic Guitar, Electric Bass, Piano, Rock Organ, Synth Lead, Synth Strings
4. **Electro** - Synth Bass 1/2, Lead 1/2, Pad 1/2, Keys, FX, Pluck
5. **Latin** - Acoustic Guitar 1/2, Electric Guitar, Acoustic Bass, Accordion, Trumpet, Tenor Sax, Flute, Grand Piano
6. **Country** - Acoustic Guitar, Electric Guitar, Pedal Steel, Fiddle, Harmonica, Banjo, Organ, Electric Bass, Honky Tonk Piano
7. **Soul** - Piano, Organ, Electric Bass, Electric Guitar, Alto Sax, Tenor Sax, Trumpet, Choir, Strings Ensemble
8. **World** - Sitar, Shamisen, Koto, Kalimba, Bagpipe, Pan Flute, Ocarina, Shakuhachi, Marimba
9. **Drum Kit** - Kick, Snare, Closed/Open Hi-Hat, Crash/Ride Cymbal, Low/High Tom, Tambourine
10. **Percussions** - Side Stick, Claves, Woodblock High/Low, Cowbell, Shaker, Tambourine, Triangle, Steel Drum
11. **Latin Drums** - High/Low Conga, Timbale High/Low, High/Low Bongo, Cabasa, Maracas, Agogo
12. **Miscellaneous** - Echoes, Helicopter, Clap, Reverse Cymbal, Seashore, Whistle, Applause, Gunshot, Breath Noise

### ShapeAnalyzer Class

**Purpose:** Converts geometric properties to musical parameters and detects multi-note gestures

**Key Methods:**

```python
def shape_to_midi(self, shape: dict) -> int
    """Y position → MIDI note (36-96) with optional scale quantization"""
    
def shape_to_velocity(self, shape: dict) -> int
    """Shape size → velocity (30-127), interpolated from base velocity"""
    
def shape_to_duration(self, shape: dict) -> float
    """Shape length → duration (0.1s - 3.0s)"""
    
def shape_to_pan(self, shape: dict) -> float
    """X position → stereo pan (0.0-1.0)"""
    
def snap_to_scale(self, midi_note: int) -> int
    """Quantize MIDI note to current scale"""
    
def set_key_signature(self, key_name: str)
    """Set current key for scale lock"""
    
def extract_note_segments(self, shape: dict) -> List[dict]
    """
    Analyze trace for vertical pitch variations
    Returns list of note segments if multi-note gesture detected
    Returns None if single note
    
    Detection criteria:
    - Y-range threshold: 35px total variation required
    - Direction change detection with smoothing
    - Minimum segment distance: 25px Y-change, 40px X-change
    - Fusion: Last two segments merged if same MIDI
    - Filtering: Consecutive duplicate MIDI notes removed
    """
```

---

## 5. Audio Engine

### FluidSynth Integration

FluidSynth provides:
- Low latency audio output (~10-20ms)
- Polyphony up to 256 simultaneous voices
- General MIDI compatibility
- Dual channel system: Channel 0 (melodic), Channel 9 (drums)

### Note Lifecycle

```
User draws trace
      ↓
Trace finalized with instrument/velocity/pan saved
      ↓
[Playback triggered]
      ↓
Calculate MIDI parameters from trace properties
      ↓
synth.noteon(channel, note, velocity)
      ↓
Store in active_notes{} with timestamp and duration
      ↓
[Note plays...]
      ↓
update() checks elapsed time
      ↓
synth.noteoff(channel, note) when duration expires
      ↓
Remove from active_notes{}
```

### Non-Blocking Architecture

```python
# Timestamp-based tracking instead of blocking delays
note_id = play_note(...)
active_notes[note_id] = {
    'midi': midi_note,
    'start_time': time.time(),
    'duration': duration,
    'channel': channel
}
# Main loop continues immediately...
# update() called every frame checks and stops expired notes
```

This architecture enables:
- Drawing while notes play
- Multiple simultaneous notes
- Responsive UI at all times

### Stereo Panning Implementation

```python
# MIDI Control Change #10 (Pan)
pan_value = int(max(0, min(127, pan * 127)))
synth.cc(channel, 10, pan_value)
# 0 = hard left, 64 = center, 127 = hard right
```

Pan can be toggled on/off with P key. When disabled, all notes play centered (pan = 0.5).

---

## 6. Shape Analysis

### Trace Properties

Each drawn trace stores:
- `points`: List of (x, y) coordinates
- `center`: Average position
- `width`, `height`: Bounding box dimensions
- `length`: Total Euclidean distance of path
- `timestamp`: Creation time
- `color`: Instrument color
- `note_position`: Last drawn point (for MIDI calculation)
- `instrument`: Instrument program or drum MIDI key
- `is_drum`: Boolean flag
- `velocity`: Saved velocity at creation time
- `pan_at_creation`: Pan enabled state when drawn

### Musical Parameter Extraction

**Pitch Mapping (Y-axis):**

```
Screen top (y=0)      → MIDI 96 (C7)  ─┐
                                        │ High register
Screen height/4       → MIDI 84 (C6)  ─┤
                                        │
Screen height/2       → MIDI 72 (C5)  ─┤ Middle register
                                        │
Screen 3*height/4     → MIDI 60 (C4)  ─┤
                                        │ Low register
Screen bottom (y=760) → MIDI 36 (C2)  ─┘
```

**Scale Quantization:**

When scale lock is enabled (M key), MIDI notes snap to nearest note in selected key:

```python
# C Major scale: C D E F G A B
semitones = [0, 2, 4, 5, 7, 9, 11]

# Example: MIDI 68 (G#) → Not in scale → snap to 69 (A)
```

Supported keys: C, G, D, F, Am, Em

**Velocity Mapping:**

```python
# Base velocity adjustable with +/- keys (30-127)
# Interpolation from base to 127 based on trace length
gamma = 0.6  # Response curve
velocity = base + (127 - base) * (normalized_length ** gamma)
```

**Duration Mapping:**

```python
# Based on total trace length (Euclidean distance)
if 'length' in shape:
    duration = max(0.1, min(3.0, length / 100))
    
# Fallback for legacy shapes without length
else:
    area = width * height
    duration = max(0.2, min(3.0, area / 6000))
```

---

## 7. Spatial Mapping

### Y-Axis → Pitch

**Rationale:** Mimics traditional music notation where higher positions = higher pitch

**Implementation:**

```python
y_normalized = 1.0 - (y_pos / screen_height)
midi_note = MIN_MIDI + y_normalized * (MAX_MIDI - MIN_MIDI)
```

**Musical Range:**
- Total range: 60 semitones (5 octaves)
- Comfortable for melody and harmony
- Avoids extreme registers (too low/high)

### X-Axis → Stereo Pan

**Rationale:** Creates spatial depth and separation between simultaneous notes

**Implementation:**

```python
pan = x_pos / screen_width  # 0.0 to 1.0
```

**Perceptual Effect:**
- Left drawings → left speaker
- Center drawings → balanced stereo
- Right drawings → right speaker
- Enables "stereo composition"

### Size → Dynamics

**Velocity (Volume):** Larger/longer traces = louder notes. Creates natural emphasis.

**Duration:** Longer traces = longer sustain. Mimics how larger objects resonate longer.

---

## 8. Multi-Note Gesture System

### Overview

Long traces with significant pitch variation are automatically segmented into multiple sequential notes, enabling expressive melodic phrases within a single gesture.

### Detection Logic

**Updated Thresholds:**
- `y_range >= 35px`: Total vertical variation required to enable multi-note detection
- `window = 5`: Smoothing window for noise reduction
- `min_y_change = 25px`: Minimum vertical difference for new segment
- `min_x_change = 40px`: Minimum horizontal spacing between segments

**Algorithm Flow:**

1. Calculate total trace length (Euclidean distance between all points)
2. Calculate Y-range (max - min Y coordinates)
3. If `y_range < 35px`: Return None (single note, horizontal line detection)
4. If `y_range >= 35px`: Analyze for direction changes
   - Smooth Y values with moving average (window=5)
   - Detect peaks/valleys (slope sign changes)
   - Filter insignificant changes (<25px Y, <40px X)
5. Create segments at extrema points
6. Always include first and last points

### Segment Processing

**MIDI Calculation:**

Each segment's MIDI note is based on the Y position of its extrema point:

```python
temp_shape = {
    'type': shape['type'],
    'center': extrema_point,
    'width': shape['width'],
    'height': shape['height']
}
midi_note = analyzer.shape_to_midi(temp_shape)
```

**Fusion Strategy:**

If the last two segments have the same MIDI note, they are merged:

```python
if len(segments) >= 2:
    if segments[-2]['midi_approx'] == segments[-1]['midi_approx']:
        # Merge last two segments
        segments[-2]['is_last'] = True
        segments.pop()
```

**Filtering:**

Consecutive segments with identical MIDI notes (except first and last) are removed:

```python
filtered = [segments[0]]
for seg in segments[1:-1]:
    if seg['midi_approx'] != filtered[-1]['midi_approx']:
        filtered.append(seg)
filtered.append(segments[-1])
```

### Duration Calculation

Each segment's duration is based on horizontal distance to next segment:

```python
# For all segments except last
duration = (next_segment_x - current_x) / pixels_per_second - 0.02

# For last segment - based on trace length to actual end
trace_length = calculate_euclidean_distance(segment_points)
duration = max(0.2, min(2.5, trace_length / 80))
```

---

## 9. Playback System

### Spatial-Temporal Mapping

The playback system treats horizontal (X) position as time:
- Leftmost trace starts playback at t=0
- Rightmost trace ends playback
- Horizontal spacing = temporal spacing (silences between notes)
- Reading speed adapts to total canvas width

### Reading Speed Calculation

**Free Mode (Quantization Disabled):**

```python
if total_width > 0:
    if total_width < 400:
        pixels_per_second = 100  # Fixed for small spaces
        target_duration = total_width / pixels_per_second
    else:
        # Adaptive: 400px=4s, 1200px=14s, 3000px=28s
        min_duration = 4.0
        max_duration = 28.0
        normalized = min(1.0, (total_width - 400) / 2600)
        target_duration = min_duration + normalized * (max_duration - min_duration)
        pixels_per_second = total_width / target_duration
else:
    pixels_per_second = 100
```

**Quantized Mode (Quantization Enabled):**

```python
# Use BPM to determine speed
beat_duration = 60.0 / bpm
pixels_per_beat = 80  # Fixed relationship
pixels_per_second = pixels_per_beat / beat_duration
```

Result: In free mode, small canvases play quickly, large canvases play proportionally longer, preserving relative timing. In quantized mode, speed is determined by tempo.

### Event Scheduling

Each trace or segment becomes a playback event with:
- `time`: Horizontal position converted to seconds
- `shape`: Reference to original trace
- `duration`: Limited to horizontal time slot (95% to avoid overlap)
- `midi_override`: Specific MIDI note (for segments)
- `original_index`: Drawing order for stable sorting
- `sub_index`: Segment order within trace

Events are sorted by: `(time, original_index, sub_index)`

### Polyphony

Traces within similar horizontal positions are quantized to same time:

```python
time_tolerance = 0.05  # 50ms tolerance
quantized_time = round(start_time / time_tolerance) * time_tolerance
```

Traces with quantized times within tolerance play simultaneously, creating chords.

### Visual Feedback

**Playhead Line:**
- Vertical line (3px wide, orange) with triangle marker at top
- Position: `current_x = min_x + (elapsed_time * pixels_per_second)`
- Semi-transparent (alpha 180)
- Only visible during active playback

**Note Highlighting:**
- Playing traces glow brighter (+80 RGB values)
- Line thickness increases to 5px
- Provides visual-audio synchronization

**Loop Behavior:**

When loop mode is enabled (L key):
- Playback automatically restarts at the beginning
- In quantized mode: Loops at measure boundary
- In free mode: Loops after all notes finish

### Immediate Event Playback

Events scheduled at t=0 are played immediately at the end of `start_playback()` to avoid timing issues:

```python
# Force immediate playback of t=0 events
for event in self.playback_events:
    if event['time'] <= 0.0:
        play_shape(event)
        self.playback_index += 1
    else:
        break

# Reset timestamp AFTER playing notes at t=0
# to avoid offset on first beat
self.playback_start_time = time.time()
```

---

## 10. Quantization & Tempo System

### Overview

The quantization system allows snapping notes to a musical grid based on tempo (BPM) and note division, enabling rhythmic precision.

### Tempo Control

**BPM (Beats Per Minute):**
- Default: 80 BPM
- Range: 60, 80, 100, 120, 140, 160, 180 BPM
- Adjusted with T key (cycles through values)
- Affects playback speed when quantization enabled

**Beat Duration Calculation:**

```python
beat_duration = 60.0 / bpm  # seconds per beat
```

### Quantization Divisions

**Available Divisions:**
- 1/2 notes (half notes)
- 1/4 notes (quarter notes)
- 1/8 notes (eighth notes)
- 1/16 notes (sixteenth notes) - default
- 1/32 notes (thirty-second notes)

Adjusted with D key (cycles through divisions)

**Quantize Step Calculation:**

```python
beat_duration = 60.0 / bpm
quantize_step = beat_duration / (division / 4)

# Example: 80 BPM, 1/16 notes
# beat_duration = 0.75s
# quantize_step = 0.75 / 4 = 0.1875s
```

### Grid System

**Visual Grid:**
- Toggle with G key
- Displays vertical lines at quantization intervals
- Stronger lines on beats, lighter lines on subdivisions
- Grid spacing: `quantize_step * pixels_per_second`

**Grid Drawing:**

```python
# Use playback parameters if available, otherwise current parameters
if hasattr(self, 'playback_pixels_per_second'):
    pixels_per_second = self.playback_pixels_per_second
else:
    pixels_per_beat = 80
    pixels_per_second = pixels_per_beat / beat_duration

# Calculate grid spacing
pixels_per_beat = pixels_per_second * beat_duration
grid_spacing = pixels_per_beat / (self.quantize_division / 4)

line_count = 0
x = 0
while x < screen_width:
    is_beat = (line_count % beat_interval) == 0
    alpha = 60 if is_beat else 30  # Beat vs subdivision
    draw_vertical_line(x, alpha)
    x += grid_spacing
    line_count += 1
```

### Quantized Playback

**When Quantization Enabled (Q key):**

**Reading Speed:**
- Based on tempo instead of canvas width
- `pixels_per_beat = 80` (fixed relationship)
- `pixels_per_second = pixels_per_beat / beat_duration`

**Event Timing:**
- All note start times rounded to nearest quantize step
- `quantized_time = round(time / quantize_step) * quantize_step`

**Playback Behavior:**
- Notes snap to grid positions
- Creates metronomic, rhythmic playback
- Preserves musical timing relationships
- Total duration rounded to complete beats

**When Quantization Disabled:**
- Free timing based on horizontal positions
- Adaptive speed calculation (existing system)
- Natural, expressive timing

### Integration with Multi-Note System

Quantization applies to each segment individually:

```python
if self.quantize_enabled:
    quantize_step = self.get_quantize_step()
    seg_time = round(seg_time / quantize_step) * quantize_step
    print(f"Quantized to {seg_time:.3f}s (grid: {quantize_step:.3f}s)")
```

This ensures melodic phrases align to the rhythmic grid while preserving their internal structure.

---

## 11. User Interface

### Visual Feedback

**Color Coding:**
- Each instrument has unique color from warm professional palette
- Active instrument displayed prominently in top-left
- Traces drawn use color of active instrument at creation time
- Selected instrument in bottom panel highlighted

**Active Note Glow:**
- Playing traces glow brighter (+80 RGB) with thicker lines (5px)
- Creates visual-audio synchronization
- Helps track polyphony visually

**Live Drawing Mode:**
- Always active, provides continuous audio feedback
- Sound follows Y-position for pitch in real-time
- Single sustained note with smooth pitch transitions
- Velocity adjusts based on drawing speed
- Stops when mouse released

### Display Elements

**Top-Left Panel (320×230px):**
- Title: "GEOMETRIC SYNTHESIZER V2"
- Current preset in brackets: [CLASSICAL]
- Active instrument name (color-coded)
- Scale status: "Scale: C" or "Scale: Free"
- Pan status: "Pan: ON/OFF"
- Loop status: "Loop: ON/OFF"
- Base velocity: "Base Velocity: 50"
- Quantize status: "Quantize: 1/16 @ 80BPM [Grid ON]" or "Quantize: OFF (80BPM)"
- Active notes: "Playing: 3 notes"
- Playback status (if active): "PLAYING (2/5)" or "PAUSED"

**Top-Right Corner:**
- Live note info (while drawing):
  - "Last note: A4 | vel:80 | pan:0.50"

**Bottom-Left Panel (450×130px):**
- Instrument panel
- Current preset name at top
- 9 instruments in 3×3 grid
- Number keys 1-9 for selection
- Selected instrument highlighted with larger font and color

**Bottom-Right Panel (720×130px):**
- Help overlay (toggle with H key)
- Semi-transparent background
- 5 adaptive-width columns:
  - MUSICAL: 1-9, Left/Right arrows, Up/Down, P, M, K
  - TEMPO: T, G, Q, D
  - EDITING: Z, Y, E, C
  - PLAYBACK: SPACE, BACKSPACE, L, X
  - OTHER: ESC

**Eraser Cursor:**
- Terra cotta circle (30px radius) when eraser active
- Crosshair at center
- Follows mouse position

**Playhead (During Playback):**
- Vertical orange line (3px wide)
- Triangle marker at top
- Moves left-to-right showing current playback position

### Color Palette

- **Background:** (250, 248, 240) - Warm cream
- **UI Background:** (240, 238, 230) - Lighter cream
- **Text:** (40, 40, 50) - Dark slate
- **Accent colors:**
  - Terra Cotta: #E07A5F (224, 122, 95)
  - Slate Blue: #3D405B (61, 64, 91)
  - Sage Green: #81B29A (129, 178, 154)
  - Warm Brown: (140, 95, 74)
  - Deep Teal: (52, 108, 117)
  - Purple: (150, 80, 150)
  - Burnt Orange: (200, 90, 60)
  - Olive Green: (105, 123, 92)
  - Golden Ochre: #DEC179 (222, 193, 121)
  - Plum Violet: #6F5777 (92, 76, 101)

---

## 12. Controls Reference

### Drawing

- **Left Click + Drag**: Draw trace with continuous pitch-following sound
- **E**: Toggle eraser mode on/off
- **Eraser mode + Click/Drag**: Remove traces within 30px radius (plays preview sound)

### Instrument Selection

- **1-9**: Select specific instrument from current preset
- **← (Left Arrow)**: Previous preset
- **→ (Right Arrow)**: Next preset

### Musical Controls

- **M**: Toggle scale lock on/off
- **K**: Cycle key signature (C → G → D → F → Am → Em → C)
- **P**: Toggle pan (stereo positioning) on/off
- **Up or =**: Increase base velocity (+10, max 127)
- **Down or -**: Decrease base velocity (-10, min 30)

### Tempo & Quantization

- **T**: Change tempo (cycles through: 60, 80, 100, 120, 140, 160, 180 BPM)
- **Q**: Toggle quantization on/off
- **D**: Change quantization division (1/2, 1/4, 1/8, 1/16, 1/32 notes)
- **G**: Toggle visual grid display on/off

### Playback

- **SPACE**: Play/Pause playback (reads left-to-right)
- **BACKSPACE**: Stop playback completely
- **L**: Toggle loop mode on/off
- Playback respects horizontal spacing as temporal spacing
- Notes at similar X positions play simultaneously (polyphony)
- Visual playhead shows current playback position

### Editing

- **Z**: Undo last action (drawing or erasing)
- **Y**: Redo last undone action
- **C**: Clear all traces from canvas (stops playback if active)

### Export

- **X**: Export composition as MIDI file (timestamped filename)

### Interface

- **H**: Toggle help overlay on/off
- **ESC**: Quit application

---

## 13. Eraser Tool

### Overview

The eraser tool allows removal of unwanted traces with audio preview feedback.

### Activation

Press **E** key to toggle eraser mode on/off. When active, cursor changes to a red circle with crossh
