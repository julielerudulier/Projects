# Draw-to-Sound Interface (Geometric Synthesizer V2) - Technical Documentation

**Version:** 2.16 (Musical Edition)  
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
13. [MIDI Export](#13-midi-export)
14. [Configuration](#14-configuration)
15. [Known Limitations](#15-known-limitations)
16. [Future Enhancements](#16-future-enhancements)
17. [Troubleshooting](#17-troubleshooting)
18. [Credits & License](#18-credits--license)
19. [Appendices](#19-appendices)

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
- **Eraser tool**: Remove unwanted traces with audio preview
- **Undo/Redo system** for trace management
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
- Eraser tool management with audio preview
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

**FluidR3_GM.sf2** (recommended, 142MB)  
Download from: https://member.keymusician.com/Member/FluidR3_GM/index.html  
Place in same directory as script

**Alternative SoundFonts:**
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

Pan can be toggled on/off with **P** key. When disabled, all notes play centered (pan = 0.5).

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
Screen top (y=0)      → MIDI 96 (C7)  ─┐ High register
Screen height/4       → MIDI 84 (C6)  ─┤
Screen height/2       → MIDI 72 (C5)  ─┤ Middle register
Screen 3*height/4     → MIDI 60 (C4)  ─┤
Screen bottom (y=760) → MIDI 36 (C2)  ─┘ Low register
```

**Scale Quantization:**

When scale lock is enabled (**M** key), MIDI notes snap to nearest note in selected key:

```python
# C Major scale: C D E F G A B
semitones = [0, 2, 4, 5, 7, 9, 11]

# Example: MIDI 68 (G#) → Not in scale → snap to 69 (A)
```

Supported keys: C, G, D, F, Am, Em

**Velocity Mapping:**

```python
# Base velocity adjustable with Up/Down keys (30-127)
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

- **Velocity (Volume):** Larger/longer traces = louder notes. Creates natural emphasis.
- **Duration:** Longer traces = longer sustain. Mimics how larger objects resonate longer.

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
   - Create segments at extrema points
   - Always include first and last points

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

**Minimum Duration Filter:**

Middle segments (not first or last) with duration < 0.15s are skipped during playback to prevent playing extremely short transitional notes and improve musical clarity.

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

**Result:** Small canvases play quickly, large canvases play proportionally longer, preserving relative timing.

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

**Metronome Extension:**
- After all events complete, playhead continues briefly before stopping
- Allows visual confirmation of playback end

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
```

### Playback End Behavior

**Quantized Mode:**
- Playback rounds up to the next complete beat
- Uses 10% beat tolerance to avoid premature rounding
- Formula: `beats_needed = ceil((last_event_end - tolerance) / beat_duration)`
- Creates clean measure boundaries for looping

**Free Mode:**
- Playback ends when all notes finish playing
- Small metronome extension after last event
- Natural, expressive endings

---

## 10. Quantization & Tempo System

### Overview

The quantization system allows snapping notes to a musical grid based on tempo (BPM) and note division, enabling rhythmic precision.

### Tempo Control

- **BPM (Beats Per Minute):**
  - Default: 80 BPM
  - Range: 60, 80, 100, 120, 140, 160, 180 BPM
  - Adjusted with **T** key (cycles through values)
  - Affects playback speed when quantization enabled

- **Beat Duration Calculation:**
  ```python
  beat_duration = 60.0 / bpm  # seconds per beat
  ```

### Quantization Divisions

**Available Divisions:**
- 1/2 notes (half notes)
- 1/4 notes (quarter notes)
- 1/8 notes (eighth notes)
- **1/16 notes** (sixteenth notes) - default
- 1/32 notes (thirty-second notes)

Adjusted with **D** key (cycles through divisions)

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
- Toggle with **G** key
- Displays vertical lines at quantization intervals
- Stronger lines on beats, lighter lines on subdivisions
- Grid spacing: `quantize_step * pixels_per_second`

**Grid Drawing:**

```python
grid_spacing = quantize_step * pixels_per_second
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

- **Reading Speed:**
  - Based on tempo instead of canvas width
  - `pixels_per_beat = 80` (fixed)
  - Uses origin `x = 0` as reference point (not leftmost shape)
  - `pixels_per_second = pixels_per_beat / beat_duration`
  - Ensures consistent timing when adding new traces

- **Event Timing:**
  - All note start times rounded to nearest quantize step
  - `quantized_time = round(time / quantize_step) * quantize_step`

- **Playback Behavior:**
  - Notes snap to grid positions
  - Creates metronomic, rhythmic playback
  - Preserves musical timing relationships

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

**Bottom-Left Panel (450×120px):**
- Instrument panel
- Current preset name at top (aligned left)
- 9 instruments in 3×3 grid
- Number keys 1-9 for selection
- Selected instrument highlighted with larger font and color

**Bottom-Right Panel (720×120px):**
- Help overlay (toggle with **H** key)
- Semi-transparent background
- 5 adaptive-width columns:
  - **MUSICAL:** 1-9, Left/Right arrows, Up/Down, P, M, K
  - **TEMPO:** T, G, Q, D
  - **EDITING:** Z, Y, E, C
  - **PLAYBACK:** SPACE, BACKSPACE, L, X
  - **OTHER:** ESC

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

| Key | Action |
|-----|--------|
| **Left Click + Drag** | Draw trace with continuous pitch-following sound |
| **E** | Toggle eraser mode on/off |
| **Eraser + Click/Drag** | Remove traces within 30px radius (plays audio preview) |

### Instrument Selection

| Key | Action |
|-----|--------|
| **1-9** | Select specific instrument from current preset |
| **← (Left Arrow)** | Previous preset |
| **→ (Right Arrow)** | Next preset |

### Musical Controls

| Key | Action |
|-----|--------|
| **M** | Toggle scale lock on/off |
| **K** | Cycle key signature (C → G → D → F → Am → Em → C) |
| **P** | Toggle pan (stereo positioning) on/off |
| **↑ (Up Arrow)** or **=** | Increase base velocity (+10, max 127) |
| **↓ (Down Arrow)** or **_** | Decrease base velocity (-10, min 30) |

### Tempo & Quantization

| Key | Action |
|-----|--------|
| **T** | Change tempo (cycles through: 60, 80, 100, 120, 140, 160, 180 BPM) |
| **Q** | Toggle quantization on/off |
| **D** | Change quantization division (1/2, 1/4, 1/8, 1/16, 1/32 notes) |
| **G** | Toggle visual grid display on/off |

### Playback

| Key | Action |
|-----|--------|
| **SPACE** | Play/Pause playback (reads left-to-right) |
| **BACKSPACE** | Stop playback completely |
| **L** | Toggle loop mode on/off |

*Playback respects horizontal spacing as temporal spacing. Notes at similar X positions play simultaneously (polyphony). Visual playhead shows current playback position.*

### Editing

| Key | Action |
|-----|--------|
| **Z** | Undo last action (drawing or erasing) |
| **Y** | Redo last undone action |
| **C** | Clear all traces from canvas (stops playback if active) |

### Export

| Key | Action |
|-----|--------|
| **X** | Export composition as MIDI file (timestamped filename) |

### Interface

| Key | Action |
|-----|--------|
| **H** | Toggle help overlay on/off |
| **ESC** | Quit application |

---

## 13. MIDI Export

### Overview

The MIDI export system converts drawn compositions into standard MIDI files (.mid) that can be opened in any DAW or music software.

### Implementation

**Required Dependency:**

```bash
pip install midiutil
```

**Export Method:**

```python
def export_to_midi(self, filename="output.mid"):
    """Export shapes as MIDI file"""
    # Creates MIDI file with one track
    # Calculates timing based on playback system
    # Adds notes with proper velocity and duration
    # Skips drum notes (channel 9) for simplicity
    # Writes to .mid file
```

### Export Process

**Timing Calculation:**
- Uses same spatial-temporal mapping as playback
- Converts X positions to time values
- Applies same reading speed calculation

**Note Extraction:**
- Processes each trace
- Uses saved instrument, velocity, pan settings
- Respects multi-note segments
- Converts to MIDI note events

**MIDI File Creation:**
- Single track for simplicity
- Tempo set to current BPM
- Channel 0 (melodic instruments only)
- Standard MIDI format type 1

**File Naming:**
- Auto-generates timestamp: `export_YYYYMMDD_HHMMSS.mid`
- Prevents accidental overwrites
- Easy to organize multiple exports

### Usage

Press **X** key during normal operation (not during playback) to export current canvas as MIDI file. Confirmation message appears in console.

### Limitations

- Drums (channel 9) not included in export
- Stereo pan information not exported (not standard MIDI)
- Single instrument per export (instrument changes not tracked)
- Multi-note segments exported as sequential notes on same channel

### Post-Export

MIDI files can be:
- Opened in any DAW (Ableton, Logic, FL Studio, GarageBand, etc.)
- Edited and refined
- Rendered to audio with different instruments
- Shared with collaborators
- Imported into notation software

---

## 14. Configuration

### Audio Settings

```
Sample Rate: 44.1 kHz (CD quality)
Latency: ~20ms (FluidSynth default)
Polyphony: Unlimited (software-tracked)
Bit Depth: 16-bit (SoundFont standard)
Channels: 2 (Channel 0 = melodic, Channel 9 = drums)
```

### Display Settings

```
Resolution: 1200×760 pixels (windowed)
Frame Rate: 60 FPS
Color Depth: 24-bit RGB
```

### MIDI Range

```
Minimum Note: MIDI 36 (C2, ~65 Hz)
Maximum Note: MIDI 96 (C7, ~2093 Hz)
Velocity Range: 30-127 (adjustable base)
```

### Customization Points

**Change SoundFont:**

```python
SOUNDFONT_PATH = "your_soundfont.sf2"
```

**Adjust MIDI Range:**

```python
self.min_midi = 48  # C3 instead of C2
self.max_midi = 84  # C6 instead of C7
```

**Modify Multi-Note Sensitivity:**

```python
y_range_threshold = 40  # Increase for less sensitivity (default 35)
window = 7  # Increase smoothing window (default 5)
min_y_change = 30  # More vertical change needed (default 25)
```

**Adjust Reading Speed:**

```python
# For faster playback
min_duration = 3.0  # Decreased from 4.0
max_duration = 20.0  # Decreased from 28.0

# Or use fixed speed regardless of canvas width
pixels_per_second = 120  # Fixed value (default 100)
```

**Change Eraser Radius:**

```python
self.eraser_radius = 50  # Increased from 30
```

**Modify Key Signatures:**

```python
# Add new key (example: Bb major)
self.key_signatures['Bb'] = [0, 2, 3, 5, 7, 8, 10]
```

**Adjust Quantization Tempos:**

```python
# In handle_events(), T key handler
tempos = [60, 70, 80, 90, 100, 120, 140, 160, 180, 200]  # Add more options
```

**Change Live Note Interval:**

```python
self.live_note_interval = 0.1  # Faster response (default 0.15)
```

**Adjust Minimum Segment Duration:**

```python
# In start_playback(), minimum duration filter section
min_duration_threshold = 0.2  # Increase to skip more short notes (default 0.15)
```

---

## 15. Known Limitations

### Technical Limitations

**SoundFont Dependency**
- Requires external .sf2 file (100-200MB)
- Audio quality depends on SoundFont quality
- Limited by available SoundFont samples

**Platform Audio Drivers**
- Linux: Requires ALSA
- macOS: Requires CoreAudio
- Windows: May require ASIO for lowest latency

**Memory Usage**
- Large canvases with many traces consume more RAM
- Each trace stores all point coordinates
- No optimization for very complex drawings

**No Audio Recording**
- Cannot record playback directly to WAV/MP3
- Would require audio capture/encoding system
- MIDI export available as alternative

### Musical Limitations

**Monophonic During Drawing**
- Only one voice active while drawing
- Polyphony available only in playback
- Intentional design for gestural clarity

**Segment Detection Sensitivity**
- Horizontal lines (<35px Y variation) won't trigger multi-note
- Very short traces need larger pitch variation
- Trade-off between sensitivity and accidental splits

**Middle Segment Duration Filter**
- Very short middle segments (< 0.15s) automatically skipped
- Can cause unexpected gaps in very fast gestural phrases
- First and last notes always preserved
- Trade-off between clarity and complete phrase capture

**Quantization Affects All Notes**
- Cannot mix quantized and free timing in same playback
- Global setting only
- Either all notes snap to grid or none do

**Reading Speed Fixed Per Canvas**
- In free mode: Speed adapts to canvas width automatically
- In quantized mode: Speed based on BPM
- No manual tempo slider during playback

**No Manual Note Editing**
- Cannot move/resize individual notes after drawing
- Cannot split/merge segments manually post-creation
- Eraser removes entire traces, not segments

**Pan Applies Globally**
- Pan toggle affects all traces during playback
- Pan state saved per-trace but global toggle overrides
- Cannot have mixed pan/mono traces in same playback

**Scale Lock Quantization**
- Quantizes to nearest scale note when enabled
- Cannot play chromatic passing tones in scale-lock mode
- Must disable scale-lock for full chromatic access

**No Articulation Control**
- No staccato/legato beyond duration calculation
- No vibrato or pitch bend during playback
- Limited to SoundFont sample characteristics

**MIDI Export Limitations**
- Drums (channel 9) not included
- Stereo pan not exported
- Single instrument per file
- No instrument changes tracked

### UI Limitations

**Fixed Window Size**
- Resolution locked at 1200×760
- Not responsive/resizable
- May not fit all screen sizes

**No Zoom/Pan Canvas**
- Cannot zoom in/out on drawing area
- Cannot pan/scroll large compositions
- Limited to visible screen area

**Help Overlay Coverage**
- Help panel may obscure some drawing area when visible
- Toggle off (**H** key) for full canvas access

---

## 16. Future Enhancements

### High Priority Features

**Save/Load System**
- Export compositions as JSON (traces + settings)
- Import previously saved compositions
- Session management with metadata
- Preserve instrument selections and settings

**Enhanced MIDI Export**
- Include drum tracks (channel 9)
- Multi-track export with instrument changes
- Export stereo pan as MIDI CC
- Export quantization settings as MIDI time signature

**Audio Export**
- Record playback to WAV/MP3 format
- Real-time or faster-than-realtime rendering
- Configurable quality settings (sample rate, bit depth)
- Direct FluidSynth audio capture

**Selection Tool**
- Click to select individual traces
- Drag to move selected traces
- Multi-select with Shift/Ctrl
- Delete selection, copy/paste

### Medium Priority Features

**Advanced Quantization**
- Swing/shuffle timing options
- Triplet subdivisions
- Custom time signatures (3/4, 6/8, 7/8, etc.)
- Per-note quantization override

**Temporal Grid Enhancements**
- Visual beat numbers
- Measure markers
- Configurable grid colors
- Grid snap strength (magnetic strength)

**Loop Region Markers**
- Define custom loop start/end points
- Not just entire canvas but specific sections
- Visual markers showing loop boundaries
- Multiple loop regions

**Per-Trace Settings**
- Individual pan override per trace
- Volume/velocity multiplier per trace
- Solo/mute functionality
- Lock/unlock traces from editing

**Velocity Curves**
- Non-linear velocity mapping options
- Customizable response curves
- Visual curve editor
- Pressure sensitivity for tablet/stylus

**Manual Tempo Control**
- Speed slider for playback
- Real-time tempo changes
- Metronome click track
- Tempo tap feature

### Low Priority Features

**Preset Management**
- Save custom instrument preset configurations
- Share presets between sessions
- Community preset library
- Import/export preset files

**Multi-Layer System**
- Separate layers with different instruments
- Show/hide layers
- Solo/mute per layer
- Layer opacity control

**Effects Processing**
- Reverb amount control
- Delay/echo effect
- Chorus depth
- Filter cutoff/resonance
- Effects per-layer or global

**Advanced Segment Editing**
- Split segments manually with click
- Merge segments with drag
- Adjust segment MIDI note after creation
- Visual segment editor overlay

**Drawing Velocity Sensing**
- Faster drawing = louder/brighter notes
- Tablet pressure sensitivity
- Configurable velocity curves
- Visual velocity feedback

**Gesture Recognition**
- Detect specific drawing patterns
- Tremolo via rapid motion
- Vibrato via wavy lines
- Glissando recognition

**Collaboration Features**
- Real-time multi-user drawing
- Network sync of compositions
- Comment/annotation system
- Version history

**Accessibility**
- Keyboard-only drawing mode
- Screen reader support
- High contrast themes
- Configurable key bindings

---

## 17. Troubleshooting

### No Sound Output

**Check:**
- SoundFont file exists at specified path
- System audio drivers installed (ALSA/CoreAudio)
- FluidSynth installed: `fluidsynth --version`
- Volume not at 0 (press Up Arrow/= to increase base velocity)
- Active notes counter shows notes playing

**Solutions:**

```bash
# Verify FluidSynth installation
python -c "import fluidsynth; print('FluidSynth OK')"

# Test SoundFont loading
python -c "import fluidsynth; s = fluidsynth.Synth(); s.sfload('FluidR3_GM.sf2')"

# Check audio output device
# Linux: aplay -l
# macOS: system_profiler SPAudioDataType
```

### Crackling/Glitchy Audio

**Solutions:**
- Increase FluidSynth buffer size (edit FluidSynth settings)
- Close other audio applications
- Update audio drivers
- Use dedicated audio interface instead of onboard audio
- Reduce polyphony if many simultaneous notes

### Traces Not Drawing

**Check:**
- Mouse button held while dragging
- Pygame window has focus (click window first)
- Not in eraser mode (press **E** to toggle off)
- No keyboard modifiers interfering
- Sufficient system resources (RAM/CPU)

**Solutions:**
- Restart application
- Check console for error messages
- Verify Pygame installation: `python -c "import pygame; print(pygame.ver)"`

### Wrong Instrument Playing

**Verify:**
- Correct key pressed (1-9)
- Check console output for instrument confirmation
- Current preset shown in UI matches expectation
- SoundFont contains requested instrument (some SoundFonts vary)

**Solutions:**
- Cycle through instruments to verify all work
- Try different SoundFont if specific instrument sounds wrong
- Check instrument preview plays when selecting (should hear note)

### Multi-Note Detection Not Working

**Symptoms:**
- Long trace with pitch variation plays as single note
- Expected melodic phrase not segmented

**Check:**
- Total Y variation ≥35px (visual vertical range)
- Console shows segment detection messages
- Trace has clear direction changes (peaks/valleys)

**Solutions:**

```python
# Reduce thresholds in extract_note_segments()
y_range_threshold = 30  # More sensitive (default 35)
window = 3  # Less smoothing (default 5)
min_y_change = 20  # Less change needed (default 25)
```

### Playback Issues

**Symptoms:**
- First note not playing
- Notes in wrong order
- All notes play at once

**Check:**
- Console shows "Playing immediate event" for t=0 notes
- Playback space and reading speed values reasonable
- Segment times calculated correctly (debug prints)

**Solutions:**
- Clear canvas (**C**) and redraw
- Restart application if playback stuck
- Check for overlapping traces (may cause simultaneous playback)
- Disable quantization (**Q**) if timing seems off

### Eraser Not Working

**Check:**
- Eraser mode enabled (**E** key pressed, status shows "Eraser: ON")
- Red circle cursor visible
- Clicking/dragging near traces (within 30px)
- Traces actually exist on canvas

**Solutions:**
- Toggle eraser off and on again (**E** key twice)
- Increase eraser radius in code if needed
- Use Undo (**Z**) as alternative
- Restart application if completely stuck

### Undo/Redo Not Working

**Symptoms:**
- Undo (**Z**) says "Nothing to undo"
- Redo (**Y**) says "Nothing to redo"

**Check:**
- History was saved before action (drawing/erasing)
- Redo stack cleared after new action (expected behavior)

**Note:**
- Drawing saves history automatically when finalizing shape
- Erasing saves history before removing shapes
- New actions clear redo stack (standard behavior)

### Performance Issues

**Symptoms:**
- Low frame rate (<30 FPS)
- Lag when drawing
- Slow playback

**Solutions:**
- Reduce number of traces on canvas (use **C** to clear)
- Close other applications
- Use faster computer/dedicated GPU
- Reduce window size if possible
- Optimize trace point count (don't draw too slowly)
- Disable grid display (**G**) when not needed

### MIDI Export Not Working

**Symptoms:**
- **X** key does nothing
- Error message about midiutil

**Check:**
- midiutil installed: `pip install midiutil`
- Console shows export confirmation or error
- File permissions for writing

**Solutions:**

```bash
# Install midiutil
pip install midiutil

# Verify installation
python -c "from midiutil import MIDIFile; print('OK')"

# Check write permissions in current directory
touch test.mid && rm test.mid
```

### Quantization Not Aligning

**Symptoms:**
- Grid visible but notes don't snap
- Timing still feels free

**Check:**
- Quantization enabled (**Q** key pressed)
- Status shows "Quantize: 1/16 @ 80BPM" not "Quantize: OFF"
- Grid lines visible (**G** key for grid display)

**Solutions:**
- Toggle quantization off and on (**Q** key twice)
- Try different division (**D** key)
- Adjust tempo (**T** key) to hear difference
- Clear and redraw traces (quantization applies at playback start)

---

## 18. Credits & License

### Development

**Developed by:** Julie Le Rudulier
**Version:** 2.16.1 (Musical Edition)  
**Date:** October 2025  
**Language:** Python 3.8+

### Dependencies

**Pygame** (LGPL)
- 2D graphics and input handling
- Cross-platform windowing
- https://www.pygame.org

**NumPy** (BSD)
- Mathematical operations
- Array processing
- https://numpy.org

**FluidSynth** (LGPL)
- Real-time software synthesizer
- SoundFont rendering
- https://www.fluidsynth.org

**pyFluidSynth** (LGPL)
- Python bindings for FluidSynth
- https://github.com/nwhitehead/pyfluidsynth

**MIDIUtil** (MIT)
- MIDI file generation
- https://github.com/MarkCWirt/MIDIUtil

### SoundFont

**FluidR3_GM** by Frank Wen (MIT License)
- General MIDI SoundFont
- 142MB, 128 instruments
- https://member.keymusician.com/Member/FluidR3_GM/

### License

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

For questions, bug reports, or contributions, please visit the project repository.

---

## 19. Appendices

### Appendix A: Code Architecture Diagram

```
GeometricSynth
│
├── __init__()
│   ├── Initialize Pygame
│   ├── Create AudioEngine
│   ├── Create ShapeAnalyzer
│   ├── Load instrument presets
│   └── Initialize UI state
│
├── run() [Main Loop @ 60 FPS]
│   ├── handle_events()
│   │   ├── Keyboard input
│   │   ├── Mouse input (drawing/eraser)
│   │   ├── Instrument selection
│   │   ├── Tempo/quantization controls
│   │   └── Export trigger
│   │
│   ├── update()
│   │   ├── audio.update() [Note lifecycle]
│   │   └── Playback logic
│   │
│   └── draw()
│       ├── Draw grid (if enabled)
│       ├── Draw traces
│       ├── Draw playhead
│       ├── Draw eraser cursor
│       └── Draw UI overlay
│
├── Drawing System
│   ├── start_live_sound()
│   ├── update_live_sound()
│   ├── stop_live_sound()
│   └── finalize_shape()
│
├── Playback System
│   ├── start_playback()
│   │   ├── Sort traces by X position
│   │   ├── Extract multi-note segments
│   │   ├── Calculate timing events
│   │   ├── Apply quantization (if enabled)
│   │   └── Force immediate t=0 events
│   │
│   ├── pause_playback()
│   ├── stop_playback()
│   └── play_shape()
│
├── Editing System
│   ├── undo()
│   ├── redo()
│   ├── clear_all()
│   └── erase_at_position()
│
├── Export System
│   └── export_to_midi()
│       ├── Calculate timing
│       ├── Extract notes
│       ├── Create MIDI file
│       └── Write to disk
│
├── Tempo/Quantization System
│   ├── get_beat_duration()
│   ├── get_quantize_step()
│   └── draw_grid()
│
└── UI System
    ├── draw_ui()
    ├── draw_help_overlay()
    └── get_shape_color()

AudioEngine
│
├── __init__()
│   ├── Initialize FluidSynth
│   ├── Load SoundFont
│   └── Setup drum channel (9)
│
├── play_note()
│   ├── Select channel (melodic/drum)
│   ├── Select program/instrument
│   ├── Apply panning (CC 10)
│   ├── Trigger note on
│   └── Track in active_notes{}
│
├── stop_note_by_id()
├── update() [Check expired notes]
├── set_instrument()
└── stop_all_notes()

ShapeAnalyzer
│
├── __init__()
│   ├── Set screen dimensions
│   ├── Initialize scales
│   └── Set velocity_base
│
├── shape_to_midi()
│   ├── Y position → MIDI note
│   └── Apply scale quantization
│
├── shape_to_velocity()
│   └── Length → velocity with curve
│
├── shape_to_duration()
│   └── Length → duration (0.1-3.0s)
│
├── shape_to_pan()
│   └── X position → pan (0.0-1.0)
│
├── snap_to_scale()
│   └── Quantize MIDI to key signature
│
├── set_key_signature()
│   └── Update current scale
│
└── extract_note_segments()
    ├── Calculate trace length
    ├── Detect Y-range (<35px = horizontal)
    ├── Smooth Y values (window=5)
    ├── Find direction changes (peaks/valleys)
    ├── Create segments at extrema
    ├── Calculate MIDI per segment
    ├── Merge last two if same MIDI
    └── Filter consecutive duplicates
```

### Appendix B: Musical Scales Reference

**Supported Key Signatures**

**C Major:**
```
Notes: C  D  E  F  G  A  B  C
Steps:   W  W  H  W  W  W  H
MIDI:  0  2  4  5  7  9  11 (12)
```

**G Major:**
```
Notes: G  A  B  C  D  E  F# G
Steps:   W  W  H  W  W  W  H
MIDI:  0  2  4  6  7  9  11 (12)
Key signature: F#
```

**D Major:**
```
Notes: D  E  F# G  A  B  C# D
Steps:   W  W  H  W  W  W  H
MIDI:  1  2  4  6  7  9  11 (13)
Key signature: F#, C#
```

**F Major:**
```
Notes: F  G  A  Bb C  D  E  F
Steps:   W  W  H  W  W  W  H
MIDI:  0  2  3  5  7  9  10 (12)
Key signature: Bb
```

**A Minor (Natural):**
```
Notes: A  B  C  D  E  F  G  A
Steps:   W  H  W  W  H  W  W
MIDI:  0  2  3  5  7  8  10 (12)
Relative major: C Major
```

**E Minor (Natural):**
```
Notes: E  F# G  A  B  C  D  E
Steps:   W  H  W  W  H  W  W
MIDI:  0  2  3  5  7  9  10 (12)
Key signature: F#
Relative major: G Major
```

**Scale Degrees:**
```
1st  - Tonic (root)
2nd  - Supertonic
3rd  - Mediant
4th  - Subdominant
5th  - Dominant
6th  - Submediant
7th  - Leading tone
8th  - Octave (tonic)
```

### Appendix C: MIDI Reference

**General MIDI Instrument Families**

```
Piano (0-7):
  0 = Acoustic Grand Piano
  1 = Bright Acoustic Piano
  4 = Electric Piano 1
  5 = Electric Piano 2

Chromatic Percussion (8-15):
  8 = Celesta
  11 = Vibraphone
  12 = Marimba

Organ (16-23):
  16 = Drawbar Organ
  18 = Rock Organ
  19 = Church Organ

Guitar (24-31):
  24 = Acoustic Guitar (nylon)
  25 = Acoustic Guitar (steel)
  27 = Electric Guitar (clean)
  30 = Distortion Guitar

Bass (32-39):
  32 = Acoustic Bass
  33 = Electric Bass (finger)
  36 = Slap Bass 1

Strings (40-47):
  40 = Violin
  42 = Cello
  46 = Orchestral Harp
  48 = String Ensemble 1

Ensemble (48-55):
  52 = Choir Aahs

Brass (56-63):
  56 = Trumpet
  57 = Trombone
  60 = French Horn
  61 = Brass Section

Reed (64-71):
  65 = Alto Sax
  66 = Tenor Sax
  68 = Oboe
  71 = Clarinet

Pipe (72-79):
  73 = Flute
  75 = Pan Flute
  79 = Ocarina

Synth Lead (80-87):
  80 = Lead 1 (square)
  81 = Lead 2 (sawtooth)

Synth Pad (88-95):
  89 = Pad 2 (warm)
  90 = Pad 3 (polysynth)

Ethnic (104-111):
  104 = Sitar
  106 = Shamisen
  107 = Koto
  108 = Kalimba
```

**Drum Kit (Channel 9)**

```
35 = Acoustic Bass Drum
36 = Bass Drum 1
38 = Acoustic Snare
40 = Electric Snare
42 = Closed Hi-Hat
44 = Pedal Hi-Hat
46 = Open Hi-Hat
49 = Crash Cymbal 1
51 = Ride Cymbal 1
54 = Tambourine
56 = Cowbell
60 = Hi Bongo
61 = Low Bongo
63 = Open High Conga
64 = Low Conga
65 = High Timbale
66 = Low Timbale
67 = High Agogo
69 = Cabasa
70 = Maracas
75 = Claves
76 = Hi Wood Block
77 = Low Wood Block
81 = Open Triangle
82 = Mute Triangle
```

### Appendix D: Quantization Reference

**Time Signatures & Divisions**

```
Division    Note Value    Beats/Bar (4/4)    Example Use
---------------------------------------------------------------
1/2         Half note     2                  Slow tempo marking
1/4         Quarter       4                  Basic beat
1/8         Eighth        8                  Faster passages
1/16        Sixteenth     16                 Detailed timing (default)
1/32        Thirty-second 32                 Very fast articulation
```

**BPM to Beat Duration Conversion**

```
BPM    Beat Duration    16th Note Duration
----------------------------------------------
60     1.000s          0.250s
80     0.750s          0.188s (default)
100    0.600s          0.150s
120    0.500s          0.125s
140    0.429s          0.107s
160    0.375s          0.094s
180    0.333s          0.083s
```

**Grid Spacing Examples (at 80 BPM, 1/16 division)**

```
Pixels/Second = 200 / 0.75 ≈ 267
Grid Spacing = 0.1875s × 267 ≈ 50 pixels

Every 4th line = Quarter note beat (stronger line)
Every 16th line = Measure boundary (if implemented)
```

---

**End of Technical Documentation**

**Total Pages:** ~40  
**Word Count:** ~15,000  
**Last Updated:** October 2025  
**Version:** 2.16.1 (includes MIDI export, quantization, eraser with audio preview, enhanced undo/redo)
