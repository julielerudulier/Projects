import pygame
import numpy as np
import math
import time
import os
import sys

try:
    import fluidsynth
except ImportError:
    print("FluidSynth not installed!")
    print("Install with: pip install pyfluidsynth")
    sys.exit(1)

os.environ['SDL_AUDIODRIVER'] = 'coreaudio'

# ======================
# CONFIGURATION
# ======================
SOUNDFONT_PATH = "FluidR3_GM.sf2"

if not os.path.exists(SOUNDFONT_PATH):
    print("SoundFont not found:", SOUNDFONT_PATH)
    print("Download 'FluidR3_GM.sf2' and place it in this folder.")
    print("Or change SOUNDFONT_PATH to point to your .sf2 file")
    sys.exit(1)

# ======================
# AUDIO ENGINE
# ======================
class AudioEngine:
    """Manages FluidSynth and active notes"""
    
    def __init__(self, soundfont_path, parent_synth=None):
        self.synth = fluidsynth.Synth()
        self.synth.start(driver="alsa" if sys.platform.startswith("linux") else "coreaudio")
        
        self.sfid = self.synth.sfload(soundfont_path)
        try:
            # Bank 128 = percussion kits, Program 0 = Standard Kit
            self.synth.program_select(9, self.sfid, 128, 0)
            print("Percussion channel initialized with Standard Kit")
        except Exception as e:
            print("Could not init drum channel:", e)
            
        self.parent_synth = parent_synth  # Reference to main synth for preset access
        
        # Active notes tracking
        self.active_notes = {}
        self.note_id_counter = 0
        
        # Current instrument override (None = use shape mapping)
        self.current_instrument = None
        # Current drum MIDI key (when using drum mode) — e.g. 35 = Acoustic Bass Drum
        self.current_drum_note = None
        
        print("Audio Engine ready (FluidSynth active)")
        
    def set_instrument(self, program_id=None, drum_note=None):
        """
        Set current instrument program (int) for melodic channels.
        If drum_note is provided, it will be stored in current_drum_note.
        None = fallback to first instrument of current preset.
        """
        self.current_instrument = program_id
        if drum_note is not None:
            self.current_drum_note = drum_note

    def play_note(self, midi_note, velocity, duration, pan=0.5, instrument=None, is_drum=False):
        """Play a note (melodic or drum) with given parameters and return note_id."""
        # Choose channel
        if is_drum:
            channel = 9  # percussion channel (0-based index)
        else:
            channel = 0

        # If this is a drum hit, decide which MIDI key to trigger (drum key numbers)
        if is_drum:
            # instrument argument may be used to pass a specific drum MIDI key
            if instrument is not None:
                midi_to_play = int(instrument)
            elif getattr(self, "current_drum_note", None) is not None:
                midi_to_play = int(self.current_drum_note)
            else:
                # fallback to provided midi_note (not ideal, but safe)
                midi_to_play = int(midi_note) # fallback
        else:
            # melodic path: decide program to select
            if instrument is not None:
                program = instrument
            elif self.current_instrument is not None:
                program = self.current_instrument
            else:
                # fallback (parent_synth may provide quick_instruments)
                program = self.parent_synth.quick_instruments[0][1] if self.parent_synth else 0

            # Select program for melodic channel only
            try:
                self.synth.program_select(channel, self.sfid, 0, program)
            except Exception:
                pass

            midi_to_play = int(midi_note)

        # Pan (control change) on same channel
        pan_value = int(max(0, min(127, pan * 127)))
        try:
            self.synth.cc(channel, 10, pan_value)
        except Exception:
            pass

        # Note on
        self.synth.noteon(channel, midi_to_play, velocity)

        # Track active note
        note_id = self.note_id_counter
        self.note_id_counter += 1

        self.active_notes[note_id] = {
            'midi': midi_to_play,
            'start_time': time.time(),
            'duration': duration,
            'channel': channel,
            'is_drum': is_drum
        }

        return note_id

    def stop_note_by_id(self, note_id):
        """Stop a specific note previously started by play_note."""
        info = self.active_notes.get(note_id)
        if not info:
            return
        try:
            self.synth.noteoff(info['channel'], info['midi'])
        except Exception:
            pass
        # remove from tracking
        del self.active_notes[note_id]
        
    def update(self):
        """Update active notes (stop expired ones)"""
        current_time = time.time()
        notes_to_remove = []
        
        for note_id, note_info in self.active_notes.items():
            elapsed = current_time - note_info['start_time']
            if elapsed >= note_info['duration']:
                # Stop note
                self.synth.noteoff(note_info['channel'], note_info['midi'])
                notes_to_remove.append(note_id)
                
        # Remove stopped notes
        for note_id in notes_to_remove:
            del self.active_notes[note_id]
            
    def stop_all_notes(self):
        """Stop all currently playing notes"""
        for note_info in self.active_notes.values():
            self.synth.noteoff(note_info['channel'], note_info['midi'])
        self.active_notes.clear()
        
    def cleanup(self):
        """Clean shutdown"""
        self.stop_all_notes()
        self.synth.delete()
        print("Audio Engine cleaned up")


# ======================
# SHAPE ANALYZER
# ======================
class ShapeAnalyzer:
    """Converts shapes to audio parameters"""
    
    def set_key_signature(self, key_name):
        """Set current key signature for scale lock"""
        if key_name in self.parent_synth.key_signatures:
            self.scale_lock_scale = self.parent_synth.key_signatures[key_name]
            self.parent_synth.current_key = key_name
            
    def __init__(self, screen_width, screen_height, parent_synth=None):
        self.width = screen_width
        self.height = screen_height
        self.parent_synth = parent_synth
        
        # Musical scales
        self.c_major_scale = [0, 2, 4, 5, 7, 9, 11]  # Semitones in C major
        self.scale_lock = True
        self.scale_lock_scale = self.c_major_scale  # Current scale for locking
        
        # MIDI note range mapping
        self.min_midi = 36   # C2 (low)
        self.max_midi = 96   # C7 (high)

    def shape_to_midi(self, shape):
        """Convert shape Y position to MIDI note"""
        # Y position determines pitch (inverted: top = high, bottom = low)
        y_pos = shape['center'][1]
        y_normalized = 1.0 - (y_pos / self.height)  # 0=bottom, 1=top
        
        # Map to MIDI range
        midi_float = self.min_midi + y_normalized * (self.max_midi - self.min_midi)
        midi_note = int(round(midi_float))
        
        # Apply scale lock if enabled
        if self.scale_lock:
            midi_note = self.snap_to_scale(midi_note)
            
        # Clamp to valid MIDI range
        return max(0, min(127, midi_note))
        
    def snap_to_scale(self, midi_note):
        """Snap MIDI note to current scale"""
        octave = midi_note // 12
        semitone = midi_note % 12
        
        # Find closest note in current scale
        closest = min(self.scale_lock_scale, key=lambda x: abs(x - semitone))
        
        return octave * 12 + closest
        
    def shape_to_velocity(self, shape):
        """Convert shape properties to note velocity (volume)"""
        # Base velocity on size
        area = shape['width'] * shape['height']
        
        if shape['type'] == 'line':
            # Lines: based on length
            length = shape.get('length', 100)
            velocity = int(40 + min(87, length / 5))
        else:
            # Shapes: based on area
            velocity = int(50 + min(77, area / 500))
            
        return max(10, min(127, velocity))
        
    def shape_to_duration(self, shape):
        """Convert shape to note duration in seconds (optimized for responsiveness)"""
        if shape['type'] == 'line':
            length = shape.get('length', 100)
            # Small lines = very short, long lines = longer (max 1.5s)
            if length < 100:
                return max(0.1, min(0.3, length / 400))
            else:
                return max(0.3, min(1.5, length / 200))
        else:
            # Shapes: based on area, capped to avoid lingering sounds
            area = shape['width'] * shape['height']
            # Typical shapes = 0.4–2.5s
            return max(0.4, min(2.5, area / 10000))
            
    def shape_to_pan(self, shape):
        """Convert X position to stereo pan (0=left, 0.5=center, 1=right)"""
        x_pos = shape['center'][0]
        return 1.0 - (x_pos / self.width)


# ======================
# MAIN SYNTHESIZER
# ======================
class GeometricSynth:
    """Main application orchestrating audio and graphics"""
    
    def __init__(self):
        pygame.init()
        
        # Display
        self.width, self.height = 1200, 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Geometric Synthesizer V2")
        
        # Colors - Professional warm palette (lighter background)
        self.BACKGROUND = (250, 248, 240)      # Lighter warm white
        self.TERRA_COTTA = (224, 122, 95)      # Terra cotta #E07A5F
        self.SLATE_BLUE = (61, 64, 91)         # Deep slate blue #3D405B
        self.SAGE_GREEN = (129, 178, 154)      # Muted sage green #81B29A
        self.WARM_BROWN = (140, 95, 74)        # Warm brown
        self.DEEP_TEAL = (52, 108, 117)        # Deep teal
        self.PURPLISH = (150, 80, 150)         # Purplish (#965096)
        self.BURNT_ORANGE = (200, 90, 60)      # Burnt orange (darker)
        self.OLIVE_GREEN = (105, 123, 92)      # Olive green
        self.GOLDEN_OCHRE = (222, 193, 121)    # Golden ochre (light) #DEC179
        self.PLUM_VIOLET = (92, 76, 101)       # Earthy plum violet (#6F5777)
        self.TEXT_DARK = (40, 40, 50)          # Dark text
        self.UI_BG = (240, 238, 230)           # UI panel background (lighter)
        
        # Legacy colors for compatibility
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (100, 100, 100)
    
        # Drawing state
        self.drawing = False
        self.current_points = []
        self.shapes = []
        
        # UI state
        self.clock = pygame.time.Clock()
        self.running = True
        self.show_help = True  # Show help overlay
        self.live_drawing_mode = True  # True by default
        self.last_live_note_time = 0
        self.live_note_interval = 0.15  # 150ms between each live note
        self.active_live_note_id = None  # Track the continuous note
        self.last_live_pitch = None
        self.base_velocity = 50 # Base velocity (30-127), adjustable with +/-
        self.current_velocity = self.base_velocity
        
        # Instrument presets by musical style
        self.instrument_presets = {
            'classical': [
                ("Piano", 0, self.DEEP_TEAL),
                ("Violin", 40, self.GOLDEN_OCHRE), 
                ("Cello", 42, self.WARM_BROWN), 
                ("Oboe", 68, self.PURPLISH),
                ("Flute", 73, self.BURNT_ORANGE),
                ("Clarinet", 71, self.SAGE_GREEN),
                ("Trumpet", 66, self.PLUM_VIOLET),
                ("Harp", 46, self.OLIVE_GREEN),
                ("Vibraphone", 11, self.TERRA_COTTA)
            ],
            'jazz': [
                ("Jazz Guitar", 26, self.DEEP_TEAL),
                ("Slap Bass", 36, self.GOLDEN_OCHRE),
                ("Acoustic Bass", 32, self.WARM_BROWN),
                ("Electric Piano", 4, self.PURPLISH),
                ("Clarinet", 71, self.BURNT_ORANGE),
                ("Alto Sax", 65, self.SAGE_GREEN),
                ("Trumpet", 66, self.PLUM_VIOLET),
                ("Trombone", 57, self.OLIVE_GREEN),
                ("Organ", 16, self.TERRA_COTTA)  
            ],
            'rock': [
                ("Electric Guitar", 27, self.DEEP_TEAL),
                ("Distortion Guitar", 30, self.GOLDEN_OCHRE),
                ("Rhythmic Guitar", 28, self.WARM_BROWN),
                ("Acoustic Guitar", 24, self.PURPLISH),
                ("Electric Bass", 33, self.BURNT_ORANGE),
                ("Piano", 1, self.SAGE_GREEN),
                ("Rock Organ", 18, self.PLUM_VIOLET),
                ("Synth Lead", 80, self.OLIVE_GREEN),
                ("Synth Strings", 50, self.TERRA_COTTA)
            ],
            'electro': [
                ("Synth Bass 1", 38, self.DEEP_TEAL),
                ("Synth Bass 2", 39, self.GOLDEN_OCHRE),
                ("Lead 1", 81, self.WARM_BROWN),
                ("Lead 2", 84, self.PURPLISH),
                ("Pad 1", 89, self.BURNT_ORANGE),
                ("Pad 2", 90, self.SAGE_GREEN),
                ("Keys", 5, self.PLUM_VIOLET),
                ("FX", 97, self.OLIVE_GREEN),
                ("Pluck", 87, self.TERRA_COTTA)
            ],
            'latin': [
                ("Acoustic Guitar 1", 24, self.DEEP_TEAL),
                ("Acoustic Guitar 2", 25, self.GOLDEN_OCHRE),
                ("Electric Guitar", 4, self.WARM_BROWN),
                ("Acoustic Bass", 32, self.PURPLISH),
                ("Conga", 64, self.BURNT_ORANGE, True),
                ("Bongo", 60, self.SAGE_GREEN, True),
                ("Timbale", 65, self.PLUM_VIOLET, True),
                ("Maracas", 70, self.OLIVE_GREEN, True),
                ("Shaker", 69, self.TERRA_COTTA, True)
            ],
            'country': [
                ("Acoustic Guitar", 25, self.DEEP_TEAL),
                ("Electric Guitar", 27, self.GOLDEN_OCHRE),
                ("Pedal Steel Guitar", 34, self.WARM_BROWN),
                ("Fiddle", 40, self.PURPLISH),
                ("Harmonica", 22, self.BURNT_ORANGE),
                ("Banjo", 105, self.SAGE_GREEN),
                ("Organ", 17, self.PLUM_VIOLET),
                ("Electric Bass", 33, self.OLIVE_GREEN),
                ("Honky Tonk Piano", 3, self.TERRA_COTTA)
            ],
            'soul': [
                ("Piano", 4, self.DEEP_TEAL),
                ("Organ", 16, self.GOLDEN_OCHRE),
                ("Electric Bass", 33, self.WARM_BROWN),
                ("Electric Guitar", 27, self.PURPLISH),
                ("Alto Sax", 65, self.BURNT_ORANGE),
                ("Tenor Sax", 66, self.SAGE_GREEN),
                ("Trumpet", 56, self.PLUM_VIOLET),
                ("Choir", 52, self.OLIVE_GREEN),
                ("Strings Ensemble", 48, self.TERRA_COTTA)
            ],
            'world': [
                ("Sitar", 104, self.DEEP_TEAL),
                ("Shamisen", 106, self.GOLDEN_OCHRE),
                ("Koto", 107, self.WARM_BROWN),
                ("Kalimba", 108, self.PURPLISH),
                ("Bagpipe", 109, self.BURNT_ORANGE),
                ("Pan Flute", 75, self.SAGE_GREEN),
                ("Ocarina", 79, self.PLUM_VIOLET),
                ("Shakuhachi", 77, self.OLIVE_GREEN),
                ("Marimba", 12, self.TERRA_COTTA)
            ],
            'drum kit': [
                ("Kick Drum", 35, self.DEEP_TEAL, True),
                ("Snare Drum", 38, self.GOLDEN_OCHRE, True),
                ("Closed Hi-Hat", 42, self.WARM_BROWN, True),
                ("Open Hi-Hat", 46, self.PURPLISH, True),
                ("Crash Cymbal", 49, self.BURNT_ORANGE, True),
                ("Ride Cymbal", 51, self.SAGE_GREEN, True),
                ("Low Tom", 45, self.PLUM_VIOLET, True),
                ("High Tom", 50, self.OLIVE_GREEN, True),
                ("Tambourine", 54, self.TERRA_COTTA, True)
            ]
        }
            
        # Components
        self.current_preset = 'classical'
        self.quick_instruments = self.instrument_presets[self.current_preset]
        self.analyzer = ShapeAnalyzer(self.width, self.height, parent_synth=self)
        self.audio = AudioEngine(SOUNDFONT_PATH, parent_synth=self)
        #self.audio.set_instrument(self.quick_instruments[0][1]) # Set default instrument to first of classical preset (Piano)
        
        # Track which instrument index is selected (0-based)
        self.selected_instrument_index = 0
        # initial selection (handle drum entries which have a fourth True flag)
        name0, program0, color0, *rest0 = self.quick_instruments[0]
        is_drum0 = rest0[0] if rest0 else False
        self.current_instrument_is_drum = is_drum0
        if is_drum0:
            # program0 here is actually the drum MIDI key
            self.audio.set_instrument(None, drum_note=program0)
        else:
            self.audio.set_instrument(program0)

        # Current key signature (for scale lock)
        self.key_signatures = {
            'C': [0, 2, 4, 5, 7, 9, 11],      # C major
            'G': [0, 2, 4, 6, 7, 9, 11],      # G major
            'D': [0, 2, 4, 6, 7, 9, 11],      # D major  
            'F': [0, 2, 3, 5, 7, 9, 10],      # F major
            'Am': [0, 2, 3, 5, 7, 8, 10],     # A minor
            'Em': [0, 2, 3, 5, 7, 8, 10],     # E minor
        }
        self.current_key = 'C'

        self.analyzer.set_key_signature(self.current_key)
        
        # Undo/Redo stacks
        self.shapes_history = []  # Stack for undo
        self.shapes_redo = []     # Stack for redo
        
        print("\n" + "="*60)
        print("GEOMETRIC SYNTHESIZER V2 - MUSICAL")
        print("="*60)
        print("\nDRAWING:")
        print("  • Click & drag to draw shapes")
        print("  • Piano is the active instrument by default")
        print("\nINSTRUMENT SELECTION:")
        print("  • Keys 1-9: Select instrument from current preset")
        print("  • Key N: Select next preset of instruments (Classical→Jazz→Rock→World)")
        print("  • Key P: Select previous preset of instruments")
        print(f"  • Current preset: {self.current_preset.upper()}")
        print("\nMUSICAL CONTROLS:")
        print("  • M: Toggle scale lock (quantize to scale)")
        print("  • K: Cycle key signature (C, G, D, F, Am, Em)")
        print("  • +/-: Increase/decrease velocity")
        print("\nEDITING:")
        print("  • Z: Undo last drawing")
        print("  • Y: Redo last drawing")
        print("  • C: Clear all shapes from canvas")
        print("\nSPATIAL MAPPING:")
        print("  • Y position → Pitch (top=high notes, bottom=low notes)")
        print("  • X position → Stereo pan (left=L speaker, right=R speaker)")
        print("  • Shape size → Duration & velocity (quicker=louder)")
        print("\nOTHER CONTROLS:")
        print("  • S: Stop all currently playing notes")
        print("  • H: Toggle help display on screen")
        print("  • ESC or Q: Quit application")
        print("="*60 + "\n")
        
    def run(self):
        """Main loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
            
        self.cleanup()
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                    
                elif event.key == pygame.K_c:
                    self.clear_all()
                    
                elif event.key == pygame.K_s:
                    self.audio.stop_all_notes()
                    print("Stopped all notes")
                    
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                    
                elif event.key == pygame.K_m:
                    self.analyzer.scale_lock = not self.analyzer.scale_lock
                    status = "ON" if self.analyzer.scale_lock else "OFF"
                    print(f"Scale lock: {status} ({self.current_key})")
                    
                elif event.key == pygame.K_k:
                    # Cycle through key signatures
                    keys = list(self.key_signatures.keys())
                    current_idx = keys.index(self.current_key)
                    next_idx = (current_idx + 1) % len(keys)
                    self.current_key = keys[next_idx]
                    self.analyzer.set_key_signature(self.current_key)
                    print(f"Key signature: {self.current_key}")
                    
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.base_velocity = min(127, self.base_velocity + 10)
                    print(f"Base velocity: {self.base_velocity}")
                    
                elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                    self.base_velocity = max(10, self.base_velocity - 10)
                    print(f"Base velocity: {self.base_velocity}")
            
                elif event.key == pygame.K_n:
                    # Cycle through presets 
                    preset_order = ['classical', 'jazz', 'rock', 'electro', 'latin', 'country', 'soul', 'world', 'drum kit']
                    current_idx = preset_order.index(self.current_preset)
                    next_idx = (current_idx + 1) % len(preset_order)
                    self.current_preset = preset_order[next_idx]
                    self.quick_instruments = self.instrument_presets[self.current_preset]
                    # reset selected index and set instrument/drum properly
                    self.selected_instrument_index = 0
                    name0, program0, color0, *rest0 = self.quick_instruments[0]
                    is_drum0 = rest0[0] if rest0 else False
                    self.current_instrument_is_drum = is_drum0
                    if is_drum0:
                        self.audio.set_instrument(None, drum_note=program0)
                    else:
                        self.audio.set_instrument(program0)
                    print(f"Preset: {self.current_preset.upper()}")

                elif event.key == pygame.K_p:
                    # Cycle through presets in reverse
                    preset_order = ['classical', 'jazz', 'rock', 'electro', 'latin', 'country', 'soul', 'world', 'drum kit']
                    current_idx = preset_order.index(self.current_preset)
                    prev_idx = (current_idx - 1) % len(preset_order)
                    self.current_preset = preset_order[prev_idx]
                    self.quick_instruments = self.instrument_presets[self.current_preset]
                    # reset selected index and set instrument/drum properly
                    self.selected_instrument_index = 0
                    name0, program0, color0, *rest0 = self.quick_instruments[0]
                    is_drum0 = rest0[0] if rest0 else False
                    self.current_instrument_is_drum = is_drum0
                    if is_drum0:
                        self.audio.set_instrument(None, drum_note=program0)
                    else:
                        self.audio.set_instrument(program0)
                    print(f"Preset: {self.current_preset.upper()}")
                               
                # Instrument selection (1-9)
                idx = event.key - pygame.K_1
                # Only accept keys 1..9 (guard against negatives)
                if 0 <= idx < len(self.quick_instruments):
                    entry = self.quick_instruments[idx]
                    name, program, color, *rest = entry
                    is_drum = rest[0] if rest else False  

                    # If already selected, do nothing (no toggle)
                    if getattr(self, "selected_instrument_index", None) == idx:
                        # already selected — ignore
                        pass
                    else:
                        # apply selection
                        self.selected_instrument_index = idx
                        self.current_instrument_is_drum = is_drum

                        if is_drum:
                            # program here is actually the drum MIDI key (e.g. 35)
                            self.audio.set_instrument(None, drum_note=program)
                        else:
                            # melodic program id
                            self.audio.set_instrument(program)
                            # clear any drum note
                            self.audio.current_drum_note = None

                    print(f"Instrument: {name} {'(Drum Kit)' if is_drum else ''}")

                # Undo/Redo
                elif event.key == pygame.K_z:
                    self.undo()
                elif event.key == pygame.K_y:
                    self.redo()
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.drawing = True
                    self.current_points = [event.pos]
                    
                    # Start continuous live note
                    if self.live_drawing_mode:
                        self.start_live_sound(event.pos)
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.drawing:
                    # Stop continuous live note
                    if self.live_drawing_mode and self.active_live_note_id is not None:
                        self.stop_live_sound()
                    
                    self.drawing = False
                    if len(self.current_points) > 1:
                        self.finalize_shape()
                    self.current_points = []
                    
            elif event.type == pygame.MOUSEMOTION:
                if self.drawing:
                    self.current_points.append(event.pos)
                    
                    # Continuous live sound during drawing
                    if self.live_drawing_mode:
                        self.update_live_sound(event.pos)
                    
    def finalize_shape(self):
        """Analyze and save the drawn shape (no sound)"""
        shape_type = 'freeform'

        # Calculate geometry
        x_coords = [p[0] for p in self.current_points]
        y_coords = [p[1] for p in self.current_points]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)

        shape = {
            'type': shape_type,
            'center': (center_x, center_y),
            'width': width,
            'height': height,
            'points': self.current_points.copy(),
            'timestamp': time.time(),
            'color': self.get_shape_color(shape_type)
        }

        if shape_type == 'line':
            shape['length'] = math.sqrt(width**2 + height**2)

        last_point = self.current_points[-1]
        shape['note_position'] = last_point
        self.shapes.append(shape)

        # Clear redo stack when new shape is added
        if self.shapes_redo:
            self.shapes_redo = []
            print("Redo history cleared")

    def start_live_sound(self, pos):
        """Start a continuous note when drawing begins (stop existing first)."""
        # Safety: Force stop any remaining live note
        if self.active_live_note_id is not None:
            self.audio.stop_note_by_id(self.active_live_note_id)
            self.active_live_note_id = None
        
        # Initialize velocity tracking
        self.last_draw_pos = pos
        self.last_draw_time = time.time()
        self.current_velocity = self.base_velocity  # Start with medium velocity
        
        temp_shape = {'type': 'line', 'center': pos, 'width': 10, 'height': 10, 'length': 10}
        midi_note = self.analyzer.shape_to_midi(temp_shape)
        velocity = self.base_velocity
        duration = 10.0 # Will be stopped manually
        pan = pos[0] / self.width

        instrument_param = (self.audio.current_drum_note if getattr(self, "current_instrument_is_drum", False)
                            else self.audio.current_instrument)

        note_id = self.audio.play_note(
            midi_note, 
            velocity, 
            duration, 
            pan,
            instrument=instrument_param,
            is_drum=getattr(self, "current_instrument_is_drum", False)
        )

        self.active_live_note_id = note_id
        self.last_live_pitch = midi_note

    def update_live_sound(self, pos):
        """Update pitch and velocity during continuous drawing"""
        current_time = time.time()
        
        # Calculate drawing speed for velocity
        if self.last_draw_pos and self.last_draw_time:
            distance = math.sqrt((pos[0] - self.last_draw_pos[0])**2 + 
                            (pos[1] - self.last_draw_pos[1])**2)
            time_delta = current_time - self.last_draw_time
            
            if time_delta > 0:
                speed = distance / time_delta
                # Map speed around base_velocity
                velocity_offset = int((speed / 7) - 30)
                target_velocity = self.base_velocity + velocity_offset
                target_velocity = max(10, min(127, target_velocity))
                
                # Smooth velocity changes (exponential moving average)
                smoothing = 0.3  # 0 = no smoothing, 1 = instant change
                self.current_velocity = int(
                self.current_velocity * (1 - smoothing) + target_velocity * smoothing
                )
        else:
            self.current_velocity = self.base_velocity
        
        # Store for next calculation
        self.last_draw_pos = pos
        self.last_draw_time = current_time
        
        temp_shape = {'type': 'line', 'center': pos, 'width': 10, 'height': 10, 'length': 10}
        midi_note = self.analyzer.shape_to_midi(temp_shape)
        
        # Only change note if pitch changed significantly
        if self.last_live_pitch is None or abs(midi_note - self.last_live_pitch) >= 1:
            if self.active_live_note_id is not None:
                self.stop_live_sound()
            
            # Start new note with calculated velocity
            velocity = self.current_velocity
            duration = 10.0
            pan = pos[0] / self.width
            
            instrument_param = (self.audio.current_drum_note if getattr(self, "current_instrument_is_drum", False)
                            else self.audio.current_instrument)

            note_id = self.audio.play_note(
                midi_note, 
                velocity, 
                duration, 
                pan,
                instrument=instrument_param,
                is_drum=getattr(self, "current_instrument_is_drum", False)
            )

            self.active_live_note_id = note_id
            self.last_live_pitch = midi_note

    def stop_live_sound(self):
        """Stop the continuous note"""
        if self.active_live_note_id is not None:
            # Use AudioEngine's method which handles everything
            self.audio.stop_note_by_id(self.active_live_note_id)
            
        self.active_live_note_id = None
        self.last_live_pitch = None
        # Reset velocity tracking
        self.last_draw_pos = None
        self.last_draw_time = 0
        
    def play_shape(self, shape):
        """Convert shape to sound and play it — returns note_id (or None)."""
        # Use last drawn point if available, otherwise center
        if 'note_position' in shape:
            note_pos_shape = {
                'type': shape['type'],
                'center': shape['note_position'],  # Use last point
                'width': shape['width'],
                'height': shape['height']
            }
            if 'length' in shape:
                note_pos_shape['length'] = shape['length']
        else:
            note_pos_shape = shape

        # Get audio parameters using last point position
        midi_note = self.analyzer.shape_to_midi(note_pos_shape)

        # Use last drawing velocity if available, otherwise calculate from shape
        velocity = getattr(self, 'current_velocity', None) or self.analyzer.shape_to_velocity(shape)

        duration = self.analyzer.shape_to_duration(shape) + 1

        # Pan based on last point X position
        if 'note_position' in shape:
            pan = shape['note_position'][0] / self.width
        else:
            pan = self.analyzer.shape_to_pan(shape)

        # Console feedback
        note_name = self.midi_to_note_name(midi_note)
        pan_str = "L" if pan < 0.4 else "R" if pan > 0.6 else "C"

        instrument_param = (self.audio.current_drum_note if getattr(self, "current_instrument_is_drum", False)
                            else self.audio.current_instrument)

        note_id = self.audio.play_note(
            midi_note, 
            velocity, 
            duration, 
            pan,
            instrument=instrument_param,
            is_drum=getattr(self, "current_instrument_is_drum", False)
        )


        print(f"{shape['type'].upper():10s} | {note_name:4s} | vel:{velocity:3d} | dur:{duration:.1f}s | pan:{pan_str}")
        return note_id
        
    def midi_to_note_name(self, midi):
        """Convert MIDI number to note name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi // 12) - 1
        note = notes[midi % 12]
        return f"{note}{octave}"
        
    def get_shape_color(self, shape_type):
        """Get color for shape type based on current instrument selection"""
        idx = getattr(self, "selected_instrument_index", 0)
        if 0 <= idx < len(self.quick_instruments):
            return self.quick_instruments[idx][2]
        return self.quick_instruments[0][2]

        
    def update(self):
        """Update game state"""
        # Update audio engine (stop finished notes)
        self.audio.update()
        
        # Safety: If live note expired naturally, reset tracking
        if self.active_live_note_id is not None:
            if self.active_live_note_id not in self.audio.active_notes:
                self.active_live_note_id = None
                self.last_live_pitch = None
        
    def undo(self):
        """Undo last shape (Ctrl+Z)"""
        if not self.shapes:
            print("Nothing to undo")
            return
        
        # Pop last shape and move to redo stack
        last_shape = self.shapes.pop()
        self.shapes_redo.append(last_shape)
        print(f"✓ Undo: {last_shape['type']}")

    def redo(self):
        """Redo last undone shape (Ctrl+Y)"""
        if not self.shapes_redo:
            print("Nothing to redo")
            return
        
        # Pop from redo stack and add back to shapes
        shape = self.shapes_redo.pop()
        self.shapes.append(shape)
        
        print(f"✓ Redo: {shape['type']}")

    def draw(self):
        """Draw everything"""
        self.screen.fill(self.BACKGROUND)  # Professional cream background
        
        # Draw all shapes
        for shape in self.shapes:
            self.draw_shape(shape)
            
        # Draw current shape being drawn
        if self.drawing and len(self.current_points) > 1:
            pygame.draw.lines(self.screen, self.TEXT_DARK, False, self.current_points, 3)
            
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
        
    def draw_shape(self, shape):
        """Draw a single shape"""
        color = shape['color']
        points = shape['points']
        
        # Check if still playing (glow effect)
        is_playing = shape.get('note_id') in self.audio.active_notes
        if is_playing:
            glow_color = tuple(min(255, c + 80) for c in color)
            pygame.draw.lines(self.screen, glow_color, False, points, 5)
            
        # Draw normal shape with thicker lines
        pygame.draw.lines(self.screen, color, False, points, 3)
        
    def draw_ui(self):
        """Draw user interface with professional layout"""
        font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 20)
        tiny_font = pygame.font.Font(None, 16)
        y_offset = 87
        preset_text = f"[{self.current_preset.upper()}]"
        preset_surface = small_font.render(preset_text, True, self.TERRA_COTTA)
        self.screen.blit(preset_surface, (10, 53))

        # ===== TOP LEFT: Title and status =====
        title = font.render("GEOMETRIC SYNTHESIZER V2", True, self.SLATE_BLUE)
        self.screen.blit(title, (10, 15))
        
        # Current instrument with highlighted color
        if self.audio.current_instrument is not None:
            instrument_name = None
            inst_color = self.TEXT_DARK
            for name, prog, color, *rest in self.quick_instruments:
                if prog == self.audio.current_instrument:
                    instrument_name = name
                    inst_color = color
                    break
            inst_text = instrument_name if instrument_name else f"Program {self.audio.current_instrument}"
            inst_color = inst_color if instrument_name else self.SLATE_BLUE
        else:
            # Fallback to first instrument
            inst_text = self.quick_instruments[0][0]
            inst_color = self.quick_instruments[0][2]
            
        inst_surface = font.render(f"{inst_text}", True, inst_color)
        x_offset = preset_surface.get_width() + 20
        self.screen.blit(inst_surface, (x_offset, 50))

        # Scale lock and key signature
        if self.analyzer.scale_lock:
            scale_text = f"Scale: {self.current_key}"
            scale_color = self.DEEP_TEAL
        else:
            scale_text = "Scale: Free"
            scale_color = self.TERRA_COTTA
        scale_surface = small_font.render(scale_text, True, scale_color)
        self.screen.blit(scale_surface, (10, y_offset))
        
        # Velocity
        vel_text = f"Base Velocity: {self.base_velocity}"
        vel_surface = small_font.render(vel_text, True, self.TEXT_DARK)
        self.screen.blit(vel_surface, (10, y_offset + 22))
        
        # Active notes
        active_count = len(self.audio.active_notes)
        active_surface = small_font.render(f"Playing: {active_count} notes", True, self.SLATE_BLUE)
        self.screen.blit(active_surface, (10, y_offset + 44))
        
        # ===== TOP RIGHT: Last shape info =====
        if self.shapes:
            last = self.shapes[-1]
            if 'midi' in last:
                note_name = self.midi_to_note_name(last['midi'])
                info = f"{note_name} | vel:{last['velocity']} | {last['duration']:.1f}s"
                info_surface = small_font.render(info, True, self.SLATE_BLUE)
                info_width = info_surface.get_width()
                self.screen.blit(info_surface, (self.width - info_width - 10, 10))
        
        # ===== BOTTOM LEFT: Instruments panel =====
        panel_width = 470
        panel_height = 100
        panel_x = 10
        panel_y = self.height - panel_height - 10
        
        panel_bg = pygame.Surface((panel_width, panel_height))
        panel_bg.set_alpha(230)
        panel_bg.fill(self.UI_BG)
        self.screen.blit(panel_bg, (panel_x, panel_y))
        
        preset_label = small_font.render(self.current_preset.upper(), True, self.TERRA_COTTA)
        inst_title = small_font.render("INSTRUMENTS:", True, self.TERRA_COTTA)
        
        x = panel_x + 10
        y = panel_y + 8
        self.screen.blit(preset_label, (x, y))
        x_next = x + preset_label.get_width() + 8
        self.screen.blit(inst_title, (x_next, y))

        # List instruments in 3 columns with reduced spacing
        start_y = panel_y + 32
        for i, (name, prog, color, *rest) in enumerate(self.quick_instruments):
            row = i // 3
            col = i % 3
            x_pos = panel_x + 15 + col * 150
            y_pos = start_y + row * 20
            
            # Highlight selection by index (works for drums too)
            if self.selected_instrument_index == i:
                text = small_font.render(f"{i+1}: {name}", True, color)
            else:
                text = tiny_font.render(f"{i+1}: {name}", True, self.TEXT_DARK)
                
            self.screen.blit(text, (x_pos, y_pos))

        
        # ===== BOTTOM RIGHT: Controls panel (horizontal layout) =====
        if self.show_help:
            self.draw_help_overlay(tiny_font, small_font, panel_height)
        
    def draw_help_overlay(self, tiny_font, small_font, instruments_height):
        """Draw help information overlay in bottom right - horizontal 4-column layout"""
        help_width = 700  # Augmenté pour 4 colonnes
        help_height = instruments_height
        help_x = self.width - help_width - 10   
        help_y = self.height - help_height - 10 
        
        help_bg = pygame.Surface((help_width, help_height))
        help_bg.set_alpha(230)
        help_bg.fill(self.UI_BG)
        self.screen.blit(help_bg, (help_x, help_y))
        
        help_title = small_font.render("CONTROLS (H to hide)", True, self.TERRA_COTTA)
        self.screen.blit(help_title, (help_x + 10, help_y + 8))
        
        # Four columns
        col_width = 170
        start_y = help_y + 32
        
        # Column 1: INSTRUMENTS
        col1_x = help_x + 15
        col1_lines = [
            ("INSTRUMENTS:", self.SLATE_BLUE),
            ("1-9: Select", self.TEXT_DARK),
            ("N: Next preset", self.TEXT_DARK),
            ("P: Previous preset", self.TEXT_DARK),
        ]
        y_off = start_y
        for text, color in col1_lines:
            surface = tiny_font.render(text, True, color)
            self.screen.blit(surface, (col1_x, y_off))
            y_off += 16
        
        # Column 2: MUSICAL
        col2_x = help_x + 15 + col_width
        col2_lines = [
            ("MUSICAL:", self.SLATE_BLUE),
            ("M: Scale lock", self.TEXT_DARK),
            ("K: Change key", self.TEXT_DARK),
            ("+/-: Velocity", self.TEXT_DARK),
        ]
        y_off = start_y
        for text, color in col2_lines:
            surface = tiny_font.render(text, True, color)
            self.screen.blit(surface, (col2_x, y_off))
            y_off += 16
        
        # Column 3: EDITING
        col3_x = help_x + 15 + col_width * 2
        col3_lines = [
            ("EDITING:", self.SLATE_BLUE),
            ("Z: Undo", self.TEXT_DARK),
            ("Y: Redo", self.TEXT_DARK),
            ("C: Clear canvas", self.TEXT_DARK),
        ]
        y_off = start_y
        for text, color in col3_lines:
            surface = tiny_font.render(text, True, color)
            self.screen.blit(surface, (col3_x, y_off))
            y_off += 16
        
        # Column 4: GENERAL
        col4_x = help_x + 15 + col_width * 3
        col4_lines = [
            ("GENERAL:", self.SLATE_BLUE),
            ("S: Stop sounds", self.TEXT_DARK),
            ("Q/ESC: Quit", self.TEXT_DARK),
        ]
        y_off = start_y
        for text, color in col4_lines:
            surface = tiny_font.render(text, True, color)
            self.screen.blit(surface, (col4_x, y_off))
            y_off += 16
                
    def clear_all(self):
        if self.shapes:
            # Save current state before clearing for potential undo
            self.shapes_history.append(self.shapes.copy())
        
        self.shapes = []
        self.shapes_redo = []  # Clear redo stack when clearing canvas
        print("Canvas cleared")
        
    def cleanup(self):
        """Clean shutdown"""
        print("\nShutting down...")
        self.audio.cleanup()
        pygame.quit()
        print("Clean exit")


# ======================
# MAIN ENTRY POINT
# ======================
if __name__ == "__main__":
    try:
        synth = GeometricSynth()
        synth.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()