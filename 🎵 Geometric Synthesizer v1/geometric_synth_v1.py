import pygame
import numpy as np
import sounddevice as sd
import threading
import math
import time

class GeometricSynth:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100
        self.buffer_size = 512
        
        # Pygame parameters
        pygame.init()
        self.width, self.height = 1000, 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Geometric Synthesizer V1")
        
        # Colors define waveforms
        self.color_palette = [
            ((255, 0, 0), 'sine', 'Pure'),
            ((0, 255, 0), 'triangle', 'Soft'),
            ((0, 0, 255), 'square', 'Bright'),
            ((255, 255, 0), 'sawtooth', 'Rich'),
            ((255, 0, 255), 'pulse', 'Hollow'),
            ((0, 255, 255), 'noise', 'Textured'),
            ((255, 128, 0), 'mixed', 'Complex')
        ]
        
        # Synthesizer states
        self.drawing = False
        self.current_stroke = None
        self.strokes = []  # All completed strokes
        self.current_color_index = 0
        self.brush_size = 3
        
        # Active oscillators with thread safety
        self.active_strokes = {}  # stroke_id -> oscillator_data
        self.stroke_counter = 0
        self.osc_lock = threading.Lock()
        
        # Audio
        self.audio_running = True
        
    def clamp_position(self, pos):
        """Ensure position stays within canvas bounds"""
        x = max(0, min(self.width - 1, pos[0]))
        y = max(0, min(self.height - 1, pos[1]))
        return (x, y)

    def start(self):
        """Starts the synthesizer"""
        print("=" * 70)
        print("GEOMETRIC SYNTHESIZER - Continuous Instrument")
        print("=" * 70)
        print("\nCONCEPT:")
        print("  A polyphonic visual theremin. Each stroke becomes a voice.")
        print("  Sound evolves continuously as you draw.")
        print("\nCONTROLS:")
        print("  Draw: Click and drag (multiple strokes = polyphony)")
        print("  Colors: Keys 1-7 (different waveforms)")
        print("  S: Save image")
        print("  Space: Clear | Escape: Quit")
        print("\nSOUND MAPPING:")
        print("  Y Position → Pitch (higher = higher)")
        print("  X Position → Pan (left/right)")
        print("  Drawing Speed → Volume (faster = louder)")
        print("  Curvature → Harmonics (curved = richer)")
        print("  Direction → Modulation (up/down/curves)")
        print("\nWAVEFORMS (Keys 1-7):")
        for i, (color, wave, desc) in enumerate(self.color_palette, 1):
            print(f"  {i}. {desc} ({wave})")
        print("=" * 70)
        
        # Start audio
        self.audio_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            callback=self.audio_callback,
            blocksize=self.buffer_size
        )
        self.audio_stream.start()
        
        self.main_loop()
        
    def main_loop(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.start_stroke(event.pos)
                        
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.drawing:
                        self.end_stroke()
                        
                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        self.continue_stroke(event.pos)
                            
                elif event.type == pygame.KEYDOWN:
                    if pygame.K_1 <= event.key <= pygame.K_7:
                        self.current_color_index = event.key - pygame.K_1
                        color, wave, desc = self.color_palette[self.current_color_index]
                        print(f"Waveform: {desc} ({wave})")
                    elif event.key == pygame.K_SPACE:
                        self.clear_all()
                    elif event.key == pygame.K_s:
                        self.save_image()
            
            self.draw_interface()
            pygame.display.flip()
            clock.tick(60)
            
        self.cleanup()
    
    def start_stroke(self, pos):
        """Start a new stroke"""
        self.drawing = True
        pos = self.clamp_position(pos)
        color, wave, _ = self.color_palette[self.current_color_index]
        
        self.current_stroke = {
            'id': self.stroke_counter,
            'points': [pos],
            'color': color,
            'waveform': wave,
            'start_time': time.time(),
            'last_update': time.time()
        }
        
        self.stroke_counter += 1
        
        # Create audio oscillator
        self.create_oscillator_for_stroke(self.current_stroke)
    
    def continue_stroke(self, pos):
        """Continue the current stroke"""
        if not self.current_stroke:
            return
        
        pos = self.clamp_position(pos)
        self.current_stroke['points'].append(pos)
        self.current_stroke['last_update'] = time.time()
        
        # Update oscillator parameters based on stroke evolution
        self.update_oscillator_for_stroke(self.current_stroke)
    
    def end_stroke(self):
        """End the current stroke"""
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            
            # Stop the oscillator after a fade
            stroke_id = self.current_stroke['id']
            with self.osc_lock:
                if stroke_id in self.active_strokes:
                    self.active_strokes[stroke_id]['ending'] = True
                    self.active_strokes[stroke_id]['end_time'] = time.time()
            
            self.current_stroke = None
        self.drawing = False
    
    def create_oscillator_for_stroke(self, stroke):
        """Create an oscillator for a stroke (initial parameters)."""
        points = stroke['points']
        if not points:
            return

        pos = points[-1]
        freq = self.pos_to_frequency(pos)
        pan = pos[0] / self.width

        waveform = stroke['waveform']
        init_target_volume = 0.1

        with self.osc_lock:
            self.active_strokes[stroke['id']] = {
                'frequency': freq,
                'target_frequency': freq,
                'volume': 0.0,
                'target_volume': init_target_volume,
                'pan': pan,
                'waveform': waveform,
                'phase': 0.0,
                'harmonics': 0.0,
                'modulation': 0.0,
                'ending': False,
                'end_time': None,
                # small one-pole filter state for gentle smoothing of noise
                'noise_prev': 0.0
            }
    
    def update_oscillator_for_stroke(self, stroke):
        """Update oscillator parameters based on stroke evolution"""
        points = stroke['points']
        if len(points) < 2:
            return

        stroke_id = stroke['id']

        with self.osc_lock:
            if stroke_id not in self.active_strokes:
                return

            osc = self.active_strokes[stroke_id]

            # Current position and small smoothing for freq target
            current_pos = points[-1]
            prev_pos = points[-2]

            freq = self.pos_to_frequency(current_pos)
            osc['target_frequency'] = freq

            osc['pan'] = current_pos[0] / self.width

            distance = math.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
            speed = min(1.0, distance / 50.0)  # Normalize
            osc['target_volume'] = 0.02 + speed * 0.06  # Ultra-low range

            if len(points) >= 3:
                curvature = self.calculate_curvature(points[-3:])
                osc['harmonics'] = curvature

            if len(points) >= 2:
                dy = current_pos[1] - prev_pos[1]
                osc['modulation'] = max(-1.0, min(1.0, dy / 20.0))
    
    def calculate_curvature(self, three_points):
        """Calculate curvature from 3 points"""
        if len(three_points) != 3:
            return 0.0
        
        p1, p2, p3 = three_points
        
        # Vectors
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 < 1 or mag2 < 1:
            return 0.0
        
        # Angle between vectors
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        angle = math.acos(cos_angle)
        
        # Normalize to 0-1
        return min(1.0, angle / (math.pi / 2))
    
    def pos_to_frequency(self, pos):
        """Convert Y position to frequency"""
        normalized = (self.height - pos[1]) / self.height
        normalized = max(0.0, min(1.0, normalized))
        
        # Free mode: full frequency range
        min_freq = 130.0
        max_freq = 1046.0
        
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        return math.exp(log_min + normalized * (log_max - log_min))
    
    def audio_callback(self, outdata, frames, time_info, status):
        """Generate audio for all active strokes with band-limited additive synthesis."""
        outdata.fill(0.0)

        t0 = time.time()

        with self.osc_lock:
            strokes_to_remove = []

            for stroke_id, osc in list(self.active_strokes.items()):
                # If ending, schedule removal after fade
                if osc['ending'] and osc['end_time']:
                    fade_duration = 0.3
                    if time.time() - osc['end_time'] > fade_duration:
                        strokes_to_remove.append(stroke_id)
                        continue

                # Very smooth parameter transitions to avoid clicks
                osc['frequency'] += (osc['target_frequency'] - osc['frequency']) * 0.02
                osc['volume'] += (osc['target_volume'] - osc['volume']) * 0.01

                # Smooth fade out if ending
                if osc['ending'] and osc['end_time']:
                    elapsed = time.time() - osc['end_time']
                    fade_duration = 0.5  # Longer fade
                    fade_factor = max(0.0, 1.0 - elapsed / fade_duration)
                    fade_factor = fade_factor * fade_factor  # Exponential fade
                    osc['volume'] *= fade_factor
                else:
                    current_volume = osc['volume']

                # For each stroke generate frame block
                wave = np.zeros(frames, dtype=np.float32)
                phase = osc.get('phase', 0.0)
                base_freq = max(20.0, min(880.0, osc['frequency']))  # clamp freq to safe range

                # Determine max harmonic to avoid aliasing: harmonic * base_freq < nyquist
                nyquist = 0.5 * self.sample_rate
                max_harmonic = max(1, int(min(31, math.floor(nyquist / base_freq))))

                # Per-sample generation
                # Precompute small LFO if modulation large
                mod = osc.get('modulation', 0.0)
                use_lfo = abs(mod) > 0.05
                if use_lfo:
                    lfo_freq = 4.5 + abs(mod) * 3.0  # 4.5..7.5Hz
                    lfo_phase = 0.0

                # noise smoothing one-pole params
                noise_prev = osc.get('noise_prev', 0.0)
                noise_alpha = 0.06  # gentle smoothing for noise

                for i in range(frames):
                    sample = 0.0
                    phase_frac = phase % 1.0

                    # Base waveform with anti-aliasing
                    current_freq = osc['frequency']
                    
                    # Skip if frequency too high (prevents aliasing)
                    if current_freq > self.sample_rate / 4:
                        current_freq = self.sample_rate / 4
                    
                    if osc['waveform'] == 'sine':
                        sample = math.sin(2 * math.pi * phase)
                    elif osc['waveform'] == 'triangle':

                        # band-limited triangle via odd harmonics with 1/h^2 rolloff
                        # sum odd harmonics up to max_harmonic
                        val = 0.0
                        for h in range(1, max_harmonic + 1, 2):
                            val += ((-1)**((h-1)//2)) * (1.0 / (h * h)) * math.sin(2.0 * math.pi * h * phase_frac)
                        sample = val * (8.0 / (math.pi**2))  # normalization factor

                    elif osc['waveform'] == 'square':
                        # band-limited square using odd harmonics (1/h)
                        val = 0.0
                        for h in range(1, max_harmonic + 1, 2):
                            val += (1.0 / h) * math.sin(2.0 * math.pi * h * phase_frac)
                        sample = val * (4.0 / math.pi)

                    elif osc['waveform'] == 'sawtooth':
                        # band-limited saw (all harmonics 1/h)
                        val = 0.0
                        for h in range(1, max_harmonic + 1):
                            val += (1.0 / h) * math.sin(2.0 * math.pi * h * phase_frac)
                        sample = val * (2.0 / math.pi)

                    elif osc['waveform'] == 'pulse':
                        # pulse with a duty (we will vary duty slightly by harmonics/hardness)
                        duty = 0.25  # 25% by default, gives hollow timbre
                        # build pulse by subtracting two saws or via harmonic series
                        val = 0.0
                        for h in range(1, max_harmonic + 1):
                            val += (1.0 / h) * math.sin(2.0 * math.pi * h * phase_frac) * math.sin(h * math.pi * duty)
                        sample = val * (2.0 / math.pi)

                    elif osc['waveform'] == 'noise':
                        # low-level smoothed noise for texture
                        raw = np.random.uniform(-1.0, 1.0) * 0.6
                        noise_prev = noise_prev * (1.0 - noise_alpha) + raw * noise_alpha
                        sample = noise_prev * 0.6  # keep lower amplitude
                    elif osc['waveform'] == 'mixed':
                        # detuned partials for richness
                        a = math.sin(2.0 * math.pi * phase_frac)
                        b = math.sin(2.0 * math.pi * (phase_frac * 1.997)) * 0.65
                        c = math.sin(2.0 * math.pi * (phase_frac * 2.003)) * 0.4
                        sample = (a * 0.5 + b + c) * 0.7
                    else:
                        sample = 0.0

                    # extra harmonics from curvature (adds subtle partials)
                    if osc.get('harmonics', 0.0) > 0.05 and osc['waveform'] != 'noise':
                        harm_gain = osc['harmonics'] * 0.08
                        # add small 2nd/3rd harmonics if allowed by nyquist
                        if base_freq * 2 < nyquist:
                            sample += harm_gain * 0.5 * math.sin(2.0 * math.pi * 2 * phase_frac)
                        if base_freq * 3 < nyquist:
                            sample += harm_gain * 0.3 * math.sin(2.0 * math.pi * 3 * phase_frac)

                    # LFO modulation amplitude (small)
                    if use_lfo:
                        lfo = math.sin(2.0 * math.pi * lfo_freq * (t0 + i / self.sample_rate))
                        sample *= 1.0 + (osc['modulation'] * 0.15) * lfo

                    wave[i] = sample

                    # increment phase by instantaneous frequency (phase stored as cycles)
                    phase += base_freq / float(self.sample_rate)
                    # keep phase small
                    if phase >= 1.0:
                        phase -= int(phase)

                # store back filtered noise state and phase
                osc['noise_prev'] = noise_prev
                osc['phase'] = phase

                # Apply overall amplitude and panning
                wave *= osc['volume'] * 0.25

                left_gain = math.sqrt(1.0 - osc['pan'])
                right_gain = math.sqrt(osc['pan'])

                outdata[:, 0] += wave * left_gain
                outdata[:, 1] += wave * right_gain

            # Remove finished strokes
            for stroke_id in strokes_to_remove:
                if stroke_id in self.active_strokes:
                    del self.active_strokes[stroke_id]

        # Soft limiting / final safety clamp
        np.clip(outdata, -0.95, 0.95, out=outdata)
    
    def draw_interface(self):
        """Draw the interface"""
        self.screen.fill((0, 0, 0))
        
        # Draw completed strokes
        for stroke in self.strokes:
            if len(stroke['points']) > 1:
                pygame.draw.lines(self.screen, stroke['color'], False, stroke['points'], self.brush_size)
        
        # Draw current stroke
        if self.current_stroke and len(self.current_stroke['points']) > 1:
            # Glowing effect for active stroke
            glow_color = tuple(min(255, c + 50) for c in self.current_stroke['color'])
            pygame.draw.lines(self.screen, glow_color, False, self.current_stroke['points'], self.brush_size + 2)
            pygame.draw.lines(self.screen, self.current_stroke['color'], False, self.current_stroke['points'], self.brush_size)
        
        self.draw_ui()
    
    def draw_ui(self):
        """Draw UI"""
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        # Title
        title = font.render("Geometric Synthesizer V1 - Continuous Instrument", True, (255, 255, 0))
        self.screen.blit(title, (10, 10))
        
        # Active voices
        with self.osc_lock:
            num_voices = len(self.active_strokes)
        
        voices_text = small_font.render(f"Active voices: {num_voices}", True, (255, 255, 255))
        self.screen.blit(voices_text, (10, 40))

        # Save button
        save_text = small_font.render("S: Save image", True, (255, 255, 255))
        self.screen.blit(save_text, (10, 80))

        # Clear command
        clear_text = small_font.render("SPACE: Clear all", True, (255, 255, 255))
        self.screen.blit(clear_text, (10, 100))
        
        # Color palette
        palette_y = self.height - 60
        for i, (color, wave, desc) in enumerate(self.color_palette):
            x = 10 + i * 45
            rect = pygame.Rect(x, palette_y, 40, 40)
            pygame.draw.rect(self.screen, color, rect)
            
            if i == self.current_color_index:
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 3)
            else:
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
            
            # Label
            label = small_font.render(str(i + 1), True, (255, 255, 255))

            self.screen.blit(label, (x + 15, palette_y + 42))
        
        # Current waveform
        _, wave, desc = self.color_palette[self.current_color_index]
        current_wave = small_font.render(f"Current: {desc} ({wave})", True, (255, 255, 255))
        self.screen.blit(current_wave, (10, self.height - 80))
    
    def save_image(self):
        """Save the current canvas with shutter sound"""
        import os
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"geometric_synth_{timestamp}.png"
        
        try:
            filepath = os.path.abspath(filename)
            pygame.image.save(self.screen, filepath)
            
            # Camera shutter click sound
            duration = 0.08
            samples = int(duration * self.sample_rate)
            t = np.linspace(0, duration, samples)
            
            # Two-tone click (mechanical shutter feel)
            click1 = np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 80)
            click2 = np.sin(2 * np.pi * 1200 * t[:len(t)//3]) * np.exp(-t[:len(t)//3] * 60)
            
            click = np.zeros(samples)
            click[:len(click2)] += click2 * 0.3
            click[len(t)//4:] += click1[:len(click[len(t)//4:])] * 0.2
            
            sd.play(click, samplerate=self.sample_rate)
            print(f"✓ Image saved: {filepath}")
        except Exception as e:
            print(f"✗ Save failed: {e}")
    
    def clear_all(self):
        """Clear everything"""
        self.strokes = []
        with self.osc_lock:
            self.active_strokes = {}
        print("Cleared!")
    
    def cleanup(self):
        """Cleanup"""
        self.audio_running = False
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        pygame.quit()
        print("Synthesizer closed.")

if __name__ == "__main__":
    synth = GeometricSynth()
    try:
        synth.start()
    except KeyboardInterrupt:
        print("\nStopping...")
        synth.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
