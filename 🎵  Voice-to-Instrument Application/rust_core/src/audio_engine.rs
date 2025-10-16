use std::collections::HashMap;
use std::time::Instant;
use fluidsynth::settings::Settings;
use fluidsynth::synth::Synth;
use fluidsynth::audio::AudioDriver;

/// Audio engine managing FluidSynth with real-time playback
pub struct AudioEngine {
    synth: Synth,
    #[allow(dead_code)]
    audio_driver: AudioDriver,
    sfid: u32,
    active_notes: HashMap<u8, Instant>,
    current_instrument: u32,
}

impl AudioEngine {
    pub fn new(soundfont_path: &str) -> Result<Self, String> {
        println!("🔧 Initializing FluidSynth...");
        
        // Create FluidSynth settings
        let mut settings = Settings::new();

        // Configure audio driver based on OS
        #[cfg(target_os = "macos")]
        {
            settings.setstr("audio.driver", "coreaudio");
            println!("🔊 Audio driver: CoreAudio (macOS)");
        }
        
        #[cfg(target_os = "linux")]
        {
            settings.setstr("audio.driver", "alsa");
            println!("🔊 Audio driver: ALSA (Linux)");
        }
        
        #[cfg(target_os = "windows")]
        {
            settings.setstr("audio.driver", "dsound");
            println!("🔊 Audio driver: DirectSound (Windows)");
        }

        // Configure audio settings
        settings.setnum("synth.sample-rate", 44100.0);
        settings.setnum("synth.gain", 0.6);
        
        println!("📊 Sample rate: 44100 Hz");
        println!("🔊 Gain: 0.6");

        // Create synthesizer
        let mut synth = Synth::new(&mut settings);
        println!("✅ Synthesizer created");

        // Load SoundFont
        let sfid = synth.sfload(soundfont_path, 1)
            .ok_or_else(|| format!("❌ Failed to load SoundFont: {}", soundfont_path))?;
        println!("✅ SoundFont loaded: {} (ID: {})", soundfont_path, sfid);

        // Select initial instrument (Grand Piano)
        synth.program_select(0, sfid, 0, 0);
        println!("🎹 Initial instrument: Grand Piano (0)");
        
        // Initialize percussion on channel 9 (General MIDI standard)
        synth.program_select(9, sfid, 128, 0); // 128 = drum bank
        println!("🥁 Percussion initialized on channel 9");

        // Create audio driver - THIS IS CRITICAL FOR SOUND OUTPUT
        // Note: AudioDriver constructor signature: new(settings, synth)
        let mut settings_for_driver = Settings::new();
        
        #[cfg(target_os = "macos")]
        settings_for_driver.setstr("audio.driver", "coreaudio");
        
        #[cfg(target_os = "linux")]
        settings_for_driver.setstr("audio.driver", "alsa");
        
        #[cfg(target_os = "windows")]
        settings_for_driver.setstr("audio.driver", "dsound");
        
        let audio_driver = AudioDriver::new(&mut settings_for_driver, &mut synth);
        println!("✅ Audio driver started - you should hear sound now!");

        Ok(Self {
            synth,
            audio_driver,
            sfid,
            active_notes: HashMap::new(),
            current_instrument: 0,
        })
    }

    /// Set instrument (General MIDI program number)
    /// See: https://en.wikipedia.org/wiki/General_MIDI#Program_change_events
    pub fn set_instrument(&mut self, program: u32) {
        self.synth.program_select(0, self.sfid, 0, program);
        self.current_instrument = program;
        
        let instrument_name = Self::get_gm_instrument_name(program);
        println!("🎹 Instrument changed to: {} ({})", instrument_name, program);
    }

    /// Play a note
    pub fn note_on(&mut self, midi_note: u8, velocity: u8) {
        // Stop previous note if playing (monophonic behavior)
        if self.active_notes.contains_key(&midi_note) {
            self.note_off(midi_note);
        }
        
        self.synth.noteon(0, midi_note as i32, velocity as i32);
        self.active_notes.insert(midi_note, Instant::now());
    }

    /// Stop a note
    pub fn note_off(&mut self, midi_note: u8) {
        self.synth.noteoff(0, midi_note as i32);
        self.active_notes.remove(&midi_note);
    }

    /// Stop all notes (panic button)
    pub fn stop_all_notes(&mut self) {
        let notes: Vec<u8> = self.active_notes.keys().cloned().collect();
        for note in notes {
            self.note_off(note);
        }
    }

    /// Get current instrument
    pub fn get_current_instrument(&self) -> u32 {
        self.current_instrument
    }
    
    /// Get General MIDI instrument name
    pub fn get_gm_instrument_name(program: u32) -> &'static str {
        match program {
            0 => "Acoustic Grand Piano",
            1 => "Bright Acoustic Piano",
            6 => "Harpsichord",
            24 => "Acoustic Guitar (nylon)",
            25 => "Acoustic Guitar (steel)",
            40 => "Violin",
            41 => "Viola",
            42 => "Cello",
            56 => "Trumpet",
            57 => "Trombone",
            60 => "French Horn",
            65 => "Soprano Sax",
            66 => "Alto Sax",
            67 => "Tenor Sax",
            68 => "Baritone Sax",
            73 => "Flute",
            74 => "Recorder",
            _ => "Unknown Instrument",
        }
    }
    
    /// Test audio output with a hard-hitting blues riff
    pub fn test_audio(&mut self) {
        println!("\n🎵 Testing audio output with blues riff...");
        
        // Dirty blues in C with approach notes and syncopation
        // Format: (note, velocity, duration_ms)
        let blues_riff = [
            (58, 110, 80),   // Bb approach
            (60, 120, 180),  // C (accent)
            (62, 80, 60),    // D ghost note
            (63, 100, 120),  // Eb (blue note)
            (64, 70, 60),    // E approach
            (65, 115, 150),  // F (accent)
            (66, 75, 50),    // F# chromatic
            (67, 120, 200),  // G (strong)
            (65, 90, 80),    // F
            (63, 100, 120),  // Eb back
            (60, 125, 250),  // C resolution (heavy)
        ];
        
        for (note, velocity, duration_ms) in blues_riff.iter() {
            self.note_on(*note, *velocity);
            std::thread::sleep(std::time::Duration::from_millis(*duration_ms as u64));
            self.note_off(*note);
            std::thread::sleep(std::time::Duration::from_millis(30)); // Tight gap
        }
        
        println!("✅ Audio test complete!\n");
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        println!("🔇 Cleaning up audio engine...");
        self.stop_all_notes();
        // audio_driver is automatically dropped
    }
}