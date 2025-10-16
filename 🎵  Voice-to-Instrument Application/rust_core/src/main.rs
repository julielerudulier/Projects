mod pitch_detector;
mod note_mapper;
mod audio_engine;

use pitch_detector::PitchDetector;
use note_mapper::{NoteMapper, NoteEvent};
use audio_engine::AudioEngine;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::io::{self, Write};

/// Display current note prominently
fn display_note(note: u8, frequency: f32, confidence: f32, instrument_name: &str) {
    // Clear screen (Unix/Mac/Windows)
    print!("\x1B[2J\x1B[1;1H");
    
    let note_name = NoteMapper::midi_to_note_name(note);
    
    println!("╔════════════════════════════════════════╗");
    println!("║   🎹 VOICE-TO-INSTRUMENT TRANSFORMER   ║");
    println!("╠════════════════════════════════════════╣");
    println!("║  Instrument: {:<26} ║", instrument_name);
    println!("╠════════════════════════════════════════╣");
    println!("║         Current Note: {:^8}        ║", note_name);
    println!("║         MIDI Number:  {:^8}        ║", note);
    println!("║         Frequency:    {:^8.1} Hz   ║", frequency);
    println!("║         Confidence:   {:^7.1}%     ║", confidence * 100.0);
    println!("║                                        ║");
    println!("╚════════════════════════════════════════╝");
    println!();
    println!("🎤 Sing or hum a note...");
    println!("💡 Press number keys to change instrument:");
    println!("   [1] Piano  [2] Violin  [3] Flute  [4] Trumpet");
    println!("   (Ctrl+C to quit)");
}

fn display_silence(instrument_name: &str) {
    print!("\x1B[2J\x1B[1;1H");
    println!("╔════════════════════════════════════════╗");
    println!("║   🎹 VOICE-TO-INSTRUMENT TRANSFORMER   ║");
    println!("╠════════════════════════════════════════╣");
    println!("║  Instrument: {:<26} ║", instrument_name);
    println!("╠════════════════════════════════════════╣");
    println!("║              🔇 SILENCE                ║");
    println!("║                                        ║");
    println!("╚════════════════════════════════════════╝");
    println!();
    println!("🎤 Sing or hum a note...");
    println!("💡 Press number keys to change instrument:");
    println!("   [1] Piano  [2] Violin  [3] Flute  [4] Trumpet");
    println!("   (Ctrl+C to quit)");
}

fn select_instrument() -> u32 {
    println!("\n🎹 Select an instrument:");
    println!("  [0]  Acoustic Grand Piano");
    println!("  [1]  Bright Piano");
    println!("  [40] Violin");
    println!("  [41] Viola");
    println!("  [42] Cello");
    println!("  [56] Trumpet");
    println!("  [60] French Horn");
    println!("  [65] Soprano Sax");
    println!("  [73] Flute");
    println!("  [74] Recorder");
    
    print!("\nEnter instrument number (default: 0): ");
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    input.trim().parse().unwrap_or(0)
}

fn main() {
    println!("╔════════════════════════════════════════╗");
    println!("║   🎵 Voice-to-Instrument Transformer   ║");
    println!("╚════════════════════════════════════════╝\n");
    
    let host = cpal::default_host();
    
    // Input device (microphone)
    let input_device = host.default_input_device()
        .expect("No microphone found");
    println!("🎤 Microphone: {}", input_device.name().unwrap());
    
    // Configuration
    let buffer_size = 1024_usize;
    
    // SoundFont path
    let soundfont_path = "FluidR3_GM.sf2";
    
    // Check SoundFont exists
    if !std::path::Path::new(soundfont_path).exists() {
        eprintln!("\n❌ SoundFont not found: {}", soundfont_path);
        eprintln!("   Download FluidR3_GM.sf2 and place it in the project folder");
        eprintln!("   Download link: https://musical-artifacts.com/artifacts/738");
        std::process::exit(1);
    }
    
    // Create audio engine with FluidSynth
    let mut audio_engine = AudioEngine::new(soundfont_path)
        .expect("Failed to create audio engine");
    
    // Test audio output
    println!("\n🔊 Testing audio output...");
    audio_engine.test_audio();
    
    // Let user select instrument
    let selected_instrument = select_instrument();
    audio_engine.set_instrument(selected_instrument);
    
    let audio_engine = Arc::new(Mutex::new(audio_engine));
    
    // Input configuration
    let input_config = input_device.default_input_config()
        .expect("Failed to configure microphone");
    
    let actual_sample_rate = input_config.sample_rate().0 as f32;
    println!("\n📊 Sample rate: {} Hz", actual_sample_rate);
    println!("📊 Buffer size: {} samples", buffer_size);
    
    let sample_rate = actual_sample_rate;
    
    // Create modules
    let mut pitch_detector = PitchDetector::new(buffer_size, sample_rate);
    let mut note_mapper = NoteMapper::new(7); // Increased smoothing window
    
    // Input accumulator buffer
    let input_accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));
    let input_accumulator_clone = Arc::clone(&input_accumulator);
    
    println!("\n▶️  Ready! Sing a note...\n");
    std::thread::sleep(std::time::Duration::from_secs(2));
    
    // Input stream (microphone)
    let input_stream = input_device.build_input_stream(
        &input_config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Accumulate samples
            let mut accumulator = input_accumulator_clone.lock().unwrap();
            accumulator.extend_from_slice(data);
        },
        |err| eprintln!("❌ Audio input error: {}", err),
        None,
    ).expect("Failed to create input stream");
    
    // Start input stream
    input_stream.play().expect("Failed to start microphone");
    
    // Main processing loop
    let mut time = 0.0;
    let mut processing_buffer = vec![0.0f32; buffer_size];
    let mut last_displayed_note: Option<u8> = None; // Track what we're displaying
    
    // Get instrument name for display
    let instrument_name = {
        let engine = audio_engine.lock().unwrap();
        AudioEngine::get_gm_instrument_name(engine.get_current_instrument())
    };
    
    loop {
        // Check if we have enough samples
        let has_enough_samples = {
            let accumulator = input_accumulator.lock().unwrap();
            accumulator.len() >= buffer_size
        };
        
        if has_enough_samples {
            // Extract samples
            {
                let mut accumulator = input_accumulator.lock().unwrap();
                processing_buffer.copy_from_slice(&accumulator[..buffer_size]);
                accumulator.drain(..buffer_size);
            }
            
            // Pitch detection
            let detected_freq = pitch_detector.detect(&processing_buffer);
            
            // Map to MIDI notes
            time += buffer_size as f32 / sample_rate;
            let event = note_mapper.process(detected_freq, time);
            
            // Control audio engine
            match event {
                NoteEvent::NoteOn { note, velocity } => {
                    let freq = detected_freq.unwrap_or(0.0);
                    let confidence = pitch_detector.get_confidence();
                    
                    display_note(note, freq, confidence, instrument_name);
                    
                    let mut engine = audio_engine.lock().unwrap();
                    engine.note_on(note, velocity);
                }
                NoteEvent::Transition { off_note, on_note, velocity } => {
                    let freq = detected_freq.unwrap_or(0.0);
                    let confidence = pitch_detector.get_confidence();
                    
                    display_note(on_note, freq, confidence, instrument_name);
                    
                    let mut engine = audio_engine.lock().unwrap();
                    engine.note_off(off_note);
                    engine.note_on(on_note, velocity);
                }
                NoteEvent::NoteOff { note } => {
                    display_silence(instrument_name);
                    
                    let mut engine = audio_engine.lock().unwrap();
                    engine.note_off(note);
                }
                NoteEvent::None => {
                    // Nothing to do
                }
            }
        }
        
        // Small sleep to avoid CPU overload
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}