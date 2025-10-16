use std::collections::VecDeque;

/// MIDI event with transition handling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoteEvent {
    NoteOn { note: u8, velocity: u8 },
    NoteOff { note: u8 },
    Transition { off_note: u8, on_note: u8, velocity: u8 },
    None,
}

/// Converts frequencies to MIDI notes with advanced smoothing
pub struct NoteMapper {
    current_note: Option<u8>,
    smoothing_window: VecDeque<f32>,
    window_size: usize,
    min_note_duration: f32,
    last_note_time: f32,
    stability_counter: usize,
    stability_threshold: usize,
    last_detected_note: Option<u8>,
}

impl NoteMapper {
    pub fn new(window_size: usize) -> Self {
        Self {
            current_note: None,
            smoothing_window: VecDeque::with_capacity(window_size),
            window_size,
            min_note_duration: 0.08, // 80ms - more responsive
            last_note_time: 0.0,
            stability_counter: 0,
            stability_threshold: 3, // Need 3 stable frames
            last_detected_note: None,
        }
    }
    
    /// Convert frequency to MIDI note
    fn frequency_to_midi(frequency: f32) -> u8 {
        // Formula: MIDI = 69 + 12 * log2(freq / 440)
        let midi_float = 69.0 + 12.0 * (frequency / 440.0).log2();
        midi_float.round().clamp(0.0, 127.0) as u8
    }
    
    /// Process new detected frequency
    pub fn process(&mut self, frequency: Option<f32>, time: f32) -> NoteEvent {
        match frequency {
            Some(freq) => {
                // Add to smoothing window
                self.smoothing_window.push_back(freq);
                if self.smoothing_window.len() > self.window_size {
                    self.smoothing_window.pop_front();
                }
                
                // Calculate median for stability
                let smoothed_freq = self.median_frequency();
                let midi_note = Self::frequency_to_midi(smoothed_freq);
                
                // Stability check - prevent flickering
                if Some(midi_note) == self.last_detected_note {
                    self.stability_counter += 1;
                } else {
                    self.stability_counter = 0;
                    self.last_detected_note = Some(midi_note);
                }
                
                // Only change note if stable enough
                if self.stability_counter < self.stability_threshold {
                    return NoteEvent::None;
                }
                
                // Check for note change
                if Some(midi_note) != self.current_note {
                    // Respect minimum duration
                    if time - self.last_note_time < self.min_note_duration {
                        return NoteEvent::None;
                    }
                    
                    self.last_note_time = time;
                    
                    // Handle transition
                    let event = if let Some(old_note) = self.current_note {
                        // Smooth transition: off old, on new
                        NoteEvent::Transition {
                            off_note: old_note,
                            on_note: midi_note,
                            velocity: 80, // TODO: calculate from amplitude
                        }
                    } else {
                        // First note
                        NoteEvent::NoteOn {
                            note: midi_note,
                            velocity: 80,
                        }
                    };
                    
                    self.current_note = Some(midi_note);
                    event
                } else {
                    // Same note continues - keep it sustained
                    // No event needed, note stays on
                    NoteEvent::None
                }
            }
            None => {
                // Silence detected
                self.stability_counter = 0;
                self.last_detected_note = None;
                
                if let Some(note) = self.current_note {
                    self.current_note = None;
                    self.smoothing_window.clear();
                    NoteEvent::NoteOff { note }
                } else {
                    NoteEvent::None
                }
            }
        }
    }
    
    /// Calculate median to filter outliers
    fn median_frequency(&self) -> f32 {
        if self.smoothing_window.is_empty() {
            return 440.0;
        }
        
        let mut sorted: Vec<f32> = self.smoothing_window.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    }
    
    /// Convert MIDI note to name (e.g., 60 → "C4")
    pub fn midi_to_note_name(midi: u8) -> String {
        let notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        let octave = (midi / 12) as i32 - 1;
        let note = notes[(midi % 12) as usize];
        format!("{}{}", note, octave)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_frequency_conversion() {
        assert_eq!(NoteMapper::frequency_to_midi(440.0), 69);  // A4
        assert_eq!(NoteMapper::frequency_to_midi(261.63), 60); // C4
    }
    
    #[test]
    fn test_note_names() {
        assert_eq!(NoteMapper::midi_to_note_name(60), "C4");
        assert_eq!(NoteMapper::midi_to_note_name(69), "A4");
    }
    
    #[test]
    fn test_stability_threshold() {
        let mut mapper = NoteMapper::new(5);
        // First detections should return None until stable
        let event1 = mapper.process(Some(440.0), 0.0);
        assert_eq!(event1, NoteEvent::None);
        
        let event2 = mapper.process(Some(440.0), 0.01);
        assert_eq!(event2, NoteEvent::None);
    }
}