use aubio_rs::{Pitch, PitchMode};

/// Pitch detector (fundamental frequency)
pub struct PitchDetector {
    detector: Pitch,
    sample_rate: f32,
    buffer_size: usize,
    min_freq: f32,
    max_freq: f32,
    confidence_threshold: f32,
}

impl PitchDetector {
    /// Create a new detector optimized for voice
    pub fn new(buffer_size: usize, sample_rate: f32) -> Self {
        // YIN algorithm = excellent for voice
        let mut detector = Pitch::new(
            PitchMode::Yin,
            buffer_size,
            buffer_size / 4,  // Hop size
            sample_rate as u32,
        ).expect("Failed to create pitch detector");
        
        // Silence threshold (-40dB is good for voice)
        detector.set_silence(-40.0);
        
        // Tolerance for YIN (lower = more accurate but less stable)
        detector.set_tolerance(0.15); // 0.15 is a good balance
        
        Self {
            detector,
            sample_rate,
            buffer_size,
            min_freq: 80.0,   // E2 - lowest reasonable singing note
            max_freq: 1200.0, // D6 - highest for most singers
            confidence_threshold: 0.8, // Require 80% confidence
        }
    }
    
    /// Detect frequency in an audio buffer
    /// Returns Some(frequency) or None if silence/unreliable
    pub fn detect(&mut self, audio_buffer: &[f32]) -> Option<f32> {
        // Check buffer size
        if audio_buffer.len() != self.buffer_size {
            eprintln!("Incorrect buffer size: {} (expected {})", 
                     audio_buffer.len(), self.buffer_size);
            return None;
        }
        
        // Detection
        let pitch = self.detector
            .do_result(audio_buffer)
            .expect("Pitch detection error");
        
        let confidence = self.detector.get_confidence();
        
        // Filter based on frequency range and confidence
        if pitch >= self.min_freq 
            && pitch <= self.max_freq 
            && confidence >= self.confidence_threshold {
            Some(pitch)
        } else {
            None
        }
    }
    
    /// Get detection confidence (0.0 - 1.0)
    pub fn get_confidence(&self) -> f32 {
        self.detector.get_confidence()
    }
    
    /// Adjust sensitivity (lower = more strict)
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
    }
    
    /// Set frequency range for detection
    pub fn set_frequency_range(&mut self, min: f32, max: f32) {
        self.min_freq = min;
        self.max_freq = max;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pitch_detector_creation() {
        let detector = PitchDetector::new(1024, 44100.0);
        assert_eq!(detector.buffer_size, 1024);
    }
    
    #[test]
    fn test_silence_detection() {
        let mut detector = PitchDetector::new(1024, 44100.0);
        let silence = vec![0.0; 1024];
        assert_eq!(detector.detect(&silence), None);
    }
    
    #[test]
    fn test_frequency_range() {
        let mut detector = PitchDetector::new(1024, 44100.0);
        detector.set_frequency_range(100.0, 1000.0);
        assert_eq!(detector.min_freq, 100.0);
        assert_eq!(detector.max_freq, 1000.0);
    }
}