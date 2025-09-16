"""
Demo script to show the Mental State Visualizer without actual Muse hardware
Simulates EEG data to demonstrate the complete pipeline
"""

import numpy as np
import time
import threading
import logging
from typing import Dict, Any

# Import our components
from advanced_signal_processor import AdvancedSignalProcessor
from mental_state_classifier import MentalStateClassifier
from realtime_visualizer import RealTimeVisualizer

class MuseSimulator:
    """Simulates Muse EEG data for demonstration purposes"""
    
    def __init__(self, sampling_rate: int = 256):
        self.sampling_rate = sampling_rate
        self.is_streaming = False
        self.current_mental_state = "neutral"
        self.time_offset = 0.0
        
        # State-specific frequency characteristics
        self.state_profiles = {
            "relaxed": {
                "alpha_amplitude": 80,    # High alpha
                "beta_amplitude": 30,     # Low beta
                "theta_amplitude": 40,
                "noise_level": 15
            },
            "focused": {
                "alpha_amplitude": 40,    # Low alpha
                "beta_amplitude": 70,     # High beta
                "theta_amplitude": 25,
                "noise_level": 20
            },
            "meditative": {
                "alpha_amplitude": 60,    # Moderate alpha
                "beta_amplitude": 25,     # Low beta
                "theta_amplitude": 80,    # High theta
                "noise_level": 10
            },
            "alert": {
                "alpha_amplitude": 50,
                "beta_amplitude": 60,
                "theta_amplitude": 30,
                "noise_level": 25
            },
            "neutral": {
                "alpha_amplitude": 50,
                "beta_amplitude": 50,
                "theta_amplitude": 40,
                "noise_level": 20
            }
        }
        
        # State transition schedule (for demo)
        self.state_schedule = [
            (0, "neutral"),
            (15, "relaxed"),
            (30, "focused"),
            (45, "meditative"),
            (60, "alert"),
            (75, "neutral")
        ]
        
        self.start_time = time.time()
    
    def start_streaming(self):
        """Start the simulated data stream"""
        self.is_streaming = True
    
    def stop_streaming(self):
        """Stop the simulated data stream"""
        self.is_streaming = False
    
    def get_eeg_data(self, num_samples: int = 64) -> np.ndarray:
        """Generate simulated EEG data"""
        if not self.is_streaming:
            return None
        
        # Update mental state based on schedule
        elapsed_time = time.time() - self.start_time
        for trigger_time, state in self.state_schedule:
            if elapsed_time >= trigger_time:
                self.current_mental_state = state
        
        # Get current state profile
        profile = self.state_profiles[self.current_mental_state]
        
        # Generate time vector
        t = np.linspace(self.time_offset, 
                       self.time_offset + num_samples/self.sampling_rate, 
                       num_samples)
        
        # Generate 4-channel EEG data
        eeg_data = np.zeros((4, num_samples))
        
        for ch in range(4):
            # Channel-specific variations
            ch_modifier = 0.8 + 0.4 * (ch / 3)  # Slight variation between channels
            
            # Alpha component (8-13 Hz, using 10 Hz)
            alpha_freq = 10 + np.random.normal(0, 0.5)  # Slight frequency variation
            alpha_component = (profile["alpha_amplitude"] * ch_modifier * 
                             np.sin(2 * np.pi * alpha_freq * t))
            
            # Beta component (13-30 Hz, using 20 Hz)
            beta_freq = 20 + np.random.normal(0, 1.0)
            beta_component = (profile["beta_amplitude"] * ch_modifier * 
                            np.sin(2 * np.pi * beta_freq * t))
            
            # Theta component (4-8 Hz, using 6 Hz)
            theta_freq = 6 + np.random.normal(0, 0.3)
            theta_component = (profile["theta_amplitude"] * ch_modifier * 
                             np.sin(2 * np.pi * theta_freq * t))
            
            # Gamma component (30-50 Hz, using 40 Hz)
            gamma_freq = 40 + np.random.normal(0, 2.0)
            gamma_component = (15 * ch_modifier * 
                             np.sin(2 * np.pi * gamma_freq * t))
            
            # Delta component (0.5-4 Hz, using 2 Hz)
            delta_freq = 2 + np.random.normal(0, 0.2)
            delta_component = (20 * ch_modifier * 
                             np.sin(2 * np.pi * delta_freq * t))
            
            # Add realistic noise
            noise = np.random.normal(0, profile["noise_level"], num_samples)
            
            # Combine all components
            eeg_data[ch, :] = (alpha_component + beta_component + theta_component + 
                             gamma_component + delta_component + noise)
            
            # Add occasional artifacts
            if np.random.random() < 0.02:  # 2% chance of artifact
                artifact_start = np.random.randint(0, num_samples - 10)
                artifact_end = min(artifact_start + 10, num_samples)
                eeg_data[ch, artifact_start:artifact_end] += np.random.normal(0, 100, artifact_end - artifact_start)
        
        # Update time offset
        self.time_offset += num_samples / self.sampling_rate
        
        return eeg_data
    
    def get_signal_quality_report(self) -> Dict[str, Any]:
        """Simulate signal quality report"""
        # Vary quality based on time (simulate contact changes)
        base_quality = 0.7 + 0.2 * np.sin(time.time() / 10)  # Slow variation
        
        return {
            'overall_quality': base_quality,
            'TP9_quality': base_quality + np.random.normal(0, 0.1),
            'AF7_quality': base_quality + np.random.normal(0, 0.1),
            'AF8_quality': base_quality + np.random.normal(0, 0.1),
            'TP10_quality': base_quality + np.random.normal(0, 0.1),
            'data_rate': self.sampling_rate,
            'connection_duration': time.time() - self.start_time,
            'is_streaming': self.is_streaming
        }

class MentalStateVisualizerDemo:
    """Demo application without real Muse hardware"""
    
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.muse_simulator = MuseSimulator(sampling_rate=256)
        self.signal_processor = AdvancedSignalProcessor(sampling_rate=256)
        self.mental_state_classifier = MentalStateClassifier()
        self.visualizer = RealTimeVisualizer(update_interval=0.1, buffer_size=3000)
        
        self.running = False
        self.processing_thread = None
    
    def run_processing_loop(self):
        """Main processing loop"""
        self.logger.info("Starting demo processing loop...")
        
        while self.running:
            try:
                # Get simulated EEG data
                eeg_data = self.muse_simulator.get_eeg_data(num_samples=64)
                
                if eeg_data is not None:
                    # Process the data
                    processing_results = self.signal_processor.process_eeg_comprehensive(eeg_data)
                    
                    if processing_results:
                        # Extract features
                        spectral_features = processing_results.get('spectral_features', {})
                        temporal_features = processing_results.get('temporal_features', {})
                        cross_features = processing_results.get('cross_channel_features', {})
                        
                        # Classify mental state
                        mental_state_result = self.mental_state_classifier.predict_mental_state(
                            spectral_features, temporal_features, cross_features
                        )
                        
                        # Update visualizer
                        timestamp = time.time()
                        self.visualizer.update_eeg_data(eeg_data, timestamp)
                        self.visualizer.update_spectral_features(spectral_features)
                        self.visualizer.update_mental_state(mental_state_result)
                        
                        # Update signal quality
                        signal_quality = self.muse_simulator.get_signal_quality_report()
                        self.visualizer.update_signal_quality(signal_quality)
                        
                        # Log current state periodically
                        if int(timestamp) % 10 == 0 and timestamp - int(timestamp) < 0.1:
                            actual_state = self.muse_simulator.current_mental_state
                            predicted_state = mental_state_result.get('state', 'unknown')
                            confidence = mental_state_result.get('confidence', 0.0)
                            
                            self.logger.info(
                                f"Simulated: {actual_state} | "
                                f"Predicted: {predicted_state} | "
                                f"Confidence: {confidence:.2f}"
                            )
                
                # Control timing
                time.sleep(0.1)  # 10 Hz processing
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
        
        self.logger.info("Demo processing loop stopped")
    
    def start(self):
        """Start the demo"""
        self.logger.info("=== Mental State Visualizer Demo ===")
        self.logger.info("This demo simulates EEG data to show the complete pipeline")
        self.logger.info("State progression: neutral → relaxed → focused → meditative → alert → neutral")
        self.logger.info("Each state lasts ~15 seconds")
        
        try:
            # Start simulator
            self.muse_simulator.start_streaming()
            
            # Initialize visualization GUI (must be on main thread)
            self.visualizer.initialize_gui()
            
            # Start processing in background thread
            self.running = True
            self.processing_thread = threading.Thread(target=self.run_processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start visualization animation
            self.visualizer.start_visualization()
            
            # Run visualization main loop (this blocks and must be on main thread)
            self.visualizer.run()
            
        except Exception as e:
            self.logger.error(f"Demo error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the demo"""
        self.logger.info("Stopping demo...")
        self.running = False
        self.muse_simulator.stop_streaming()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)


def main():
    """Run the demo"""
    print("🧠 Mental State Visualizer - DEMO MODE")
    print("=" * 50)
    print("This demo simulates Muse EEG data to demonstrate the system.")
    print("No actual Muse headset is required.")
    print()
    print("Features demonstrated:")
    print("✅ Real-time EEG signal simulation")
    print("✅ Advanced signal processing")
    print("✅ Mental state classification")
    print("✅ Real-time visualization")
    print("✅ Signal quality monitoring")
    print()
    print("The demo will cycle through different mental states:")
    print("Neutral → Relaxed → Focused → Meditative → Alert → Neutral")
    print()
    input("Press Enter to start the demo...")
    
    demo = MentalStateVisualizerDemo()
    demo.start()

if __name__ == "__main__":
    main()
