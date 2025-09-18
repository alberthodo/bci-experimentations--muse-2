#!/usr/bin/env python3
"""
BCI Control System Console Visualizer
Tracks 5 states: normal, double blink, jaw clench, sustained focus, sustained relaxation
"""

import sys
import os
import time
import threading
from collections import deque
import logging

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
mental_state_path = os.path.join(project_root, 'mental_state_visualizer')
sys.path.extend([current_dir, project_root, mental_state_path])

# Import advanced components from mental_state_visualizer
try:
    from enhanced_muse_connector import EnhancedMuseConnector
    from mental_state_classifier import MentalStateClassifier
    print("Successfully imported components from mental_state_visualizer")
except ImportError as e:
    print(f"Warning: Could not import some components: {e}")
    EnhancedMuseConnector = None
    MentalStateClassifier = None

# Import our custom artifact preserving processor
from artifact_preserving_processor import ArtifactPreservingProcessor

class BCIStateTracker:
    """
    Tracks 5 BCI states: normal, double blink, jaw clench, sustained focus, sustained relaxation
    """
    
    def __init__(self):
        # System parameters
        self.sampling_rate = 256
        self.running = False
        
        # State tracking
        self.current_state = "normal"
        self.state_history = deque(maxlen=100)
        self.state_start_time = time.time()
        
        # State durations for sustained states
        self.focus_duration_threshold = 3.0  # seconds
        self.relaxation_duration_threshold = 3.0  # seconds
        
        # Initialize components
        self.signal_processor = None
        self.muse_connector = None
        self.mental_state_classifier = None
        
        # Event tracking
        self.last_double_blink_time = 0
        self.last_jaw_clench_time = 0
        self.double_blink_cooldown = 2.0  # seconds
        self.jaw_clench_cooldown = 2.0  # seconds
        
        # Mental state tracking
        self.mental_state_buffer = deque(maxlen=30)  # ~1 second at 30Hz
        self.current_mental_state = "neutral"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def initialize_components(self):
        """Initialize all BCI components"""
        try:
            # Initialize artifact preserving signal processor
            self.signal_processor = ArtifactPreservingProcessor(sampling_rate=self.sampling_rate)
            self.logger.info("Artifact preserving signal processor initialized")
                
            if EnhancedMuseConnector is not None:
                self.muse_connector = EnhancedMuseConnector(board_type="MUSE_2")
                self.logger.info("Enhanced Muse connector initialized")
                
            if MentalStateClassifier is not None:
                self.mental_state_classifier = MentalStateClassifier()
                self.logger.info("Mental state classifier initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            
    def connect_muse(self):
        """Connect to Muse headband"""
        if self.muse_connector is None:
            self.logger.error("Muse connector not available")
            return False
            
        try:
            success = self.muse_connector.connect()
            if success:
                self.logger.info("Connected to Muse headband")
                return True
            else:
                self.logger.error("Failed to connect to Muse")
                return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
            
    def start_streaming(self):
        """Start EEG data streaming"""
        if self.muse_connector is None:
            self.logger.error("Muse connector not available")
            return False
            
        try:
            success = self.muse_connector.start_streaming()
            if success:
                self.logger.info("Started EEG streaming")
                return True
            else:
                self.logger.error("Failed to start streaming")
                return False
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            return False
            
    def process_eeg_data(self, eeg_data):
        """Process EEG data and detect state changes"""
        if eeg_data is None or eeg_data.size == 0:
            return
            
        current_time = time.time()
        
        # Process with artifact preserving signal processor
        if self.signal_processor is not None:
            results = self.signal_processor.process_eeg_sample(eeg_data)
            
            # Extract double blink information
            if 'double_blinks' in results:
                double_blinks = results['double_blinks']
                if double_blinks and (current_time - self.last_double_blink_time) > self.double_blink_cooldown:
                    self._handle_double_blink(current_time)
                    
            # Extract jaw clench information
            if 'jaw_clenches' in results:
                jaw_clenches = results['jaw_clenches']
                if jaw_clenches and (current_time - self.last_jaw_clench_time) > self.jaw_clench_cooldown:
                    self._handle_jaw_clench(current_time)
                    
            # Extract mental state information
            if 'mental_state' in results:
                mental_state = results['mental_state']
                overall_state = mental_state.get('overall_state', 'neutral')
                self.mental_state_buffer.append(overall_state)
                
        # Update mental state classification
        self._update_mental_state_classification(current_time)
        
        
    def _handle_double_blink(self, current_time):
        """Handle double blink detection"""
        self.current_state = "double_blink"
        self.state_start_time = current_time
        self.last_double_blink_time = current_time
        self.state_history.append(("double_blink", current_time))
        self.logger.info(f"State: DOUBLE BLINK detected at {current_time:.1f}s")
        
    def _handle_jaw_clench(self, current_time):
        """Handle jaw clench detection"""
        self.current_state = "jaw_clench"
        self.state_start_time = current_time
        self.last_jaw_clench_time = current_time
        self.state_history.append(("jaw_clench", current_time))
        self.logger.info(f"State: JAW CLENCH detected at {current_time:.1f}s")
        
    def _update_mental_state_classification(self, current_time):
        """Update mental state classification and detect sustained states"""
        if len(self.mental_state_buffer) < 10:  # Need enough samples
            return
            
        # Get most common mental state from recent buffer
        recent_states = list(self.mental_state_buffer)[-10:]
        most_common_state = max(set(recent_states), key=recent_states.count)
        
        # Check for sustained focus
        if most_common_state == "focused":
            if self.current_state == "sustained_focus":
                # Already in sustained focus, check duration
                duration = current_time - self.state_start_time
                if duration >= self.focus_duration_threshold:
                    return  # Keep sustained focus
            else:
                # Start tracking potential sustained focus
                self.current_state = "sustained_focus"
                self.state_start_time = current_time
                self.logger.info(f"State: SUSTAINED FOCUS started at {current_time:.1f}s")
                
        # Check for sustained relaxation
        elif most_common_state == "relaxed":
            if self.current_state == "sustained_relaxation":
                # Already in sustained relaxation, check duration
                duration = current_time - self.state_start_time
                if duration >= self.relaxation_duration_threshold:
                    return  # Keep sustained relaxation
            else:
                # Start tracking potential sustained relaxation
                self.current_state = "sustained_relaxation"
                self.state_start_time = current_time
                self.logger.info(f"State: SUSTAINED RELAXATION started at {current_time:.1f}s")
                
        # Return to normal state
        else:
            if self.current_state in ["sustained_focus", "sustained_relaxation"]:
                duration = current_time - self.state_start_time
                if duration >= 1.0:  # Minimum duration before returning to normal
                    self.current_state = "normal"
                    self.state_history.append(("normal", current_time))
                    self.logger.info(f"State: NORMAL at {current_time:.1f}s (was {self.current_state})")
                    
    def get_current_state(self):
        """Get current BCI state"""
        return self.current_state
        
    def get_state_history(self):
        """Get recent state history"""
        return list(self.state_history)
        
    def run(self):
        """Main processing loop"""
        self.logger.info("Starting BCI Control System...")
        
        # Initialize components
        self.initialize_components()
        
        # Connect to Muse
        if not self.connect_muse():
            self.logger.error("Failed to connect to Muse. Exiting.")
            return
            
        # Start streaming
        if not self.start_streaming():
            self.logger.error("Failed to start streaming. Exiting.")
            return
            
        self.running = True
        self.logger.info("BCI Control System running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                # Get EEG data
                eeg_data = self.muse_connector.get_eeg_data()
                
                if eeg_data is not None and eeg_data.size > 0:
                    # Process the data
                    self.process_eeg_data(eeg_data)
                    
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            self.logger.info("Stopping BCI Control System...")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        if self.muse_connector:
            try:
                self.muse_connector.disconnect()
                self.logger.info("Disconnected from Muse")
            except:
                pass
                
        self.logger.info("BCI Control System stopped")

def main():
    """Main function"""
    print("=== BCI Control System Console Visualizer ===")
    print("Tracking 5 states:")
    print("1. Normal (default)")
    print("2. Double Blink")
    print("3. Jaw Clench")
    print("4. Sustained Focus")
    print("5. Sustained Relaxation")
    print()
    
    # Create and run BCI state tracker
    tracker = BCIStateTracker()
    tracker.run()

if __name__ == "__main__":
    main()
