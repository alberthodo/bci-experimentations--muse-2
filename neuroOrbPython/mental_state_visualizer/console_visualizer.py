"""
Console-based Mental State Visualizer
Real-time mental state monitoring without GUI - shows data in terminal
"""

import time
import logging
import signal
import sys
from typing import Dict, Any
import numpy as np

from enhanced_muse_connector import EnhancedMuseConnector
from advanced_signal_processor import AdvancedSignalProcessor
from mental_state_classifier import MentalStateClassifier


class ConsoleMentalStateVisualizer:
    """
    Console-based mental state visualizer - no GUI, just terminal output
    """
    
    def __init__(self, board_type: str = "MUSE_2", mac_address: str = ""):
        """
        Initialize the console visualizer
        
        Args:
            board_type: Type of Muse board
            mac_address: MAC address of device (empty for auto-scan)
        """
        self.board_type = board_type
        self.mac_address = mac_address
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.muse_connector = None
        self.signal_processor = None
        self.mental_state_classifier = None
        
        self.running = False
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'mental_states': {},
            'start_time': None,
            'last_state': 'unknown',
            'state_changes': 0
        }
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Stopping Mental State Visualizer...")
        self.running = False
    
    def initialize_components(self) -> bool:
        """Initialize all components"""
        try:
            print("🔧 Initializing components...")
            
            # Initialize Muse connector
            self.muse_connector = EnhancedMuseConnector(
                board_type=self.board_type,
                mac_address=self.mac_address
            )
            
            # Initialize signal processor
            self.signal_processor = AdvancedSignalProcessor(sampling_rate=256)
            
            # Initialize mental state classifier
            self.mental_state_classifier = MentalStateClassifier()
            
            print("✅ Components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            return False
    
    def connect_to_muse(self) -> bool:
        """Connect to Muse headset"""
        try:
            print("🔌 Connecting to Muse headset...")
            
            if not self.muse_connector.connect():
                print("❌ Failed to connect to Muse headset")
                return False
            
            if not self.muse_connector.start_streaming():
                print("❌ Failed to start EEG streaming")
                return False
            
            # Update signal processor sampling rate
            actual_sampling_rate = self.muse_connector.sampling_rate
            if actual_sampling_rate:
                self.signal_processor.sampling_rate = actual_sampling_rate
            
            print(f"✅ Connected to Muse! Sampling at {actual_sampling_rate} Hz")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to Muse: {e}")
            return False
    
    def display_header(self):
        """Display the console header"""
        print("\n" + "="*80)
        print("🧠 MENTAL STATE VISUALIZER - CONSOLE MODE")
        print("="*80)
        print("Real-time mental state monitoring from your Muse headset")
        print("Press Ctrl+C to stop")
        print("-"*80)
        print(f"{'Time':<12} {'Mental State':<15} {'Conf.':<6} {'Quality':<8} {'Samples':<8} {'Notes':<20}")
        print("-"*80)
    
    def format_mental_state(self, state: str) -> str:
        """Format mental state with emoji"""
        state_emojis = {
            'relaxed': '😌 Relaxed',
            'focused': '🎯 Focused', 
            'meditative': '🧘 Meditative',
            'alert': '⚡ Alert',
            'neutral': '😐 Neutral',
            'stressed': '😰 Stressed',
            'unknown': '❓ Unknown'
        }
        return state_emojis.get(state, f"❓ {state}")
    
    def format_quality(self, quality: float) -> str:
        """Format signal quality with color coding"""
        if quality > 0.8:
            return f"🟢 {quality:.2f}"
        elif quality > 0.6:
            return f"🟡 {quality:.2f}"
        else:
            return f"🔴 {quality:.2f}"
    
    def update_statistics(self, mental_state: str):
        """Update session statistics"""
        self.stats['total_samples'] += 1
        
        # Track mental state frequency
        if mental_state in self.stats['mental_states']:
            self.stats['mental_states'][mental_state] += 1
        else:
            self.stats['mental_states'][mental_state] = 1
        
        # Track state changes
        if mental_state != self.stats['last_state'] and self.stats['last_state'] != 'unknown':
            self.stats['state_changes'] += 1
        
        self.stats['last_state'] = mental_state
    
    def display_real_time_data(self, mental_state_result: Dict[str, Any], 
                              signal_quality: Dict[str, Any], sample_count: int):
        """Display real-time data in console"""
        
        # Extract key information
        state = mental_state_result.get('state', 'unknown')
        confidence = mental_state_result.get('confidence', 0.0)
        overall_quality = signal_quality.get('overall_quality', 0.0)
        
        # Update statistics
        self.update_statistics(state)
        
        # Calculate session duration
        elapsed = time.time() - self.stats['start_time']
        time_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        # Generate notes
        notes = []
        if confidence > 0.8:
            notes.append("High conf.")
        if overall_quality < 0.5:
            notes.append("Poor signal")
        if self.stats['state_changes'] > 0 and sample_count % 10 == 0:
            notes.append(f"{self.stats['state_changes']} changes")
        
        notes_str = " | ".join(notes[:2])  # Limit to 2 notes
        
        # Display the line
        print(f"{time_str:<12} {self.format_mental_state(state):<15} "
              f"{confidence:.2f}  {self.format_quality(overall_quality):<8} "
              f"{sample_count:<8} {notes_str:<20}")
    
    def display_summary(self):
        """Display session summary"""
        elapsed = time.time() - self.stats['start_time']
        
        print("\n" + "="*80)
        print("📊 SESSION SUMMARY")
        print("="*80)
        print(f"Session Duration: {int(elapsed//60):02d}:{int(elapsed%60):02d}")
        print(f"Total Samples: {self.stats['total_samples']}")
        print(f"State Changes: {self.stats['state_changes']}")
        print(f"Average Rate: {self.stats['total_samples']/elapsed:.1f} samples/sec")
        
        print("\n📈 Mental State Distribution:")
        total_states = sum(self.stats['mental_states'].values())
        for state, count in sorted(self.stats['mental_states'].items(), 
                                 key=lambda x: x[1], reverse=True):
            percentage = (count / total_states) * 100
            bar_length = int(percentage / 2)  # Scale for display
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {self.format_mental_state(state):<15} {count:>4} ({percentage:>5.1f}%) {bar}")
        
        print("\n🎯 Most Common State:", self.format_mental_state(
            max(self.stats['mental_states'], key=self.stats['mental_states'].get)
        ))
        print("="*80)
    
    def run(self) -> bool:
        """Run the console visualizer"""
        try:
            print("🧠 Mental State Visualizer - Console Mode")
            print("=" * 50)
            
            # Initialize components
            if not self.initialize_components():
                return False
            
            # Connect to Muse
            if not self.connect_to_muse():
                return False
            
            # Display header
            self.display_header()
            
            # Start monitoring
            self.running = True
            self.stats['start_time'] = time.time()
            
            sample_count = 0
            last_display_time = time.time()
            
            while self.running:
                try:
                    # Get EEG data
                    eeg_data = self.muse_connector.get_eeg_data(num_samples=64)
                    
                    if eeg_data is not None and eeg_data.size > 0:
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
                            
                            # Get signal quality
                            signal_quality = self.muse_connector.get_signal_quality_report()
                            
                            # Display update every 2 seconds
                            current_time = time.time()
                            if current_time - last_display_time >= 2.0:
                                self.display_real_time_data(
                                    mental_state_result, signal_quality, sample_count
                                )
                                last_display_time = current_time
                            
                            sample_count += 1
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ Processing error: {e}")
                    time.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"❌ Application error: {e}")
            return False
        
        finally:
            # Display summary
            if self.stats['start_time']:
                self.display_summary()
            
            # Cleanup
            if self.muse_connector:
                self.muse_connector.disconnect()
            
            print("👋 Mental State Visualizer stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Console Mental State Visualizer")
    parser.add_argument("--board-type", default="MUSE_2", 
                       choices=["MUSE_2", "MUSE_S", "MUSE_2016"],
                       help="Type of Muse board")
    parser.add_argument("--mac-address", default="", 
                       help="MAC address of Muse device")
    
    args = parser.parse_args()
    
    # Create and run visualizer
    visualizer = ConsoleMentalStateVisualizer(
        board_type=args.board_type,
        mac_address=args.mac_address
    )
    
    success = visualizer.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
