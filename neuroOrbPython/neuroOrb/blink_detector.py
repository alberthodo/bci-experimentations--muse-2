"""
Blink Detector for BCI Orb Control Project
Implements voluntary blink and double blink detection from EOG signals
"""

import numpy as np
from scipy import signal
from typing import List, Dict, Any
import logging


class BlinkDetector:
    """
    Handles detection of voluntary blinks and double blinks from EOG signals
    """
    
    def __init__(self, sampling_rate: int = 256):
        """
        Initialize blink detector
        
        Args:
            sampling_rate: EEG sampling rate in Hz (default 256 for Muse)
        """
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
        
        # Blink detection parameters
        self.blink_threshold = 200.0  # Amplitude threshold for voluntary blinks (microvolts)
        self.double_blink_time_window = 1.2  # Maximum time between blinks for double blink (seconds)
        self.min_blink_interval = 0.2  # Minimum time between blinks (seconds)
        self.min_double_blink_interval = 0.2  # Minimum time for double blink (seconds)
        
        # Blink duration constraints
        self.min_blink_duration = 0.05  # 50ms minimum blink duration
        self.max_blink_duration = 0.3   # 300ms maximum blink duration
        
    def extract_eog_signals(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract EOG signals from frontal electrodes for blink detection
        
        Args:
            eeg_data: EEG data with shape (channels, samples)
            
        Returns:
            EOG signals from frontal channels (AF7, AF8)
        """
        if eeg_data is None or eeg_data.size == 0:
            return np.array([])
            
        try:
            # Extract frontal channels (AF7, AF8) - channels 1 and 2 for Muse
            if eeg_data.shape[0] >= 3:
                # Use AF7 and AF8 channels
                eog_signals = eeg_data[1:3, :]  # Channels 1 and 2
            else:
                # Fallback to first two channels
                eog_signals = eeg_data[:2, :]
                
            return eog_signals
            
        except Exception as e:
            self.logger.error(f"Error extracting EOG signals: {e}")
            return np.array([])
    
    def detect_blinks(self, eog_data: np.ndarray, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Detect voluntary blinks in EOG signals
        
        Args:
            eog_data: EOG signals with shape (channels, samples)
            threshold: Amplitude threshold for voluntary blink detection (microvolts)
            
        Returns:
            List of detected voluntary blinks with timing information
        """
        if eog_data is None or eog_data.size == 0:
            return []
            
        if threshold is None:
            threshold = self.blink_threshold
            
        try:
            blinks = []
            
            for channel_idx in range(eog_data.shape[0]):
                channel_data = eog_data[channel_idx, :]
                
                # Use a higher threshold for voluntary blinks and look for negative peaks
                # Voluntary blinks typically show as negative deflections in EOG
                negative_data = -channel_data  # Invert to find negative peaks
                
                # Find peaks with stricter criteria for voluntary blinks
                peaks, properties = signal.find_peaks(
                    negative_data, 
                    height=threshold,  # Higher threshold for voluntary blinks
                    distance=int(self.sampling_rate * self.min_blink_interval),  # Minimum time between blinks
                    prominence=threshold * 0.5,  # Require significant prominence
                    width=(int(self.sampling_rate * self.min_blink_duration), 
                          int(self.sampling_rate * self.max_blink_duration))  # Blink duration constraints
                )
                
                # Convert peak indices to time
                for peak_idx in peaks:
                    blink_time = peak_idx / self.sampling_rate
                    amplitude = channel_data[peak_idx]  # Original amplitude (negative)
                    
                    # Only include blinks that are strong enough to be voluntary
                    if abs(amplitude) > threshold:
                        blinks.append({
                            'channel': channel_idx,
                            'time': blink_time,
                            'amplitude': amplitude,
                            'sample_index': peak_idx
                        })
                    
            return blinks
            
        except Exception as e:
            self.logger.error(f"Error in blink detection: {e}")
            return []
    
    def detect_double_blinks(self, blinks: List[Dict[str, Any]], 
                           time_window: float = None) -> List[Dict[str, Any]]:
        """
        Detect voluntary double blinks within a time window
        
        Args:
            blinks: List of detected voluntary blinks
            time_window: Maximum time between blinks to count as double blink (seconds)
            
        Returns:
            List of detected voluntary double blinks
        """
        if not blinks:
            return []
            
        if time_window is None:
            time_window = self.double_blink_time_window
            
        try:
            double_blinks = []
            blinks_sorted = sorted(blinks, key=lambda x: x['time'])
            
            i = 0
            while i < len(blinks_sorted) - 1:
                current_blink = blinks_sorted[i]
                next_blink = blinks_sorted[i + 1]
                
                time_diff = next_blink['time'] - current_blink['time']
                
                # Stricter criteria for voluntary double blinks
                if self.min_double_blink_interval <= time_diff <= time_window:
                    # Both blinks should be strong enough (voluntary)
                    if (abs(current_blink['amplitude']) > 150 and 
                        abs(next_blink['amplitude']) > 150):
                        
                        # Found a voluntary double blink
                        double_blink = {
                            'first_blink': current_blink,
                            'second_blink': next_blink,
                            'time_interval': time_diff,
                            'detection_time': current_blink['time'],
                            'strength': (abs(current_blink['amplitude']) + abs(next_blink['amplitude'])) / 2
                        }
                        double_blinks.append(double_blink)
                        i += 2  # Skip both blinks
                    else:
                        i += 1
                else:
                    i += 1
                    
            return double_blinks
            
        except Exception as e:
            self.logger.error(f"Error in double blink detection: {e}")
            return []
    
    def process_eog_data(self, eeg_data: np.ndarray) -> Dict[str, Any]:
        """
        Complete EOG processing pipeline for blink detection
        
        Args:
            eeg_data: Raw EEG data with shape (channels, samples)
            
        Returns:
            Dictionary with blink detection results
        """
        try:
            # Extract EOG signals
            eog_signals = self.extract_eog_signals(eeg_data)
            
            if eog_signals.size == 0:
                return {
                    'eog_signals': np.array([]),
                    'blinks': [],
                    'double_blinks': []
                }
            
            # Detect blinks
            blinks = self.detect_blinks(eog_signals)
            
            # Detect double blinks
            double_blinks = self.detect_double_blinks(blinks)
            
            return {
                'eog_signals': eog_signals,
                'blinks': blinks,
                'double_blinks': double_blinks
            }
            
        except Exception as e:
            self.logger.error(f"Error in EOG processing: {e}")
            return {
                'eog_signals': np.array([]),
                'blinks': [],
                'double_blinks': []
            }
    
    def update_thresholds(self, blink_threshold: float = None, 
                         double_blink_window: float = None,
                         min_blink_interval: float = None):
        """
        Update detection thresholds
        
        Args:
            blink_threshold: New amplitude threshold for blinks
            double_blink_window: New time window for double blinks
            min_blink_interval: New minimum interval between blinks
        """
        if blink_threshold is not None:
            self.blink_threshold = blink_threshold
            
        if double_blink_window is not None:
            self.double_blink_time_window = double_blink_window
            
        if min_blink_interval is not None:
            self.min_blink_interval = min_blink_interval
            
        self.logger.info(f"Updated thresholds: blink={self.blink_threshold}, "
                        f"double_window={self.double_blink_time_window}, "
                        f"min_interval={self.min_blink_interval}")


def main():
    """
    Test function for blink detector
    """
    import time
    
    print("=== Blink Detector Test ===")
    
    # Create blink detector
    detector = BlinkDetector(sampling_rate=256)
    
    # Generate test EOG data with simulated blinks
    test_data = np.random.randn(4, 256) * 30  # 4 channels, 256 samples, ~30 microvolts
    
    # Add simulated voluntary blinks (negative deflections)
    test_data[1, 50:70] -= 250  # Simulate blink in AF7 channel
    test_data[1, 80:100] -= 200  # Simulate second blink
    test_data[2, 120:140] -= 300  # Simulate blink in AF8 channel
    
    print(f"Test data shape: {test_data.shape}")
    
    # Test EOG extraction
    print("\nTesting EOG extraction...")
    eog_signals = detector.extract_eog_signals(test_data)
    print(f"EOG signals shape: {eog_signals.shape}")
    
    # Test blink detection
    print("\nTesting blink detection...")
    blinks = detector.detect_blinks(eog_signals)
    print(f"Detected {len(blinks)} blinks")
    for i, blink in enumerate(blinks):
        print(f"  Blink {i+1}: {blink['amplitude']:.1f}μV at {blink['time']:.2f}s")
    
    # Test double blink detection
    print("\nTesting double blink detection...")
    double_blinks = detector.detect_double_blinks(blinks)
    print(f"Detected {len(double_blinks)} double blinks")
    for i, db in enumerate(double_blinks):
        print(f"  Double blink {i+1}: {db['time_interval']:.3f}s interval")
    
    # Test complete processing
    print("\nTesting complete EOG processing...")
    results = detector.process_eog_data(test_data)
    print(f"Processing results keys: {list(results.keys())}")
    
    print("\nBlink detector test completed!")


if __name__ == "__main__":
    main()
