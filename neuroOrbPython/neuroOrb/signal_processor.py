"""
Signal Processor for BCI Orb Control Project
Implements EEG analysis including preprocessing, feature extraction, and blink detection
"""

import numpy as np
import time
from scipy import signal
from scipy.signal import butter, filtfilt, welch
from typing import Dict, List, Optional, Tuple
import logging
from blink_detector import BlinkDetector


class SignalProcessor:
    """
    Handles EEG signal processing including filtering, feature extraction, and blink detection
    """
    
    def __init__(self, sampling_rate: int = 256):
        """
        Initialize signal processor
        
        Args:
            sampling_rate: EEG sampling rate in Hz (default 256 for Muse)
        """
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
        
        # Frequency bands for EEG analysis
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Channel mapping for Muse (based on typical EEG electrode positions)
        self.channel_mapping = {
            0: "TP9",   # Left temporal
            1: "AF7",   # Left frontal (good for EOG/blink detection)
            2: "AF8",   # Right frontal (good for EOG/blink detection)
            3: "TP10",  # Right temporal
        }
        
        # Initialize filters
        self._setup_filters()
        
        # Initialize blink detector
        self.blink_detector = BlinkDetector(sampling_rate=sampling_rate)
        
        # Buffer for continuous processing
        self.buffer_size = sampling_rate * 2  # 2 seconds buffer
        self.eeg_buffer = None
        self.eog_buffer = None
        
    def _setup_filters(self):
        """Setup digital filters for preprocessing"""
        nyquist = self.sampling_rate / 2
        
        # Notch filter for 50/60 Hz power line noise
        self.notch_freq = 50.0  # Can be adjusted for 60Hz if needed
        self.notch_quality = 30.0
        
        # Bandpass filter for EEG (1-50 Hz)
        self.lowcut = 1.0
        self.highcut = 50.0
        
        # Design notch filter
        b_notch, a_notch = signal.iirnotch(self.notch_freq, self.notch_quality, self.sampling_rate)
        self.notch_filter = (b_notch, a_notch)
        
        # Design bandpass filter
        b_band, a_band = butter(4, [self.lowcut/nyquist, self.highcut/nyquist], btype='band')
        self.bandpass_filter = (b_band, a_band)
        
        self.logger.info(f"Filters initialized: notch={self.notch_freq}Hz, bandpass={self.lowcut}-{self.highcut}Hz")
    
    def preprocess_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing filters to EEG data
        
        Args:
            eeg_data: Raw EEG data with shape (channels, samples)
            
        Returns:
            Preprocessed EEG data
        """
        if eeg_data is None or eeg_data.size == 0:
            return eeg_data
            
        try:
            processed_data = np.zeros_like(eeg_data)
            
            for channel in range(eeg_data.shape[0]):
                channel_data = eeg_data[channel, :]
                
                # Apply notch filter to remove power line noise
                filtered_data = signal.filtfilt(self.notch_filter[0], self.notch_filter[1], channel_data)
                
                # Apply bandpass filter
                filtered_data = signal.filtfilt(self.bandpass_filter[0], self.bandpass_filter[1], filtered_data)
                
                processed_data[channel, :] = filtered_data
                
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            return eeg_data
    
    def extract_power_spectral_density(self, eeg_data: np.ndarray, window_size: int = 256) -> Dict[str, np.ndarray]:
        """
        Calculate power spectral density for each frequency band
        
        Args:
            eeg_data: Preprocessed EEG data with shape (channels, samples)
            window_size: Window size for PSD calculation
            
        Returns:
            Dictionary with power values for each frequency band per channel
        """
        if eeg_data is None or eeg_data.size == 0:
            return {}
            
        try:
            psd_results = {}
            
            for channel in range(eeg_data.shape[0]):
                channel_data = eeg_data[channel, :]
                
                # Use Welch's method for PSD estimation
                freqs, psd = welch(channel_data, fs=self.sampling_rate, nperseg=window_size)
                
                # Calculate power in each frequency band
                band_powers = {}
                for band_name, (low_freq, high_freq) in self.freq_bands.items():
                    # Find frequency indices
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.sum(psd[freq_mask])
                    band_powers[band_name] = band_power
                
                psd_results[f"channel_{channel}"] = band_powers
                
            return psd_results
            
        except Exception as e:
            self.logger.error(f"Error in PSD extraction: {e}")
            return {}
    
    def calculate_alpha_beta_ratio(self, psd_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate alpha/beta ratio for focus/relaxation classification
        
        Args:
            psd_results: PSD results from extract_power_spectral_density
            
        Returns:
            Dictionary with alpha/beta ratios for each channel
        """
        ratios = {}
        
        try:
            for channel_key, band_powers in psd_results.items():
                alpha_power = band_powers.get('alpha', 0)
                beta_power = band_powers.get('beta', 0)
                
                # Avoid division by zero
                if beta_power > 0:
                    ratio = alpha_power / beta_power
                else:
                    ratio = alpha_power if alpha_power > 0 else 0
                    
                ratios[channel_key] = ratio
                
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating alpha/beta ratio: {e}")
            return {}
    
    
    
    
    def classify_mental_state(self, psd_results: Dict[str, Dict[str, float]], 
                            relaxed_threshold: float = 0.6, 
                            focused_threshold: float = 0.4) -> Dict[str, str]:
        """
        Classify mental state based on average alpha/beta ratio across all channels
        
        Args:
            psd_results: PSD results from extract_power_spectral_density
            relaxed_threshold: Threshold above which state is "relaxed"
            focused_threshold: Threshold below which state is "focused"
            
        Returns:
            Dictionary with overall mental state classification
        """
        try:
            ratios = self.calculate_alpha_beta_ratio(psd_results)
            
            if not ratios:
                return {"overall_state": "unknown", "average_ratio": 0.0}
            
            # Calculate average ratio across all channels
            ratio_values = list(ratios.values())
            average_ratio = sum(ratio_values) / len(ratio_values)
            
            # Classify based on average ratio
            if average_ratio >= relaxed_threshold:
                overall_state = "relaxed"
            else:
                overall_state = "focused"
            
            return {
                "overall_state": overall_state,
                "average_ratio": average_ratio,
                "individual_ratios": ratios
            }
            
        except Exception as e:
            self.logger.error(f"Error in mental state classification: {e}")
            return {"overall_state": "unknown", "average_ratio": 0.0}
    
    def process_eeg_sample(self, eeg_data: np.ndarray) -> Dict[str, any]:
        """
        Complete processing pipeline for a single EEG sample
        
        Args:
            eeg_data: Raw EEG data with shape (channels, samples)
            
        Returns:
            Dictionary with all processing results
        """
        try:
            # Preprocess the data
            processed_data = self.preprocess_eeg(eeg_data)
            
            # Extract PSD features
            psd_results = self.extract_power_spectral_density(processed_data)
            
            # Calculate alpha/beta ratios
            ratios = self.calculate_alpha_beta_ratio(psd_results)
            
            # Classify mental state
            mental_state = self.classify_mental_state(psd_results)
            
            # Process EOG data for blink detection
            blink_results = self.blink_detector.process_eog_data(processed_data)
            eog_signals = blink_results['eog_signals']
            blinks = blink_results['blinks']
            double_blinks = blink_results['double_blinks']
            
            return {
                'processed_data': processed_data,
                'psd_results': psd_results,
                'alpha_beta_ratios': ratios,
                'mental_state': mental_state,
                'eog_signals': eog_signals,
                'blinks': blinks,
                'double_blinks': double_blinks,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error in complete processing: {e}")
            return {}


def main():
    """
    Test function for signal processor
    """
    import time
    
    print("=== Signal Processor Test ===")
    
    # Create signal processor
    processor = SignalProcessor(sampling_rate=256)
    
    # Generate test EEG data (simulate 4 channels, 1 second of data)
    test_data = np.random.randn(4, 256) * 50  # 4 channels, 256 samples, ~50 microvolts
    
    print(f"Test data shape: {test_data.shape}")
    
    # Test preprocessing
    print("\nTesting preprocessing...")
    processed = processor.preprocess_eeg(test_data)
    print(f"Processed data shape: {processed.shape}")
    
    # Test PSD extraction
    print("\nTesting PSD extraction...")
    psd_results = processor.extract_power_spectral_density(processed)
    for channel, bands in psd_results.items():
        print(f"{channel}: {bands}")
    
    # Test alpha/beta ratios
    print("\nTesting alpha/beta ratios...")
    ratios = processor.calculate_alpha_beta_ratio(psd_results)
    for channel, ratio in ratios.items():
        print(f"{channel}: {ratio:.3f}")
    
    # Test mental state classification
    print("\nTesting mental state classification...")
    mental_state = processor.classify_mental_state(psd_results)
    for channel, state in mental_state.items():
        print(f"{channel}: {state}")
    
    # Test EOG extraction
    print("\nTesting EOG extraction...")
    eog_signals = processor.extract_eog_signals(processed)
    print(f"EOG signals shape: {eog_signals.shape}")
    
    # Test blink detection
    print("\nTesting blink detection...")
    blinks = processor.detect_blinks(eog_signals)
    print(f"Detected {len(blinks)} blinks")
    
    # Test complete processing
    print("\nTesting complete processing pipeline...")
    results = processor.process_eeg_sample(test_data)
    print(f"Processing results keys: {list(results.keys())}")
    
    print("\nSignal processor test completed!")


if __name__ == "__main__":
    main()
