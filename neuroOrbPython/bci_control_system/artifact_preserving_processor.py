#!/usr/bin/env python3
"""
Artifact Preserving Signal Processor for BCI Control System
Preserves artifacts in separate channels before removing them for blink and jaw clench detection
"""

import numpy as np
import time
from scipy import signal
from scipy.signal import butter, filtfilt, welch, hilbert, periodogram
from scipy.stats import entropy
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class ArtifactPreservingProcessor:
    """
    Signal processor that preserves artifacts in separate channels before removing them
    """
    
    def __init__(self, sampling_rate: int = 256, buffer_duration: float = 10.0):
        """
        Initialize artifact preserving signal processor
        
        Args:
            sampling_rate: EEG sampling rate in Hz (default 256 for Muse)
            buffer_duration: Duration of signal buffer in seconds
        """
        self.sampling_rate = sampling_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sampling_rate * buffer_duration)
        self.logger = logging.getLogger(__name__)
        
        # Enhanced frequency bands for comprehensive analysis
        self.freq_bands = {
            'delta': (0.5, 4),      # Deep sleep, unconscious processes
            'theta': (4, 8),        # Deep meditation, creativity, intuition
            'alpha': (8, 13),       # Relaxed awareness, flow states
            'beta': (13, 30),       # Active concentration, analytical thinking
            'gamma': (30, 50),      # Higher cognitive processes, consciousness
            'low_gamma': (30, 40),  # Focused attention
            'high_gamma': (40, 50)  # Higher-order cognitive processing
        }
        
        # Channel mapping for Muse (based on 10-20 electrode positions)
        self.channel_mapping = {
            0: "TP9",   # Left temporal (emotion, memory)
            1: "AF7",   # Left frontal (executive function, attention)
            2: "AF8",   # Right frontal (executive function, attention)
            3: "TP10",  # Right temporal (emotion, memory)
        }
        
        # Initialize filters
        self._setup_advanced_filters()
        
        # Data buffers for continuous processing
        self.eeg_buffer = {ch: deque(maxlen=self.buffer_size) for ch in range(4)}
        self.filtered_buffer = {ch: deque(maxlen=self.buffer_size) for ch in range(4)}
        self.artifact_buffer = {ch: deque(maxlen=self.buffer_size) for ch in range(4)}
        
        # Feature history for temporal analysis
        self.feature_history = deque(maxlen=100)  # Keep last 100 feature vectors
        self.mental_state_history = deque(maxlen=50)  # Keep last 50 mental state classifications
        
        # Initialize feature scaler for normalization
        self.feature_scaler = StandardScaler()
        self.scaler_fitted = False
        
    def _setup_advanced_filters(self):
        """Setup comprehensive digital filters for advanced preprocessing"""
        nyquist = self.sampling_rate / 2
        
        # Multi-stage filtering approach
        
        # 1. High-pass filter to remove DC drift (0.5 Hz)
        self.highpass_freq = 0.5
        b_hp, a_hp = butter(4, self.highpass_freq/nyquist, btype='high')
        self.highpass_filter = (b_hp, a_hp)
        
        # 2. Low-pass filter for anti-aliasing (50 Hz)
        self.lowpass_freq = 50.0
        b_lp, a_lp = butter(4, self.lowpass_freq/nyquist, btype='low')
        self.lowpass_filter = (b_lp, a_lp)
        
        # 3. Notch filters for power line interference
        self.notch_freqs = [50.0, 60.0]  # 50Hz (EU) and 60Hz (US)
        self.notch_quality = 30.0
        self.notch_filters = []
        
        for freq in self.notch_freqs:
            if freq < nyquist:
                b_notch, a_notch = signal.iirnotch(freq, self.notch_quality, self.sampling_rate)
                self.notch_filters.append((b_notch, a_notch))
        
        # 4. Band-specific filters for feature extraction
        self.band_filters = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            if high_freq < nyquist:
                b_band, a_band = butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
                self.band_filters[band_name] = (b_band, a_band)
        
        # 5. EMG filter for jaw clench detection (30-100 Hz)
        self.emg_low = 30.0
        self.emg_high = 100.0
        if self.emg_high < nyquist:
            b_emg, a_emg = butter(4, [self.emg_low/nyquist, self.emg_high/nyquist], btype='band')
            self.emg_filter = (b_emg, a_emg)
        else:
            self.emg_filter = None
        
        self.logger.info(f"Advanced filters initialized: HP={self.highpass_freq}Hz, LP={self.lowpass_freq}Hz")
        self.logger.info(f"Notch filters: {self.notch_freqs}Hz, EMG filter: {self.emg_low}-{self.emg_high}Hz")
    
    def preprocess_eeg_with_artifacts(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply preprocessing pipeline while preserving artifacts in separate channels
        
        Args:
            eeg_data: Raw EEG data with shape (channels, samples)
            
        Returns:
            Dictionary containing:
            - 'clean_data': Preprocessed EEG data with artifacts removed
            - 'artifact_data': Preserved artifacts for blink/jaw clench detection
            - 'emg_data': EMG signals for jaw clench detection
        """
        if eeg_data is None or eeg_data.size == 0:
            return {
                'clean_data': eeg_data,
                'artifact_data': eeg_data,
                'emg_data': eeg_data
            }
            
        try:
            processed_data = np.zeros_like(eeg_data)
            artifact_data = np.zeros_like(eeg_data)
            emg_data = np.zeros_like(eeg_data)
            
            for channel in range(eeg_data.shape[0]):
                channel_data = eeg_data[channel, :].copy()
                original_data = channel_data.copy()
                
                # 1. Remove DC component and apply high-pass filter
                if len(channel_data) > 30:  # Ensure minimum length for filtering (padlen is ~15-27)
                    channel_data = signal.filtfilt(self.highpass_filter[0], self.highpass_filter[1], channel_data)
                
                # 2. Apply low-pass filter
                if len(channel_data) > 30:
                    channel_data = signal.filtfilt(self.lowpass_filter[0], self.lowpass_filter[1], channel_data)
                
                # 3. Apply notch filters for power line noise
                for b_notch, a_notch in self.notch_filters:
                    if len(channel_data) > 30:
                        channel_data = signal.filtfilt(b_notch, a_notch, channel_data)
                
                # 4. Extract EMG signals for jaw clench detection (temporal channels)
                if channel in [0, 3] and self.emg_filter is not None:  # TP9 and TP10
                    if len(original_data) > 30:
                        emg_signal = signal.filtfilt(self.emg_filter[0], self.emg_filter[1], original_data)
                        emg_data[channel, :] = emg_signal
                
                # 5. Detect and preserve artifacts before removal
                # Use lower threshold for better artifact detection
                amplitude_threshold = 1.5 * np.std(channel_data)  # Lower threshold to catch more artifacts
                artifact_mask = np.abs(channel_data) > amplitude_threshold
                
                # Store artifacts in separate channel - preserve original signal for blinks
                artifact_signal = np.zeros_like(channel_data)
                artifact_signal[artifact_mask] = original_data[artifact_mask]  # Use original data, not filtered
                artifact_data[channel, :] = artifact_signal
                
                
                # 6. Remove artifacts from clean signal
                if np.any(artifact_mask):
                    # Simple linear interpolation for artifact removal
                    channel_data[artifact_mask] = np.interp(
                        np.where(artifact_mask)[0],
                        np.where(~artifact_mask)[0],
                        channel_data[~artifact_mask]
                    )
                
                processed_data[channel, :] = channel_data
                
                # Update buffers
                self.eeg_buffer[channel].extend(eeg_data[channel, :])
                self.filtered_buffer[channel].extend(channel_data)
                self.artifact_buffer[channel].extend(artifact_signal)
                
            return {
                'clean_data': processed_data,
                'artifact_data': artifact_data,
                'emg_data': emg_data
            }
            
        except Exception as e:
            self.logger.error(f"Error in artifact preserving preprocessing: {e}")
            return {
                'clean_data': eeg_data,
                'artifact_data': eeg_data,
                'emg_data': eeg_data
            }
    
    def extract_blink_artifacts(self, artifact_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract blink artifacts using proper EOG detection from project1
        
        Args:
            artifact_data: Artifact data with shape (channels, samples)
            
        Returns:
            Dictionary with blink detection results
        """
        if artifact_data is None or artifact_data.size == 0:
            return {'blinks': [], 'double_blinks': []}
            
        try:
            # Extract EOG signals from frontal channels (AF7, AF8) - channels 1 and 2
            if artifact_data.shape[0] >= 3:
                eog_signals = artifact_data[1:3, :]  # Channels 1 and 2 (AF7, AF8)
            else:
                eog_signals = artifact_data[:2, :]  # Fallback to first two channels
                
            if eog_signals.size == 0:
                return {'blinks': [], 'double_blinks': []}
            
            # Detect blinks using proper EOG criteria from project1
            blinks = self._detect_blinks_from_eog(eog_signals)
            
            # Detect double blinks using proper criteria from project1
            double_blinks = self._detect_double_blinks_from_blinks(blinks)
            
            return {
                'blinks': blinks,
                'double_blinks': double_blinks
            }
            
        except Exception as e:
            self.logger.error(f"Error in blink artifact extraction: {e}")
            return {'blinks': [], 'double_blinks': []}
    
    def _detect_blinks_from_eog(self, eog_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect voluntary blinks in EOG signals using proper criteria from project1
        
        Args:
            eog_data: EOG signals with shape (channels, samples)
            
        Returns:
            List of detected voluntary blinks with timing information
        """
        if eog_data is None or eog_data.size == 0:
            return []
            
        try:
            blinks = []
            blink_threshold = 200.0  # Higher threshold for voluntary blinks (microvolts)
            min_blink_interval = 0.2  # Minimum time between blinks (seconds)
            min_blink_duration = 0.05  # 50ms minimum blink duration
            max_blink_duration = 0.3   # 300ms maximum blink duration
            
            for channel_idx in range(eog_data.shape[0]):
                channel_data = eog_data[channel_idx, :]
                
                # Use a higher threshold for voluntary blinks and look for negative peaks
                # Voluntary blinks typically show as negative deflections in EOG
                negative_data = -channel_data  # Invert to find negative peaks
                
                # Find peaks with stricter criteria for voluntary blinks
                peaks, properties = signal.find_peaks(
                    negative_data, 
                    height=blink_threshold,  # Higher threshold for voluntary blinks
                    distance=int(self.sampling_rate * min_blink_interval),  # Minimum time between blinks
                    prominence=blink_threshold * 0.5,  # Require significant prominence
                    width=(int(self.sampling_rate * min_blink_duration), 
                          int(self.sampling_rate * max_blink_duration))  # Blink duration constraints
                )
                
                # Convert peak indices to time
                for peak_idx in peaks:
                    blink_time = peak_idx / self.sampling_rate
                    amplitude = channel_data[peak_idx]  # Original amplitude (negative)
                    
                    # Only include blinks that are strong enough to be voluntary
                    if abs(amplitude) > blink_threshold:
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
    
    def _detect_double_blinks_from_blinks(self, blinks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect voluntary double blinks within a time window using proper criteria from project1
        
        Args:
            blinks: List of detected voluntary blinks
            
        Returns:
            List of detected voluntary double blinks
        """
        if not blinks:
            return []
            
        try:
            double_blinks = []
            double_blink_time_window = 1.2  # Maximum time between blinks for double blink (seconds)
            min_double_blink_interval = 0.2  # Minimum time for double blink (seconds)
            
            blinks_sorted = sorted(blinks, key=lambda x: x['time'])
            
            i = 0
            while i < len(blinks_sorted) - 1:
                current_blink = blinks_sorted[i]
                next_blink = blinks_sorted[i + 1]
                
                time_diff = next_blink['time'] - current_blink['time']
                
                # Stricter criteria for voluntary double blinks
                if min_double_blink_interval <= time_diff <= double_blink_time_window:
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
    
    def extract_jaw_clench_artifacts(self, emg_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract jaw clench artifacts from temporal channels (TP9, TP10)
        
        Args:
            emg_data: EMG data with shape (channels, samples)
            
        Returns:
            Dictionary with jaw clench detection results
        """
        if emg_data is None or emg_data.size == 0:
            return {'jaw_clenches': []}
            
        try:
            jaw_clenches = []
            
            # Focus on temporal channels for jaw clench detection (TP9, TP10)
            temporal_channels = [0, 3]  # TP9, TP10
            
            for channel_idx in temporal_channels:
                if channel_idx < emg_data.shape[0]:
                    channel_data = emg_data[channel_idx, :]
                    
                    # Calculate EMG power
                    emg_power = np.abs(channel_data)
                    
                    # Find sustained high EMG activity
                    threshold = 2 * np.std(emg_power)  # Adaptive threshold
                    high_activity = emg_power > threshold
                    
                    # Find continuous periods of high activity
                    if np.any(high_activity):
                        # Find start and end of activity periods
                        diff = np.diff(high_activity.astype(int))
                        starts = np.where(diff == 1)[0]
                        ends = np.where(diff == -1)[0]
                        
                        # Handle edge cases
                        if len(starts) > len(ends):
                            ends = np.append(ends, len(high_activity) - 1)
                        if len(ends) > len(starts):
                            starts = np.append(0, starts)
                        
                        # Check each period for jaw clench criteria
                        for start, end in zip(starts, ends):
                            duration = (end - start) / self.sampling_rate
                            avg_power = np.mean(emg_power[start:end])
                            
                            # Jaw clench criteria: 0.5-3 seconds, high power
                            if 0.5 <= duration <= 3.0 and avg_power > threshold:
                                clench_time = start / self.sampling_rate
                                
                                jaw_clenches.append({
                                    'channel': channel_idx,
                                    'time': clench_time,
                                    'duration': duration,
                                    'power': avg_power,
                                    'start_sample': start,
                                    'end_sample': end
                                })
            
            return {'jaw_clenches': jaw_clenches}
            
        except Exception as e:
            self.logger.error(f"Error in jaw clench artifact extraction: {e}")
            return {'jaw_clenches': []}
    
    def extract_spectral_features(self, clean_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract spectral features from clean EEG data for mental state classification
        
        Args:
            clean_data: Clean EEG data with shape (channels, samples)
            
        Returns:
            Dictionary with spectral features
        """
        if clean_data is None or clean_data.size == 0:
            return {}
            
        try:
            spectral_features = {}
            window_size = min(256, clean_data.shape[1])  # Adaptive window size
            
            for channel in range(clean_data.shape[0]):
                channel_name = self.channel_mapping.get(channel, f"Ch{channel}")
                channel_data = clean_data[channel, :]
                
                if len(channel_data) < window_size:
                    continue
                
                # Calculate power spectral density
                freqs, psd = welch(channel_data, fs=self.sampling_rate, nperseg=window_size)
                
                # Calculate power in each frequency band
                band_powers = {}
                for band_name, (low_freq, high_freq) in self.freq_bands.items():
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.sum(psd[freq_mask])
                    band_powers[band_name] = band_power
                
                spectral_features[f"channel_{channel}"] = band_powers
                
            return spectral_features
            
        except Exception as e:
            self.logger.error(f"Error in spectral feature extraction: {e}")
            return {}
    
    def classify_mental_state(self, spectral_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify mental state based on spectral features
        
        Args:
            spectral_features: Spectral features from extract_spectral_features
            
        Returns:
            Dictionary with mental state classification
        """
        try:
            if not spectral_features:
                return {"overall_state": "unknown", "confidence": 0.0}
            
            # Calculate alpha/beta ratios for each channel
            ratios = {}
            for channel_key, band_powers in spectral_features.items():
                alpha_power = band_powers.get('alpha', 0)
                beta_power = band_powers.get('beta', 0)
                
                if beta_power > 0:
                    ratio = alpha_power / beta_power
                else:
                    ratio = alpha_power if alpha_power > 0 else 0
                    
                ratios[channel_key] = ratio
            
            # Calculate average ratio across all channels
            if ratios:
                average_ratio = sum(ratios.values()) / len(ratios)
                
                # Simple classification based on alpha/beta ratio
                if average_ratio >= 1.2:
                    overall_state = "relaxed"
                    confidence = min(1.0, (average_ratio - 1.2) / 0.5)
                elif average_ratio <= 0.8:
                    overall_state = "focused"
                    confidence = min(1.0, (0.8 - average_ratio) / 0.3)
                else:
                    overall_state = "neutral"
                    confidence = 0.5
                    
                return {
                    "overall_state": overall_state,
                    "average_ratio": average_ratio,
                    "confidence": confidence,
                    "individual_ratios": ratios
                }
            else:
                return {"overall_state": "unknown", "confidence": 0.0}
                
        except Exception as e:
            self.logger.error(f"Error in mental state classification: {e}")
            return {"overall_state": "unknown", "confidence": 0.0}
    
    def process_eeg_sample(self, eeg_data: np.ndarray) -> Dict[str, Any]:
        """
        Complete processing pipeline for a single EEG sample
        
        Args:
            eeg_data: Raw EEG data with shape (channels, samples)
            
        Returns:
            Dictionary with all processing results
        """
        try:
            # Preprocess while preserving artifacts
            processed_results = self.preprocess_eeg_with_artifacts(eeg_data)
            
            # Extract blink artifacts
            blink_results = self.extract_blink_artifacts(processed_results['artifact_data'])
            
            # Extract jaw clench artifacts
            jaw_clench_results = self.extract_jaw_clench_artifacts(processed_results['emg_data'])
            
            # Extract spectral features from clean data
            spectral_features = self.extract_spectral_features(processed_results['clean_data'])
            
            # Classify mental state
            mental_state = self.classify_mental_state(spectral_features)
            
            return {
                'clean_data': processed_results['clean_data'],
                'artifact_data': processed_results['artifact_data'],
                'emg_data': processed_results['emg_data'],
                'blinks': blink_results['blinks'],
                'double_blinks': blink_results['double_blinks'],
                'jaw_clenches': jaw_clench_results['jaw_clenches'],
                'spectral_features': spectral_features,
                'mental_state': mental_state,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error in complete processing: {e}")
            return {}
