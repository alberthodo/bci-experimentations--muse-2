"""
Advanced Signal Processor for Mental State Visualization
Implements comprehensive EEG preprocessing, feature extraction, and advanced signal analysis
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


class AdvancedSignalProcessor:
    """
    Advanced signal processor with comprehensive feature extraction and preprocessing
    """
    
    def __init__(self, sampling_rate: int = 256, buffer_duration: float = 10.0):
        """
        Initialize advanced signal processor
        
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
        
        # Advanced feature extraction parameters
        self.feature_params = {
            'window_size': 2.0,     # Window size in seconds for feature extraction
            'overlap': 0.5,         # Window overlap (50%)
            'min_freq': 0.5,        # Minimum frequency for analysis
            'max_freq': 50.0,       # Maximum frequency for analysis
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
        
        # Feature history for temporal analysis
        self.feature_history = deque(maxlen=100)  # Keep last 100 feature vectors
        self.mental_state_history = deque(maxlen=50)  # Keep last 50 mental state classifications
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'relaxation': {'alpha_threshold': 0.6, 'confidence': 0.0},
            'focused': {'beta_threshold': 0.4, 'confidence': 0.0},
            'meditative': {'theta_threshold': 0.5, 'confidence': 0.0},
            'alert': {'gamma_threshold': 0.3, 'confidence': 0.0}
        }
        
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
        
        self.logger.info(f"Advanced filters initialized: HP={self.highpass_freq}Hz, LP={self.lowpass_freq}Hz")
        self.logger.info(f"Notch filters: {self.notch_freqs}Hz, Band filters: {list(self.band_filters.keys())}")
    
    def preprocess_eeg_advanced(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply advanced preprocessing pipeline to EEG data
        
        Args:
            eeg_data: Raw EEG data with shape (channels, samples)
            
        Returns:
            Comprehensively preprocessed EEG data
        """
        if eeg_data is None or eeg_data.size == 0:
            return eeg_data
            
        try:
            processed_data = np.zeros_like(eeg_data)
            
            for channel in range(eeg_data.shape[0]):
                channel_data = eeg_data[channel, :].copy()
                
                # 1. Remove DC component and apply high-pass filter
                if len(channel_data) > 10:  # Ensure minimum length for filtering
                    channel_data = signal.filtfilt(self.highpass_filter[0], self.highpass_filter[1], channel_data)
                
                # 2. Apply low-pass filter
                if len(channel_data) > 10:
                    channel_data = signal.filtfilt(self.lowpass_filter[0], self.lowpass_filter[1], channel_data)
                
                # 3. Apply notch filters for power line noise
                for b_notch, a_notch in self.notch_filters:
                    if len(channel_data) > 10:
                        channel_data = signal.filtfilt(b_notch, a_notch, channel_data)
                
                # 4. Artifact rejection (simple amplitude thresholding)
                amplitude_threshold = 5 * np.std(channel_data)
                artifact_mask = np.abs(channel_data) > amplitude_threshold
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
                
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in advanced preprocessing: {e}")
            return eeg_data
    
    def extract_spectral_features(self, eeg_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive spectral features from EEG data
        
        Args:
            eeg_data: Preprocessed EEG data with shape (channels, samples)
            
        Returns:
            Dictionary with detailed spectral features
        """
        if eeg_data is None or eeg_data.size == 0:
            return {}
            
        try:
            spectral_features = {}
            window_size = min(256, eeg_data.shape[1])  # Adaptive window size
            
            for channel in range(eeg_data.shape[0]):
                channel_name = self.channel_mapping.get(channel, f"Ch{channel}")
                channel_data = eeg_data[channel, :]
                
                if len(channel_data) < window_size:
                    continue
                
                # 1. Power Spectral Density using Welch's method
                freqs, psd = welch(channel_data, fs=self.sampling_rate, 
                                 nperseg=window_size, noverlap=window_size//2)
                
                # 2. Extract power in each frequency band
                band_powers = {}
                for band_name, (low_freq, high_freq) in self.freq_bands.items():
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(freq_mask):
                        band_power = np.sum(psd[freq_mask])
                        band_powers[band_name] = band_power
                    else:
                        band_powers[band_name] = 0.0
                
                # 3. Relative power (normalized by total power)
                total_power = sum(band_powers.values())
                relative_powers = {}
                if total_power > 0:
                    for band_name, power in band_powers.items():
                        relative_powers[f"{band_name}_rel"] = power / total_power
                else:
                    for band_name in band_powers.keys():
                        relative_powers[f"{band_name}_rel"] = 0.0
                
                # 4. Peak frequency in each band
                peak_freqs = {}
                for band_name, (low_freq, high_freq) in self.freq_bands.items():
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(freq_mask):
                        band_freqs = freqs[freq_mask]
                        band_psd = psd[freq_mask]
                        if len(band_psd) > 0:
                            peak_idx = np.argmax(band_psd)
                            peak_freqs[f"{band_name}_peak"] = band_freqs[peak_idx]
                        else:
                            peak_freqs[f"{band_name}_peak"] = (low_freq + high_freq) / 2
                    else:
                        peak_freqs[f"{band_name}_peak"] = (low_freq + high_freq) / 2
                
                # 5. Spectral edge frequency (frequency below which 95% of power lies)
                cumulative_power = np.cumsum(psd)
                total_power_welch = cumulative_power[-1]
                if total_power_welch > 0:
                    edge_idx = np.where(cumulative_power >= 0.95 * total_power_welch)[0]
                    spectral_edge = freqs[edge_idx[0]] if len(edge_idx) > 0 else freqs[-1]
                else:
                    spectral_edge = freqs[-1]
                
                # 6. Spectral entropy (measure of signal complexity)
                if total_power_welch > 0:
                    normalized_psd = psd / total_power_welch
                    spectral_entropy = entropy(normalized_psd + 1e-12)  # Add small value to avoid log(0)
                else:
                    spectral_entropy = 0.0
                
                # Store features for this channel
                spectral_features[channel_name] = {
                    **band_powers,
                    **relative_powers,
                    **peak_freqs,
                    'spectral_edge': spectral_edge,
                    'spectral_entropy': spectral_entropy,
                    'total_power': total_power
                }
                
            return spectral_features
            
        except Exception as e:
            self.logger.error(f"Error in spectral feature extraction: {e}")
            return {}
    
    def extract_temporal_features(self, eeg_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract temporal domain features from EEG data
        
        Args:
            eeg_data: Preprocessed EEG data with shape (channels, samples)
            
        Returns:
            Dictionary with temporal features
        """
        if eeg_data is None or eeg_data.size == 0:
            return {}
            
        try:
            temporal_features = {}
            
            for channel in range(eeg_data.shape[0]):
                channel_name = self.channel_mapping.get(channel, f"Ch{channel}")
                channel_data = eeg_data[channel, :]
                
                if len(channel_data) == 0:
                    continue
                
                # 1. Statistical moments
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                variance = np.var(channel_data)
                skewness = self._calculate_skewness(channel_data)
                kurtosis = self._calculate_kurtosis(channel_data)
                
                # 2. Signal complexity measures
                zero_crossings = self._count_zero_crossings(channel_data)
                hjorth_activity, hjorth_mobility, hjorth_complexity = self._hjorth_parameters(channel_data)
                
                # 3. Amplitude features
                rms = np.sqrt(np.mean(channel_data**2))
                peak_to_peak = np.max(channel_data) - np.min(channel_data)
                
                # 4. Envelope features (using Hilbert transform)
                try:
                    analytic_signal = hilbert(channel_data)
                    envelope = np.abs(analytic_signal)
                    envelope_mean = np.mean(envelope)
                    envelope_std = np.std(envelope)
                except:
                    envelope_mean = 0.0
                    envelope_std = 0.0
                
                temporal_features[channel_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'variance': variance,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'rms': rms,
                    'peak_to_peak': peak_to_peak,
                    'zero_crossings': zero_crossings,
                    'hjorth_activity': hjorth_activity,
                    'hjorth_mobility': hjorth_mobility,
                    'hjorth_complexity': hjorth_complexity,
                    'envelope_mean': envelope_mean,
                    'envelope_std': envelope_std
                }
                
            return temporal_features
            
        except Exception as e:
            self.logger.error(f"Error in temporal feature extraction: {e}")
            return {}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness (third moment)"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis (fourth moment)"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
    
    def _count_zero_crossings(self, data: np.ndarray) -> int:
        """Count zero crossings in the signal"""
        if len(data) < 2:
            return 0
        return np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
    
    def _hjorth_parameters(self, data: np.ndarray) -> Tuple[float, float, float]:
        """Calculate Hjorth parameters (activity, mobility, complexity)"""
        if len(data) < 2:
            return 0.0, 0.0, 0.0
            
        # Activity (variance)
        activity = np.var(data)
        
        # First derivative
        diff1 = np.diff(data)
        if len(diff1) == 0:
            return activity, 0.0, 0.0
            
        # Mobility
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0.0
        
        # Second derivative
        diff2 = np.diff(diff1)
        if len(diff2) == 0:
            return activity, mobility, 0.0
            
        # Complexity
        mobility2 = np.sqrt(np.var(diff2) / np.var(diff1)) if np.var(diff1) > 0 else 0.0
        complexity = mobility2 / mobility if mobility > 0 else 0.0
        
        return activity, mobility, complexity
    
    def calculate_cross_channel_features(self, eeg_data: np.ndarray) -> Dict[str, Any]:
        """
        Calculate features based on relationships between channels
        
        Args:
            eeg_data: Preprocessed EEG data with shape (channels, samples)
            
        Returns:
            Dictionary with cross-channel features
        """
        if eeg_data is None or eeg_data.shape[0] < 2:
            return {}
            
        try:
            cross_features = {}
            
            # 1. Inter-channel correlations
            correlations = {}
            for i in range(eeg_data.shape[0]):
                for j in range(i+1, eeg_data.shape[0]):
                    ch1_name = self.channel_mapping.get(i, f"Ch{i}")
                    ch2_name = self.channel_mapping.get(j, f"Ch{j}")
                    corr_key = f"{ch1_name}_{ch2_name}_corr"
                    
                    if eeg_data.shape[1] > 1:
                        correlation = np.corrcoef(eeg_data[i, :], eeg_data[j, :])[0, 1]
                        correlations[corr_key] = correlation if not np.isnan(correlation) else 0.0
                    else:
                        correlations[corr_key] = 0.0
            
            # 2. Hemispheric asymmetry (frontal: AF7 vs AF8, temporal: TP9 vs TP10)
            asymmetry = {}
            
            # Frontal asymmetry (AF7 vs AF8) - channels 1 and 2
            if eeg_data.shape[0] > 2:
                left_frontal = eeg_data[1, :]   # AF7
                right_frontal = eeg_data[2, :]  # AF8
                
                left_power = np.var(left_frontal)
                right_power = np.var(right_frontal)
                
                if left_power + right_power > 0:
                    frontal_asymmetry = (right_power - left_power) / (right_power + left_power)
                else:
                    frontal_asymmetry = 0.0
                    
                asymmetry['frontal_asymmetry'] = frontal_asymmetry
            
            # Temporal asymmetry (TP9 vs TP10) - channels 0 and 3
            if eeg_data.shape[0] > 3:
                left_temporal = eeg_data[0, :]   # TP9
                right_temporal = eeg_data[3, :]  # TP10
                
                left_power = np.var(left_temporal)
                right_power = np.var(right_temporal)
                
                if left_power + right_power > 0:
                    temporal_asymmetry = (right_power - left_power) / (right_power + left_power)
                else:
                    temporal_asymmetry = 0.0
                    
                asymmetry['temporal_asymmetry'] = temporal_asymmetry
            
            cross_features.update(correlations)
            cross_features.update(asymmetry)
            
            return cross_features
            
        except Exception as e:
            self.logger.error(f"Error in cross-channel feature extraction: {e}")
            return {}
    
    def classify_mental_state_advanced(self, spectral_features: Dict[str, Any], 
                                     temporal_features: Dict[str, Any],
                                     cross_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced mental state classification using multiple feature types
        
        Args:
            spectral_features: Spectral domain features
            temporal_features: Temporal domain features  
            cross_features: Cross-channel features
            
        Returns:
            Dictionary with detailed mental state classification
        """
        try:
            # Combine features from all channels
            all_features = {}
            
            # Aggregate spectral features across channels
            spectral_summary = self._aggregate_channel_features(spectral_features)
            temporal_summary = self._aggregate_channel_features(temporal_features)
            
            all_features.update(spectral_summary)
            all_features.update(temporal_summary)
            all_features.update(cross_features)
            
            # Calculate key ratios and indices
            ratios = self._calculate_mental_state_ratios(spectral_summary)
            
            # Advanced mental state classification
            mental_states = self._classify_multiple_states(ratios, all_features)
            
            # Update adaptive thresholds
            self._update_adaptive_thresholds(mental_states, ratios)
            
            # Store feature vector for temporal analysis
            self.feature_history.append(all_features)
            self.mental_state_history.append(mental_states)
            
            # Add temporal consistency score
            consistency_score = self._calculate_temporal_consistency()
            
            result = {
                'primary_state': mental_states.get('primary', 'neutral'),
                'secondary_state': mental_states.get('secondary', 'neutral'),
                'confidence': mental_states.get('confidence', 0.0),
                'consistency': consistency_score,
                'state_probabilities': mental_states.get('probabilities', {}),
                'key_ratios': ratios,
                'feature_summary': {
                    'spectral': spectral_summary,
                    'temporal': temporal_summary,
                    'cross_channel': cross_features
                },
                'adaptive_thresholds': dict(self.adaptive_thresholds),
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in advanced mental state classification: {e}")
            return {'primary_state': 'unknown', 'confidence': 0.0}
    
    def _aggregate_channel_features(self, channel_features: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate features across channels"""
        aggregated = {}
        
        if not channel_features:
            return aggregated
        
        # Get all feature names from first channel
        first_channel = list(channel_features.values())[0]
        feature_names = list(first_channel.keys())
        
        # Calculate statistics across channels for each feature
        for feature_name in feature_names:
            values = []
            for channel_data in channel_features.values():
                if feature_name in channel_data:
                    values.append(channel_data[feature_name])
            
            if values:
                aggregated[f"{feature_name}_mean"] = np.mean(values)
                aggregated[f"{feature_name}_std"] = np.std(values)
                aggregated[f"{feature_name}_max"] = np.max(values)
                aggregated[f"{feature_name}_min"] = np.min(values)
        
        return aggregated
    
    def _calculate_mental_state_ratios(self, spectral_summary: Dict[str, float]) -> Dict[str, float]:
        """Calculate key ratios for mental state classification"""
        ratios = {}
        
        # Get mean power values
        alpha_power = spectral_summary.get('alpha_mean', 0.0)
        beta_power = spectral_summary.get('beta_mean', 0.0)
        theta_power = spectral_summary.get('theta_mean', 0.0)
        gamma_power = spectral_summary.get('gamma_mean', 0.0)
        delta_power = spectral_summary.get('delta_mean', 0.0)
        
        # Calculate ratios (with safety checks)
        ratios['alpha_beta_ratio'] = alpha_power / (beta_power + 1e-8)
        ratios['theta_alpha_ratio'] = theta_power / (alpha_power + 1e-8)
        ratios['beta_theta_ratio'] = beta_power / (theta_power + 1e-8)
        ratios['gamma_beta_ratio'] = gamma_power / (beta_power + 1e-8)
        
        # Engagement index (beta / (alpha + theta))
        ratios['engagement_index'] = beta_power / (alpha_power + theta_power + 1e-8)
        
        # Relaxation index (alpha / (beta + gamma))
        ratios['relaxation_index'] = alpha_power / (beta_power + gamma_power + 1e-8)
        
        # Meditation index (theta / (alpha + beta))
        ratios['meditation_index'] = theta_power / (alpha_power + beta_power + 1e-8)
        
        # Alertness index (gamma / (delta + theta))
        ratios['alertness_index'] = gamma_power / (delta_power + theta_power + 1e-8)
        
        return ratios
    
    def _classify_multiple_states(self, ratios: Dict[str, float], 
                                all_features: Dict[str, float]) -> Dict[str, Any]:
        """Classify multiple mental states simultaneously"""
        
        # State probability calculation
        state_probs = {}
        
        # Relaxation state (high alpha, low beta)
        relaxation_score = (ratios.get('alpha_beta_ratio', 0) * 0.4 + 
                           ratios.get('relaxation_index', 0) * 0.6)
        state_probs['relaxed'] = min(relaxation_score / 2.0, 1.0)
        
        # Focused state (high beta, moderate alpha)
        focus_score = (ratios.get('engagement_index', 0) * 0.6 + 
                      (1.0 / (ratios.get('alpha_beta_ratio', 1) + 1e-8)) * 0.4)
        state_probs['focused'] = min(focus_score / 2.0, 1.0)
        
        # Meditative state (high theta, balanced alpha)
        meditation_score = (ratios.get('meditation_index', 0) * 0.5 + 
                           ratios.get('theta_alpha_ratio', 0) * 0.5)
        state_probs['meditative'] = min(meditation_score / 2.0, 1.0)
        
        # Alert state (high gamma, high beta)
        alert_score = (ratios.get('alertness_index', 0) * 0.4 + 
                      ratios.get('gamma_beta_ratio', 0) * 0.6)
        state_probs['alert'] = min(alert_score / 2.0, 1.0)
        
        # Neutral state (balanced activity)
        balance_score = 1.0 - max(state_probs.values())
        state_probs['neutral'] = max(balance_score, 0.0)
        
        # Normalize probabilities
        total_prob = sum(state_probs.values())
        if total_prob > 0:
            state_probs = {k: v/total_prob for k, v in state_probs.items()}
        
        # Determine primary and secondary states
        sorted_states = sorted(state_probs.items(), key=lambda x: x[1], reverse=True)
        primary_state = sorted_states[0][0]
        secondary_state = sorted_states[1][0] if len(sorted_states) > 1 else 'neutral'
        
        # Calculate confidence based on separation between top states
        confidence = sorted_states[0][1] - sorted_states[1][1] if len(sorted_states) > 1 else sorted_states[0][1]
        
        return {
            'primary': primary_state,
            'secondary': secondary_state,
            'confidence': confidence,
            'probabilities': state_probs
        }
    
    def _update_adaptive_thresholds(self, mental_states: Dict[str, Any], ratios: Dict[str, float]):
        """Update adaptive thresholds based on recent classifications"""
        primary_state = mental_states.get('primary', 'neutral')
        confidence = mental_states.get('confidence', 0.0)
        
        # Update thresholds only for high-confidence classifications
        if confidence > 0.7:
            if primary_state in self.adaptive_thresholds:
                # Exponential moving average for threshold adaptation
                alpha = 0.1  # Learning rate
                
                if primary_state == 'relaxed':
                    current_ratio = ratios.get('alpha_beta_ratio', 0.0)
                    self.adaptive_thresholds[primary_state]['alpha_threshold'] = (
                        (1 - alpha) * self.adaptive_thresholds[primary_state]['alpha_threshold'] +
                        alpha * current_ratio
                    )
                
                # Update confidence
                self.adaptive_thresholds[primary_state]['confidence'] = (
                    (1 - alpha) * self.adaptive_thresholds[primary_state]['confidence'] +
                    alpha * confidence
                )
    
    def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of mental state classifications"""
        if len(self.mental_state_history) < 2:
            return 0.0
        
        # Count how often the primary state remains the same
        recent_states = [state.get('primary', 'neutral') for state in list(self.mental_state_history)[-10:]]
        
        if not recent_states:
            return 0.0
        
        # Calculate consistency as percentage of same states
        most_common_state = max(set(recent_states), key=recent_states.count)
        consistency = recent_states.count(most_common_state) / len(recent_states)
        
        return consistency
    
    def process_eeg_comprehensive(self, eeg_data: np.ndarray) -> Dict[str, Any]:
        """
        Complete comprehensive processing pipeline
        
        Args:
            eeg_data: Raw EEG data with shape (channels, samples)
            
        Returns:
            Dictionary with all processing results
        """
        try:
            # 1. Advanced preprocessing
            processed_data = self.preprocess_eeg_advanced(eeg_data)
            
            # 2. Comprehensive feature extraction
            spectral_features = self.extract_spectral_features(processed_data)
            temporal_features = self.extract_temporal_features(processed_data)
            cross_features = self.calculate_cross_channel_features(processed_data)
            
            # 3. Advanced mental state classification
            mental_state_result = self.classify_mental_state_advanced(
                spectral_features, temporal_features, cross_features
            )
            
            return {
                'processed_data': processed_data,
                'spectral_features': spectral_features,
                'temporal_features': temporal_features,
                'cross_channel_features': cross_features,
                'mental_state': mental_state_result,
                'signal_quality': self._assess_processing_quality(processed_data),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive processing: {e}")
            return {}
    
    def _assess_processing_quality(self, processed_data: np.ndarray) -> Dict[str, float]:
        """Assess quality of processed signal"""
        if processed_data is None or processed_data.size == 0:
            return {'overall_quality': 0.0}
        
        quality_metrics = {}
        channel_qualities = []
        
        for channel in range(processed_data.shape[0]):
            channel_data = processed_data[channel, :]
            
            # Signal-to-noise ratio estimate
            signal_power = np.var(channel_data)
            noise_estimate = np.var(np.diff(channel_data))  # High-frequency content
            snr = signal_power / (noise_estimate + 1e-8)
            
            # Normality check (should be somewhat normal for good EEG)
            kurtosis = self._calculate_kurtosis(channel_data)
            normality_score = 1.0 / (1.0 + abs(kurtosis))  # Lower kurtosis is better
            
            # Stability check (low variance in envelope)
            try:
                envelope = np.abs(hilbert(channel_data))
                stability = 1.0 / (1.0 + np.std(envelope) / np.mean(envelope))
            except:
                stability = 0.5
            
            channel_quality = np.mean([min(snr/10, 1.0), normality_score, stability])
            channel_qualities.append(channel_quality)
            
            channel_name = self.channel_mapping.get(channel, f"Ch{channel}")
            quality_metrics[f"{channel_name}_quality"] = channel_quality
        
        quality_metrics['overall_quality'] = np.mean(channel_qualities)
        
        return quality_metrics


def main():
    """
    Test function for advanced signal processor
    """
    print("=== Advanced Signal Processor Test ===")
    
    # Create advanced signal processor
    processor = AdvancedSignalProcessor(sampling_rate=256)
    
    # Generate test EEG data with realistic characteristics
    duration = 2.0  # 2 seconds
    samples = int(256 * duration)
    
    # Simulate realistic EEG with different frequency components
    t = np.linspace(0, duration, samples)
    test_data = np.zeros((4, samples))
    
    for ch in range(4):
        # Base signal with multiple frequency components
        alpha_component = 50 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        beta_component = 30 * np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
        theta_component = 40 * np.sin(2 * np.pi * 6 * t)   # 6 Hz theta
        
        # Add some noise and artifacts
        noise = np.random.normal(0, 15, samples)
        
        test_data[ch, :] = alpha_component + beta_component + theta_component + noise
    
    print(f"Test data shape: {test_data.shape}")
    
    # Test comprehensive processing
    print("\nTesting comprehensive processing pipeline...")
    start_time = time.time()
    
    results = processor.process_eeg_comprehensive(test_data)
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.3f} seconds")
    
    # Display results
    if results:
        print(f"\nProcessing results keys: {list(results.keys())}")
        
        mental_state = results.get('mental_state', {})
        print(f"\nMental State Analysis:")
        print(f"Primary State: {mental_state.get('primary_state', 'unknown')}")
        print(f"Secondary State: {mental_state.get('secondary_state', 'unknown')}")
        print(f"Confidence: {mental_state.get('confidence', 0.0):.3f}")
        print(f"Consistency: {mental_state.get('consistency', 0.0):.3f}")
        
        state_probs = mental_state.get('state_probabilities', {})
        if state_probs:
            print("\nState Probabilities:")
            for state, prob in state_probs.items():
                print(f"  {state}: {prob:.3f}")
        
        key_ratios = mental_state.get('key_ratios', {})
        if key_ratios:
            print("\nKey Ratios:")
            for ratio_name, value in key_ratios.items():
                print(f"  {ratio_name}: {value:.3f}")
        
        signal_quality = results.get('signal_quality', {})
        if signal_quality:
            print(f"\nSignal Quality:")
            print(f"Overall Quality: {signal_quality.get('overall_quality', 0.0):.3f}")
    
    print("\nAdvanced signal processor test completed!")


if __name__ == "__main__":
    main()
