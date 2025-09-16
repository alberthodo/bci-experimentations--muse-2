"""
Enhanced Muse Connector for Mental State Visualization Project
Provides enhanced signal quality monitoring and more robust data streaming
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.board_shim import BrainFlowError
import threading
from collections import deque


class EnhancedMuseConnector:
    """
    Enhanced Muse connector with improved signal quality monitoring and data buffering
    """
    
    def __init__(self, board_type: str = "MUSE_2", mac_address: str = "", buffer_size: int = 5120):
        """
        Initialize enhanced Muse connector
        
        Args:
            board_type: Type of Muse board ("MUSE_2", "MUSE_S", "MUSE_2016")
            mac_address: MAC address of the Muse device (empty for auto-scan)
            buffer_size: Size of the circular buffer for continuous data storage
        """
        self.board = None
        self.board_id = self._get_board_id(board_type)
        self.params = BrainFlowInputParams()
        self.params.mac_address = mac_address
        self.is_streaming = False
        self.is_connected = False
        self.sample_count = 0
        
        # Enhanced buffer management
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Channel information
        self.eeg_channels = None
        self.sampling_rate = None
        self.channel_names = {}
        
        # Signal quality monitoring - will be initialized after connection
        self.signal_quality = {}
        
        # Signal quality thresholds
        self.quality_thresholds = {
            'min_amplitude': 10.0,    # Minimum signal amplitude (µV)
            'max_amplitude': 1000.0,  # Maximum signal amplitude (µV) 
            'noise_threshold': 5.0,   # Maximum acceptable noise level
            'saturation_threshold': 800.0  # Signal saturation threshold
        }
        
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'dropped_samples': 0,
            'quality_scores': deque(maxlen=1000),
            'connection_time': None,
            'last_data_time': None
        }
        
    def _get_board_id(self, board_type: str) -> int:
        """Get Brainflow board ID for Muse type"""
        board_mapping = {
            "MUSE_2": BoardIds.MUSE_2_BOARD.value,
            "MUSE_S": BoardIds.MUSE_S_BOARD.value,
            "MUSE_2016": BoardIds.MUSE_2016_BOARD.value
        }
        return board_mapping.get(board_type, BoardIds.MUSE_2_BOARD.value)
    
    def connect(self) -> bool:
        """
        Establish connection to Muse headset with enhanced error handling
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Attempting to connect to Muse headset...")
            self.logger.info(f"Board ID: {self.board_id}")
            self.logger.info(f"MAC Address: {self.params.mac_address or 'Auto-scan'}")
            
            # Create board instance
            self.board = BoardShim(self.board_id, self.params)
            
            # Prepare session
            self.board.prepare_session()
            self.is_connected = True
            self.stats['connection_time'] = time.time()
            
            # Get channel information after connection
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            
            # Create channel names mapping (Muse specific)
            muse_channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
            self.channel_names = {i: muse_channel_names[i] for i in range(len(self.eeg_channels))}
            
            # Initialize signal quality tracking for connected channels
            for i, ch in enumerate(self.eeg_channels):
                channel_name = muse_channel_names[i] if i < len(muse_channel_names) else f"Ch{ch}"
                self.signal_quality[channel_name] = {
                    'good_samples': 0, 'total_samples': 0, 'quality': 0.0
                }
            
            self.logger.info("Successfully connected to Muse headset!")
            self.logger.info(f"EEG Channels: {self.eeg_channels}")
            self.logger.info(f"Channel Mapping: {self.channel_names}")
            self.logger.info(f"Sampling Rate: {self.sampling_rate} Hz")
            
            return True
            
        except BrainFlowError as e:
            self.logger.error(f"Brainflow error during connection: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during connection: {e}")
            self.is_connected = False
            return False
    
    def start_streaming(self) -> bool:
        """
        Start EEG data streaming with enhanced monitoring
        
        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        if not self.is_connected:
            self.logger.error("Not connected to Muse. Call connect() first.")
            return False
            
        try:
            self.board.start_stream()
            self.is_streaming = True
            self.stats['last_data_time'] = time.time()
            self.logger.info("Started EEG data streaming with enhanced monitoring")
            
            return True
            
        except BrainFlowError as e:
            self.logger.error(f"Error starting stream: {e}")
            return False
    
    def get_eeg_data(self, num_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get EEG data with enhanced quality monitoring
        
        Args:
            num_samples: Number of samples to retrieve (None for all available)
            
        Returns:
            np.ndarray: EEG data with shape (channels, samples) or None if error
        """
        if not self.is_streaming:
            self.logger.warning("Not currently streaming. Call start_streaming() first.")
            return None
            
        try:
            if num_samples is None:
                num_samples = 256  # Default to 256 samples (1 second at 256Hz)
            
            data = self.board.get_current_board_data(num_samples)
            
            if data.size == 0:
                return None
                
            # Extract EEG channels
            if self.eeg_channels is not None:
                eeg_data = data[self.eeg_channels, :]
            else:
                # Fallback to first 4 channels
                eeg_data = data[:4, :]
            
            if eeg_data is not None and eeg_data.size > 0:
                # Update statistics
                self.stats['total_samples'] += eeg_data.shape[1]
                self.stats['last_data_time'] = time.time()
                
                # Assess signal quality
                quality_score = self._assess_signal_quality(eeg_data)
                self.stats['quality_scores'].append(quality_score)
                
                # Store in buffer
                with self.buffer_lock:
                    self.data_buffer.extend(eeg_data.T)  # Store samples as rows
                
                self.sample_count += 1
                
            return eeg_data
            
        except BrainFlowError as e:
            self.logger.error(f"Error getting EEG data: {e}")
            self.stats['dropped_samples'] += 1
            return None
    
    def _assess_signal_quality(self, eeg_data: np.ndarray) -> float:
        """
        Assess signal quality for current data sample
        
        Args:
            eeg_data: EEG data with shape (channels, samples)
            
        Returns:
            float: Overall quality score (0.0 to 1.0)
        """
        if eeg_data is None or eeg_data.size == 0:
            return 0.0
        
        channel_qualities = []
        
        for ch_idx in range(eeg_data.shape[0]):
            channel_data = eeg_data[ch_idx, :]
            channel_name = self.channel_names.get(ch_idx, f"Ch{ch_idx}")
            
            # Calculate quality metrics
            amplitude = np.std(channel_data)
            max_val = np.max(np.abs(channel_data))
            noise_level = np.std(np.diff(channel_data))  # High-frequency noise estimate
            
            # Quality criteria
            amplitude_ok = (self.quality_thresholds['min_amplitude'] <= 
                          amplitude <= self.quality_thresholds['max_amplitude'])
            not_saturated = max_val < self.quality_thresholds['saturation_threshold']
            low_noise = noise_level < self.quality_thresholds['noise_threshold']
            
            # Calculate channel quality score
            quality = 0.0
            if amplitude_ok:
                quality += 0.4
            if not_saturated:
                quality += 0.3
            if low_noise:
                quality += 0.3
                
            channel_qualities.append(quality)
            
            # Update signal quality tracking
            self.signal_quality[channel_name]['total_samples'] += channel_data.shape[0]
            if quality > 0.7:
                self.signal_quality[channel_name]['good_samples'] += channel_data.shape[0]
            
            # Update running quality average
            total = self.signal_quality[channel_name]['total_samples']
            good = self.signal_quality[channel_name]['good_samples']
            self.signal_quality[channel_name]['quality'] = good / total if total > 0 else 0.0
        
        return np.mean(channel_qualities)
    
    def get_buffered_data(self, num_samples: int) -> Optional[np.ndarray]:
        """
        Get recent data from the circular buffer
        
        Args:
            num_samples: Number of recent samples to retrieve
            
        Returns:
            np.ndarray: Buffered data with shape (samples, channels) or None if insufficient data
        """
        with self.buffer_lock:
            if len(self.data_buffer) < num_samples:
                return None
            
            # Get the most recent samples
            recent_data = list(self.data_buffer)[-num_samples:]
            return np.array(recent_data)
    
    def get_signal_quality_report(self) -> Dict[str, Any]:
        """
        Get comprehensive signal quality report
        
        Returns:
            Dict containing detailed signal quality information
        """
        current_time = time.time()
        connection_duration = (current_time - self.stats['connection_time'] 
                             if self.stats['connection_time'] else 0)
        
        # Calculate recent quality (last 100 samples)
        recent_quality = (np.mean(list(self.stats['quality_scores'])[-100:]) 
                         if self.stats['quality_scores'] else 0.0)
        
        # Calculate data rate
        data_rate = (self.stats['total_samples'] / connection_duration 
                    if connection_duration > 0 else 0)
        
        return {
            'overall_quality': recent_quality,
            'channel_qualities': dict(self.signal_quality),
            'connection_duration': connection_duration,
            'total_samples': self.stats['total_samples'],
            'dropped_samples': self.stats['dropped_samples'],
            'data_rate': data_rate,
            'buffer_fill': len(self.data_buffer),
            'is_streaming': self.is_streaming,
            'last_data_age': (current_time - self.stats['last_data_time'] 
                            if self.stats['last_data_time'] else float('inf'))
        }
    
    def update_quality_thresholds(self, **kwargs):
        """Update signal quality thresholds"""
        for key, value in kwargs.items():
            if key in self.quality_thresholds:
                self.quality_thresholds[key] = value
                self.logger.info(f"Updated quality threshold {key} to {value}")
    
    def stop_streaming(self):
        """Stop EEG data streaming"""
        if self.is_streaming and self.board:
            try:
                self.board.stop_stream()
                self.is_streaming = False
                self.logger.info("Stopped EEG data streaming")
                    
            except BrainFlowError as e:
                self.logger.error(f"Error stopping stream: {e}")
    
    def disconnect(self):
        """Disconnect from Muse headset and cleanup"""
        if self.board:
            try:
                if self.is_streaming:
                    self.stop_streaming()
                    
                self.board.release_session()
                self.is_connected = False
                self.logger.info("Disconnected from Muse headset")
                
                # Log final statistics
                quality_report = self.get_signal_quality_report()
                self.logger.info(f"Session summary: {quality_report}")
                
            except BrainFlowError as e:
                self.logger.error(f"Error during disconnect: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.disconnect()


def main():
    """
    Test function for enhanced Muse connector
    """
    print("=== Enhanced Muse Connector Test ===")
    print("Testing enhanced signal quality monitoring")
    print("Press Ctrl+C to stop")
    
    with EnhancedMuseConnector(board_type="MUSE_2", mac_address="") as muse:
        try:
            # Connect to Muse
            if not muse.connect():
                print("Failed to connect to Muse headset")
                return
            
            # Start streaming
            if not muse.start_streaming():
                print("Failed to start streaming")
                return
            
            print("Connected and streaming with enhanced monitoring!")
            print("-" * 60)
            
            start_time = time.time()
            last_report_time = time.time()
            
            while True:
                # Get EEG data
                eeg_data = muse.get_eeg_data()
                
                if eeg_data is not None and eeg_data.size > 0:
                    # Print quality report every 5 seconds
                    current_time = time.time()
                    if current_time - last_report_time >= 5.0:
                        quality_report = muse.get_signal_quality_report()
                        print(f"\n=== Quality Report (t={current_time-start_time:.1f}s) ===")
                        print(f"Overall Quality: {quality_report['overall_quality']:.2f}")
                        print(f"Data Rate: {quality_report['data_rate']:.1f} Hz")
                        print(f"Buffer Fill: {quality_report['buffer_fill']}/{muse.buffer_size}")
                        
                        for ch_name, ch_quality in quality_report['channel_qualities'].items():
                            print(f"{ch_name}: {ch_quality['quality']:.2f}")
                        
                        last_report_time = current_time
                
                time.sleep(0.01)  # Small delay
            
        except KeyboardInterrupt:
            print(f"\nStopped by user")
            
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    print("Test completed!")


if __name__ == "__main__":
    main()
