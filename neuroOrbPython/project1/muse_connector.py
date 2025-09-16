
"""
Muse Connector for BCI Orb Control Project
Implements MVP 1.1: Muse Connection and Data Streaming
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Any
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.board_shim import BrainFlowError


class MuseConnector:
    """
    Handles connection to Muse headset and EEG data streaming
    """
    
    def __init__(self, board_type: str = "MUSE_2", mac_address: str = ""):
        """
        Initialize Muse connector
        
        Args:
            board_type: Type of Muse board ("MUSE_2", "MUSE_S", "MUSE_2016")
            mac_address: MAC address of the Muse device (empty for auto-scan)
        """
        self.board = None
        self.board_id = self._get_board_id(board_type)
        self.params = BrainFlowInputParams()
        self.params.mac_address = mac_address
        self.is_streaming = False
        self.is_connected = False
        self.sample_count = 0
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Get actual channel information from Brainflow
        self.eeg_channels = None
        self.sampling_rate = None
        self.channel_names = {}
        
    def _get_board_id(self, board_type: str) -> int:
        """Get Brainflow board ID for Muse type"""
        board_mapping = {
            "MUSE_2": BoardIds.MUSE_2_BOARD.value,
            "MUSE_S": BoardIds.MUSE_S_BOARD.value,
            "MUSE_2016": BoardIds.MUSE_2016_BOARD.value
        }
        return board_mapping.get(board_type, BoardIds.MUSE_2_BOARD.value)
    
    def _get_board_type_name(self) -> str:
        """Get board type name from board ID"""
        board_id_mapping = {
            BoardIds.MUSE_2_BOARD.value: "MUSE_2",
            BoardIds.MUSE_S_BOARD.value: "MUSE_S", 
            BoardIds.MUSE_2016_BOARD.value: "MUSE_2016"
        }
        return board_id_mapping.get(self.board_id, "UNKNOWN")
    
    def connect(self) -> bool:
        """
        Establish connection to Muse headset
        
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
            
            # Get channel information after connection
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            
            # Create channel names mapping
            self.channel_names = {i: f"Ch{ch}" for i, ch in enumerate(self.eeg_channels)}
            
            self.logger.info("Successfully connected to Muse headset!")
            self.logger.info(f"EEG Channels: {self.eeg_channels}")
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
        Start EEG data streaming
        
        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        if not self.is_connected:
            self.logger.error("Not connected to Muse. Call connect() first.")
            return False
            
        try:
            self.board.start_stream()
            self.is_streaming = True
            self.logger.info("Started EEG data streaming")
            
            return True
            
        except BrainFlowError as e:
            self.logger.error(f"Error starting stream: {e}")
            return False
    
    def get_eeg_data(self, num_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get EEG data from the stream
        
        Args:
            num_samples: Number of samples to retrieve (None for all available)
            
        Returns:
            np.ndarray: EEG data with shape (channels, samples) or None if error
        """
        if not self.is_streaming:
            self.logger.warning("Not currently streaming. Call start_streaming() first.")
            return None
            
        try:
            # Use get_current_board_data like the working code
            if num_samples is None:
                num_samples = 256  # Default to 256 samples (1 second at 256Hz)
            
            data = self.board.get_current_board_data(num_samples)
            
            if data.size == 0:
                self.logger.debug("No data available")
                return None
                
            # Extract EEG channels using the actual channel indices
            if self.eeg_channels is not None:
                eeg_data = data[self.eeg_channels, :]
            else:
                # Fallback to first 4 channels if eeg_channels not available
                eeg_data = data[:4, :]
            
            # Increment sample count
            if eeg_data is not None and eeg_data.size > 0:
                self.sample_count += 1
                
            return eeg_data
            
        except BrainFlowError as e:
            self.logger.error(f"Error getting EEG data: {e}")
            return None
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the current data stream
        
        Returns:
            Dict containing stream information
        """
        if not self.is_streaming:
            return {"error": "Not currently streaming"}
            
        try:
            # Get a small sample to check data availability
            data = self.board.get_current_board_data(10)
            if data.size == 0:
                return {"samples": 0, "channels": 0}
                
            return {
                "samples": data.shape[1],
                "channels": len(self.eeg_channels) if self.eeg_channels is not None else 0,
                "sampling_rate": self.sampling_rate,
                "eeg_channels": list(self.eeg_channels) if self.eeg_channels is not None else [],
                "channel_names": list(self.channel_names.values())
            }
            
        except BrainFlowError as e:
            return {"error": str(e)}
    
    def log_data_sample(self, data: np.ndarray, sample_count: int = 0):
        """
        Log a sample of EEG data for debugging
        
        Args:
            data: EEG data array
            sample_count: Current sample count for logging
        """
        if data is not None and data.size > 0:
            # Get the latest sample from each channel
            latest_sample = data[:, -1] if data.shape[1] > 0 else data[:, 0]
            
            # Create channel info string
            channel_info = []
            for i, value in enumerate(latest_sample):
                channel_name = self.channel_names.get(i, f"Ch{i}")
                channel_info.append(f"{channel_name}={value:.3f}")
            
            self.logger.info(f"Sample {sample_count}: {' | '.join(channel_info)}")
    
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
    Simple Muse EEG data streaming test
    """
    print("=== Simple Muse EEG Data Streaming Test ===")
    print("Press Ctrl+C to stop")
    
    # Use context manager for automatic cleanup
    with MuseConnector(board_type="MUSE_2", mac_address="") as muse:
        try:
            # Connect to Muse
            if not muse.connect():
                print("Failed to connect to Muse headset")
                return
            
            # Start streaming
            if not muse.start_streaming():
                print("Failed to start streaming")
                return
            
            print("Connected and streaming! Showing raw EEG data...")
            print("Press Ctrl+C to stop")
            print("-" * 60)
            
            sample_count = 0
            last_display_time = time.time()
            
            while True:  # Run continuously
                # Get EEG data
                eeg_data = muse.get_eeg_data()
                
                if eeg_data is not None and eeg_data.size > 0:
                    # Display data every 0.5 seconds to avoid spam
                    current_time = time.time()
                    if current_time - last_display_time >= 0.5:
                        muse.log_data_sample(eeg_data, sample_count)
                        last_display_time = current_time
                    
                    sample_count += 1
                else:
                    # If no data, wait a bit longer
                    time.sleep(0.1)
                    continue
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            
        except KeyboardInterrupt:
            print(f"\nStopped by user. Total samples collected: {sample_count}")
            
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    print("Disconnected from Muse headset!")


if __name__ == "__main__":
    main()
