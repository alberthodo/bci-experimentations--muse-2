"""
Simple test script to verify Muse connection using the working pattern
"""

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time

def test_muse_connection():
    """Test Muse connection using the working pattern"""
    print("=== Testing Muse Connection ===")
    
    # Setup Muse 2 connection (same as working code)
    params = BrainFlowInputParams()
    board_id = BoardIds.MUSE_2_BOARD
    board = BoardShim(board_id, params)
    
    # Get channel information
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    
    print(f"EEG Channels: {eeg_channels}")
    print(f"Sampling Rate: {sampling_rate} Hz")
    
    try:
        # Connect to board
        board.prepare_session()
        print("Connected to Muse 2! Starting data stream...")
        
        # Start streaming
        board.start_stream()
        
        print("Streaming data for 5 seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < 5:  # Run for 5 seconds
            try:
                # Get latest data (same as working code)
                data = board.get_current_board_data(256)  # Get up to 256 samples
                
                if data.shape[1] > 0:
                    # Extract EEG channels
                    eeg_data = data[eeg_channels, :]
                    
                    # Print sample info every 50 iterations
                    if sample_count % 50 == 0:
                        print(f"Sample {sample_count}: Data shape {eeg_data.shape}")
                        if eeg_data.size > 0:
                            latest_sample = eeg_data[:, -1]
                            print(f"Latest values: {latest_sample}")
                    
                    sample_count += 1
                
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                print(f"Error getting data: {e}")
                break
        
        print(f"\nTest completed! Collected {sample_count} data samples")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        try:
            if board.is_prepared():
                board.stop_stream()
                board.release_session()
            print("Disconnected from Muse 2")
        except:
            pass

if __name__ == "__main__":
    test_muse_connection()
