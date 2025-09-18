"""
Main application for BCI Orb Control Project
Combines Muse Connector, Signal Processor, and WebSocket Server for real-time EEG analysis
"""

import time
import logging
import asyncio
from muse_connector import MuseConnector
from signal_processor import SignalProcessor
from websocket_server import BCIWebSocketServer


async def main():
    """
    Main application: Real-time EEG analysis with Muse headset and WebSocket communication
    """
    print("=== BCI Orb Control - Real-time EEG Analysis with WebSocket ===")
    print("WebSocket server: ws://localhost:8765")
    print("Press Ctrl+C to stop")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize WebSocket server
    websocket_server = BCIWebSocketServer(host="localhost", port=8765)
    
    # Start WebSocket server in background
    server_task = asyncio.create_task(websocket_server.start_server())
    
    # Give server time to start
    await asyncio.sleep(2)
    
    # Test WebSocket server before proceeding with Muse
    print("Testing WebSocket server...")
    try:
        import websockets
        test_uri = "ws://localhost:8765"
        
        # Quick connection test
        async with websockets.connect(test_uri) as test_websocket:
            print("✅ WebSocket server is working!")
            
            # Test receiving a message (with timeout)
            try:
                message = await asyncio.wait_for(test_websocket.recv(), timeout=3.0)
                print("✅ WebSocket communication verified")
            except asyncio.TimeoutError:
                print("⚠️ WebSocket connected but no welcome message (still OK)")
    
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        print("Cannot proceed without working WebSocket server.")
        print("Stopping and exiting...")
        server_task.cancel()
        return
    
    print("WebSocket verified! Starting Muse connection...")
    print("-" * 50)
    
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
            
            # Initialize signal processor with the actual sampling rate
            signal_processor = SignalProcessor(sampling_rate=muse.sampling_rate)
            
            # Send system status
            await websocket_server.send_system_status("muse_connected", "Muse headset connected and streaming")
            
            print("Connected and streaming! Sending data via WebSocket...")
            print("Press Ctrl+C to stop")
            print("-" * 80)
            
            sample_count = 0
            last_display_time = time.time()
            double_blink_count = 0
            
            while True:  # Run continuously
                # Get EEG data from Muse
                eeg_data = muse.get_eeg_data()
                
                if eeg_data is not None and eeg_data.size > 0:
                    # Process EEG data with signal processor
                    processing_results = signal_processor.process_eeg_sample(eeg_data)
                    
                    # Send data via WebSocket every 0.5 seconds
                    current_time = time.time()
                    if current_time - last_display_time >= 0.5 and processing_results:
                        # Send mental state data
                        mental_state = processing_results.get('mental_state', {})
                        if mental_state:
                            overall_state = mental_state.get('overall_state', 'unknown')
                            average_ratio = mental_state.get('average_ratio', 0.0)
                            individual_ratios = mental_state.get('individual_ratios', {})
                            
                            # Send via WebSocket
                            await websocket_server.send_mental_state(
                                overall_state, average_ratio, individual_ratios
                            )
                            
                            # Display in console
                            print(f"Sample {sample_count}: {overall_state} (ratio: {average_ratio:.3f})")
                        
                        # Send blink data
                        blinks = processing_results.get('blinks', [])
                        double_blinks = processing_results.get('double_blinks', [])
                        
                        # Send single blinks
                        for blink in blinks:
                            await websocket_server.send_single_blink(blink)
                        
                        # Send double blinks
                        for db in double_blinks:
                            double_blink_count += 1
                            await websocket_server.send_double_blink(db)
                            print(f"🎯 DOUBLE BLINK #{double_blink_count}: {db['time_interval']:.3f}s interval")
                        
                        last_display_time = current_time
                    
                    sample_count += 1
                else:
                    # If no data, wait a bit longer
                    await asyncio.sleep(0.1)
                    continue
                
                await asyncio.sleep(0.01)  # Small delay to prevent excessive CPU usage
            
        except KeyboardInterrupt:
            print(f"\nStopped by user. Total samples processed: {sample_count}")
            print(f"Total double blinks detected: {double_blink_count}")
            await websocket_server.send_system_status("system_stopped", "BCI system stopped by user")
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            await websocket_server.send_system_status("error", f"System error: {str(e)}")
    
    print("Disconnected from Muse headset!")
    
    # Stop WebSocket server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
