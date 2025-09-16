"""
Test GUI components only without Muse connection
"""

import time
import threading
import numpy as np
from realtime_visualizer import RealTimeVisualizer

def test_gui():
    """Test GUI initialization and basic functionality"""
    print("Testing GUI initialization...")
    
    try:
        # Create visualizer
        visualizer = RealTimeVisualizer(update_interval=0.1, buffer_size=100)
        
        # Initialize GUI
        print("Initializing GUI...")
        visualizer.initialize_gui()
        print("✅ GUI initialized successfully!")
        
        # Start animation
        print("Starting animation...")
        visualizer.start_visualization()
        print("✅ Animation started!")
        
        # Generate some test data in background
        def generate_test_data():
            for i in range(100):
                if not visualizer.running:
                    break
                
                # Generate test EEG data
                eeg_data = np.random.randn(4, 10) * 50
                visualizer.update_eeg_data(eeg_data, time.time())
                
                # Generate test spectral features
                spectral_features = {
                    'TP9': {'alpha': np.random.random(), 'beta': np.random.random(), 'theta': np.random.random()},
                    'AF7': {'alpha': np.random.random(), 'beta': np.random.random(), 'theta': np.random.random()},
                    'AF8': {'alpha': np.random.random(), 'beta': np.random.random(), 'theta': np.random.random()},
                    'TP10': {'alpha': np.random.random(), 'beta': np.random.random(), 'theta': np.random.random()}
                }
                visualizer.update_spectral_features(spectral_features)
                
                # Generate test mental state
                states = ['relaxed', 'focused', 'meditative', 'alert', 'neutral']
                mental_state = {
                    'primary_state': np.random.choice(states),
                    'confidence': np.random.random()
                }
                visualizer.update_mental_state(mental_state)
                
                # Generate test signal quality
                signal_quality = {
                    'TP9_quality': np.random.random(),
                    'AF7_quality': np.random.random(),
                    'AF8_quality': np.random.random(),
                    'TP10_quality': np.random.random()
                }
                visualizer.update_signal_quality(signal_quality)
                
                time.sleep(0.2)
        
        # Start data generation thread
        data_thread = threading.Thread(target=generate_test_data, daemon=True)
        data_thread.start()
        
        print("🎉 GUI test running! Close the window to stop.")
        
        # Run the GUI
        visualizer.run()
        
        print("✅ GUI test completed successfully!")
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gui()
