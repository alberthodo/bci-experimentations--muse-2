"""
Test script to verify all components work correctly
"""

import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_signal_processor():
    """Test the advanced signal processor"""
    print("=== Testing Advanced Signal Processor ===")
    
    try:
        from advanced_signal_processor import AdvancedSignalProcessor
        
        # Create processor
        processor = AdvancedSignalProcessor(sampling_rate=256)
        
        # Generate test data
        duration = 2.0
        samples = int(256 * duration)
        t = np.linspace(0, duration, samples)
        
        # Create realistic EEG with different frequency components
        test_data = np.zeros((4, samples))
        for ch in range(4):
            alpha_component = 50 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
            beta_component = 30 * np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
            theta_component = 40 * np.sin(2 * np.pi * 6 * t)   # 6 Hz theta
            noise = np.random.normal(0, 15, samples)
            test_data[ch, :] = alpha_component + beta_component + theta_component + noise
        
        # Process data
        results = processor.process_eeg_comprehensive(test_data)
        
        if results:
            mental_state = results.get('mental_state', {})
            print(f"✅ Signal Processor: Primary state = {mental_state.get('primary_state', 'unknown')}")
            print(f"   Confidence = {mental_state.get('confidence', 0.0):.3f}")
            return True
        else:
            print("❌ Signal Processor: No results returned")
            return False
            
    except Exception as e:
        print(f"❌ Signal Processor Error: {e}")
        return False

def test_mental_state_classifier():
    """Test the mental state classifier"""
    print("\n=== Testing Mental State Classifier ===")
    
    try:
        from mental_state_classifier import MentalStateClassifier
        
        # Create classifier
        classifier = MentalStateClassifier()
        
        # Generate mock features
        spectral_features = {
            'TP9': {'alpha': 0.8, 'beta': 0.3, 'theta': 0.4, 'gamma': 0.2, 'delta': 0.1},
            'AF7': {'alpha': 0.7, 'beta': 0.4, 'theta': 0.3, 'gamma': 0.3, 'delta': 0.2},
            'AF8': {'alpha': 0.9, 'beta': 0.2, 'theta': 0.5, 'gamma': 0.1, 'delta': 0.1},
            'TP10': {'alpha': 0.6, 'beta': 0.5, 'theta': 0.4, 'gamma': 0.4, 'delta': 0.3}
        }
        
        temporal_features = {
            'TP9': {'std': 0.5, 'skewness': 0.1, 'kurtosis': 0.2, 'hjorth_activity': 0.3, 'hjorth_mobility': 0.4, 'hjorth_complexity': 0.5, 'rms': 0.6, 'zero_crossings': 10},
            'AF7': {'std': 0.4, 'skewness': 0.2, 'kurtosis': 0.1, 'hjorth_activity': 0.4, 'hjorth_mobility': 0.3, 'hjorth_complexity': 0.6, 'rms': 0.5, 'zero_crossings': 12},
            'AF8': {'std': 0.6, 'skewness': 0.0, 'kurtosis': 0.3, 'hjorth_activity': 0.2, 'hjorth_mobility': 0.5, 'hjorth_complexity': 0.4, 'rms': 0.7, 'zero_crossings': 8},
            'TP10': {'std': 0.3, 'skewness': 0.3, 'kurtosis': 0.0, 'hjorth_activity': 0.5, 'hjorth_mobility': 0.2, 'hjorth_complexity': 0.7, 'rms': 0.4, 'zero_crossings': 15}
        }
        
        cross_features = {'frontal_asymmetry': 0.1, 'temporal_asymmetry': -0.2}
        
        # Test prediction
        prediction = classifier.predict_mental_state(spectral_features, temporal_features, cross_features)
        
        if prediction and 'state' in prediction:
            print(f"✅ Mental State Classifier: State = {prediction['state']}")
            print(f"   Confidence = {prediction.get('confidence', 0.0):.3f}")
            return True
        else:
            print("❌ Mental State Classifier: Invalid prediction")
            return False
            
    except Exception as e:
        print(f"❌ Mental State Classifier Error: {e}")
        return False

def test_enhanced_muse_connector():
    """Test the enhanced Muse connector (without actual hardware)"""
    print("\n=== Testing Enhanced Muse Connector ===")
    
    try:
        from enhanced_muse_connector import EnhancedMuseConnector
        
        # Create connector (don't try to connect without hardware)
        connector = EnhancedMuseConnector(board_type="MUSE_2", mac_address="")
        
        # Test initialization
        if connector.board_id is not None:
            print("✅ Enhanced Muse Connector: Initialization successful")
            print(f"   Board ID: {connector.board_id}")
            return True
        else:
            print("❌ Enhanced Muse Connector: Failed to initialize")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced Muse Connector Error: {e}")
        return False

def test_realtime_visualizer():
    """Test the real-time visualizer (GUI components)"""
    print("\n=== Testing Real-time Visualizer ===")
    
    try:
        from realtime_visualizer import RealTimeVisualizer
        
        # Create visualizer
        visualizer = RealTimeVisualizer(update_interval=0.1, buffer_size=100)
        
        # Test data updates
        test_eeg = np.random.randn(4, 10) * 50
        timestamp = time.time()
        
        visualizer.update_eeg_data(test_eeg, timestamp)
        
        # Test spectral features update
        spectral_features = {
            'TP9': {'alpha': 0.5, 'beta': 0.3, 'theta': 0.2},
            'AF7': {'alpha': 0.4, 'beta': 0.4, 'theta': 0.3},
            'AF8': {'alpha': 0.6, 'beta': 0.2, 'theta': 0.4},
            'TP10': {'alpha': 0.3, 'beta': 0.5, 'theta': 0.2}
        }
        visualizer.update_spectral_features(spectral_features)
        
        # Test mental state update
        mental_state = {'primary_state': 'relaxed', 'confidence': 0.8}
        visualizer.update_mental_state(mental_state)
        
        # Test signal quality update
        signal_quality = {
            'TP9_quality': 0.9, 'AF7_quality': 0.8,
            'AF8_quality': 0.7, 'TP10_quality': 0.85
        }
        visualizer.update_signal_quality(signal_quality)
        
        print("✅ Real-time Visualizer: Data updates successful")
        print(f"   Current state: {visualizer.current_mental_state}")
        print(f"   Buffer sizes: EEG={len(visualizer.eeg_buffers['TP9'])}")
        return True
        
    except Exception as e:
        print(f"❌ Real-time Visualizer Error: {e}")
        return False

def main():
    """Run all component tests"""
    print("🧠 Mental State Visualizer - Component Tests")
    print("=" * 50)
    
    tests = [
        test_signal_processor,
        test_mental_state_classifier,
        test_enhanced_muse_connector,
        test_realtime_visualizer
    ]
    
    results = []
    
    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"❌ Test Error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("🔍 Test Summary")
    print("-" * 20)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Advanced Signal Processor",
        "Mental State Classifier", 
        "Enhanced Muse Connector",
        "Real-time Visualizer"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All components are working correctly!")
        print("\nTo run the full application:")
        print("python main_visualizer.py")
    else:
        print("⚠️  Some components have issues. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main()
