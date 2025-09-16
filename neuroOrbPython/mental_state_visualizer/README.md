
# Mental State Visualizer

A comprehensive real-time mental state monitoring and visualization system using Muse EEG headsets. This project provides advanced signal processing, machine learning-based classification, and interactive visualization of mental states.

## Features

### 🧠 Advanced EEG Processing
- **Enhanced Muse Connectivity**: Robust connection with signal quality monitoring
- **Comprehensive Preprocessing**: Multi-stage filtering, artifact removal, notch filtering
- **Rich Feature Extraction**: Spectral, temporal, and cross-channel features
- **Real-time Processing**: Low-latency signal processing pipeline

### 🎯 Mental State Classification
- **Multiple Mental States**: Relaxed, Focused, Meditative, Alert, Neutral, Stressed
- **Machine Learning Models**: Ensemble of Random Forest, SVM, and Logistic Regression
- **Rule-based Fallback**: Works without training data
- **Adaptive Thresholds**: Self-adjusting classification parameters
- **Temporal Consistency**: Smoothing to reduce prediction jitter

### 📊 Real-time Visualization
- **Live EEG Signals**: Real-time plot of all 4 channels (TP9, AF7, AF8, TP10)
- **Frequency Band Power**: Dynamic visualization of brain wave activity
- **Mental State Timeline**: Track state changes over time
- **Classification Confidence**: Real-time confidence monitoring
- **Signal Quality Indicators**: Per-channel quality assessment
- **Session Statistics**: Duration, state distribution, and performance metrics

### 🔧 Advanced Features
- **Session Recording**: Save complete sessions to JSON files
- **Model Training**: Collect labeled data and train custom classifiers
- **Performance Monitoring**: Real-time processing speed and quality metrics
- **Configurable Settings**: Adjustable update rates and buffer sizes

## Installation

### Prerequisites
- Python 3.8 or higher
- Muse headset (Muse 2, Muse S, or Muse 2016)
- Bluetooth connectivity

### Setup
1. Clone or download the project:
```bash
cd mental_state_visualizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your Muse headset is paired with your computer via Bluetooth.

## Usage

### Basic Usage
Run the mental state visualizer with default settings:
```bash
python main_visualizer.py
```

### Advanced Usage
```bash
# Specify Muse model
python main_visualizer.py --board-type MUSE_S

# Use specific MAC address
python main_visualizer.py --mac-address "00:11:22:33:44:55"

# Enable training mode
python main_visualizer.py --enable-training

# Load pre-trained model
python main_visualizer.py --model-file "my_model.pkl"

# Set logging level
python main_visualizer.py --log-level DEBUG
```

### Command Line Options
- `--board-type`: Muse model (MUSE_2, MUSE_S, MUSE_2016)
- `--mac-address`: Specific device MAC address
- `--enable-training`: Enable training data collection
- `--model-file`: Path to pre-trained classifier model
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## GUI Interface

### Main Dashboard
The visualization dashboard contains six main panels:

1. **EEG Signals (Top Left)**: Real-time raw EEG from all channels
2. **Frequency Band Power (Top Right)**: Delta, Theta, Alpha, Beta, Gamma activity
3. **Mental State Timeline (Middle Left)**: State changes over time
4. **Classification Confidence (Middle Right)**: Real-time confidence scores
5. **Signal Quality (Bottom Left)**: Per-channel quality indicators
6. **State Distribution (Bottom Right)**: Session state percentage breakdown

### Status Bar
- **Current Mental State**: Live classification result
- **Confidence**: Classification certainty (0-100%)
- **Signal Quality**: Overall signal assessment
- **Session Duration**: Time elapsed since start

### Controls
- **Start/Stop Monitoring**: Toggle real-time processing
- **Reset Data**: Clear all buffers and restart session
- **Save Session**: Export session data to JSON file
- **Settings**: Adjust visualization parameters

## Mental States

### Defined States
- **Relaxed**: High alpha waves, low beta activity
- **Focused**: High beta waves, moderate alpha activity  
- **Meditative**: High theta waves, balanced alpha
- **Alert**: High gamma waves, elevated beta
- **Neutral**: Balanced activity across all bands
- **Stressed**: Irregular patterns, high noise

### Classification Features
The system uses over 100 features including:
- **Spectral Features**: Power in each frequency band, peak frequencies, spectral entropy
- **Temporal Features**: Statistical moments, complexity measures, amplitude characteristics
- **Cross-Channel Features**: Inter-channel correlations, hemispheric asymmetry

## Signal Quality

### Quality Metrics
- **Signal Amplitude**: Appropriate voltage ranges
- **Noise Level**: High-frequency interference assessment
- **Saturation Detection**: Artifact identification
- **Contact Quality**: Electrode-skin interface evaluation

### Quality Indicators
- **Green**: Excellent signal quality (>80%)
- **Orange**: Good signal quality (60-80%)
- **Red**: Poor signal quality (<60%)

## Data Export

### Session Files
Sessions are saved as JSON files containing:
```json
{
  "session_info": {
    "start_time": "timestamp",
    "total_samples": 12345,
    "state_durations": {...}
  },
  "mental_states": [...],
  "confidences": [...],
  "spectral_features": {...},
  "signal_quality": {...},
  "timestamps": [...]
}
```

## Troubleshooting

### Common Issues

**Muse Won't Connect**
- Ensure headset is paired via Bluetooth
- Check if headset is already connected to another app
- Try specifying MAC address manually
- Restart Bluetooth service

**Poor Signal Quality**
- Ensure electrodes have good skin contact
- Clean electrode contacts with alcohol
- Adjust headset position
- Check for electromagnetic interference

**Low Classification Confidence**
- Allow 2-3 minutes for signal stabilization
- Ensure proper electrode contact
- Try different mental activities
- Consider training custom models

**Visualization Lag**
- Reduce update interval in settings
- Close unnecessary applications
- Check CPU usage
- Reduce buffer sizes

### Debug Mode
Enable detailed logging:
```bash
python main_visualizer.py --log-level DEBUG
```

## Architecture

### Component Overview
```
main_visualizer.py          # Main orchestration
├── enhanced_muse_connector.py    # Hardware interface
├── advanced_signal_processor.py # Signal processing
├── mental_state_classifier.py   # ML classification
└── realtime_visualizer.py       # GUI visualization
```

### Data Flow
1. **Muse Connector**: Acquires raw EEG data at 256 Hz
2. **Signal Processor**: Applies filtering, extracts features
3. **Classifier**: Predicts mental state from features  
4. **Visualizer**: Updates real-time displays

### Threading Model
- **Main Thread**: GUI event loop
- **Processing Thread**: Signal processing and classification
- **Animation Thread**: Real-time plot updates

## Performance

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 100MB for application + session data
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Performance Metrics
- **Processing Latency**: <100ms typical
- **Update Rate**: 10 FPS visualization
- **Memory Usage**: ~200MB steady state
- **CPU Usage**: 10-20% on modern processors

## Customization

### Training Custom Models
1. Enable training mode: `--enable-training`
2. Collect labeled data during sessions
3. Train models using provided tools
4. Save and load custom classifiers

### Extending Mental States
Modify `mental_state_classifier.py` to add new states:
```python
self.mental_states = {
    0: "neutral",
    1: "focused", 
    2: "relaxed",
    3: "meditative",
    4: "alert",
    5: "stressed",
    6: "your_new_state"  # Add here
}
```

### Custom Visualizations
Extend `realtime_visualizer.py` to add new plots or modify existing displays.

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Run tests: `pytest`
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Add unit tests for new features

## License

This project is open source. See LICENSE file for details.

## Acknowledgments

- **Brainflow**: EEG data acquisition framework
- **Muse**: InteraXon EEG headset platform
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib**: Visualization framework

## Support

For issues and questions:
1. Check troubleshooting section
2. Review debug logs
3. Search existing issues
4. Create new issue with logs and system info

## Future Enhancements

### Planned Features
- **Cloud Integration**: Remote monitoring and data sync
- **Mobile App**: Companion smartphone application
- **Advanced ML**: Deep learning models for classification
- **Biofeedback**: Audio/visual feedback for training
- **Multi-user**: Support for multiple simultaneous users
- **VR Integration**: Virtual reality mental training environments

### Research Applications
- **Meditation Training**: Real-time feedback for mindfulness practice
- **Cognitive Load**: Mental workload assessment
- **Attention Training**: Focus enhancement protocols
- **Stress Monitoring**: Continuous stress level tracking
- **Sleep Research**: Pre-sleep mental state analysis
