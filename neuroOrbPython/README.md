# BCI Orb Control Project - MVP 1.1

This project implements a brain-computer interface system using a Muse headset to control a 3D orb in Unity through brainwave analysis and voluntary eye blinks.

## MVP 1.1: Muse Connection and Data Streaming

This phase establishes the basic communication pipeline with the Muse headset and implements data logging functionality.

### Features Implemented

- ✅ **Muse Headset Connection**: Reliable connection to Muse 2, Muse S, or Muse 2016
- ✅ **EEG Data Streaming**: Continuous streaming of raw EEG data at 256 Hz
- ✅ **Data Logging**: Comprehensive logging of EEG data and system events
- ✅ **Error Handling**: Robust error handling for connection issues
- ✅ **Session Management**: Proper session lifecycle management with cleanup

### Files

- `muse_connector.py` - Main Muse connector class with connection and streaming functionality
- `data_logger.py` - Data logging utility for EEG data and system events
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your Muse headset is paired with your computer via Bluetooth

### Usage

#### Basic Usage

```python
from muse_connector import MuseConnector

# Create connector with logging enabled
with MuseConnector(board_type="MUSE_2", enable_logging=True) as muse:
    # Connect to Muse
    if muse.connect():
        # Start streaming
        if muse.start_streaming():
            # Get EEG data
            eeg_data = muse.get_eeg_data()
            print(f"EEG data shape: {eeg_data.shape}")
```

#### Running the Test

```bash
python muse_connector.py
```

This will:
1. Connect to your Muse headset
2. Start streaming EEG data
3. Collect data for 10 seconds
4. Log all data to CSV and JSON files
5. Display session summary

### Data Output

The system creates a `logs/` directory with the following files:

- `eeg_data_YYYYMMDD_HHMMSS.csv` - Raw EEG data with timestamps
- `events_YYYYMMDD_HHMMSS.json` - System events and errors
- `session_YYYYMMDD_HHMMSS.json` - Complete session metadata

### EEG Channel Mapping

For Muse 2 headset:
- **TP9**: Left temporal electrode
- **AF7**: Left frontal electrode  
- **AF8**: Right frontal electrode
- **TP10**: Right temporal electrode

### Configuration

You can customize the connector:

```python
# Different Muse models
muse = MuseConnector(board_type="MUSE_S")  # Muse S
muse = MuseConnector(board_type="MUSE_2016")  # Muse 2016

# Specific MAC address
muse = MuseConnector(mac_address="00:11:22:33:44:55")

# Disable logging
muse = MuseConnector(enable_logging=False)
```

### Troubleshooting

1. **Connection Issues**: Ensure Muse is paired and in range
2. **No Data**: Check electrode contact and signal quality
3. **Permission Errors**: Ensure Bluetooth permissions are granted
4. **Import Errors**: Verify all dependencies are installed

### Next Steps (MVP 1.2)

The next phase will implement:
- WebSocket communication layer
- Unity WebSocket client
- Basic message passing protocol

### Technical Details

- **Sampling Rate**: 256 Hz
- **Data Format**: NumPy arrays with shape (4, samples)
- **Units**: Microvolts (μV)
- **Connection**: Bluetooth Low Energy (BLE)
