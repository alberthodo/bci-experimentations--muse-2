# BCI Control System

A Brain-Computer Interface control system that tracks 5 distinct states for control applications.

## Features

Tracks the following 5 states:
1. **Normal** (default state)
2. **Double Blink** (input command)
3. **Jaw Clench** (input command)
4. **Sustained Focus** (mental state)
5. **Sustained Relaxation** (mental state)

## Components

This project imports and integrates:
- Advanced Signal Processor (from project1)
- Advanced Muse Connector (from project1)
- Mental State Classifier (from mental_state_visualizer)
- Blink Detector (from project1)

## Usage

```bash
python3 console_visualizer.py
```

## Requirements

```bash
pip install -r requirements.txt
```

## State Detection

- **Double Blink**: Detected using EOG signals from frontal electrodes
- **Jaw Clench**: Detected using muscle artifacts from temporal electrodes
- **Sustained Focus**: Detected through sustained alpha/beta ratio analysis
- **Sustained Relaxation**: Detected through sustained alpha/beta ratio analysis
- **Normal**: Default state when no other states are active
