"""
Real-time Mental State Visualizer
Creates interactive dashboard for visualizing mental states and EEG features in real-time
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import colorsys
import json


class RealTimeVisualizer:
    """
    Real-time visualization dashboard for mental state monitoring
    """
    
    def __init__(self, update_interval: float = 0.1, buffer_size: int = 1000):
        """
        Initialize real-time visualizer
        
        Args:
            update_interval: Update interval in seconds
            buffer_size: Size of data buffer for plots
        """
        self.update_interval = update_interval
        self.buffer_size = buffer_size
        self.logger = logging.getLogger(__name__)
        
        # Data buffers
        self.time_buffer = deque(maxlen=buffer_size)
        self.eeg_buffers = {
            'TP9': deque(maxlen=buffer_size),
            'AF7': deque(maxlen=buffer_size), 
            'AF8': deque(maxlen=buffer_size),
            'TP10': deque(maxlen=buffer_size)
        }
        
        # Feature buffers
        self.spectral_buffers = {
            'alpha': deque(maxlen=buffer_size),
            'beta': deque(maxlen=buffer_size),
            'theta': deque(maxlen=buffer_size),
            'gamma': deque(maxlen=buffer_size),
            'delta': deque(maxlen=buffer_size)
        }
        
        # Mental state buffers
        self.mental_state_buffer = deque(maxlen=buffer_size)
        self.confidence_buffer = deque(maxlen=buffer_size)
        self.state_history = deque(maxlen=50)
        
        # Signal quality buffers
        self.quality_buffers = {
            'TP9': deque(maxlen=buffer_size),
            'AF7': deque(maxlen=buffer_size),
            'AF8': deque(maxlen=buffer_size), 
            'TP10': deque(maxlen=buffer_size)
        }
        
        # GUI components
        self.root = None
        self.canvas = None
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.texts = {}
        self.bars = {}
        
        # Animation and threading
        self.animation = None
        self.running = False
        self.data_lock = threading.Lock()
        
        # Visualization settings
        self.colors = {
            'TP9': '#FF6B6B',     # Red
            'AF7': '#4ECDC4',     # Teal
            'AF8': '#45B7D1',     # Blue
            'TP10': '#96CEB4',     # Green
            'alpha': '#FFD93D',   # Yellow
            'beta': '#FF6B6B',    # Red
            'theta': '#6BCF7F',   # Green
            'gamma': '#A8E6CF',   # Light green
            'delta': '#DDA0DD'    # Plum
        }
        
        # Mental state colors
        self.state_colors = {
            'relaxed': '#87CEEB',     # Sky blue
            'focused': '#FF6347',     # Tomato
            'meditative': '#DDA0DD',  # Plum
            'alert': '#FFD700',       # Gold
            'neutral': '#D3D3D3',     # Light gray
            'stressed': '#DC143C',    # Crimson
            'unknown': '#696969'      # Dim gray
        }
        
        # Current state tracking
        self.current_mental_state = 'neutral'
        self.current_confidence = 0.0
        self.current_features = {}
        self.current_signal_quality = {}
        
        # Statistics
        self.session_stats = {
            'start_time': time.time(),
            'total_samples': 0,
            'state_durations': {state: 0.0 for state in self.state_colors.keys()},
            'last_update_time': time.time()
        }
        
    def initialize_gui(self):
        """Initialize the GUI components"""
        try:
            # Create main window
            self.root = tk.Tk()
            self.root.title("Mental State Visualizer - Real-time EEG Monitor")
            self.root.geometry("1400x1000")
            self.root.configure(bg='#2C3E50')
            
            # Create main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Configure grid weights
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(1, weight=1)
            
            # Create status frame
            self._create_status_frame(main_frame)
            
            # Create matplotlib figure
            self._create_plot_figure(main_frame)
            
            # Create control frame
            self._create_control_frame(main_frame)
            
            self.logger.info("GUI initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing GUI: {e}")
    
    def _create_status_frame(self, parent):
        """Create status information frame"""
        status_frame = ttk.LabelFrame(parent, text="Real-time Status", padding="10")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Current mental state display
        state_frame = ttk.Frame(status_frame)
        state_frame.grid(row=0, column=0, padx=(0, 20))
        
        ttk.Label(state_frame, text="Mental State:", font=('Arial', 12, 'bold')).grid(row=0, column=0)
        self.texts['current_state'] = ttk.Label(state_frame, text="Neutral", 
                                               font=('Arial', 16, 'bold'), foreground='blue')
        self.texts['current_state'].grid(row=1, column=0)
        
        # Confidence display
        conf_frame = ttk.Frame(status_frame)
        conf_frame.grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(conf_frame, text="Confidence:", font=('Arial', 12, 'bold')).grid(row=0, column=0)
        self.texts['confidence'] = ttk.Label(conf_frame, text="0.0%", 
                                            font=('Arial', 14), foreground='green')
        self.texts['confidence'].grid(row=1, column=0)
        
        # Signal quality display
        quality_frame = ttk.Frame(status_frame)
        quality_frame.grid(row=0, column=2, padx=(0, 20))
        
        ttk.Label(quality_frame, text="Signal Quality:", font=('Arial', 12, 'bold')).grid(row=0, column=0)
        self.texts['signal_quality'] = ttk.Label(quality_frame, text="Good", 
                                                font=('Arial', 14), foreground='green')
        self.texts['signal_quality'].grid(row=1, column=0)
        
        # Session duration
        session_frame = ttk.Frame(status_frame)
        session_frame.grid(row=0, column=3)
        
        ttk.Label(session_frame, text="Session Duration:", font=('Arial', 12, 'bold')).grid(row=0, column=0)
        self.texts['session_duration'] = ttk.Label(session_frame, text="00:00:00", 
                                                  font=('Arial', 14))
        self.texts['session_duration'].grid(row=1, column=0)
    
    def _create_plot_figure(self, parent):
        """Create matplotlib figure with subplots"""
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(3, 2, figsize=(14, 10))
        self.fig.patch.set_facecolor('#2C3E50')
        self.fig.suptitle('Real-time Mental State Monitoring', fontsize=16, color='white')
        
        # Configure subplot layout
        plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, 
                           hspace=0.4, wspace=0.3)
        
        # EEG Signals Plot (top left)
        ax_eeg = self.axes[0, 0]
        ax_eeg.set_title('EEG Signals (Real-time)', color='white', fontsize=12)
        ax_eeg.set_xlabel('Time (s)', color='white')
        ax_eeg.set_ylabel('Amplitude (μV)', color='white')
        ax_eeg.set_facecolor('#34495E')
        ax_eeg.grid(True, alpha=0.3)
        
        # Initialize EEG lines
        for channel, color in [('TP9', self.colors['TP9']), ('AF7', self.colors['AF7']), 
                              ('AF8', self.colors['AF8']), ('TP10', self.colors['TP10'])]:
            line, = ax_eeg.plot([], [], color=color, label=channel, linewidth=1.5)
            self.lines[f'eeg_{channel}'] = line
        ax_eeg.legend(loc='upper right')
        
        # Spectral Power Plot (top right)
        ax_spectral = self.axes[0, 1]
        ax_spectral.set_title('Frequency Band Power', color='white', fontsize=12)
        ax_spectral.set_xlabel('Time (s)', color='white')
        ax_spectral.set_ylabel('Relative Power', color='white')
        ax_spectral.set_facecolor('#34495E')
        ax_spectral.grid(True, alpha=0.3)
        
        # Initialize spectral lines
        for band, color in self.colors.items():
            if band in ['alpha', 'beta', 'theta', 'gamma', 'delta']:
                line, = ax_spectral.plot([], [], color=color, label=band, linewidth=2)
                self.lines[f'spectral_{band}'] = line
        ax_spectral.legend(loc='upper right')
        
        # Mental State Timeline (middle left)
        ax_states = self.axes[1, 0]
        ax_states.set_title('Mental State Timeline', color='white', fontsize=12)
        ax_states.set_xlabel('Time (s)', color='white')
        ax_states.set_ylabel('Mental State', color='white')
        ax_states.set_facecolor('#34495E')
        ax_states.grid(True, alpha=0.3)
        
        # Mental State Confidence (middle right)
        ax_confidence = self.axes[1, 1]
        ax_confidence.set_title('Classification Confidence', color='white', fontsize=12)
        ax_confidence.set_xlabel('Time (s)', color='white')
        ax_confidence.set_ylabel('Confidence', color='white')
        ax_confidence.set_facecolor('#34495E')
        ax_confidence.grid(True, alpha=0.3)
        ax_confidence.set_ylim(0, 1)
        
        line, = ax_confidence.plot([], [], color='#FFD93D', linewidth=2)
        self.lines['confidence'] = line
        
        # Signal Quality Indicators (bottom left)
        ax_quality = self.axes[2, 0]
        ax_quality.set_title('Signal Quality by Channel', color='white', fontsize=12)
        ax_quality.set_xlabel('Channel', color='white')
        ax_quality.set_ylabel('Quality Score', color='white')
        ax_quality.set_facecolor('#34495E')
        ax_quality.set_ylim(0, 1)
        
        # Create bars for signal quality
        channels = ['TP9', 'AF7', 'AF8', 'TP10']
        x_pos = np.arange(len(channels))
        self.bars['quality'] = ax_quality.bar(x_pos, [0]*len(channels), 
                                            color=[self.colors[ch] for ch in channels])
        ax_quality.set_xticks(x_pos)
        ax_quality.set_xticklabels(channels)
        
        # State Distribution (bottom right)
        ax_distribution = self.axes[2, 1]
        ax_distribution.set_title('Session State Distribution', color='white', fontsize=12)
        ax_distribution.set_facecolor('#34495E')
        
        # This will be updated with pie chart (store reference without indexing)
        self.distribution_ax = ax_distribution
        
        # Style all axes
        for i in range(self.axes.shape[0]):
            for j in range(self.axes.shape[1]):
                ax = self.axes[i, j]
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def _create_control_frame(self, parent):
        """Create control buttons frame"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Monitoring", 
                                      command=self.toggle_monitoring)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        # Reset button
        ttk.Button(control_frame, text="Reset Data", 
                  command=self.reset_data).grid(row=0, column=1, padx=(0, 10))
        
        # Save session button
        ttk.Button(control_frame, text="Save Session", 
                  command=self.save_session).grid(row=0, column=2, padx=(0, 10))
        
        # Settings button
        ttk.Button(control_frame, text="Settings", 
                  command=self.show_settings).grid(row=0, column=3)
    
    def update_eeg_data(self, eeg_data: np.ndarray, timestamp: float):
        """
        Update EEG data buffers
        
        Args:
            eeg_data: EEG data with shape (channels, samples)
            timestamp: Timestamp of the data
        """
        with self.data_lock:
            try:
                if eeg_data is not None and eeg_data.size > 0:
                    # Add timestamp
                    self.time_buffer.append(timestamp)
                    
                    # Add latest sample from each channel
                    channels = ['TP9', 'AF7', 'AF8', 'TP10']
                    for i, channel in enumerate(channels):
                        if i < eeg_data.shape[0]:
                            # Take the last sample from this channel
                            latest_sample = eeg_data[i, -1] if eeg_data.shape[1] > 0 else 0.0
                            self.eeg_buffers[channel].append(latest_sample)
                        else:
                            self.eeg_buffers[channel].append(0.0)
                    
                    self.session_stats['total_samples'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error updating EEG data: {e}")
    
    def update_spectral_features(self, spectral_features: Dict[str, Any]):
        """
        Update spectral feature buffers
        
        Args:
            spectral_features: Spectral features from signal processor
        """
        with self.data_lock:
            try:
                # Average power across channels for each frequency band
                band_averages = {}
                for band in ['alpha', 'beta', 'theta', 'gamma', 'delta']:
                    total_power = 0.0
                    channel_count = 0
                    
                    for channel in ['TP9', 'AF7', 'AF8', 'TP10']:
                        if channel in spectral_features:
                            channel_data = spectral_features[channel]
                            if band in channel_data:
                                total_power += channel_data[band]
                                channel_count += 1
                    
                    if channel_count > 0:
                        band_averages[band] = total_power / channel_count
                    else:
                        band_averages[band] = 0.0
                
                # Normalize by total power for relative values
                total_power = sum(band_averages.values())
                if total_power > 0:
                    for band in band_averages:
                        self.spectral_buffers[band].append(band_averages[band] / total_power)
                else:
                    for band in band_averages:
                        self.spectral_buffers[band].append(0.0)
                
                self.current_features = band_averages
                
            except Exception as e:
                self.logger.error(f"Error updating spectral features: {e}")
    
    def update_mental_state(self, mental_state_result: Dict[str, Any]):
        """
        Update mental state information
        
        Args:
            mental_state_result: Mental state classification result
        """
        with self.data_lock:
            try:
                primary_state = mental_state_result.get('primary_state', 'neutral')
                confidence = mental_state_result.get('confidence', 0.0)
                
                self.mental_state_buffer.append(primary_state)
                self.confidence_buffer.append(confidence)
                
                self.current_mental_state = primary_state
                self.current_confidence = confidence
                
                # Update state duration statistics
                current_time = time.time()
                time_delta = current_time - self.session_stats['last_update_time']
                
                if primary_state in self.session_stats['state_durations']:
                    self.session_stats['state_durations'][primary_state] += time_delta
                
                self.session_stats['last_update_time'] = current_time
                
                # Add to history for state transitions
                self.state_history.append({
                    'state': primary_state,
                    'confidence': confidence,
                    'timestamp': current_time
                })
                
            except Exception as e:
                self.logger.error(f"Error updating mental state: {e}")
    
    def update_signal_quality(self, signal_quality: Dict[str, float]):
        """
        Update signal quality information
        
        Args:
            signal_quality: Signal quality metrics per channel
        """
        with self.data_lock:
            try:
                for channel in ['TP9', 'AF7', 'AF8', 'TP10']:
                    quality_key = f"{channel}_quality"
                    if quality_key in signal_quality:
                        quality_value = signal_quality[quality_key]
                        self.quality_buffers[channel].append(quality_value)
                    else:
                        self.quality_buffers[channel].append(0.5)  # Default medium quality
                
                self.current_signal_quality = signal_quality
                
            except Exception as e:
                self.logger.error(f"Error updating signal quality: {e}")
    
    def _animate(self, frame):
        """Animation function for real-time updates"""
        try:
            with self.data_lock:
                if not self.time_buffer:
                    return
                
                # Convert time buffer to relative time (seconds from start)
                start_time = self.time_buffer[0] if self.time_buffer else time.time()
                time_array = np.array([t - start_time for t in self.time_buffer])
                
                # Update EEG signals
                for channel in ['TP9', 'AF7', 'AF8', 'TP10']:
                    if channel in self.eeg_buffers and self.eeg_buffers[channel]:
                        eeg_data = np.array(self.eeg_buffers[channel])
                        self.lines[f'eeg_{channel}'].set_data(time_array, eeg_data)
                
                # Update EEG axis limits
                if len(time_array) > 0:
                    self.axes[0, 0].set_xlim(max(0, time_array[-1] - 30), time_array[-1] + 1)
                    
                    # Calculate y-limits based on data
                    all_eeg_data = []
                    for channel in ['TP9', 'AF7', 'AF8', 'TP10']:
                        if self.eeg_buffers[channel]:
                            all_eeg_data.extend(list(self.eeg_buffers[channel])[-100:])  # Last 100 samples
                    
                    if all_eeg_data:
                        y_min, y_max = np.min(all_eeg_data), np.max(all_eeg_data)
                        y_range = max(y_max - y_min, 50)  # Minimum range of 50μV
                        y_center = (y_min + y_max) / 2
                        self.axes[0, 0].set_ylim(y_center - y_range/2, y_center + y_range/2)
                
                # Update spectral power
                for band in ['alpha', 'beta', 'theta', 'gamma', 'delta']:
                    if band in self.spectral_buffers and self.spectral_buffers[band]:
                        spectral_data = np.array(self.spectral_buffers[band])
                        self.lines[f'spectral_{band}'].set_data(time_array, spectral_data)
                
                # Update spectral axis limits
                if len(time_array) > 0:
                    self.axes[0, 1].set_xlim(max(0, time_array[-1] - 30), time_array[-1] + 1)
                    self.axes[0, 1].set_ylim(0, 1)
                
                # Update mental state timeline
                self._update_mental_state_plot(time_array)
                
                # Update confidence plot
                if self.confidence_buffer:
                    confidence_data = np.array(self.confidence_buffer)
                    self.lines['confidence'].set_data(time_array, confidence_data)
                    
                    if len(time_array) > 0:
                        self.axes[1, 1].set_xlim(max(0, time_array[-1] - 30), time_array[-1] + 1)
                
                # Update signal quality bars
                self._update_signal_quality_bars()
                
                # Update state distribution pie chart
                self._update_state_distribution()
                
                # Update status text
                self._update_status_text()
                
        except Exception as e:
            self.logger.error(f"Error in animation update: {e}")
    
    def _update_mental_state_plot(self, time_array):
        """Update mental state timeline plot"""
        try:
            ax = self.axes[1, 0]
            ax.clear()
            
            if not self.mental_state_buffer or len(time_array) == 0:
                return
            
            # Create state mapping for y-axis
            unique_states = list(set(self.mental_state_buffer))
            state_to_y = {state: i for i, state in enumerate(unique_states)}
            
            # Plot state changes as colored segments
            for i in range(len(self.mental_state_buffer)):
                if i < len(time_array):
                    state = self.mental_state_buffer[i]
                    y_pos = state_to_y[state]
                    color = self.state_colors.get(state, '#696969')
                    
                    # Plot as scatter point
                    ax.scatter(time_array[i], y_pos, c=color, s=30, alpha=0.7)
            
            # Set up axis
            ax.set_xlim(max(0, time_array[-1] - 30), time_array[-1] + 1)
            ax.set_ylim(-0.5, len(unique_states) - 0.5)
            ax.set_yticks(range(len(unique_states)))
            ax.set_yticklabels(unique_states)
            ax.set_title('Mental State Timeline', color='white', fontsize=12)
            ax.set_xlabel('Time (s)', color='white')
            ax.set_ylabel('Mental State', color='white')
            ax.set_facecolor('#34495E')
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='white')
            
        except Exception as e:
            self.logger.error(f"Error updating mental state plot: {e}")
    
    def _update_signal_quality_bars(self):
        """Update signal quality bar chart"""
        try:
            channels = ['TP9', 'AF7', 'AF8', 'TP10']
            
            for i, channel in enumerate(channels):
                if channel in self.quality_buffers and self.quality_buffers[channel]:
                    # Get recent average quality
                    recent_quality = np.mean(list(self.quality_buffers[channel])[-10:])
                    self.bars['quality'][i].set_height(recent_quality)
                    
                    # Color based on quality level
                    if recent_quality > 0.8:
                        color = '#2ECC71'  # Green
                    elif recent_quality > 0.6:
                        color = '#F39C12'  # Orange
                    else:
                        color = '#E74C3C'  # Red
                    
                    self.bars['quality'][i].set_color(color)
                else:
                    self.bars['quality'][i].set_height(0)
            
        except Exception as e:
            self.logger.error(f"Error updating signal quality bars: {e}")
    
    def _update_state_distribution(self):
        """Update state distribution pie chart"""
        try:
            ax = self.distribution_ax
            ax.clear()
            
            # Calculate state durations
            durations = self.session_stats['state_durations']
            
            # Filter out states with zero duration
            filtered_durations = {state: duration for state, duration in durations.items() 
                                if duration > 0}
            
            if not filtered_durations:
                ax.text(0.5, 0.5, 'No data yet', transform=ax.transAxes, 
                       ha='center', va='center', color='white', fontsize=14)
                return
            
            # Create pie chart
            states = list(filtered_durations.keys())
            values = list(filtered_durations.values())
            colors = [self.state_colors.get(state, '#696969') for state in states]
            
            wedges, texts, autotexts = ax.pie(values, labels=states, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            
            # Style the text
            for text in texts:
                text.set_color('white')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Session State Distribution', color='white', fontsize=12)
            
        except Exception as e:
            self.logger.error(f"Error updating state distribution: {e}")
    
    def _update_status_text(self):
        """Update status text displays"""
        try:
            # Update mental state
            if 'current_state' in self.texts:
                self.texts['current_state'].config(
                    text=self.current_mental_state.capitalize(),
                    foreground=self.state_colors.get(self.current_mental_state, 'blue')
                )
            
            # Update confidence
            if 'confidence' in self.texts:
                self.texts['confidence'].config(text=f"{self.current_confidence:.1%}")
            
            # Update signal quality
            if 'signal_quality' in self.texts:
                overall_quality = 0.0
                if self.current_signal_quality:
                    quality_values = [v for k, v in self.current_signal_quality.items() 
                                    if 'quality' in k]
                    overall_quality = np.mean(quality_values) if quality_values else 0.0
                
                if overall_quality > 0.8:
                    quality_text = "Excellent"
                    quality_color = "green"
                elif overall_quality > 0.6:
                    quality_text = "Good"
                    quality_color = "orange"
                elif overall_quality > 0.4:
                    quality_text = "Fair"
                    quality_color = "orange"
                else:
                    quality_text = "Poor"
                    quality_color = "red"
                
                self.texts['signal_quality'].config(text=quality_text, foreground=quality_color)
            
            # Update session duration
            if 'session_duration' in self.texts:
                duration = time.time() - self.session_stats['start_time']
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                self.texts['session_duration'].config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
        except Exception as e:
            self.logger.error(f"Error updating status text: {e}")
    
    def start_visualization(self):
        """Start the real-time visualization"""
        if self.root is None:
            self.initialize_gui()
        
        try:
            self.running = True
            
            # Start animation
            self.animation = animation.FuncAnimation(
                self.fig, self._animate, interval=int(self.update_interval * 1000),
                blit=False, cache_frame_data=False
            )
            
            self.logger.info("Visualization started")
            self.start_button.config(text="Stop Monitoring")
            
        except Exception as e:
            self.logger.error(f"Error starting visualization: {e}")
    
    def stop_visualization(self):
        """Stop the real-time visualization"""
        try:
            self.running = False
            
            if self.animation:
                self.animation.event_source.stop()
                self.animation = None
            
            self.logger.info("Visualization stopped")
            if hasattr(self, 'start_button') and self.start_button:
                self.start_button.config(text="Start Monitoring")
            
        except Exception as e:
            self.logger.error(f"Error stopping visualization: {e}")
    
    def toggle_monitoring(self):
        """Toggle monitoring on/off"""
        if self.running:
            self.stop_visualization()
        else:
            self.start_visualization()
    
    def reset_data(self):
        """Reset all data buffers"""
        try:
            with self.data_lock:
                # Clear all buffers
                self.time_buffer.clear()
                for buffer in self.eeg_buffers.values():
                    buffer.clear()
                for buffer in self.spectral_buffers.values():
                    buffer.clear()
                for buffer in self.quality_buffers.values():
                    buffer.clear()
                
                self.mental_state_buffer.clear()
                self.confidence_buffer.clear()
                self.state_history.clear()
                
                # Reset statistics
                self.session_stats = {
                    'start_time': time.time(),
                    'total_samples': 0,
                    'state_durations': {state: 0.0 for state in self.state_colors.keys()},
                    'last_update_time': time.time()
                }
            
            self.logger.info("Data buffers reset")
            messagebox.showinfo("Reset", "All data has been reset.")
            
        except Exception as e:
            self.logger.error(f"Error resetting data: {e}")
            messagebox.showerror("Error", f"Error resetting data: {e}")
    
    def save_session(self):
        """Save session data to file"""
        try:
            from tkinter import filedialog
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Session Data"
            )
            
            if filename:
                # Prepare session data
                session_data = {
                    'session_info': dict(self.session_stats),
                    'mental_states': list(self.mental_state_buffer),
                    'confidences': list(self.confidence_buffer),
                    'state_history': list(self.state_history),
                    'spectral_features': {band: list(buffer) 
                                        for band, buffer in self.spectral_buffers.items()},
                    'signal_quality': {channel: list(buffer) 
                                     for channel, buffer in self.quality_buffers.items()},
                    'timestamps': list(self.time_buffer)
                }
                
                # Save to file
                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2, default=str)
                
                self.logger.info(f"Session saved to {filename}")
                messagebox.showinfo("Save Complete", f"Session data saved to {filename}")
                
        except Exception as e:
            self.logger.error(f"Error saving session: {e}")
            messagebox.showerror("Save Error", f"Error saving session: {e}")
    
    def show_settings(self):
        """Show settings dialog"""
        try:
            settings_window = tk.Toplevel(self.root)
            settings_window.title("Visualization Settings")
            settings_window.geometry("400x300")
            settings_window.configure(bg='#2C3E50')
            
            # Update interval setting
            ttk.Label(settings_window, text="Update Interval (seconds):").grid(row=0, column=0, padx=10, pady=5)
            interval_var = tk.DoubleVar(value=self.update_interval)
            ttk.Entry(settings_window, textvariable=interval_var).grid(row=0, column=1, padx=10, pady=5)
            
            # Buffer size setting
            ttk.Label(settings_window, text="Buffer Size:").grid(row=1, column=0, padx=10, pady=5)
            buffer_var = tk.IntVar(value=self.buffer_size)
            ttk.Entry(settings_window, textvariable=buffer_var).grid(row=1, column=1, padx=10, pady=5)
            
            # Apply button
            def apply_settings():
                self.update_interval = interval_var.get()
                self.buffer_size = buffer_var.get()
                settings_window.destroy()
                messagebox.showinfo("Settings", "Settings applied successfully!")
            
            ttk.Button(settings_window, text="Apply", command=apply_settings).grid(row=2, column=0, columnspan=2, pady=20)
            
        except Exception as e:
            self.logger.error(f"Error showing settings: {e}")
    
    def run(self):
        """Run the visualization GUI"""
        if self.root is None:
            self.initialize_gui()
        
        try:
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error running visualization: {e}")
        finally:
            self.stop_visualization()


def main():
    """
    Test function for real-time visualizer
    """
    print("=== Real-time Mental State Visualizer Test ===")
    
    # Create visualizer
    visualizer = RealTimeVisualizer()
    
    # Start visualization
    visualizer.start_visualization()
    
    # Simulate some data
    def simulate_data():
        import threading
        import random
        
        def data_generator():
            while visualizer.running:
                # Generate mock EEG data
                eeg_data = np.random.randn(4, 10) * 50  # 4 channels, 10 samples
                visualizer.update_eeg_data(eeg_data, time.time())
                
                # Generate mock spectral features
                spectral_features = {
                    'TP9': {'alpha': random.random(), 'beta': random.random(), 'theta': random.random()},
                    'AF7': {'alpha': random.random(), 'beta': random.random(), 'theta': random.random()},
                    'AF8': {'alpha': random.random(), 'beta': random.random(), 'theta': random.random()},
                    'TP10': {'alpha': random.random(), 'beta': random.random(), 'theta': random.random()}
                }
                visualizer.update_spectral_features(spectral_features)
                
                # Generate mock mental state
                states = ['relaxed', 'focused', 'meditative', 'alert', 'neutral']
                mental_state = {
                    'primary_state': random.choice(states),
                    'confidence': random.random()
                }
                visualizer.update_mental_state(mental_state)
                
                # Generate mock signal quality
                signal_quality = {
                    'TP9_quality': random.random(),
                    'AF7_quality': random.random(),
                    'AF8_quality': random.random(),
                    'TP10_quality': random.random()
                }
                visualizer.update_signal_quality(signal_quality)
                
                time.sleep(0.1)
        
        thread = threading.Thread(target=data_generator, daemon=True)
        thread.start()
    
    # Start data simulation
    simulate_data()
    
    # Run the GUI
    visualizer.run()
    
    print("Visualizer test completed!")


if __name__ == "__main__":
    main()
