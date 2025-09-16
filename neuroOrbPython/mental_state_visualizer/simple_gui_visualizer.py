"""
Simple GUI Mental State Visualizer
Avoids complex threading issues while providing visual feedback
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import time
import logging
import threading
from typing import Dict, Any
import queue
from collections import deque

from enhanced_muse_connector import EnhancedMuseConnector
from advanced_signal_processor import AdvancedSignalProcessor
from mental_state_classifier import MentalStateClassifier


class SimpleGUIVisualizer:
    """
    Simple GUI-based mental state visualizer that avoids complex matplotlib threading
    """
    
    def __init__(self, board_type: str = "MUSE_2", mac_address: str = ""):
        """
        Initialize the simple GUI visualizer
        
        Args:
            board_type: Type of Muse board
            mac_address: MAC address of device (empty for auto-scan)
        """
        self.board_type = board_type
        self.mac_address = mac_address
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.muse_connector = None
        self.signal_processor = None
        self.mental_state_classifier = None
        
        # GUI components
        self.root = None
        self.running = False
        self.processing_thread = None
        
        # Data queue for thread-safe GUI updates
        self.data_queue = queue.Queue()
        
        # Current state tracking
        self.current_state = {
            'mental_state': 'unknown',
            'confidence': 0.0,
            'signal_quality': 0.0,
            'sample_count': 0,
            'session_duration': 0.0
        }
        
        # Session statistics
        self.session_stats = {
            'start_time': None,
            'state_counts': {},
            'total_samples': 0
        }
        
        # State history for trend analysis
        self.state_history = deque(maxlen=20)
        
        # Mental state colors and emojis
        self.state_info = {
            'relaxed': {'color': '#87CEEB', 'emoji': '😌', 'desc': 'Relaxed & Calm'},
            'focused': {'color': '#FF6347', 'emoji': '🎯', 'desc': 'Focused & Alert'},
            'meditative': {'color': '#DDA0DD', 'emoji': '🧘', 'desc': 'Deep Meditation'},
            'alert': {'color': '#FFD700', 'emoji': '⚡', 'desc': 'High Alertness'},
            'neutral': {'color': '#D3D3D3', 'emoji': '😐', 'desc': 'Neutral State'},
            'stressed': {'color': '#DC143C', 'emoji': '😰', 'desc': 'Stress Detected'},
            'unknown': {'color': '#696969', 'emoji': '❓', 'desc': 'Processing...'}
        }
    
    def initialize_components(self) -> bool:
        """Initialize all processing components"""
        try:
            self.logger.info("Initializing components...")
            
            # Initialize Muse connector
            self.muse_connector = EnhancedMuseConnector(
                board_type=self.board_type,
                mac_address=self.mac_address
            )
            
            # Initialize signal processor
            self.signal_processor = AdvancedSignalProcessor(sampling_rate=256)
            
            # Initialize mental state classifier
            self.mental_state_classifier = MentalStateClassifier()
            
            self.logger.info("Components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    def create_gui(self):
        """Create the simple GUI interface"""
        self.root = tk.Tk()
        self.root.title("Mental State Monitor - Simple GUI")
        self.root.geometry("600x700")
        self.root.configure(bg='#2C3E50')
        
        # Configure style for dark theme
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2C3E50', foreground='white')
        style.configure('Status.TLabel', font=('Arial', 14), background='#2C3E50', foreground='white')
        style.configure('Data.TLabel', font=('Arial', 12), background='#2C3E50', foreground='white')
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2C3E50', pady=10)
        title_frame.pack(fill='x')
        
        ttk.Label(title_frame, text="🧠 Mental State Monitor", style='Title.TLabel').pack()
        ttk.Label(title_frame, text="Real-time EEG Analysis with Muse", style='Data.TLabel').pack()
        
        # Current state display (large)
        self.state_frame = tk.Frame(self.root, bg='#34495E', pady=20, padx=20)
        self.state_frame.pack(fill='x', padx=20, pady=10)
        
        self.state_emoji_label = tk.Label(self.state_frame, text="❓", font=('Arial', 48), 
                                         bg='#34495E', fg='white')
        self.state_emoji_label.pack()
        
        self.state_name_label = tk.Label(self.state_frame, text="Processing...", 
                                        font=('Arial', 20, 'bold'), bg='#34495E', fg='white')
        self.state_name_label.pack()
        
        self.confidence_label = tk.Label(self.state_frame, text="Confidence: 0%", 
                                        font=('Arial', 14), bg='#34495E', fg='#BDC3C7')
        self.confidence_label.pack()
        
        # Status panel
        status_frame = tk.LabelFrame(self.root, text="System Status", font=('Arial', 12, 'bold'),
                                   bg='#2C3E50', fg='white', pady=10, padx=10)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        # Status grid
        status_grid = tk.Frame(status_frame, bg='#2C3E50')
        status_grid.pack(fill='x')
        
        # Connection status
        tk.Label(status_grid, text="Connection:", font=('Arial', 10, 'bold'), 
                bg='#2C3E50', fg='white').grid(row=0, column=0, sticky='w', padx=5)
        self.connection_label = tk.Label(status_grid, text="Disconnected", font=('Arial', 10),
                                       bg='#2C3E50', fg='red')
        self.connection_label.grid(row=0, column=1, sticky='w', padx=5)
        
        # Signal quality
        tk.Label(status_grid, text="Signal Quality:", font=('Arial', 10, 'bold'),
                bg='#2C3E50', fg='white').grid(row=0, column=2, sticky='w', padx=15)
        self.quality_label = tk.Label(status_grid, text="0%", font=('Arial', 10),
                                    bg='#2C3E50', fg='gray')
        self.quality_label.grid(row=0, column=3, sticky='w', padx=5)
        
        # Sample count
        tk.Label(status_grid, text="Samples:", font=('Arial', 10, 'bold'),
                bg='#2C3E50', fg='white').grid(row=1, column=0, sticky='w', padx=5)
        self.samples_label = tk.Label(status_grid, text="0", font=('Arial', 10),
                                    bg='#2C3E50', fg='white')
        self.samples_label.grid(row=1, column=1, sticky='w', padx=5)
        
        # Session duration
        tk.Label(status_grid, text="Duration:", font=('Arial', 10, 'bold'),
                bg='#2C3E50', fg='white').grid(row=1, column=2, sticky='w', padx=15)
        self.duration_label = tk.Label(status_grid, text="00:00", font=('Arial', 10),
                                     bg='#2C3E50', fg='white')
        self.duration_label.grid(row=1, column=3, sticky='w', padx=5)
        
        # Recent history
        history_frame = tk.LabelFrame(self.root, text="Recent Mental States", 
                                    font=('Arial', 12, 'bold'), bg='#2C3E50', fg='white',
                                    pady=10, padx=10)
        history_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.history_text = scrolledtext.ScrolledText(history_frame, height=8, width=60,
                                                     bg='#34495E', fg='white', font=('Courier', 10),
                                                     insertbackground='white')
        self.history_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Controls
        control_frame = tk.Frame(self.root, bg='#2C3E50', pady=10)
        control_frame.pack(fill='x', padx=20)
        
        self.start_button = tk.Button(control_frame, text="Start Monitoring", 
                                     command=self.toggle_monitoring, font=('Arial', 12, 'bold'),
                                     bg='#27AE60', fg='white', pady=5, padx=20)
        self.start_button.pack(side='left', padx=5)
        
        tk.Button(control_frame, text="Clear History", command=self.clear_history,
                 font=('Arial', 12), bg='#E74C3C', fg='white', pady=5, padx=20).pack(side='left', padx=5)
        
        tk.Button(control_frame, text="Save Session", command=self.save_session,
                 font=('Arial', 12), bg='#3498DB', fg='white', pady=5, padx=20).pack(side='left', padx=5)
        
        # Add some initial text
        self.add_history_message("🔧 Mental State Monitor initialized")
        self.add_history_message("🔌 Connect your Muse headset and press 'Start Monitoring'")
        
        # Set up periodic GUI updates
        self.update_gui()
    
    def add_history_message(self, message: str):
        """Add a message to the history display"""
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.history_text.insert('end', full_message)
        self.history_text.see('end')
        
        # Limit history to last 100 lines
        lines = self.history_text.get('1.0', 'end').count('\n')
        if lines > 100:
            self.history_text.delete('1.0', '2.0')
    
    def update_gui(self):
        """Update GUI with latest data"""
        try:
            # Process any queued data updates
            while not self.data_queue.empty():
                try:
                    data = self.data_queue.get_nowait()
                    self._process_gui_update(data)
                except queue.Empty:
                    break
            
            # Schedule next update
            if self.root:
                self.root.after(100, self.update_gui)  # Update every 100ms
                
        except Exception as e:
            self.logger.error(f"Error updating GUI: {e}")
    
    def _process_gui_update(self, data: Dict[str, Any]):
        """Process a single GUI update"""
        update_type = data.get('type')
        
        if update_type == 'mental_state':
            self._update_mental_state_display(data)
        elif update_type == 'connection_status':
            self._update_connection_status(data)
        elif update_type == 'session_stats':
            self._update_session_stats(data)
    
    def _update_mental_state_display(self, data: Dict[str, Any]):
        """Update the mental state display"""
        state = data.get('state', 'unknown')
        confidence = data.get('confidence', 0.0)
        
        # Update current state tracking
        self.current_state['mental_state'] = state
        self.current_state['confidence'] = confidence
        
        # Get state info
        state_info = self.state_info.get(state, self.state_info['unknown'])
        
        # Update main display
        self.state_emoji_label.config(text=state_info['emoji'])
        self.state_name_label.config(text=state_info['desc'], fg=state_info['color'])
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
        
        # Update background color based on confidence
        if confidence > 0.8:
            bg_color = '#27AE60'  # High confidence - green
        elif confidence > 0.5:
            bg_color = '#F39C12'  # Medium confidence - orange
        else:
            bg_color = '#34495E'  # Low confidence - gray
        
        self.state_frame.config(bg=bg_color)
        self.state_emoji_label.config(bg=bg_color)
        self.state_name_label.config(bg=bg_color)
        self.confidence_label.config(bg=bg_color)
        
        # Add to history
        self.state_history.append(state)
        self.add_history_message(f"{state_info['emoji']} {state_info['desc']} (Confidence: {confidence:.1%})")
        
        # Update session stats
        if state in self.session_stats['state_counts']:
            self.session_stats['state_counts'][state] += 1
        else:
            self.session_stats['state_counts'][state] = 1
    
    def _update_connection_status(self, data: Dict[str, Any]):
        """Update connection status"""
        connected = data.get('connected', False)
        signal_quality = data.get('signal_quality', 0.0)
        
        if connected:
            self.connection_label.config(text="Connected ✓", fg='green')
        else:
            self.connection_label.config(text="Disconnected ✗", fg='red')
        
        # Update signal quality
        quality_pct = signal_quality * 100
        if quality_pct > 70:
            quality_color = 'green'
            quality_text = f"{quality_pct:.0f}% ✓"
        elif quality_pct > 40:
            quality_color = 'orange'
            quality_text = f"{quality_pct:.0f}% ⚠"
        else:
            quality_color = 'red'
            quality_text = f"{quality_pct:.0f}% ✗"
        
        self.quality_label.config(text=quality_text, fg=quality_color)
    
    def _update_session_stats(self, data: Dict[str, Any]):
        """Update session statistics"""
        sample_count = data.get('sample_count', 0)
        duration = data.get('duration', 0)
        
        self.current_state['sample_count'] = sample_count
        self.current_state['session_duration'] = duration
        
        # Update labels
        self.samples_label.config(text=str(sample_count))
        
        # Format duration
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        self.duration_label.config(text=f"{minutes:02d}:{seconds:02d}")
    
    def processing_loop(self):
        """Main processing loop running in background thread"""
        self.logger.info("Starting processing loop...")
        
        start_time = time.time()
        sample_count = 0
        
        while self.running:
            try:
                # Get EEG data
                eeg_data = self.muse_connector.get_eeg_data(num_samples=64)
                
                if eeg_data is not None and eeg_data.size > 0:
                    # Process the data
                    processing_results = self.signal_processor.process_eeg_comprehensive(eeg_data)
                    
                    if processing_results:
                        # Extract features
                        spectral_features = processing_results.get('spectral_features', {})
                        temporal_features = processing_results.get('temporal_features', {})
                        cross_features = processing_results.get('cross_channel_features', {})
                        
                        # Classify mental state
                        mental_state_result = self.mental_state_classifier.predict_mental_state(
                            spectral_features, temporal_features, cross_features
                        )
                        
                        # Get signal quality
                        signal_quality = self.muse_connector.get_signal_quality_report()
                        
                        # Queue GUI updates
                        self.data_queue.put({
                            'type': 'mental_state',
                            'state': mental_state_result.get('state', 'unknown'),
                            'confidence': mental_state_result.get('confidence', 0.0)
                        })
                        
                        self.data_queue.put({
                            'type': 'connection_status',
                            'connected': True,
                            'signal_quality': signal_quality.get('overall_quality', 0.0)
                        })
                        
                        self.data_queue.put({
                            'type': 'session_stats',
                            'sample_count': sample_count,
                            'duration': time.time() - start_time
                        })
                        
                        sample_count += 1
                
                time.sleep(0.1)  # Control loop timing
                
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                time.sleep(0.5)
        
        self.logger.info("Processing loop stopped")
    
    def toggle_monitoring(self):
        """Toggle monitoring on/off"""
        if not self.running:
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Start monitoring"""
        try:
            self.add_history_message("🔧 Initializing components...")
            
            # Initialize components
            if not self.initialize_components():
                self.add_history_message("❌ Failed to initialize components")
                return
            
            self.add_history_message("🔌 Connecting to Muse headset...")
            
            # Connect to Muse
            if not self.muse_connector.connect():
                self.add_history_message("❌ Failed to connect to Muse headset")
                return
            
            if not self.muse_connector.start_streaming():
                self.add_history_message("❌ Failed to start EEG streaming")
                return
            
            # Update signal processor sampling rate
            actual_sampling_rate = self.muse_connector.sampling_rate
            if actual_sampling_rate:
                self.signal_processor.sampling_rate = actual_sampling_rate
            
            self.add_history_message(f"✅ Connected! Sampling at {actual_sampling_rate} Hz")
            
            # Start processing
            self.running = True
            self.session_stats['start_time'] = time.time()
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Update button
            self.start_button.config(text="Stop Monitoring", bg='#E74C3C')
            
            self.add_history_message("🧠 Mental state monitoring started!")
            
        except Exception as e:
            self.add_history_message(f"❌ Error: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        try:
            self.running = False
            
            if self.muse_connector:
                self.muse_connector.disconnect()
            
            # Update GUI
            self.start_button.config(text="Start Monitoring", bg='#27AE60')
            
            # Update connection status
            self.data_queue.put({
                'type': 'connection_status',
                'connected': False,
                'signal_quality': 0.0
            })
            
            self.add_history_message("🛑 Monitoring stopped")
            
            # Show session summary
            if self.session_stats['state_counts']:
                self.add_history_message("📊 Session Summary:")
                for state, count in self.session_stats['state_counts'].items():
                    emoji = self.state_info.get(state, {}).get('emoji', '❓')
                    self.add_history_message(f"   {emoji} {state.title()}: {count} detections")
            
        except Exception as e:
            self.add_history_message(f"❌ Error stopping: {e}")
    
    def clear_history(self):
        """Clear the history display"""
        self.history_text.delete('1.0', 'end')
        self.add_history_message("🧹 History cleared")
    
    def save_session(self):
        """Save session data"""
        try:
            import json
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Session Data"
            )
            
            if filename:
                session_data = {
                    'session_stats': self.session_stats,
                    'current_state': self.current_state,
                    'state_history': list(self.state_history)
                }
                
                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2, default=str)
                
                self.add_history_message(f"💾 Session saved to {filename}")
        
        except Exception as e:
            self.add_history_message(f"❌ Save error: {e}")
    
    def run(self):
        """Run the GUI application"""
        try:
            self.create_gui()
            self.logger.info("Starting Simple GUI Mental State Monitor")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"GUI error: {e}")
        finally:
            if self.running:
                self.stop_monitoring()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple GUI Mental State Monitor")
    parser.add_argument("--board-type", default="MUSE_2", 
                       choices=["MUSE_2", "MUSE_S", "MUSE_2016"],
                       help="Type of Muse board")
    parser.add_argument("--mac-address", default="", 
                       help="MAC address of Muse device")
    
    args = parser.parse_args()
    
    # Create and run visualizer
    app = SimpleGUIVisualizer(
        board_type=args.board_type,
        mac_address=args.mac_address
    )
    
    app.run()


if __name__ == "__main__":
    main()
