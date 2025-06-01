# Real-time Progress Bar Solutions for Streamlit

import streamlit as st
import time
import asyncio
from typing import Optional, Any
import threading
from queue import Queue

# Solution 1: Force UI Updates with Container Management
class RealTimeProgressBar:
    """Enhanced progress bar that forces real-time updates in Streamlit"""
    
    def __init__(self, container=None):
        self.container = container or st.empty()
        self.progress_bar = None
        self.status_text = None
        self.current_progress = 0
        
    def initialize(self):
        """Initialize the progress bar components"""
        with self.container.container():
            self.status_text = st.empty()
            self.progress_bar = st.progress(0)
        return self
    
    def update(self, progress: int, status: str = ""):
        """Update progress with forced UI refresh"""
        if self.progress_bar and self.status_text:
            self.status_text.text(status)
            self.progress_bar.progress(progress)
            self.current_progress = progress
            
            # Force Streamlit to update the UI
            time.sleep(0.01)  # Small delay to allow UI update
            
    def complete(self, final_message: str = "Complete!"):
        """Mark as complete and clean up"""
        if self.progress_bar and self.status_text:
            self.status_text.text(final_message)
            self.progress_bar.progress(100)
            time.sleep(1)  # Show completion for 1 second
            
    def cleanup(self):
        """Clean up the progress bar"""
        if self.container:
            self.container.empty()

# Solution 2: Thread-Safe Progress with Queue
class ThreadSafeProgress:
    """Thread-safe progress bar using queue for updates"""
    
    def __init__(self):
        self.progress_queue = Queue()
        self.is_running = False
        self.progress_container = st.empty()
        self.status_container = st.empty()
        
    def start_progress_monitor(self):
        """Start monitoring progress updates"""
        self.is_running = True
        
        # Create initial progress bar
        with self.progress_container.container():
            progress_bar = st.progress(0)
            
        with self.status_container.container():
            status_text = st.empty()
            
        # Monitor queue for updates
        while self.is_running:
            try:
                if not self.progress_queue.empty():
                    update_data = self.progress_queue.get_nowait()
                    
                    if update_data['type'] == 'progress':
                        progress_bar.progress(update_data['value'])
                    elif update_data['type'] == 'status':
                        status_text.text(update_data['message'])
                    elif update_data['type'] == 'complete':
                        self.is_running = False
                        
                time.sleep(0.1)  # Check queue every 100ms
                
            except:
                time.sleep(0.1)
                
    def update_progress(self, value: int):
        """Thread-safe progress update"""
        self.progress_queue.put({'type': 'progress', 'value': value})
        
    def update_status(self, message: str):
        """Thread-safe status update"""
        self.progress_queue.put({'type': 'status', 'message': message})
        
    def complete(self):
        """Signal completion"""
        self.progress_queue.put({'type': 'complete'})

# Solution 3: Enhanced Async Progress with Proper State Management
async def enhanced_progress_function(progress_container, status_container, steps):
    """Enhanced async function with real-time progress updates"""
    
    # Initialize progress components
    with progress_container.container():
        progress_bar = st.progress(0)
        
    with status_container.container():
        status_text = st.empty()
    
    total_steps = len(steps)
    
    for i, step in enumerate(steps):
        # Update status
        status_text.text(f"🔄 {step['message']}")
        
        # Update progress
        progress_percent = int((i / total_steps) * 100)
        progress_bar.progress(progress_percent)
        
        # Force UI update by yielding control
        await asyncio.sleep(0.01)
        
        # Simulate work
        await asyncio.sleep(step.get('duration', 0.5))
        
        # Update to show completion of this step
        status_text.text(f"✅ {step['message']} - Complete")
        await asyncio.sleep(0.1)
    
    # Final update
    progress_bar.progress(100)
    status_text.text("🎉 All steps completed!")
    
    return "Process completed successfully"

# Solution 4: WebSocket-like Real-time Updates
class StreamlitProgressManager:
    """Manager for handling multiple progress bars with real-time updates"""
    
    def __init__(self):
        self.progress_bars = {}
        self.status_texts = {}
        self.containers = {}
        
    def create_progress_bar(self, name: str, container=None):
        """Create a new progress bar"""
        if container is None:
            container = st.empty()
            
        self.containers[name] = container
        
        with container.container():
            self.status_texts[name] = st.empty()
            self.progress_bars[name] = st.progress(0)
    
    def update_progress(self, name: str, value: int, status: str = ""):
        """Update a specific progress bar"""
        if name in self.progress_bars:
            if status:
                self.status_texts[name].text(status)
            self.progress_bars[name].progress(value)
            
            # Force update
            time.sleep(0.01)
    
    def complete_progress(self, name: str, final_message: str = "Complete!"):
        """Complete a progress bar"""
        if name in self.progress_bars:
            self.status_texts[name].text(final_message)
            self.progress_bars[name].progress(100)
            time.sleep(0.5)
    
    def cleanup_progress(self, name: str):
        """Clean up a progress bar"""
        if name in self.containers:
            self.containers[name].empty()
            del self.progress_bars[name]
            del self.status_texts[name]
            del self.containers[name]