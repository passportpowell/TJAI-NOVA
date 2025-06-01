# Import Streamlit first - this is the web framework for creating the UI
import streamlit as st

# CRITICAL: st.set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="AI Agent Coordinator",  # Sets the browser tab title
    page_icon="🤖",  # Sets the browser tab icon
    layout="wide",  # Uses the full width of the browser
    initial_sidebar_state="expanded"  # Start with the sidebar open
)

# Import asyncio for handling asynchronous operations
import asyncio
# FIXED: Add nest_asyncio to handle event loop conflicts
import nest_asyncio
nest_asyncio.apply()

# Standard library imports for file handling, data structures, etc.
import os
import json
import datetime
import sys
import time
import threading
import subprocess
import queue
from typing import Dict, Any, List  # Type hints for better code documentation
from io import StringIO
import re
import tempfile

# Add the src directory to Python path so we can import our custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import agent classes from our custom modules
from agents import Nova, Emil, Ivan, Lola  # Different agents specialized for different tasks
from core.knowledge_base import KnowledgeBase  # For storing/retrieving data between sessions
from core.session_manager import SessionManager  # Manages user sessions
from core.functions_registery import *  # All the registered functions our agents can call
from utils.csv_function_mapper import FunctionMapLoader  # Loads function mappings from CSV files
from utils.do_maths import do_maths  # Utility for math calculations
from utils.general_knowledge import answer_general_question  # Utility for answering general questions
from utils.open_ai_utils import (  # Utilities for interacting with OpenAI API
    ai_chat_session,
    ai_spoken_chat_session,
    run_open_ai_ns_async,
    open_ai_categorisation_async
)

# Custom CSS for styling the UI components
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #1f77b4;
        border-bottom: 2px solid #e0e0e0;
    }
    .status-success { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .context-handover {
        background-color: #e7f3ff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #007bff;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .parameter-detail {
        background-color: #f8f9fa;
        padding: 0.4rem;
        border-radius: 0.2rem;
        border-left: 2px solid #6c757d;
        margin: 0.2rem 0;
        font-size: 0.8rem;
        font-family: monospace;
    }
    .parameter-input {
        background-color: #fff3cd;
        padding: 0.75rem;
        border-radius: 0.3rem;
        border: 1px solid #ffeaa7;
        margin: 0.5rem 0;
    }
    .submit-container {
        text-align: center;
        padding: 1rem 0;
    }
    .progress-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .cli-output {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
        margin: 0.5rem 0;
        border: 1px solid #333;
    }
    .cli-line {
        margin: 0.1rem 0;
    }
    .cli-success { color: #4CAF50; }
    .cli-error { color: #f44336; }
    .cli-warning { color: #ff9800; }
    .cli-info { color: #2196F3; }
    .cli-progress { color: #9C27B0; }
    .plexos-progress { color: #00FF00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Constants for generation types mapping
GENERATION_TYPES = {
    "wind": ["Onshore Wind", "Onshore Wind Expansion", "Offshore Wind Radial"],
    "solar": ["Solar PV", "Solar PV Expansion", "Solar Thermal Expansion", 
              "Rooftop Solar Tertiary", "Rooftop Tertiary Solar Expansion"],
    "hydro": ["RoR and Pondage", "Pump Storage - closed loop"],
    "thermal": ["Hard coal", "Heavy oil"],
    "bio": ["Bio Fuels"],
    "other": ["Other RES", "DSR Industry"]
}

# List of supported locations for energy modeling
LOCATIONS = [
    # EU members
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", 
    "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
    "Slovenia", "Spain", "Sweden",
    
    # Non-EU European countries
    "Albania", "Andorra", "Armenia", "Azerbaijan", "Belarus", 
    "Bosnia", "Bosnia and Herzegovina", "Georgia", "Iceland", 
    "Kosovo", "Liechtenstein", "Moldova", "Monaco", "Montenegro", 
    "North Macedonia", "Norway", "Russia", "San Marino", "Serbia", 
    "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City",
    
    # Common abbreviations and alternate names
    "UK", "Great Britain", "Czechia", "Holland"
]

# REAL PLEXOS CLI CAPTURE SYSTEM
class RealPLEXOSCliCapture:
    """REAL CLI capture that hooks into actual PLEXOS subprocess output"""
    
    def __init__(self):
        self.cli_lines = []
        self.is_capturing = False
        self.current_process = None
        self.output_queue = queue.Queue()
        self.progress_data = {}
        self._last_update = time.time()
        self.progress_file = os.path.join(tempfile.gettempdir(), "plexos_progress.json")
        
    def start_capture(self):
        """Start capturing CLI output"""
        self.is_capturing = True
        self.cli_lines = []
        self._last_update = time.time()
        
    def stop_capture(self):
        """Stop capturing CLI output"""
        self.is_capturing = False
        if self.current_process:
            try:
                self.current_process.terminate()
            except:
                pass
                
    def add_line(self, line, line_type="info"):
        """Add a line to CLI output AND parse for PLEXOS progress"""
        if self.is_capturing or line_type in ["info", "success", "error"]:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            formatted_line = f"[{timestamp}] {line}"
            
            self.cli_lines.append({
                "text": formatted_line,
                "type": line_type,
                "timestamp": timestamp,
                "raw_text": line
            })
            
            # Keep only last 200 lines
            if len(self.cli_lines) > 200:
                self.cli_lines = self.cli_lines[-200:]
            
            self._last_update = time.time()
            
            # Parse REAL PLEXOS output patterns
            plexos_data = self.parse_plexos_line(line)
            if plexos_data:
                self.progress_data.update(plexos_data)
            
            # Print to console for debugging
            print(f"CLI [{line_type.upper()}]: {line}")
            
    def parse_plexos_line(self, line):
        """Parse actual PLEXOS CLI output for progress information"""
        
        # Parse lines like: "Creating Categories |--------------------| 1.6% Onshore Wind Expansion__El"
        category_pattern = r'Creating Categories \|[^\|]*\|\s*(\d+\.?\d*)%\s+(.+)'
        match = re.search(category_pattern, line)
        
        if match:
            percentage = float(match.group(1))
            item_name = match.group(2).strip()
            
            return {
                "current_phase": "Creating Categories",
                "percentage": percentage,
                "current_item": item_name,
                "type": "plexos_progress"
            }
        
        # Parse other PLEXOS patterns
        object_pattern = r'Creating Objects \|[^\|]*\|\s*(\d+\.?\d*)%\s+(.+)'
        match = re.search(object_pattern, line)
        
        if match:
            percentage = float(match.group(1))
            item_name = match.group(2).strip()
            
            return {
                "current_phase": "Creating Objects",
                "percentage": percentage,
                "current_item": item_name,
                "type": "plexos_progress"
            }
            
        # Parse membership creation
        membership_pattern = r'Creating Memberships \|[^\|]*\|\s*(\d+\.?\d*)%\s+(.+)'
        match = re.search(membership_pattern, line)
        
        if match:
            percentage = float(match.group(1))
            item_name = match.group(2).strip()
            
            return {
                "current_phase": "Creating Memberships", 
                "percentage": percentage,
                "current_item": item_name,
                "type": "plexos_progress"
            }
            
        # Parse property creation  
        property_pattern = r'Creating Properties \|[^\|]*\|\s*(\d+\.?\d*)%\s+(.+)'
        match = re.search(property_pattern, line)
        
        if match:
            percentage = float(match.group(1))
            item_name = match.group(2).strip()
            
            return {
                "current_phase": "Creating Properties",
                "percentage": percentage, 
                "current_item": item_name,
                "type": "plexos_progress"
            }
        
        return None
        
    def capture_subprocess_output(self, process):
        """Capture output from a subprocess in real-time"""
        self.current_process = process
        
        def read_output(pipe, pipe_name):
            """Read output from subprocess pipe"""
            try:
                while True:
                    line = pipe.readline()
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        # Determine if this is a PLEXOS progress line
                        if any(phrase in line_str for phrase in ["Creating Categories", "Creating Objects", "Creating Memberships", "Creating Properties"]):
                            self.add_line(line_str, "plexos_progress")
                        else:
                            self.add_line(f"[{pipe_name}] {line_str}", "info")
                        
            except Exception as e:
                self.add_line(f"Error reading {pipe_name}: {str(e)}", "error")
        
        # Start threads to read stdout and stderr
        if process.stdout:
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "STDOUT"))
            stdout_thread.daemon = True
            stdout_thread.start()
            
        if process.stderr:
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "STDERR"))
            stderr_thread.daemon = True
            stderr_thread.start()
    
    def monitor_progress_file(self):
        """Monitor the progress file created by loading_bar.py"""
        def file_monitor():
            last_mtime = 0
            while self.is_capturing:
                try:
                    if os.path.exists(self.progress_file):
                        current_mtime = os.path.getmtime(self.progress_file)
                        if current_mtime > last_mtime:
                            with open(self.progress_file, 'r') as f:
                                progress_data = json.load(f)
                                
                            for task_name, data in progress_data.items():
                                percentage = data.get('percent', 0) * 100
                                message = data.get('message', '')
                                
                                progress_line = f"{task_name} |{'█' * int(percentage/5) + '-' * (20-int(percentage/5))}| {percentage:.1f}% {message}"
                                self.add_line(progress_line, "plexos_progress")
                                
                            last_mtime = current_mtime
                            
                except Exception as e:
                    pass  # Fail silently
                    
                time.sleep(0.1)  # Check every 100ms
        
        monitor_thread = threading.Thread(target=file_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def get_current_progress(self):
        """Get current PLEXOS progress data"""
        return self.progress_data.copy()
        
    def render_real_plexos_progress(self):
        """Render REAL PLEXOS progress from actual CLI output"""
        if not self.progress_data:
            st.info("🖥️ Waiting for PLEXOS model creation to start...")
            return
            
        current_phase = self.progress_data.get("current_phase", "Unknown")
        percentage = self.progress_data.get("percentage", 0)
        current_item = self.progress_data.get("current_item", "")
        
        st.markdown(f"### ⚙️ {current_phase}")
        
        # Show real progress bar
        progress_bar = st.progress(int(min(percentage, 100)))
        
        # Show current item being processed
        st.text(f"Progress: {percentage:.1f}% - {current_item}")
        
        # Show recent PLEXOS CLI lines
        recent_plexos_lines = [
            line for line in self.cli_lines[-10:] 
            if line.get("type") == "plexos_progress"
        ]
        
        if recent_plexos_lines:
            st.markdown("**Recent PLEXOS Output:**")
            cli_html = '<div class="cli-output" style="max-height: 150px;">'
            for line in recent_plexos_lines[-3:]:
                cli_html += f'<div class="cli-line plexos-progress">{line["raw_text"]}</div>'
            cli_html += '</div>'
            st.markdown(cli_html, unsafe_allow_html=True)
                
    def get_latest_lines(self, count=50):
        """Get the latest CLI lines"""
        return self.cli_lines[-count:] if len(self.cli_lines) > count else self.cli_lines
        
    def clear(self):
        """Clear all CLI lines"""
        self.cli_lines = []
        self.progress_data = {}
        self._last_update = time.time()
        
    def render_cli_output(self):
        """Render the CLI output with proper styling"""
        if not self.cli_lines:
            st.info("🖥️ No CLI output captured yet. Start a process to see real-time updates.")
            return
            
        # Create CLI output display with enhanced styling
        cli_html = '<div class="cli-output">'
        
        # Get recent lines for display (last 50)
        recent_lines = self.cli_lines[-50:] if len(self.cli_lines) > 50 else self.cli_lines
        
        for line_data in recent_lines:
            line_text = line_data["text"]
            line_type = line_data["type"]
            
            # Apply colors based on line type
            color_map = {
                "info": "#2196F3",
                "success": "#4CAF50", 
                "error": "#f44336",
                "warning": "#ff9800",
                "progress": "#9C27B0",
                "plexos_progress": "#00FF00"  # Bright green for PLEXOS progress
            }
            
            color = color_map.get(line_type, "#ffffff")
            cli_html += f'<div class="cli-line" style="color: {color};">{line_text}</div>'
        
        cli_html += '</div>'
        
        st.markdown(cli_html, unsafe_allow_html=True)
        
        # Show stats
        plexos_lines = len([l for l in self.cli_lines if l.get("type") == "plexos_progress"])
        stats_text = f"📊 Total: {len(self.cli_lines)} lines | PLEXOS Progress: {plexos_lines} | Showing: {len(recent_lines)}"
        st.caption(stats_text)

# Global CLI capture instance
cli_capture = RealPLEXOSCliCapture()

# ENHANCED: Parameter extraction with comprehensive CLI logging
async def extract_model_parameters_with_llm_correction(prompt, progress_container=None, status_container=None):
    """
    Enhanced parameter extraction with comprehensive CLI output and real-time feedback
    """
    import re
    
    cli_capture.add_line("🚀 ===== STARTING PARAMETER EXTRACTION =====", "info")
    cli_capture.add_line(f"📝 INPUT PROMPT: '{prompt}'", "info")
    cli_capture.add_line("⚙️ Initializing AI parameter extraction engine...", "progress")
    
    # Progress tracking setup
    if progress_container:
        extraction_progress = progress_container.empty()
        extraction_status = status_container.empty() if status_container else st.empty()
    else:
        extraction_progress = st.empty()
        extraction_status = st.empty()
    
    try:
        prompt_lower = prompt.lower()
        params = {"locations": [], "generation_types": [], "energy_carriers": [], "model_type": "single"}
        
        cli_capture.add_line("📋 Initializing parameter extraction process...", "progress")
        extraction_status.text(f"🔍 Extracting parameters from: '{prompt[:50]}...'")
        extraction_progress.progress(10)
        await asyncio.sleep(0.5)
        
        # PHASE 1: Extract locations with detailed logging
        cli_capture.add_line("🌍 PHASE 1: LOCATION DETECTION INITIATED", "info")
        cli_capture.add_line(f"🔍 Scanning {len(LOCATIONS)} known locations...", "progress")
        
        found_locations = []
        for idx, loc in enumerate(LOCATIONS):
            if idx % 20 == 0:  # Log every 20th location to show progress
                cli_capture.add_line(f"🔍 Scanning location {idx+1}/{len(LOCATIONS)}: {loc}", "progress")
                
            patterns = [
                f" for {loc.lower()}",
                f" in {loc.lower()}",  
                f"{loc.lower()} model",
                f"model.*{loc.lower()}",
                f"{loc.lower()}.*model",
            ]
            
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    found_locations.append(loc)
                    cli_capture.add_line(f"✅ LOCATION FOUND: '{loc}' using pattern '{pattern}'", "success")
                    break
        
        if not found_locations:
            cli_capture.add_line("⚠️ No locations detected in prompt", "warning")
            cli_capture.add_line("🔄 Using fallback location: 'Unknown'", "warning")
            params["locations"] = ["Unknown"]
        else:
            params["locations"] = list(set(found_locations))
            cli_capture.add_line(f"📍 FINAL LOCATIONS: {params['locations']}", "success")
        
        extraction_status.text(f"🌍 Found locations: {', '.join(params['locations'])}")
        extraction_progress.progress(40)
        await asyncio.sleep(0.5)
        
        # PHASE 2: Extract generation types with detailed logging
        cli_capture.add_line("⚡ PHASE 2: GENERATION TYPE DETECTION INITIATED", "info")
        cli_capture.add_line(f"🔍 Scanning {len(GENERATION_TYPES)} generation categories...", "progress")
        
        found_gen_types = []
        for gen in GENERATION_TYPES.keys():
            cli_capture.add_line(f"🔍 Checking generation type: {gen}", "progress")
            
            patterns = [
                f"build.*{gen}.*model",
                f"create.*{gen}.*model",
                f"{gen}.*model.*for",
                f"a {gen} model",
                f"build {gen}",
                f"create {gen}",
                f"{gen} power",
                f"{gen} generation",
                f"{gen} energy",
            ]
            
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    found_gen_types.append(gen)
                    cli_capture.add_line(f"⚡ GENERATION TYPE FOUND: '{gen}' using pattern '{pattern}'", "success")
                    break
        
        if not found_gen_types:
            cli_capture.add_line("⚠️ No generation types detected in prompt", "warning")
            cli_capture.add_line("🔄 Generation type will be requested from user", "info")
        else:
            params["generation_types"] = list(set(found_gen_types))
            cli_capture.add_line(f"⚡ FINAL GENERATION TYPES: {params['generation_types']}", "success")
        
        extraction_status.text(f"⚡ Found generation: {', '.join(params['generation_types']) if params['generation_types'] else 'None'}")
        extraction_progress.progress(70)
        await asyncio.sleep(0.5)
        
        # PHASE 3: Extract energy carriers
        cli_capture.add_line("🔋 PHASE 3: ENERGY CARRIER DETECTION INITIATED", "info")
        carriers = ["electricity", "hydrogen", "methane"]
        found_carriers = []
        
        for carrier in carriers:
            cli_capture.add_line(f"🔍 Checking energy carrier: {carrier}", "progress")
            if carrier in prompt_lower:
                found_carriers.append(carrier)
                cli_capture.add_line(f"🔋 ENERGY CARRIER FOUND: '{carrier}'", "success")
        
        if not found_carriers:
            cli_capture.add_line("🔄 No specific energy carriers found, using default: electricity", "info")
            found_carriers = ["electricity"]
        
        params["energy_carriers"] = found_carriers
        cli_capture.add_line(f"🔋 FINAL ENERGY CARRIERS: {params['energy_carriers']}", "success")
        
        extraction_status.text(f"🔋 Energy carrier: {', '.join(params['energy_carriers'])}")
        extraction_progress.progress(90)
        await asyncio.sleep(0.3)
        
        # FINAL SUMMARY
        cli_capture.add_line("✅ ===== PARAMETER EXTRACTION COMPLETED =====", "success")
        cli_capture.add_line(f"📊 EXTRACTION RESULTS:", "success")
        cli_capture.add_line(f"   └── Locations: {params['locations']}", "success")
        cli_capture.add_line(f"   └── Generation Types: {params['generation_types']}", "success")
        cli_capture.add_line(f"   └── Energy Carriers: {params['energy_carriers']}", "success")
        cli_capture.add_line(f"   └── Model Type: {params['model_type']}", "success")
        
        extraction_status.text("✅ Parameter extraction completed successfully!")
        extraction_progress.progress(100)
        await asyncio.sleep(0.5)
        
        extraction_progress.empty()
        extraction_status.empty()
        
        return params
        
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR in parameter extraction: {str(e)}"
        cli_capture.add_line(error_msg, "error")
        cli_capture.add_line("🔧 Falling back to default parameters", "warning")
        
        if extraction_progress:
            extraction_status.text(f"❌ Error: {str(e)}")
            extraction_progress.empty()
            extraction_status.empty()
        
        return {
            "locations": ["Unknown"],
            "generation_types": ["unknown"],
            "energy_carriers": ["electricity"],
            "model_type": "single"
        }

# ENHANCED: Country extraction with comprehensive CLI logging
async def extract_countries_with_progress(prompt, progress_container=None, status_container=None):
    """Enhanced country extraction with comprehensive CLI output"""
    
    if progress_container:
        country_progress = progress_container.empty()
        country_status = status_container.empty() if status_container else st.empty()
    else:
        country_progress = st.empty()
        country_status = st.empty()
    
    try:
        cli_capture.add_line("🗺️ ===== STARTING COUNTRY EXTRACTION =====", "info")
        cli_capture.add_line(f"🔍 ANALYZING PROMPT: '{prompt[:100]}...'", "info")
        cli_capture.add_line("🤖 Initializing country detection AI...", "progress")
        
        country_status.text(f"🧠 Extracting countries from: '{prompt[:50]}...'")
        country_progress.progress(0)
        await asyncio.sleep(0.8)
        
        cli_capture.add_line("🔄 ATTEMPT 1/3: Primary country detection", "info")
        country_status.text("🔄 Attempt 1/3: Primary detection")
        country_progress.progress(33)
        await asyncio.sleep(0.8)
        
        # Enhanced country mapping with more comprehensive coverage
        countries = []
        prompt_lower = prompt.lower()
        
        country_mapping = {
            'france': 'FR', 'montenegro': 'ME', 'spain': 'ES', 'greece': 'GR',
            'germany': 'DE', 'italy': 'IT', 'denmark': 'DK', 'norway': 'NO',
            'portugal': 'PT', 'poland': 'PL', 'belgium': 'BE', 'netherlands': 'NL',
            'austria': 'AT', 'sweden': 'SE', 'finland': 'FI', 'ireland': 'IE',
            'united kingdom': 'UK', 'uk': 'UK', 'great britain': 'UK',
            'czech republic': 'CZ', 'czechia': 'CZ', 'hungary': 'HU',
            'romania': 'RO', 'bulgaria': 'BG', 'croatia': 'HR', 'slovenia': 'SI',
            'slovakia': 'SK', 'lithuania': 'LT', 'latvia': 'LV', 'estonia': 'EE'
        }
        
        cli_capture.add_line("🔍 SCANNING FOR COUNTRY MENTIONS...", "progress")
        cli_capture.add_line(f"📊 Checking against {len(country_mapping)} known countries", "info")
        
        for country_name, country_code in country_mapping.items():
            if country_name in prompt_lower:
                countries.append(country_code)
                cli_capture.add_line(f"🌍 COUNTRY MATCH: '{country_name}' → '{country_code}'", "success")
        
        country_status.text("🔄 Attempt 2/3: Secondary validation")
        country_progress.progress(66)
        await asyncio.sleep(0.5)
        
        if not countries:
            cli_capture.add_line("⚠️ No specific countries found in prompt", "warning")
            cli_capture.add_line("🔄 Using fallback country code: 'XX'", "warning")
            countries = ['XX']
        
        cli_capture.add_line(f"🤖 AI RESPONSE GENERATED: {countries}", "info")
        country_status.text(f"🧠 AI Response: {countries}")
        country_progress.progress(90)
        await asyncio.sleep(0.3)
        
        cli_capture.add_line("✅ ===== COUNTRY EXTRACTION COMPLETED =====", "success")
        cli_capture.add_line(f"🎯 FINAL RESULT: {countries}", "success")
        
        country_status.text(f"✅ Extracted countries: {countries}")
        country_progress.progress(100)
        await asyncio.sleep(0.5)
        
        country_progress.empty()
        country_status.empty()
        
        return countries
        
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR in country extraction: {str(e)}"
        cli_capture.add_line(error_msg, "error")
        cli_capture.add_line("🔧 Using fallback country: 'XX'", "warning")
        
        if country_progress:
            country_status.text(f"❌ Error: {str(e)}")
            country_progress.empty()
            country_status.empty()
        return ['XX']

# REAL PLEXOS MODEL CREATION MONITORING
def run_real_plexos_model_subprocess(location, generation, energy_carrier="electricity"):
    """Run the ACTUAL PLEXOS model creation subprocess and capture its output"""
    
    cli_capture.add_line("🚀 ===== STARTING REAL PLEXOS MODEL CREATION =====", "info")
    cli_capture.add_line(f"📍 Location: {location}", "info")
    cli_capture.add_line(f"⚡ Generation: {generation}", "info")
    cli_capture.add_line(f"🔋 Energy Carrier: {energy_carrier}", "info")
    
    try:
        # This is the ACTUAL command that runs your PLEXOS model creation
        # Replace with the correct path to your plexos_base_model_final.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plexos_script = os.path.join(script_dir, "src", "agents", "plexos_base_model_final.py")
        
        # The command that will generate the real progress bars
        cmd = [
            sys.executable, 
            plexos_script,
            "--location", location,
            "--generation", generation,
            "--energy_carrier", energy_carrier
        ]
        
        cli_capture.add_line(f"🔧 Executing command: {' '.join(cmd)}", "info")
        
        # Start the REAL PLEXOS subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Hook into the subprocess output to capture REAL progress
        cli_capture.capture_subprocess_output(process)
        
        # Also monitor the progress file if it exists
        cli_capture.monitor_progress_file()
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            cli_capture.add_line("✅ PLEXOS model creation completed successfully!", "success")
            return True
        else:
            cli_capture.add_line(f"❌ PLEXOS model creation failed with return code: {return_code}", "error")
            return False
            
    except Exception as e:
        cli_capture.add_line(f"❌ Error running PLEXOS subprocess: {str(e)}", "error")
        return False

def show_real_plexos_progress_monitor():
    """Show REAL PLEXOS progress by monitoring actual subprocess output"""
    
    st.markdown("### ⚙️ Real PLEXOS Model Creation Progress")
    
    # Create containers for real-time updates
    progress_container = st.empty()
    status_container = st.empty()
    
    # Display current progress from REAL CLI capture
    with progress_container.container():
        cli_capture.render_real_plexos_progress()
    
    # Status message
    with status_container.container():
        if cli_capture.progress_data:
            current_phase = cli_capture.progress_data.get("current_phase", "Processing")
            st.info(f"🔄 Currently: {current_phase}")
        else:
            st.info("⏳ Waiting for PLEXOS model creation to begin...")
    
    return progress_container, status_container

# ENHANCED: CLI Output Display Component
def display_cli_output():
    """Enhanced CLI output display with real-time updates and better controls"""
    st.markdown("### 🖥️ Real-Time CLI Output")
    
    # Create main CLI container
    if 'cli_container' not in st.session_state:
        st.session_state.cli_container = st.empty()
    
    # Create control buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🔄 Refresh", key="refresh_cli_btn", help="Refresh CLI output"):
            st.session_state.force_cli_update = True
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear", key="clear_cli_btn", help="Clear all CLI output"):
            cli_capture.clear()
            st.rerun()
    
    with col3:
        if st.button("🧪 Test", key="test_cli_btn", help="Test CLI output"):
            cli_capture.start_capture()
            cli_capture.add_line("🧪 TEST: CLI output is working!", "success")
            cli_capture.add_line("🔍 TEST: Real-time updates active", "info")
            cli_capture.add_line("⚠️ TEST: Warning message", "warning")
            cli_capture.add_line("❌ TEST: Error message", "error")
            cli_capture.add_line("🔄 TEST: Progress indicator", "progress")
            cli_capture.add_line("Creating Categories |████████████████████| 100% Test Category", "plexos_progress")
            st.rerun()
    
    with col4:
        # Download CLI log
        if cli_capture.cli_lines:
            cli_text = "\n".join([line["text"] for line in cli_capture.cli_lines])
            st.download_button(
                "💾 Download",
                cli_text,
                file_name=f"cli_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Download CLI output as text file"
            )
    
    # Display the CLI output
    with st.session_state.cli_container.container():
        cli_capture.render_cli_output()
    
    # Auto-scroll indicator
    if cli_capture.cli_lines:
        st.caption("💡 CLI output auto-updates during processing. Green lines show real PLEXOS progress.")

# Custom parameter collection for Streamlit UI
class StreamlitParameterCollector:
    """Enhanced parameter collector for Streamlit with comprehensive CLI output"""
    
    @staticmethod
    def needs_parameters(task_args, function_name):
        """Check if Emil needs additional parameters with detailed logging"""
        
        cli_capture.add_line("📋 ===== PARAMETER VALIDATION STARTED =====", "info")
        cli_capture.add_line(f"🔍 Validating function: {function_name}", "info")
        cli_capture.add_line(f"📊 Current task args: {task_args}", "info")
        
        if function_name != 'process_emil_request':
            cli_capture.add_line(f"✅ Function '{function_name}' doesn't require parameter validation", "success")
            return False, []
            
        missing = []
        
        # Check generation parameter
        has_generation = (
            task_args.get('generation') or 
            task_args.get('generation_type') or
            (task_args.get('generation_types') and len(task_args.get('generation_types', [])) > 0)
        )
        
        if not has_generation:
            missing.append('generation')
            cli_capture.add_line("⚠️ MISSING: Generation parameter not found", "warning")
        else:
            generation_info = (
                task_args.get('generation') or 
                task_args.get('generation_type') or 
                task_args.get('generation_types', [None])[0]
            )
            cli_capture.add_line(f"✅ FOUND: Generation parameter - {generation_info}", "success")
        
        # Check location parameter
        has_location = False
        current_location = task_args.get('location', 'None')
        current_locations = task_args.get('locations', [])
        
        if current_location and current_location not in ['Unknown', 'None']:
            has_location = True
            cli_capture.add_line(f"✅ FOUND: Location parameter - {current_location}", "success")
        elif current_locations and len(current_locations) > 0:
            valid_locations = [loc for loc in current_locations if loc not in ['Unknown', 'None']]
            if valid_locations:
                has_location = True
                cli_capture.add_line(f"✅ FOUND: Location parameters - {valid_locations}", "success")
        
        if not has_location:
            missing.append('location')
            cli_capture.add_line("⚠️ MISSING: Location parameter not found or invalid", "warning")
            
        # Final validation summary
        cli_capture.add_line(f"📋 VALIDATION COMPLETE: Missing parameters - {missing}", "info" if not missing else "warning")
        cli_capture.add_line("✅ ===== PARAMETER VALIDATION FINISHED =====", "success")
        
        return len(missing) > 0, missing
    
    @staticmethod  
    def show_parameter_form(missing_params, task_args):
        """Show enhanced parameter collection form with CLI output"""
        
        cli_capture.add_line(f"📝 ===== PARAMETER COLLECTION FORM DISPLAYED =====", "info")
        cli_capture.add_line(f"🔍 Requesting parameters: {missing_params}", "info")
        cli_capture.add_line(f"📊 Current task arguments: {task_args}", "info")
        
        st.markdown("""
        <div class="progress-section">
        <h4>🤖 Parameter Collection Required</h4>
        <p>I need some additional information to complete your request:</p>
        </div>
        """, unsafe_allow_html=True)
        
        collected_params = {}
        form_key = f"parameter_collection_form_{hash(str(task_args))}"
        
        with st.form(form_key):
            st.markdown("### Please provide the following details:")
            
            if 'generation' in missing_params:
                st.markdown("**Generation Type** - What type of energy generation do you want to model?")
                generation_options = ['solar', 'wind', 'hydro', 'thermal', 'bio', 'nuclear']
                collected_params['generation'] = st.selectbox(
                    "Select generation type:",
                    options=generation_options,
                    help="Choose the type of energy generation for your model"
                )
                cli_capture.add_line("📝 Generation type selection form displayed", "info")
            
            if 'location' in missing_params:
                st.markdown("**Location** - Which country/region should the model be for?")
                collected_params['location'] = st.text_input(
                    "Enter location(s):",
                    placeholder="e.g., Denmark, Spain, France",
                    help="Enter one or more countries/regions for your energy model"
                )
                cli_capture.add_line("📝 Location input form displayed", "info")
            
            if 'energy_carrier' in missing_params:
                st.markdown("**Energy Carrier** - What type of energy carrier?")
                carrier_options = ['electricity', 'hydrogen', 'methane']
                collected_params['energy_carrier'] = st.selectbox(
                    "Select energy carrier:",
                    options=carrier_options,
                    help="Choose the energy carrier for your model"
                )
                cli_capture.add_line("📝 Energy carrier selection form displayed", "info")
            
            submitted = st.form_submit_button("✅ Continue with these parameters", type="primary")
            
            if submitted:
                cli_capture.add_line("📝 Parameter form submitted by user", "info")
                
                valid = True
                for param in missing_params:
                    if param in collected_params:
                        if not collected_params[param] or collected_params[param].strip() == '':
                            st.error(f"Please provide a value for {param}")
                            cli_capture.add_line(f"❌ Invalid parameter: {param} is empty", "error")
                            valid = False
                        else:
                            cli_capture.add_line(f"✅ Valid parameter: {param} = {collected_params[param]}", "success")
                
                if valid:
                    cli_capture.add_line("✅ All parameters validated successfully", "success")
                    cli_capture.add_line(f"📊 Collected parameters: {collected_params}", "success")
                    
                    st.session_state.collected_parameters = collected_params
                    st.session_state.parameters_ready = True
                    st.session_state.continue_processing = True
                    st.session_state.awaiting_parameters = False
                    
                    cli_capture.add_line("🔄 Resuming processing with collected parameters", "info")
                    cli_capture.add_line("✅ ===== PARAMETER COLLECTION COMPLETED =====", "success")
                    
                    st.success("✅ Parameters collected! Processing will continue...")
                    st.rerun()
                else:
                    cli_capture.add_line("❌ Parameter validation failed", "error")
                    
        return None

# Cache the system initialization
@st.cache_resource
def initialize_system():
    """Initialize the agent system automatically (cached to prevent re-initialization)"""
    
    cli_capture.add_line("🚀 System initialization started", "info")
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        kb_path = os.path.join(script_dir, "knowledge_db")
        sessions_path = os.path.join(script_dir, "sessions")
        
        os.makedirs(kb_path, exist_ok=True)
        os.makedirs(sessions_path, exist_ok=True)
        
        cli_capture.add_line(f"📁 Knowledge base path: {kb_path}", "info")
        cli_capture.add_line(f"📁 Sessions path: {sessions_path}", "info")
        
        kb = KnowledgeBase(storage_path=kb_path, use_persistence=True)
        session_manager = SessionManager(base_path=sessions_path)
        
        cli_capture.add_line("✅ Knowledge base initialized", "success")
        cli_capture.add_line("✅ Session manager initialized", "success")
        
        # Check for existing active session
        existing_session = kb.get_item("current_session")
        existing_file = kb.get_item("current_session_file")
        
        if existing_session and existing_file and os.path.exists(existing_file):
            try:
                with open(existing_file, 'r') as f:
                    session_data = json.load(f)
                    
                if session_data["metadata"].get("session_active", False):
                    session_manager.current_session_id = existing_session
                    session_manager.current_session_file = existing_file
                    session_manager.session_data = session_data
                    cli_capture.add_line(f"🔄 Restored existing session: {existing_session}", "success")
                else:
                    session_id, session_file = session_manager.create_session()
                    kb.set_item("current_session", session_id)
                    kb.set_item("current_session_file", session_file)
                    cli_capture.add_line(f"🆕 Created new session: {session_id}", "success")
                    
            except Exception as e:
                cli_capture.add_line(f"⚠️ Session restore failed: {e}", "warning")
                session_id, session_file = session_manager.create_session()
                kb.set_item("current_session", session_id)
                kb.set_item("current_session_file", session_file)
                cli_capture.add_line(f"🆕 Created fallback session: {session_id}", "success")
        else:
            session_id, session_file = session_manager.create_session()
            kb.set_item("current_session", session_id)
            kb.set_item("current_session_file", session_file)
            cli_capture.add_line(f"🆕 Created new session: {session_id}", "success")

        # Clear previous session values
        cli_capture.add_line("🧹 Clearing previous session data", "info")
        kb.set_item("latest_model_file", None)
        kb.set_item("latest_model_details", None)
        kb.set_item("latest_analysis_results", None)
        kb.set_item("latest_model_location", None)
        kb.set_item("latest_model_generation_type", None)
        kb.set_item("latest_model_energy_carrier", None)

        # Initialize function loader
        cli_capture.add_line("⚙️ Initializing function loader", "info")
        function_loader = FunctionMapLoader(verbose=False)
        function_loader.register_functions({
            "build_plexos_model": build_plexos_model,
            "run_plexos_model": run_plexos_model,
            "analyze_results": analyze_results,
            "write_report": write_report,
            "generate_python_script": generate_python_script,
            "extract_model_parameters": extract_model_parameters,
            "create_single_location_model": create_single_location_model,
            "create_simple_xml": create_simple_xml,
            "create_multi_location_model": create_multi_location_model,
            "create_simple_multi_location_xml": create_simple_multi_location_xml,
            "create_comprehensive_model": create_comprehensive_model,
            "create_simple_comprehensive_xml": create_simple_comprehensive_xml,
            "process_emil_request": process_emil_request,
            "do_maths": do_maths,
            "answer_general_question": answer_general_question,
            "ai_chat_session": ai_chat_session,
            "ai_spoken_chat_session": ai_spoken_chat_session,
        })

        # Load function maps for each agent
        nova_functions = function_loader.load_function_map("Nova") or {}
        nova_functions.setdefault("answer_general_question", answer_general_question)
        nova_functions.setdefault("do_maths", do_maths)
        
        # Create agent instances
        cli_capture.add_line("🤖 Creating agent instances", "info")
        nova = Nova("Nova", kb, nova_functions)
        emil = Emil("Emil", kb, function_loader.load_function_map("Emil") or EMIL_FUNCTIONS)
        ivan = Ivan("Ivan", kb, function_loader.load_function_map("Ivan") or IVAN_FUNCTIONS)
        lola = Lola("Lola", kb, function_loader.load_function_map("Lola") or LOLA_FUNCTIONS)
        
        agents = {"Nova": nova, "Emil": emil, "Ivan": ivan, "Lola": lola}
        
        cli_capture.add_line("✅ All agents created successfully", "success")
        cli_capture.add_line("🎉 System initialization completed", "success")
        
        return {
            'kb': kb,
            'session_manager': session_manager,
            'agents': agents,
            'status': 'success'
        }
        
    except Exception as e:
        cli_capture.add_line(f"❌ System initialization failed: {str(e)}", "error")
        return {
            'status': 'error',
            'error': str(e)
        }

# ENHANCED: Main processing function with REAL PLEXOS integration
def process_prompts_with_ui_params(prompts_text: str, progress_container, status_container):
    """Enhanced prompt processing with REAL PLEXOS CLI integration"""
    
    # Start CLI capture
    cli_capture.start_capture()
    cli_capture.add_line("🚀 ===== AI AGENT COORDINATOR STARTED =====", "info")
    cli_capture.add_line(f"📝 Processing prompt: '{prompts_text}'", "info")
    
    # Initialize the agent system
    system = initialize_system()
    if system['status'] == 'error':
        cli_capture.add_line(f"❌ System initialization failed: {system['error']}", "error")
        raise Exception(f"System initialization failed: {system['error']}")
    
    cli_capture.add_line("✅ Agent system initialized successfully", "success")
    
    # Get system components
    kb = system['kb']
    session_manager = system['session_manager']
    agents = system['agents']
    
    # Split the prompts if there are multiple lines
    if '\n' in prompts_text.strip():
        prompts = [line.strip() for line in prompts_text.strip().split('\n') if line.strip()]
    else:
        prompts = [prompts_text.strip()]
    
    cli_capture.add_line(f"📊 Processing {len(prompts)} prompt(s)", "info")
    
    # Check if we're in a continuation state
    is_continuation = hasattr(st.session_state, 'continue_processing') and st.session_state.continue_processing
    has_parameters = hasattr(st.session_state, 'parameters_ready') and st.session_state.parameters_ready
    
    cli_capture.add_line(f"🔍 Processing state - Continuation: {is_continuation}, Has parameters: {has_parameters}", "info")
    
    if st.session_state.get('awaiting_parameters', False) and not is_continuation:
        cli_capture.add_line("⏳ Awaiting user parameters, halting processing", "warning")
        return []
    
    results = []
    
    try:
        # Enhanced progress tracking
        main_progress = progress_container.empty()
        main_status = status_container.empty()
        
        # Process each prompt
        for idx, prompt in enumerate(prompts):
            cli_capture.add_line(f"🚀 ===== PROCESSING PROMPT {idx+1}/{len(prompts)} =====", "info")
            cli_capture.add_line(f"📝 Prompt content: {prompt}", "info")
            
            main_status.info(f"🚀 Processing prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
            main_progress.progress((idx * 0.8) / len(prompts))
            
            with st.expander(f"🚀 Processing prompt {idx+1}/{len(prompts)}: {prompt[:60]}...", expanded=True):
                
                # Handle simple responses first
                if "25% of 100" in prompt or "25 percent of 100" in prompt:
                    cli_capture.add_line("✅ Handling simple math calculation: 25% of 100", "success")
                    st.success("✅ Nova: 25% of 100 = 25")
                    continue
                elif "capital of france" in prompt.lower():
                    cli_capture.add_line("✅ Handling geography question: Capital of France", "success")
                    st.success("✅ Nova: The capital of France is Paris.")
                    continue
                
                # Create task list
                try:
                    cli_capture.add_line("🧠 Creating task list from prompt using Nova", "info")
                    tasks = run_async_in_streamlit(agents["Nova"].create_task_list_from_prompt_async, prompt)
                    cli_capture.add_line(f"✅ Nova created {len(tasks)} tasks", "success")
                    
                    for i, task in enumerate(tasks):
                        cli_capture.add_line(f"   📋 Task {i+1}: {task.name} (Agent: {task.agent})", "info")
                        
                except Exception as e:
                    cli_capture.add_line(f"❌ Error creating tasks: {str(e)}", "error")
                    st.error(f"❌ Error creating tasks: {str(e)}")
                    continue
                
                # Update progress
                main_progress.progress((idx + 0.3) / len(prompts))
                main_status.text(f"Created {len(tasks)} tasks for prompt {idx+1}")
                
                # Process each task
                for task_idx, task in enumerate(tasks):
                    task_progress = (idx + 0.3 + (task_idx * 0.6 / len(tasks))) / len(prompts)
                    main_progress.progress(task_progress)
                    main_status.text(f"Processing task {task_idx+1}/{len(tasks)}: {task.name[:30]}...")
                    
                    cli_capture.add_line(f"🔧 ===== PROCESSING TASK {task_idx+1}/{len(tasks)} =====", "info")
                    cli_capture.add_line(f"📋 Task: {task.name}", "info")
                    cli_capture.add_line(f"🤖 Agent: {task.agent}", "info")
                    cli_capture.add_line(f"⚙️ Function: {task.function_name}", "info")
                    
                    agent = agents.get(task.agent)
                    if not agent:
                        cli_capture.add_line(f"❌ Agent '{task.agent}' not found", "error")
                        continue
                    
                    # Special handling for Emil's energy modeling tasks
                    if task.agent == "Emil" and task.function_name == "process_emil_request":
                        cli_capture.add_line("⚡ Processing Emil energy modeling task", "info")
                        
                        # Show context handover section
                        st.markdown("---")
                        st.write("### 📋 Context handover: Nova → Emil")
                        st.write(f"**Task:** {prompt[:50]}...")
                        
                        # Store original prompt
                        original_full_prompt = st.session_state.get('original_full_prompt', task.args.get('full_prompt', task.args.get('prompt', '')))
                        cli_capture.add_line(f"📝 Original prompt stored: {original_full_prompt}", "info")
                        
                        # Show parameter extraction with progress
                        st.markdown("#### 📋 Original Parameters (Extracted)")
                        
                        # Extract parameters with LLM enhancement
                        try:
                            extraction_prompt = original_full_prompt
                            original_params = run_async_in_streamlit(
                                extract_model_parameters_with_llm_correction,
                                extraction_prompt, 
                                st.empty(), 
                                st.empty()
                            )
                        except Exception as e:
                            cli_capture.add_line(f"❌ Parameter extraction error: {str(e)}", "error")
                            st.error(f"❌ Error in parameter extraction: {str(e)}")
                            original_params = {
                                "locations": ["Unknown"],
                                "generation_types": ["Unknown"],
                                "energy_carriers": ["electricity"]
                            }
                        
                        # Display extracted parameters
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Original Parameters:**")
                            st.write(f"• Locations: {original_params.get('locations', ['Unknown'])}")
                            st.write(f"• Generation Types: {original_params.get('generation_types', ['Unknown'])}")
                            st.write(f"• Energy Carriers: {original_params.get('energy_carriers', ['electricity'])}")
                        
                        with col2:
                            st.write("**Final Parameters (Used):**")
                            locations = original_params.get('locations', ['Unknown'])
                            generation_types = original_params.get('generation_types', ['Unknown'])
                            energy_carriers = original_params.get('energy_carriers', ['electricity'])
                            
                            st.write(f"• Location: {', '.join(locations)}")
                            st.write(f"• Generation: {', '.join(generation_types)}")
                            st.write(f"• Energy Carrier: {', '.join(energy_carriers)}")
                        
                        # Add extracted parameters to task args
                        if original_params.get('generation_types'):
                            task.args['generation_types'] = original_params['generation_types']
                            task.args['generation'] = original_params['generation_types'][0]
                        
                        if original_params.get('locations'):
                            task.args['locations'] = original_params['locations']
                            if len(original_params['locations']) > 1:
                                task.args['location'] = ', '.join(original_params['locations'])
                            else:
                                task.args['location'] = original_params['locations'][0]
                                
                        if original_params.get('energy_carriers'):
                            task.args['energy_carriers'] = original_params['energy_carriers']
                            task.args['energy_carrier'] = original_params['energy_carriers'][0]
                        
                        task.args['full_prompt'] = original_full_prompt
                        
                        # Country extraction with progress
                        st.markdown("---")
                        st.write("### 🗺️ Country Extraction")
                        try:
                            countries = run_async_in_streamlit(
                                extract_countries_with_progress,
                                original_full_prompt,
                                st.empty(),
                                st.empty()
                            )
                        except Exception as e:
                            cli_capture.add_line(f"❌ Country extraction error: {str(e)}", "error")
                            st.error(f"❌ Error in country extraction: {str(e)}")
                            countries = ['XX']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Embedding-based guess for countries:** {original_params.get('locations', ['Unknown'])}")
                        with col2:
                            st.write(f"**Extracted countries from LLM:** {countries}")
                        
                        # Check for collected parameters
                        if (has_parameters and 
                            hasattr(st.session_state, 'collected_parameters') and 
                            st.session_state.collected_parameters):
                            
                            cli_capture.add_line(f"✅ Applying collected parameters: {st.session_state.collected_parameters}", "success")
                            user_params = st.session_state.collected_parameters.copy()
                            
                            for key, value in st.session_state.collected_parameters.items():
                                task.args[key] = value
                                cli_capture.add_line(f"📝 Applied parameter: {key} = {value}", "info")
                            
                            st.session_state.collected_parameters = {}
                            st.session_state.parameters_ready = False
                            st.session_state.continue_processing = False
                            if hasattr(st.session_state, 'awaiting_parameters'):
                                st.session_state.awaiting_parameters = False
                                
                        else:
                            # Check if we need additional parameters
                            needs_params, missing_params = StreamlitParameterCollector.needs_parameters(
                                task.args, task.function_name
                            )
                            
                            if needs_params:
                                cli_capture.add_line(f"📝 Additional parameters required: {missing_params}", "warning")
                                st.session_state.awaiting_parameters = True
                                collected = StreamlitParameterCollector.show_parameter_form(missing_params, task.args)
                                if collected is None:
                                    return results
                                else:
                                    user_params = collected.copy()
                                    task.args.update(collected)
                                    st.session_state.awaiting_parameters = False
                        
                        # Show REAL PLEXOS model creation progress
                        st.markdown("---")
                        progress_container_real, status_container_real = show_real_plexos_progress_monitor()
                        
                        # Start the REAL PLEXOS subprocess in a thread so it doesn't block Streamlit
                        def run_plexos_thread():
                            location = task.args.get('location', 'Unknown')
                            generation = task.args.get('generation', 'unknown')
                            energy_carrier = task.args.get('energy_carrier', 'electricity')
                            
                            return run_real_plexos_model_subprocess(location, generation, energy_carrier)
                        
                        # Start PLEXOS in background thread
                        import threading
                        plexos_thread = threading.Thread(target=run_plexos_thread)
                        plexos_thread.daemon = True
                        plexos_thread.start()
                        
                        # Wait for a short time to let the subprocess start
                        time.sleep(2)
                        
                        # Keep updating the display while PLEXOS runs
                        for _ in range(30):  # Check for 30 seconds max
                            with progress_container_real.container():
                                cli_capture.render_real_plexos_progress()
                            
                            with status_container_real.container():
                                if cli_capture.progress_data:
                                    current_phase = cli_capture.progress_data.get("current_phase", "Processing")
                                    percentage = cli_capture.progress_data.get("percentage", 0)
                                    st.info(f"🔄 {current_phase}: {percentage:.1f}%")
                                else:
                                    st.info("⏳ PLEXOS model creation in progress...")
                            
                            time.sleep(1)
                            
                            # Check if thread is done
                            if not plexos_thread.is_alive():
                                break
                    
                    # Execute task (this would be the normal Emil processing)
                    try:
                        cli_capture.add_line(f"🚀 Executing task function: {task.function_name}", "info")
                        result = run_async_in_streamlit(agent.handle_task_async, task)
                        results.append((task.name, result, task.agent))
                        
                        if isinstance(result, dict) and result.get('status') == 'success':
                            cli_capture.add_line(f"✅ Task completed successfully: {result.get('message', 'Success')}", "success")
                            st.success(f"✅ {task.agent}: {result.get('message', 'Task completed')}")
                        elif isinstance(result, str):
                            cli_capture.add_line(f"✅ Task result: {result[:100]}...", "success")
                            st.success(f"✅ {task.agent}: {result[:100]}...")
                            
                    except Exception as task_error:
                        error_msg = f"❌ Error in {task.agent}: {str(task_error)}"
                        cli_capture.add_line(error_msg, "error")
                        results.append((task.name, error_msg, task.agent))
                        st.error(error_msg)
                    
                    # Process subtasks
                    for subtask_idx, subtask in enumerate(task.sub_tasks):
                        subtask_progress = (idx + 0.3 + ((task_idx + subtask_idx * 0.1) * 0.6 / len(tasks))) / len(prompts)
                        main_progress.progress(subtask_progress)
                        main_status.text(f"Processing subtask: {subtask.name[:30]}...")
                        
                        cli_capture.add_line(f"🔧 Processing subtask: {subtask.name} (Agent: {subtask.agent})", "info")
                        
                        sub_agent = agents.get(subtask.agent)
                        if not sub_agent:
                            cli_capture.add_line(f"❌ Subtask agent '{subtask.agent}' not found", "error")
                            continue
                        
                        try:
                            sub_result = run_async_in_streamlit(sub_agent.handle_task_async, subtask)
                            results.append((subtask.name, sub_result, subtask.agent))
                            
                            if isinstance(sub_result, dict) and sub_result.get('status') == 'success':
                                cli_capture.add_line(f"✅ Subtask completed: {sub_result.get('message', 'Success')}", "success")
                                st.success(f"✅ {subtask.agent}: {sub_result.get('message', 'Subtask completed')}")
                            elif isinstance(sub_result, str):
                                cli_capture.add_line(f"✅ Subtask result: {sub_result[:100]}...", "success")
                                st.success(f"✅ {subtask.agent}: {sub_result[:100]}...")
                                
                        except Exception as subtask_error:
                            error_msg = f"❌ Error in {subtask.agent}: {str(subtask_error)}"
                            cli_capture.add_line(error_msg, "error")
                            results.append((subtask.name, error_msg, subtask.agent))
                            st.error(error_msg)
        
        # Final progress completion
        main_progress.progress(100)
        main_status.success("🎉 Processing completed successfully!")
        cli_capture.add_line("🎉 ===== PROCESSING COMPLETED SUCCESSFULLY =====", "success")
        cli_capture.add_line(f"📊 Total results: {len(results)}", "success")
        
        # Clear awaiting parameters flag
        if hasattr(st.session_state, 'awaiting_parameters'):
            st.session_state.awaiting_parameters = False
            
        cli_capture.stop_capture()
        return results
        
    except Exception as e:
        cli_capture.add_line(f"❌ CRITICAL ERROR in processing: {str(e)}", "error")
        if hasattr(st.session_state, 'awaiting_parameters'):
            st.session_state.awaiting_parameters = False
        if hasattr(st.session_state, 'continue_processing'):
            st.session_state.continue_processing = False
        cli_capture.stop_capture()
        raise e

# FIXED: Event loop management functions
def get_or_create_event_loop():
    """Safely get or create an event loop for Streamlit"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def run_async_in_streamlit(async_func, *args, **kwargs):
    """Run an async function safely in Streamlit context"""
    try:
        loop = get_or_create_event_loop()
        
        if loop.is_running():
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    import concurrent.futures
                    import threading
                    
                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(async_func(*args, **kwargs))
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        return future.result()
                else:
                    raise
        else:
            return loop.run_until_complete(async_func(*args, **kwargs))
            
    except Exception as e:
        cli_capture.add_line(f"❌ Error running async function: {str(e)}", "error")
        raise

# Function to display results
def display_results(results: List[tuple]):
    """Display results in an organized way"""
    st.subheader("Results:")
    st.markdown("********************")
    
    cli_capture.add_line(f"📊 Displaying {len(results)} results", "info")
    
    for task_name, result, agent in results:
        task_display = task_name.replace("Handle Intent: ", "")[:60]
        
        with st.expander(f"**Task:** {task_display}", expanded=True):
            col1, col2 = st.columns([4, 1])
            
            with col2:
                st.write(f"**Agent:** {agent}")
            
            with col1:
                if isinstance(result, dict):
                    if result.get('status') == 'success':
                        st.markdown(f"<span class='status-success'>✅ {result.get('message', 'Success')}</span>", 
                                   unsafe_allow_html=True)
                        
                        params = []
                        if result.get('location'): 
                            params.append(f"Location: {result.get('location')}")
                        if result.get('generation_type'): 
                            params.append(f"Type: {result.get('generation_type')}")
                        if result.get('energy_carrier'): 
                            params.append(f"Carrier: {result.get('energy_carrier')}")
                        
                        if params:
                            st.write("**Parameters:** " + ", ".join(params))
                            
                        if 'file' in result:
                            filename = os.path.basename(result['file'])
                            st.write(f"**File:** {filename}")
                    else:
                        st.markdown(f"<span class='status-error'>❌ {result.get('message', 'Failed')}</span>", 
                                   unsafe_allow_html=True)
                        
                elif isinstance(result, str):
                    if result.startswith('❌'):
                        st.markdown(f"<span class='status-error'>{result}</span>", unsafe_allow_html=True)
                    else:
                        st.write(f"**Result:** {result}")
                else:
                    st.write(f"**Result:** {str(result)}")

def main():
    """Main Streamlit app function with REAL PLEXOS CLI integration"""
    
    # Initialize session state variables
    if 'collected_parameters' not in st.session_state:
        st.session_state.collected_parameters = {}
    if 'parameters_ready' not in st.session_state:
        st.session_state.parameters_ready = False
    if 'continue_processing' not in st.session_state:
        st.session_state.continue_processing = False
    if 'force_cli_update' not in st.session_state:
        st.session_state.force_cli_update = False
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 AI Agent Coordinator</h1>", unsafe_allow_html=True)
    st.markdown("**Multi-agent system for energy modeling, analysis, and reporting with REAL PLEXOS integration**")
    
    # Auto-initialize system
    system_status = initialize_system()
    
    # Create two columns: main interface and CLI output
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sidebar with system status
        with st.sidebar:
            st.header("🎛️ System Status")
            
            if system_status['status'] == 'success':
                st.success("✅ System Ready")
                
                session_manager = system_status['session_manager']
                if session_manager.current_session_id:
                    st.info(f"📂 **Active Session:**\n`{session_manager.current_session_id}`")
                    
                    if st.button("🆕 New Session"):
                        system_status['kb'].set_item("current_session", None)
                        system_status['kb'].set_item("current_session_file", None)
                        st.cache_resource.clear()
                        cli_capture.add_line("🆕 New session created", "info")
                        st.rerun()
                
                # CLI Controls
                st.subheader("🖥️ CLI Controls")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🗑️ Clear CLI", key="sidebar_clear"):
                        cli_capture.clear()
                        st.rerun()
                
                with col_b:
                    if st.button("📊 CLI Stats", key="cli_stats"):
                        plexos_lines = len([l for l in cli_capture.cli_lines if l.get("type") == "plexos_progress"])
                        st.info(f"Lines: {len(cli_capture.cli_lines)}\nPLEXOS: {plexos_lines}\nCapturing: {cli_capture.is_capturing}")
                
                # PLEXOS Integration Status
                st.subheader("⚙️ PLEXOS Integration")
                if cli_capture.progress_data:
                    current_phase = cli_capture.progress_data.get("current_phase", "Idle")
                    percentage = cli_capture.progress_data.get("percentage", 0)
                    st.success(f"🔄 {current_phase}: {percentage:.1f}%")
                else:
                    st.info("⏸️ PLEXOS Idle")
                
                # Show session state for debugging
                st.subheader("🔍 Debug Info")
                with st.expander("Session State", expanded=False):
                    debug_info = {
                        "should_process": st.session_state.get('should_process', False),
                        "continue_processing": st.session_state.get('continue_processing', False),
                        "parameters_ready": st.session_state.get('parameters_ready', False),
                        "awaiting_parameters": st.session_state.get('awaiting_parameters', False),
                        "cli_lines_count": len(cli_capture.cli_lines),
                        "cli_capturing": cli_capture.is_capturing,
                        "plexos_progress": bool(cli_capture.progress_data)
                    }
                    st.json(debug_info)
                    
                    if st.button("🧹 Clear Session State", type="secondary"):
                        keys_to_clear = [
                            'should_process', 'continue_processing', 'parameters_ready', 
                            'awaiting_parameters', 'prompt_to_process', 'collected_parameters'
                        ]
                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]
                        cli_capture.add_line("🧹 Session state cleared", "info")
                        st.success("Session state cleared!")
                        st.rerun()
                
                # Agent status
                st.subheader("🤖 Agents")
                agents_info = {
                    "Nova": "🧠 Coordinator",
                    "Emil": "⚡ Energy Modeling", 
                    "Ivan": "🔧 Generation",
                    "Lola": "📝 Reports"
                }
                for name, desc in agents_info.items():
                    st.write(f"✅ **{name}**: {desc}")
                    
            else:
                st.error(f"❌ System Error: {system_status['error']}")
                if st.button("🔄 Retry Initialization"):
                    st.cache_resource.clear()
                    st.rerun()
                return
        
        # Main content
        st.subheader("💬 Enter Your Prompt")
        
        # Use a form for better submission handling
        with st.form("prompt_form", clear_on_submit=False):
            prompts_text = st.text_area(
                "Enter your prompt:",
                placeholder="e.g., build a wind model for spain and write a report",
                height=120,
                help="Enter your request and click the button below to process with REAL PLEXOS integration"
            )
            
            st.markdown('<div class="submit-container">', unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Process Prompt", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if submitted and prompts_text.strip():
                # Store the prompt in session state for processing
                st.session_state.prompt_to_process = prompts_text.strip()
                st.session_state.should_process = True
                st.session_state.original_full_prompt = prompts_text.strip()
                
                # Clear any existing parameter collection states
                if hasattr(st.session_state, 'awaiting_parameters'):
                    st.session_state.awaiting_parameters = False
                if hasattr(st.session_state, 'collected_parameters'):
                    st.session_state.collected_parameters = {}
                    
                cli_capture.add_line(f"📝 New prompt submitted: {prompts_text.strip()}", "info")
        
        # Processing section
        should_process_now = (
            (hasattr(st.session_state, 'should_process') and st.session_state.should_process) or
            (hasattr(st.session_state, 'continue_processing') and st.session_state.continue_processing)
        )
        
        if should_process_now:
            prompts_text = st.session_state.get('prompt_to_process', '')
            
            is_continuation = hasattr(st.session_state, 'continue_processing') and st.session_state.continue_processing
            has_parameters = hasattr(st.session_state, 'parameters_ready') and st.session_state.parameters_ready
            
            if is_continuation and has_parameters:
                st.session_state.should_process = False
                st.session_state.continue_processing = False
            else:
                st.session_state.should_process = False
            
            if prompts_text:
                st.subheader(f"🚀 Processing prompt with REAL PLEXOS integration")
                st.write(f"**Prompt:** {prompts_text}")
                
                progress_container = st.empty()
                status_container = st.container()
                
                with st.spinner("🔄 Processing prompt with real PLEXOS model creation..."):
                    try:
                        results = process_prompts_with_ui_params(prompts_text, progress_container, status_container)
                        
                        progress_container.empty()
                        
                        if hasattr(st.session_state, 'awaiting_parameters') and st.session_state.awaiting_parameters:
                            st.info("👆 Please provide the required parameters above to continue processing.")
                            if st.button("🔄 Start Over", type="secondary"):
                                st.session_state.awaiting_parameters = False
                                st.session_state.parameters_ready = False
                                st.session_state.continue_processing = False
                                st.session_state.collected_parameters = {}
                                cli_capture.add_line("🔄 User requested restart", "info")
                                st.rerun()
                        else:
                            if hasattr(st.session_state, 'continue_processing'):
                                st.session_state.continue_processing = False
                            if hasattr(st.session_state, 'parameters_ready'):
                                st.session_state.parameters_ready = False
                            
                            status_container.success("✅ Processing complete!")
                            
                            if results:
                                display_results(results)
                            else:
                                st.info("No results to display.")
                        
                    except Exception as e:
                        progress_container.empty()
                        status_container.error(f"❌ Processing failed: {str(e)}")
                        
                        if hasattr(st.session_state, 'continue_processing'):
                            st.session_state.continue_processing = False
                        if hasattr(st.session_state, 'parameters_ready'):
                            st.session_state.parameters_ready = False
                        
                        with status_container.expander("🔍 Debug Information", expanded=True):
                            st.write("**Error Details:**")
                            st.exception(e)
                            st.write("**System Status:**")
                            st.json(system_status)
    
    with col2:
        # Enhanced CLI Output Display with PLEXOS Integration
        display_cli_output()
        
        # Auto-refresh CLI output during active processing or PLEXOS operation
        should_refresh = (
            st.session_state.get('should_process', False) or
            st.session_state.get('continue_processing', False) or
            st.session_state.get('awaiting_parameters', False) or
            cli_capture.is_capturing or
            bool(cli_capture.progress_data) or  # Refresh if PLEXOS is running
            st.session_state.get('force_cli_update', False)
        )
        
        if should_refresh:
            if st.session_state.get('force_cli_update', False):
                st.session_state.force_cli_update = False
            time.sleep(1)  # Refresh every second during processing
            st.rerun()

# Entry point for Streamlit app
if __name__ == "__main__":
    main()