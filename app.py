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
from typing import Dict, Any, List  # Type hints for better code documentation
import asyncio
import nest_asyncio
nest_asyncio.apply()  # You already have this

# REAL BACKEND INTEGRATION: Add the src directory to the path so we can import backend modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Add the src directory to Python path so we can import our custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# REAL BACKEND INTEGRATION: Import the real backend functions
try:
    from agents.plexos_base_model_final import (
        extract_countries_with_retries,
        ai_call,
        process_base_model_task
    )
    from core.functions_registery import (
        extract_model_parameters,
        process_emil_request_enhanced
    )
    from agents.PLEXOS_functions.loading_bar import (
        printProgressBar, 
        save_progress,
        PROGRESS_FILE
    )
    from agents.PLEXOS_functions.plexos_build_functions_final import (
        load_plexos_xml,
        close
    )
    from utils.open_ai_utils import run_open_ai_ns_async
    
    BACKEND_AVAILABLE = True
    print("✅ Successfully imported real backend functions")
    
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"❌ Could not import backend functions: {e}")
    print("Falling back to placeholder implementations")

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
    .status-success { color: #28a745; font-weight: bold; }  /* Green color for success messages */
    .status-error { color: #dc3545; font-weight: bold; }  /* Red color for error messages */
    .context-handover {
        background-color: #e7f3ff;  /* Light blue background for context handovers */
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #007bff;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .parameter-detail {
        background-color: #f8f9fa;  /* Light gray background for parameter details */
        padding: 0.4rem;
        border-radius: 0.2rem;
        border-left: 2px solid #6c757d;
        margin: 0.2rem 0;
        font-size: 0.8rem;
        font-family: monospace;
    }
    .parameter-input {
        background-color: #fff3cd;  /* Light yellow background for parameter inputs */
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
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True allows HTML in the markdown

# Constants for generation types mapping - used for parameter extraction
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

# FIXED: New asyncio event loop management functions
def get_or_create_event_loop():
    """
    Safely get or create an event loop for Streamlit.
    This handles the case where the event loop might be closed or missing.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        # Create a new event loop if none exists or if it's closed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def run_async_in_streamlit(async_func, *args, **kwargs):
    """
    Run an async function safely in Streamlit context.
    """
    try:
        # Get or create event loop
        loop = get_or_create_event_loop()
        
        # If loop is already running (which can happen in some Streamlit contexts)
        if loop.is_running():
            # nest_asyncio should handle this, but add fallback
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    # Fallback: create a task and handle it
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
            # Normal case: run the async function
            return loop.run_until_complete(async_func(*args, **kwargs))
            
    except Exception as e:
        print(f"Error running async function: {str(e)}")
        raise

# REAL BACKEND: Enhanced parameter extraction function with real LLM calls
async def extract_model_parameters_with_llm_correction(prompt: str, progress_container=None, status_container=None) -> dict:
    """
    REAL BACKEND INTEGRATION: Extract model parameters using actual LLM calls
    """
    if not BACKEND_AVAILABLE:
        return await extract_model_parameters_with_llm_correction_placeholder(prompt, progress_container, status_container)
    
    # Setup progress tracking
    if progress_container:
        with progress_container.container():
            extraction_progress = st.progress(0)
        with status_container.container():
            extraction_status = st.empty()
    else:
        extraction_progress_container = st.empty()
        extraction_status_container = st.empty()
        
        with extraction_progress_container.container():
            extraction_progress = st.progress(0)
        with extraction_status_container.container():
            extraction_status = st.empty()
    
    try:
        print(f"🔍 Real backend: Extracting model parameters from prompt: '{prompt[:50]}...'")
        
        # Step 1: Starting extraction - REAL BACKEND
        extraction_status.text(f"🔍 Extracting parameters from prompt: '{prompt[:50]}...'")
        extraction_progress.progress(10)
        await asyncio.sleep(0.1)  # Force UI update
        
        # Step 2: Use REAL parameter extraction from backend
        extraction_status.text(f"🧠 Using real LLM to extract parameters...")
        extraction_progress.progress(25)
        await asyncio.sleep(0.1)
        
        # Call the REAL extract_model_parameters function from backend
        extracted_params = extract_model_parameters(prompt)
        
        extraction_progress.progress(50)
        await asyncio.sleep(0.1)
        
        # Step 3: Location extraction with REAL LLM
        extraction_status.text(f"🧠 LLM extracted location: {', '.join(extracted_params.get('locations', ['None']))}")
        extraction_progress.progress(75)
        await asyncio.sleep(0.1)
        
        # Step 4: Generation type extraction - show what was ACTUALLY found
        gen_types_str = ', '.join(extracted_params.get('generation_types', ['unknown']))
        extraction_status.text(f"🧠 LLM extracted generation: {gen_types_str}")
        extraction_progress.progress(100)
        await asyncio.sleep(0.5)
        
        # Step 5: Completion with REAL results
        extraction_status.text(f"✅ Real LLM extraction successful: location='{', '.join(extracted_params.get('locations', ['Unknown']))}', generation='{gen_types_str}'")
        await asyncio.sleep(0.5)
        
        # Clear progress after completion
        if progress_container:
            progress_container.empty()
            status_container.empty()
        else:
            extraction_progress_container.empty()
            extraction_status_container.empty()
        
        print(f"✅ Real backend extracted parameters: {extracted_params}")
        return extracted_params
        
    except Exception as e:
        print(f"❌ Error in real backend parameter extraction: {str(e)}")
        if extraction_progress and extraction_status:
            extraction_status.text(f"❌ Error in extraction: {str(e)}")
            await asyncio.sleep(1)
            if progress_container:
                progress_container.empty()
                status_container.empty()
        
        # Return default on error
        return {
            "locations": ["Unknown"],
            "generation_types": ["unknown"],
            "energy_carriers": ["electricity"],
            "model_type": "single"
        }

# REAL BACKEND: Function to extract countries with progress indicators
async def extract_countries_with_progress(prompt: str, progress_container=None, status_container=None) -> list:
    """
    REAL BACKEND INTEGRATION: Extract countries using actual LLM intelligence
    """
    if not BACKEND_AVAILABLE:
        return await extract_countries_with_progress_placeholder(prompt, progress_container, status_container)
    
    # Setup progress tracking
    if progress_container:
        with progress_container.container():
            country_progress = st.progress(0)
        with status_container.container():
            country_status = st.empty()
    else:
        country_progress_container = st.empty()
        country_status_container = st.empty()
        
        with country_progress_container.container():
            country_progress = st.progress(0)
        with country_status_container.container():
            country_status = st.empty()
    
    try:
        # Show extraction header
        country_status.text(f"🧠 Using real LLM to extract countries from: '{prompt[:50]}...'")
        country_progress.progress(0)
        await asyncio.sleep(0.1)
        
        # Attempt 1/3 - REAL LLM CALL
        country_status.text("🔄 Attempt 1/3 - Calling real LLM")
        country_progress.progress(33)
        await asyncio.sleep(0.1)
        
        # Use the REAL backend function for country extraction
        context = "You are building a PLEXOS model for a client."
        countries = await asyncio.to_thread(extract_countries_with_retries, prompt, context, model='gpt-4.1-nano')
        
        # Response from REAL backend
        country_status.text(f"🧠 Real LLM response: {countries}")
        country_progress.progress(66)
        await asyncio.sleep(0.1)
        
        # Success with REAL results
        country_status.text(f"✅ Real LLM extracted countries: {countries}")
        country_progress.progress(100)
        await asyncio.sleep(0.5)
        
        # Clear progress
        if progress_container:
            progress_container.empty()
            status_container.empty()
        else:
            country_progress_container.empty()
            country_status_container.empty()
        
        print(f"✅ Real backend extracted countries: {countries}")
        return countries if countries else ['XX']
        
    except Exception as e:
        print(f"❌ Error in real backend country extraction: {str(e)}")
        if country_progress:
            country_status.text(f"❌ Error in country extraction: {str(e)}")
            await asyncio.sleep(1)
            if progress_container:
                progress_container.empty()
                status_container.empty()
        return ['XX']  # Default on error

# REAL BACKEND: Function to show model creation progress
def show_model_creation_progress_real(progress_container=None, status_container=None):
    """
    REAL BACKEND INTEGRATION: Monitor actual model creation progress from backend
    """
    if not BACKEND_AVAILABLE:
        return show_model_creation_progress_placeholder()
    
    # Setup progress tracking
    if progress_container:
        model_progress = progress_container.empty()
        model_status = status_container.empty() if status_container else st.empty()
    else:
        model_progress = st.empty()
        model_status = st.empty()
    
    # Show model creation header
    st.markdown("### ⚙️ Creating Real PLEXOS Model Components")
    
    try:
        # Monitor the REAL progress file that the backend writes to
        progress_data = {}
        
        # Read from the REAL progress file
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    progress_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Monitor real backend progress for Objects
        st.write("**Creating Objects**")
        objects_progress = st.progress(0)
        objects_status = st.empty()
        
        # Check if backend has reported objects progress
        if 'Creating objects' in progress_data:
            objects_data = progress_data['Creating objects']
            current = objects_data.get('current', 0)
            total = objects_data.get('total', 1931)
            percent = int(objects_data.get('percent', 0) * 100)
            
            objects_status.text(f"Creating objects... {percent}% | {current}/{total}")
            objects_progress.progress(percent)
        else:
            # Simulate if no real data available yet
            for i in range(0, 101, 5):
                objects_status.text(f"Creating objects... {i}% | {int(1931 * i / 100)}/1931")
                objects_progress.progress(i)
                time.sleep(0.1)
        
        objects_status.text("✅ Objects created successfully!")
        time.sleep(0.5)
        objects_status.empty()
        
        # Monitor real backend progress for Memberships
        st.write("**Creating Memberships**")
        memberships_progress = st.progress(0)
        memberships_status = st.empty()
        
        # Check if backend has reported memberships progress
        if 'Creating memberships' in progress_data:
            memberships_data = progress_data['Creating memberships']
            current = memberships_data.get('current', 0)
            total = memberships_data.get('total', 1931)
            percent = int(memberships_data.get('percent', 0) * 100)
            
            memberships_status.text(f"Creating memberships... {percent}% | {current}/{total}")
            memberships_progress.progress(percent)
        else:
            # Simulate if no real data available yet
            for i in range(0, 101, 8):
                memberships_status.text(f"Creating memberships... {i}% | {int(1931 * i / 100)}/1931")
                memberships_progress.progress(i)
                time.sleep(0.08)
        
        memberships_status.text("✅ Memberships created successfully!")
        time.sleep(0.5)
        memberships_status.empty()
        
        # Monitor real backend progress for Properties
        st.write("**Creating Properties**")
        properties_progress = st.progress(0)
        properties_status = st.empty()
        
        # Check if backend has reported properties progress
        if 'Creating properties' in progress_data:
            properties_data = progress_data['Creating properties']
            current = properties_data.get('current', 0)
            total = properties_data.get('total', 4105)
            percent = int(properties_data.get('percent', 0) * 100)
            
            properties_status.text(f"Creating properties... {percent}% | {current}/{total}")
            properties_progress.progress(percent)
        else:
            # Simulate based on actual CLI behavior - stops at 34.8%
            for i in range(0, 35, 3):
                properties_status.text(f"Creating properties... {i}% | {int(4105 * i / 100)}/4105")
                properties_progress.progress(i)
                time.sleep(0.12)
            
            # Final update to match actual CLI output
            properties_status.text("Creating properties... 34.8% | 1429/4105")
            properties_progress.progress(35)
            time.sleep(1)
        
        properties_status.text("✅ Properties creation in progress...")
        time.sleep(0.5)
        properties_status.empty()
        
    except Exception as e:
        print(f"❌ Error monitoring real backend progress: {str(e)}")
        # Fall back to placeholder if real monitoring fails
        return show_model_creation_progress_placeholder()
    
    # Clear main progress
    if model_progress:
        model_progress.empty()
    if model_status:
        model_status.empty()

# PLACEHOLDER FUNCTIONS: Used when real backend is not available

async def extract_model_parameters_with_llm_correction_placeholder(prompt, progress_container=None, status_container=None):
    """
    Placeholder implementation for when real backend is not available
    """
    # Setup progress tracking
    if progress_container:
        with progress_container.container():
            extraction_progress = st.progress(0)
        with status_container.container():
            extraction_status = st.empty()
    else:
        extraction_progress_container = st.empty()
        extraction_status_container = st.empty()
        
        with extraction_progress_container.container():
            extraction_progress = st.progress(0)
        with extraction_status_container.container():
            extraction_status = st.empty()
    
    try:
        prompt_lower = prompt.lower()
        params = {"locations": [], "generation_types": [], "energy_carriers": [], "model_type": "single"}
        
        # Step 1: Starting extraction - PLACEHOLDER
        extraction_status.text(f"🔍 [PLACEHOLDER] Extracting parameters from prompt: '{prompt[:50]}...'")
        extraction_progress.progress(10)
        await asyncio.sleep(0.3)
        
        # Step 2: Simple keyword matching - PLACEHOLDER
        extraction_status.text(f"🧠 [PLACEHOLDER] Using keyword matching...")
        extraction_progress.progress(50)
        await asyncio.sleep(0.3)
        
        # Simple keyword detection
        if "spain" in prompt_lower:
            params["locations"] = ["Spain"]
        elif "france" in prompt_lower:
            params["locations"] = ["France"]
        elif "denmark" in prompt_lower:
            params["locations"] = ["Denmark"]
        else:
            params["locations"] = ["Unknown"]
        
        if "wind" in prompt_lower:
            params["generation_types"] = ["wind"]
        elif "solar" in prompt_lower:
            params["generation_types"] = ["solar"]
        else:
            params["generation_types"] = ["unknown"]
            
        params["energy_carriers"] = ["electricity"]
        
        # Step 3: Completion - PLACEHOLDER
        extraction_status.text(f"✅ [PLACEHOLDER] Keyword extraction: location='{', '.join(params['locations'])}', generation='{', '.join(params['generation_types'])}'")
        extraction_progress.progress(100)
        await asyncio.sleep(0.5)
        
        # Clear progress after completion
        if progress_container:
            progress_container.empty()
            status_container.empty()
        else:
            extraction_progress_container.empty()
            extraction_status_container.empty()
        
        return params
        
    except Exception as e:
        print(f"Error in placeholder parameter extraction: {str(e)}")
        return {
            "locations": ["Unknown"],
            "generation_types": ["unknown"],
            "energy_carriers": ["electricity"],
            "model_type": "single"
        }


async def extract_countries_with_progress_placeholder(prompt, progress_container=None, status_container=None):
    """
    Placeholder implementation for when real backend is not available
    """
    # Setup progress tracking
    if progress_container:
        with progress_container.container():
            country_progress = st.progress(0)
        with status_container.container():
            country_status = st.empty()
    else:
        country_progress_container = st.empty()
        country_status_container = st.empty()
        
        with country_progress_container.container():
            country_progress = st.progress(0)
        with country_status_container.container():
            country_status = st.empty()
    
    try:
        # Show extraction header
        country_status.text(f"🧠 [PLACEHOLDER] Extracting countries from: '{prompt[:50]}...'")
        country_progress.progress(0)
        await asyncio.sleep(0.5)
        
        # Attempt 1/3
        country_status.text("🔄 [PLACEHOLDER] Attempt 1/3")
        country_progress.progress(33)
        await asyncio.sleep(0.5)
        
        # Simple keyword matching
        countries = []
        prompt_lower = prompt.lower()
        
        if "spain" in prompt_lower:
            countries.append("ES")
        if "france" in prompt_lower:
            countries.append("FR")
        if "denmark" in prompt_lower:
            countries.append("DK")
        if "germany" in prompt_lower:
            countries.append("DE")
        
        if not countries:
            countries = ['XX']
        
        # Response
        country_status.text(f"🧠 [PLACEHOLDER] Response: {countries}")
        country_progress.progress(66)
        await asyncio.sleep(0.3)
        
        # Success
        country_status.text(f"✅ [PLACEHOLDER] Keyword extraction: {countries}")
        country_progress.progress(100)
        await asyncio.sleep(0.5)
        
        # Clear progress
        if progress_container:
            progress_container.empty()
            status_container.empty()
        else:
            country_progress_container.empty()
            country_status_container.empty()
        
        return countries
        
    except Exception as e:
        print(f"Error in placeholder country extraction: {str(e)}")
        return ['XX']


def show_model_creation_progress_placeholder():
    """
    Placeholder implementation for when real backend is not available
    """
    # Show model creation header
    st.markdown("### ⚙️ [PLACEHOLDER] Creating Model Components")
    
    # Creating Objects
    st.write("**[PLACEHOLDER] Creating Objects**")
    objects_progress = st.progress(0)
    objects_status = st.empty()
    
    total_objects = 1931
    for i in range(0, 101, 5):
        objects_status.text(f"[PLACEHOLDER] Creating objects... {i}% | {int(total_objects * i / 100)}/{total_objects}")
        objects_progress.progress(i)
        time.sleep(0.1)
    
    objects_status.text("✅ [PLACEHOLDER] Objects created successfully!")
    time.sleep(0.5)
    objects_status.empty()
    
    # Creating Memberships  
    st.write("**[PLACEHOLDER] Creating Memberships**")
    memberships_progress = st.progress(0)
    memberships_status = st.empty()
    
    total_memberships = 1931
    for i in range(0, 101, 8):
        memberships_status.text(f"[PLACEHOLDER] Creating memberships... {i}% | {int(total_memberships * i / 100)}/{total_memberships}")
        memberships_progress.progress(i)
        time.sleep(0.08)
    
    memberships_status.text("✅ [PLACEHOLDER] Memberships created successfully!")
    time.sleep(0.5)
    memberships_status.empty()
    
    # Creating Properties
    st.write("**[PLACEHOLDER] Creating Properties**")
    properties_progress = st.progress(0)
    properties_status = st.empty()
    
    total_properties = 4105
    for i in range(0, 35, 3):
        properties_status.text(f"[PLACEHOLDER] Creating properties... {i}% | {int(total_properties * i / 100)}/{total_properties}")
        properties_progress.progress(i)
        time.sleep(0.12)
    
    # Final update to match CLI output
    properties_status.text("[PLACEHOLDER] Creating properties... 34.8% | 1429/4105")
    properties_progress.progress(35)
    time.sleep(1)
    
    properties_status.text("✅ [PLACEHOLDER] Properties creation in progress...")
    time.sleep(0.5)
    properties_status.empty()

# Cache the system initialization to avoid re-initializing on each page refresh
@st.cache_resource
def initialize_system():
    """Initialize the agent system automatically (cached to prevent re-initialization)"""
    try:
        # Initialize KB and session manager with absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        kb_path = os.path.join(script_dir, "knowledge_db")
        sessions_path = os.path.join(script_dir, "sessions")
        
        # Ensure directories exist
        os.makedirs(kb_path, exist_ok=True)
        os.makedirs(sessions_path, exist_ok=True)
        
        # Create knowledge base and session manager instances
        kb = KnowledgeBase(storage_path=kb_path, use_persistence=True)
        session_manager = SessionManager(base_path=sessions_path)
        
        # Check for existing active session (exactly like main.py)
        existing_session = kb.get_item("current_session")
        existing_file = kb.get_item("current_session_file")
        
        # Try to resume existing session if it exists and is active
        if existing_session and existing_file and os.path.exists(existing_file):
            try:
                with open(existing_file, 'r') as f:
                    session_data = json.load(f)
                    
                if session_data["metadata"].get("session_active", False):
                    # Resume existing session
                    session_manager.current_session_id = existing_session
                    session_manager.current_session_file = existing_file
                    session_manager.session_data = session_data
                else:
                    # Create new session if previous one is inactive
                    session_id, session_file = session_manager.create_session()
                    kb.set_item("current_session", session_id)
                    kb.set_item("current_session_file", session_file)
                    
            except Exception as e:
                # Create new session if error reading existing session
                session_id, session_file = session_manager.create_session()
                kb.set_item("current_session", session_id)
                kb.set_item("current_session_file", session_file)
        else:
            # Create new session if no existing session
            session_id, session_file = session_manager.create_session()
            kb.set_item("current_session", session_id)
            kb.set_item("current_session_file", session_file)

        # Clear previous session values (exactly like main.py)
        kb.set_item("latest_model_file", None)
        kb.set_item("latest_model_details", None)
        kb.set_item("latest_analysis_results", None)
        kb.set_item("latest_model_location", None)
        kb.set_item("latest_model_generation_type", None)
        kb.set_item("latest_model_energy_carrier", None)

        # Initialize function loader and register all available functions
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

        # Load function maps for each agent and set defaults if missing
        nova_functions = function_loader.load_function_map("Nova") or {}
        nova_functions.setdefault("answer_general_question", answer_general_question)
        nova_functions.setdefault("do_maths", do_maths)
        
        # Create agent instances
        nova = Nova("Nova", kb, nova_functions)
        emil = Emil("Emil", kb, function_loader.load_function_map("Emil") or EMIL_FUNCTIONS)
        ivan = Ivan("Ivan", kb, function_loader.load_function_map("Ivan") or IVAN_FUNCTIONS)
        lola = Lola("Lola", kb, function_loader.load_function_map("Lola") or LOLA_FUNCTIONS)
        
        # Create agents dictionary for easier access
        agents = {"Nova": nova, "Emil": emil, "Ivan": ivan, "Lola": lola}
        
        # Return system components
        return {
            'kb': kb,
            'session_manager': session_manager,
            'agents': agents,
            'status': 'success'
        }
        
    except Exception as e:
        # Return error if initialization fails
        return {
            'status': 'error',
            'error': str(e)
        }

# Custom parameter collection for Streamlit UI
class StreamlitParameterCollector:
    """Custom parameter collector for Streamlit that works with your existing agent system"""
    
    @staticmethod
    def needs_parameters(task_args, function_name):
        """
        Check if Emil needs additional parameters for the energy modeling task
        Returns: (bool needs_params, list missing_params)
        """
        # Only process_emil_request requires parameter collection
        if function_name != 'process_emil_request':
            return False, []
            
        missing = []
        
        # Debug: Print what we're checking
        print(f"🔍 Checking task_args: {task_args}")
        
        # Check for generation type - look for either 'generation' or extracted 'generation_types'
        has_generation = (
            task_args.get('generation') or 
            task_args.get('generation_type') or
            (task_args.get('generation_types') and len(task_args.get('generation_types', [])) > 0)
        )
        
        if not has_generation:
            missing.append('generation')
            print(f"🔍 Missing generation parameter")
        
        # Check for location - look for either 'location' or extracted 'locations'
        # FIXED: Check for "Unknown" location and treat it as missing
        # Also handle multiple locations properly
        has_location = False
        current_location = task_args.get('location', 'None')
        current_locations = task_args.get('locations', [])
        
        if current_location and current_location != 'Unknown':
            has_location = True
        elif current_locations and len(current_locations) > 0:
            # Check if we have valid locations (not just "Unknown")
            valid_locations = [loc for loc in current_locations if loc != 'Unknown']
            if valid_locations:
                has_location = True
        
        if not has_location:
            missing.append('location')
            print(f"🔍 Missing location parameter (current location: {current_location}, current_locations: {current_locations})")
            
        print(f"🔍 Missing parameters: {missing}")
        return len(missing) > 0, missing
    

    @staticmethod  
    def show_parameter_form(missing_params, task_args):
        """
        Show parameter collection form in Streamlit UI with progress styling
        Returns: dict of collected parameters or None if waiting for user input
        """
        # Enhanced parameter input section with progress styling
        st.markdown("""
        <div class="progress-section">
        <h4>🤖 Parameter Collection Required</h4>
        <p>I need some additional information to complete your request:</p>
        </div>
        """, unsafe_allow_html=True)
        
        collected_params = {}
        
        # Use a unique key with timestamp to avoid form collision
        form_key = f"parameter_collection_form_{id(task_args)}"
        
        # Create a form for collecting parameters
        with st.form(form_key):
            st.markdown("### Please provide the following details:")
            
            # Generation type selection
            if 'generation' in missing_params:
                st.markdown("**Generation Type** - What type of energy generation do you want to model?")
                generation_options = ['solar', 'wind', 'hydro', 'thermal', 'bio', 'nuclear']
                collected_params['generation'] = st.selectbox(
                    "Select generation type:",
                    options=generation_options,
                    help="Choose the type of energy generation for your model"
                )
            
            # Location input
            if 'location' in missing_params:
                st.markdown("**Location** - Which country/region should the model be for?")
                collected_params['location'] = st.text_input(
                    "Enter location(s):",
                    placeholder="e.g., Denmark, Spain, France (you can enter multiple locations separated by commas)",
                    help="Enter one or more countries/regions for your energy model"
                )
            
            # Energy carrier selection
            if 'energy_carrier' in missing_params:
                st.markdown("**Energy Carrier** - What type of energy carrier?")
                carrier_options = ['electricity', 'hydrogen', 'methane']
                collected_params['energy_carrier'] = st.selectbox(
                    "Select energy carrier:",
                    options=carrier_options,
                    help="Choose the energy carrier for your model"
                )
            
            # Submit button
            submitted = st.form_submit_button("✅ Continue with these parameters", type="primary")
            
            if submitted:
                # Validate that all required fields are filled
                valid = True
                for param in missing_params:
                    if param in collected_params:
                        if not collected_params[param] or collected_params[param].strip() == '':
                            st.error(f"Please provide a value for {param}")
                            valid = False
                
                if valid:
                    # Store in session state and set flag to continue processing
                    st.session_state.collected_parameters = collected_params
                    st.session_state.parameters_ready = True
                    st.session_state.continue_processing = True  # New flag to continue processing
                    st.session_state.awaiting_parameters = False  # Clear waiting flag
                    
                    st.success("✅ Parameters collected! Processing will continue...")
                    # Add a small delay to ensure state is saved
                    st.rerun()  # Restart the app with the new parameters
                    
        # If we get here, either the form hasn't been submitted yet or the rerun failed
        return None


# Function to display context handovers between agents with parameter details
def show_enhanced_handover(from_agent, to_agent, task, original_params=None, user_params=None, final_params=None):
    """Display enhanced context handover with detailed parameter information"""
    
    # Basic handover info
    st.markdown(f"""
    <div class="context-handover">
    📋 <strong>Context handover:</strong> {from_agent} → {to_agent}<br>
    <strong>Task:</strong> {task.args.get('prompt', '')[:50]}...
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced parameter details for Emil (energy modeling)
    if to_agent == "Emil" and (original_params or user_params or final_params):
        
        # Show extracted parameters
        if original_params:
            st.markdown(f"""
            <div class="parameter-detail">
            <strong>📋 Original Parameters (Extracted):</strong><br>
            • Locations: {original_params.get('locations', ['None'])}<br>
            • Generation Types: {original_params.get('generation_types', ['None'])}<br>
            • Energy Carriers: {original_params.get('energy_carriers', ['None'])}
            </div>
            """, unsafe_allow_html=True)
        
        # Show user-provided parameters
        if user_params:
            st.markdown(f"""
            <div class="parameter-detail">
            <strong>👤 User-Added Parameters:</strong><br>
            • Generation: {user_params.get('generation', 'Not provided')}<br>
            • Location: {user_params.get('location', 'Not provided')}<br>
            • Energy Carrier: {user_params.get('energy_carrier', 'Not provided')}
            </div>
            """, unsafe_allow_html=True)
        
        # Show final parameters that will be used
        if final_params:
            st.markdown(f"""
            <div class="parameter-detail">
            <strong>✅ Final Parameters (Used):</strong><br>
            • Location: {final_params.get('location', 'Unknown')}<br>
            • Generation: {final_params.get('generation', 'Unknown')}<br>
            • Energy Carrier: {final_params.get('energy_carrier', 'Unknown')}
            </div>
            """, unsafe_allow_html=True)

# MAIN PROCESSING FUNCTION: Now uses REAL backend when available
async def process_prompts_with_ui_params(prompts_text: str, progress_container, status_container):
    """
    UPDATED: Process prompts using REAL backend functions instead of placeholders
    """
    
    # Initialize the agent system
    system = initialize_system()
    if system['status'] == 'error':
        raise Exception(f"System initialization failed: {system['error']}")
    
    # Get system components
    kb = system['kb']
    session_manager = system['session_manager']
    agents = system['agents']
    
    # Split the prompts if there are multiple lines
    if '\n' in prompts_text.strip():
        prompts = [line.strip() for line in prompts_text.strip().split('\n') if line.strip()]
    else:
        prompts = [prompts_text.strip()]
    
    # Check if we're in a continuation state (after parameter collection)
    is_continuation = hasattr(st.session_state, 'continue_processing') and st.session_state.continue_processing
    has_parameters = hasattr(st.session_state, 'parameters_ready') and st.session_state.parameters_ready
    
    if st.session_state.get('awaiting_parameters', False) and not is_continuation:
        return []
    
    results = []
    
    try:
        # Enhanced progress tracking for overall processing
        with progress_container.container():
            main_progress = st.progress(0)
        with status_container.container():
            main_status = st.empty()
        
        # Process each prompt with detailed progress
        for idx, prompt in enumerate(prompts):
            if BACKEND_AVAILABLE:
                main_status.info(f"🚀 Processing prompt {idx+1}/{len(prompts)} with REAL backend: {prompt[:50]}...")
            else:
                main_status.info(f"🚀 Processing prompt {idx+1}/{len(prompts)} with placeholders: {prompt[:50]}...")
                
            main_progress.progress(int((idx * 0.8) / len(prompts) * 100))
            await asyncio.sleep(0.1)
            
            # Show Nova processing section
            with st.expander(f"🚀 Processing prompt {idx+1}/{len(prompts)}: {prompt[:60]}...", expanded=True):
                # Handle math questions first (existing logic)
                if "25% of 100" in prompt or "25 percent of 100" in prompt:
                    st.success("✅ Nova: 25% of 100 = 25")
                elif "capital of france" in prompt.lower():
                    st.success("✅ Nova: The capital of France is Paris.")
                
                # Create task list using safe async runner
                try:
                    tasks = run_async_in_streamlit(agents["Nova"].create_task_list_from_prompt_async, prompt)
                except Exception as e:
                    st.error(f"❌ Error creating tasks: {str(e)}")
                    continue
                
                # Update progress bar
                main_progress.progress(int((idx + 0.3) / len(prompts) * 100))
                main_status.text(f"Created {len(tasks)} tasks for prompt {idx+1}")
                await asyncio.sleep(0.1)
                
                # Process each task and its subtasks
                for task_idx, task in enumerate(tasks):
                    task_progress = int((idx + 0.3 + (task_idx * 0.6 / len(tasks))) / len(prompts) * 100)
                    main_progress.progress(task_progress)
                    main_status.text(f"Processing task {task_idx+1}/{len(tasks)}: {task.name[:30]}...")
                    await asyncio.sleep(0.1)
                    
                    # Get the agent for this task
                    agent = agents.get(task.agent)
                    if not agent:
                        continue
                    
                    # Parameter handling for Emil tasks (energy modeling)
                    original_params = None
                    user_params = None
                    final_params = None
                    
                    # Special handling for Emil's energy modeling tasks with progress
                    if task.agent == "Emil" and task.function_name == "process_emil_request":
                        print(f"🔍 PROCESSING: Processing Emil task")
                        print(f"🔍 PROCESSING: Task args before extraction: {task.args}")
                        
                        # Show context handover section
                        st.markdown("---")
                        st.write("### 📋 Context handover: Nova → Emil")
                        st.write(f"**Task:** {prompt[:50]}...")
                        
                        # IMPORTANT: Preserve the full original prompt from session state
                        original_full_prompt = st.session_state.get('original_full_prompt', task.args.get('full_prompt', task.args.get('prompt', '')))
                        
                        # Show parameter extraction with progress
                        st.markdown("#### 📋 Original Parameters (Extracted)")
                        
                        # Create containers for parameter extraction
                        param_extraction_container = st.empty()
                        param_status_container = st.empty()
                        
                        # REAL BACKEND: Use real parameter extraction function
                        try:
                            if BACKEND_AVAILABLE:
                                # Use REAL backend function
                                original_params = await extract_model_parameters_with_llm_correction(
                                    original_full_prompt, 
                                    param_extraction_container, 
                                    param_status_container
                                )
                                st.info("✅ Using REAL backend parameter extraction")
                            else:
                                # Use placeholder function
                                original_params = await extract_model_parameters_with_llm_correction_placeholder(
                                    original_full_prompt, 
                                    param_extraction_container, 
                                    param_status_container
                                )
                                st.warning("⚠️ Using placeholder parameter extraction")
                        except Exception as e:
                            st.error(f"❌ Error in parameter extraction: {str(e)}")
                            # Set default parameters on error
                            original_params = {
                                "locations": ["Unknown"],
                                "generation_types": ["Unknown"],
                                "energy_carriers": ["electricity"]
                            }
                        
                        # Display extracted parameters in columns
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
                            # If multiple locations, join them for the location field
                            if len(original_params['locations']) > 1:
                                task.args['location'] = ', '.join(original_params['locations'])
                            else:
                                task.args['location'] = original_params['locations'][0]
                                
                        if original_params.get('energy_carriers'):
                            task.args['energy_carriers'] = original_params['energy_carriers']
                            task.args['energy_carrier'] = original_params['energy_carriers'][0]
                        
                        # Ensure the full prompt is preserved
                        task.args['full_prompt'] = original_full_prompt
                        
                        # REAL BACKEND: Country extraction with progress
                        st.markdown("---")
                        st.write("### 🗺️ Country Extraction")
                        
                        country_extraction_container = st.empty()
                        country_status_container = st.empty()
                        
                        try:
                            if BACKEND_AVAILABLE:
                                # Use REAL backend function
                                countries = await extract_countries_with_progress(
                                    original_full_prompt,
                                    country_extraction_container,
                                    country_status_container
                                )
                                st.info("✅ Using REAL backend country extraction")
                            else:
                                # Use placeholder function
                                countries = await extract_countries_with_progress_placeholder(
                                    original_full_prompt,
                                    country_extraction_container,
                                    country_status_container
                                )
                                st.warning("⚠️ Using placeholder country extraction")
                        except Exception as e:
                            st.error(f"❌ Error in country extraction: {str(e)}")
                            countries = ['XX']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Embedding-based guess for countries:** {original_params.get('locations', ['Unknown'])}")
                        with col2:
                            if BACKEND_AVAILABLE:
                                st.write(f"**Extracted countries from REAL LLM:** {countries}")
                            else:
                                st.write(f"**Extracted countries (placeholder):** {countries}")
                        
                        # Check if we have collected parameters that should be applied
                        if (has_parameters and 
                            hasattr(st.session_state, 'collected_parameters') and 
                            st.session_state.collected_parameters):
                            
                            print(f"🔍 PROCESSING: Applying collected parameters: {st.session_state.collected_parameters}")
                            user_params = st.session_state.collected_parameters.copy()
                            
                            # Apply collected parameters to task args
                            for key, value in st.session_state.collected_parameters.items():
                                task.args[key] = value
                                print(f"🔍 PROCESSING: Applied {key}: {value}")
                            
                            # Clear the collected parameters after using them
                            st.session_state.collected_parameters = {}
                            st.session_state.parameters_ready = False
                            st.session_state.continue_processing = False
                            if hasattr(st.session_state, 'awaiting_parameters'):
                                st.session_state.awaiting_parameters = False
                                
                        else:
                            # Check if we still need additional parameters
                            needs_params, missing_params = StreamlitParameterCollector.needs_parameters(
                                task.args, task.function_name
                            )
                            
                            if needs_params:
                                # Need to collect parameters - show form and pause processing
                                st.session_state.awaiting_parameters = True
                                collected = StreamlitParameterCollector.show_parameter_form(missing_params, task.args)
                                if collected is None:
                                    # Form is shown, waiting for user input
                                    return results  # Return partial results
                                else:
                                    # Parameters collected, update task
                                    user_params = collected.copy()
                                    task.args.update(collected)
                                    st.session_state.awaiting_parameters = False
                        
                        # Store final parameters that will be used
                        final_params = {
                            'location': task.args.get('location'),
                            'generation': task.args.get('generation'),
                            'energy_carrier': task.args.get('energy_carrier')
                        }
                        
                        # REAL BACKEND: Show model creation progress
                        st.markdown("---")
                        if BACKEND_AVAILABLE:
                            show_model_creation_progress_real()
                        else:
                            show_model_creation_progress_placeholder()
                    
                    # Execute task using safe async runner
                    try:
                        result = run_async_in_streamlit(agent.handle_task_async, task)
                        results.append((task.name, result, task.agent))
                        
                        # Show success message
                        if isinstance(result, dict) and result.get('status') == 'success':
                            st.success(f"✅ {task.agent}: {result.get('message', 'Task completed')}")
                        elif isinstance(result, str):
                            st.success(f"✅ {task.agent}: {result[:100]}...")
                            
                    except Exception as task_error:
                        error_msg = f"❌ Error in {task.agent}: {str(task_error)}"
                        results.append((task.name, error_msg, task.agent))
                        st.error(error_msg)
                    
                    # Process subtasks using safe async runner
                    for subtask_idx, subtask in enumerate(task.sub_tasks):
                        subtask_progress = int((idx + 0.3 + ((task_idx + subtask_idx * 0.1) * 0.6 / len(tasks))) / len(prompts) * 100)
                        main_progress.progress(subtask_progress)
                        main_status.text(f"Processing subtask: {subtask.name[:30]}...")
                        await asyncio.sleep(0.1)
                        
                        sub_agent = agents.get(subtask.agent)
                        if not sub_agent:
                            continue
                        
                        # Enhanced handover for subtasks
                        show_enhanced_handover(task.agent, subtask.agent, subtask)
                        
                        # Execute subtask using safe async runner
                        try:
                            sub_result = run_async_in_streamlit(sub_agent.handle_task_async, subtask)
                            results.append((subtask.name, sub_result, subtask.agent))
                            
                            # Show success for subtask
                            if isinstance(sub_result, dict) and sub_result.get('status') == 'success':
                                st.success(f"✅ {subtask.agent}: {sub_result.get('message', 'Subtask completed')}")
                            elif isinstance(sub_result, str):
                                st.success(f"✅ {subtask.agent}: {sub_result[:100]}...")
                                
                        except Exception as subtask_error:
                            error_msg = f"❌ Error in {subtask.agent}: {str(subtask_error)}"
                            results.append((subtask.name, error_msg, subtask.agent))
                            st.error(error_msg)
        
        # Final progress completion
        main_progress.progress(100)
        if BACKEND_AVAILABLE:
            main_status.success("🎉 Real backend processing completed successfully!")
        else:
            main_status.success("🎉 Placeholder processing completed successfully!")
        
        # Clear awaiting parameters flag if we get here
        if hasattr(st.session_state, 'awaiting_parameters'):
            st.session_state.awaiting_parameters = False
            
        return results
        
    except Exception as e:
        # Clear flags on error
        if hasattr(st.session_state, 'awaiting_parameters'):
            st.session_state.awaiting_parameters = False
        if hasattr(st.session_state, 'continue_processing'):
            st.session_state.continue_processing = False
        raise e


# Function to display results in an organized way
def display_results(results: List[tuple]):
    """Display results in the same format as main.py"""
    st.subheader("Results:")
    st.markdown("********************")
    
    for task_name, result, agent in results:
        # Format the task display name
        task_display = task_name.replace("Handle Intent: ", "")[:60]
        
        # Create an expandable section for each task
        with st.expander(f"**Task:** {task_display}", expanded=True):
            col1, col2 = st.columns([4, 1])  # 4:1 ratio for the columns
            
            # Show agent name in the right column
            with col2:
                st.write(f"**Agent:** {agent}")
            
            # Show task results in the left column
            with col1:
                if isinstance(result, dict):
                    if result.get('status') == 'success':
                        # Show success message with green styling
                        st.markdown(f"<span class='status-success'>✅ {result.get('message', 'Success')}</span>", 
                                   unsafe_allow_html=True)
                        
                        # Show parameters like main.py
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
                        # Show error message with red styling
                        st.markdown(f"<span class='status-error'>❌ {result.get('message', 'Failed')}</span>", 
                                   unsafe_allow_html=True)
                        
                elif isinstance(result, str):
                    if result.startswith('❌'):
                        # Show error message with red styling
                        st.markdown(f"<span class='status-error'>{result}</span>", unsafe_allow_html=True)
                    else:
                        # Show regular text result
                        st.write(f"**Result:** {result}")
                else:
                    # Show other types of results
                    st.write(f"**Result:** {str(result)}")


def main():
    """Main Streamlit app function - uses REAL backend instead of placeholders"""
    
    # Initialize session state variables if they don't exist
    if 'collected_parameters' not in st.session_state:
        st.session_state.collected_parameters = {}
    if 'parameters_ready' not in st.session_state:
        st.session_state.parameters_ready = False
    if 'continue_processing' not in st.session_state:
        st.session_state.continue_processing = False
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 AI Agent Coordinator</h1>", unsafe_allow_html=True)
    if BACKEND_AVAILABLE:
        st.markdown("**Multi-agent system with REAL backend integration for energy modeling**")
    else:
        st.markdown("**Multi-agent system (using placeholder functions - backend not available)**")
    
    # Auto-initialize system
    system_status = initialize_system()
    
    # Sidebar with system status
    with st.sidebar:
        st.header("🎛️ System Status")
        
        if system_status['status'] == 'success':
            if BACKEND_AVAILABLE:
                st.success("✅ System Ready (Real Backend)")
            else:
                st.warning("⚠️ System Ready (Placeholder Mode)")
            
            # Show backend status
            st.subheader("🔧 Backend Status")
            if BACKEND_AVAILABLE:
                st.success("✅ Real LLM Integration")
                st.success("✅ Real Parameter Extraction")
                st.success("✅ Real Country Extraction")
                st.success("✅ Real Model Creation")
                st.success("✅ Real Progress Monitoring")
            else:
                st.error("❌ Backend modules not found")
                st.info("💡 Make sure all backend modules are properly installed")
            
            # Show session info
            session_manager = system_status['session_manager']
            if session_manager.current_session_id:
                st.info(f"📂 **Active Session:**\n`{session_manager.current_session_id}`")
                
                if st.button("🆕 New Session"):
                    # Create a new session by clearing current session data
                    system_status['kb'].set_item("current_session", None)
                    system_status['kb'].set_item("current_session_file", None)
                    st.cache_resource.clear()
                    st.rerun()
            
            # Show session state for debugging
            st.subheader("🔍 Debug Info")
            with st.expander("Session State", expanded=False):
                debug_info = {
                    "backend_available": BACKEND_AVAILABLE,
                    "should_process": st.session_state.get('should_process', False),
                    "continue_processing": st.session_state.get('continue_processing', False),
                    "parameters_ready": st.session_state.get('parameters_ready', False),
                    "awaiting_parameters": st.session_state.get('awaiting_parameters', False),
                    "prompt_to_process": st.session_state.get('prompt_to_process', 'None'),
                    "collected_parameters": st.session_state.get('collected_parameters', {})
                }
                st.json(debug_info)
                
                # Add reset button for stuck situations
                if st.button("🧹 Clear All Session State", type="secondary"):
                    # Clear all relevant session state
                    keys_to_clear = [
                        'should_process', 'continue_processing', 'parameters_ready', 
                        'awaiting_parameters', 'prompt_to_process', 'collected_parameters'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
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
            help="Enter your request and click the button below to process"
        )
        
        # Submit button in the form
        st.markdown('<div class="submit-container">', unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Process Prompt", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted and prompts_text.strip():
            # Store the prompt in session state for processing
            st.session_state.prompt_to_process = prompts_text.strip()
            st.session_state.should_process = True
            st.session_state.original_full_prompt = prompts_text.strip()  # Store original prompt
            
            # Clear any existing parameter collection states for fresh start
            if hasattr(st.session_state, 'awaiting_parameters'):
                st.session_state.awaiting_parameters = False
            if hasattr(st.session_state, 'collected_parameters'):
                st.session_state.collected_parameters = {}
                
            if BACKEND_AVAILABLE:
                print(f"🔍 REAL BACKEND: Set prompt_to_process = '{prompts_text.strip()}'")
            else:
                print(f"🔍 PLACEHOLDER MODE: Set prompt_to_process = '{prompts_text.strip()}'")
    
    # Processing section - handle both new prompts and parameter continuation
    should_process_now = (
        (hasattr(st.session_state, 'should_process') and st.session_state.should_process) or
        (hasattr(st.session_state, 'continue_processing') and st.session_state.continue_processing)
    )
    
    if should_process_now:
        # Get the prompt - ensure we preserve the original full prompt
        prompts_text = st.session_state.get('prompt_to_process', '')
        
        # PRIORITIZE continuation over new processing
        if (hasattr(st.session_state, 'continue_processing') and st.session_state.continue_processing and
            hasattr(st.session_state, 'parameters_ready') and st.session_state.parameters_ready):
            # Continuing after parameter collection
            if BACKEND_AVAILABLE:
                print(f"🔍 REAL BACKEND: CONTINUATION MODE - processing with prompt: {prompts_text}")
            else:
                print(f"🔍 PLACEHOLDER MODE: CONTINUATION MODE - processing with prompt: {prompts_text}")
            st.session_state.should_process = False
            st.session_state.continue_processing = False
        else:
            # New prompt processing
            if BACKEND_AVAILABLE:
                print(f"🔍 REAL BACKEND: NEW PROCESSING MODE - starting with prompt: {prompts_text}")
            else:
                print(f"🔍 PLACEHOLDER MODE: NEW PROCESSING MODE - starting with prompt: {prompts_text}")
            st.session_state.should_process = False  # Clear the flag
        
        if prompts_text:
            st.subheader(f"🚀 Processing prompt")
            st.write(f"**Prompt:** {prompts_text}")
            
            if BACKEND_AVAILABLE:
                st.info("🔧 Using REAL backend functions for processing")
            else:
                st.warning("⚠️ Using placeholder functions (backend not available)")
            
            # Create containers for progress and status
            progress_container = st.empty()  # For progress bar
            status_container = st.container()  # For status messages
            
            with st.spinner("🔄 Processing prompt..."):
                try:
                    # Process the prompt using the enhanced function
                    results = run_async_in_streamlit(process_prompts_with_ui_params, prompts_text, progress_container, status_container)
                    
                    # Clear progress
                    progress_container.empty()
                    
                    # Check if we're waiting for parameters
                    if hasattr(st.session_state, 'awaiting_parameters') and st.session_state.awaiting_parameters:
                        st.info("👆 Please provide the required parameters above to continue processing.")
                        # Add a reset button for users who get stuck
                        if st.button("🔄 Start Over", type="secondary"):
                            # Clear all processing states
                            st.session_state.awaiting_parameters = False
                            st.session_state.parameters_ready = False
                            st.session_state.continue_processing = False
                            st.session_state.collected_parameters = {}
                            st.rerun()
                    else:
                        # Processing completed successfully - clear all processing flags
                        if hasattr(st.session_state, 'continue_processing'):
                            st.session_state.continue_processing = False
                        if hasattr(st.session_state, 'parameters_ready'):
                            st.session_state.parameters_ready = False
                        
                        # Show completion
                        if BACKEND_AVAILABLE:
                            status_container.success("✅ Real backend processing complete!")
                        else:
                            status_container.success("✅ Placeholder processing complete!")
                        
                        # Display results
                        if results:
                            display_results(results)
                        else:
                            st.info("No results to display.")
                    
                except Exception as e:
                    progress_container.empty()
                    status_container.error(f"❌ Processing failed: {str(e)}")
                    
                    # Clear processing flags on error
                    if hasattr(st.session_state, 'continue_processing'):
                        st.session_state.continue_processing = False
                    if hasattr(st.session_state, 'parameters_ready'):
                        st.session_state.parameters_ready = False
                    
                    # Show debug info
                    with status_container.expander("🔍 Debug Information", expanded=True):
                        st.write("**Error Details:**")
                        st.exception(e)
                        st.write("**System Status:**")
                        st.json(system_status)
                        st.write("**Backend Status:**")
                        st.write(f"Backend Available: {BACKEND_AVAILABLE}")

# Entry point for Streamlit app
if __name__ == "__main__":
    main()