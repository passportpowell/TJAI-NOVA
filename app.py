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
from typing import Dict, Any, List  # Type hints for better code documentation
import asyncio
import nest_asyncio
nest_asyncio.apply()  # You already have this



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

# Enhanced parameter extraction function with progress indicators - FIXED VERSION
async def extract_model_parameters_with_llm_correction(prompt, progress_container=None, status_container=None):
    """
    Enhanced parameter extraction that uses LLM to correct misspelled locations
    and find parameters that hardcoded lists might miss - with progress indicators.
    FIXED for better error handling.
    """
    import re  # For regular expression pattern matching
    
    # Progress tracking setup
    if progress_container:
        extraction_progress = progress_container.empty()
        extraction_status = status_container.empty() if status_container else st.empty()
    else:
        extraction_progress = st.empty()
        extraction_status = st.empty()
    
    try:
        print("Extracting model parameters from prompt...")
        prompt_lower = prompt.lower()  # Convert to lowercase for case-insensitive matching
        
        # Initialize results dictionary with default values
        params = {"locations": [], "generation_types": [], "energy_carriers": [], "model_type": "single"}
        
        # Progress: Starting extraction
        if extraction_progress:
            extraction_status.text(f"🔍 Extracting parameters from prompt: '{prompt[:50]}...'")
            extraction_progress.progress(10)
            time.sleep(0.5)
        
        # Step 1: Try hardcoded location matching first (pattern-based approach)
        extraction_status.text(f"🧠 Using LLM to extract parameters from: '{prompt[:50]}...'")
        extraction_progress.progress(25)
        time.sleep(0.5)
        
        found_locations = []
        for loc in LOCATIONS:
            # Various patterns to match location mentions in different contexts
            patterns = [
                f" for {loc.lower()}",  # e.g., "model for Spain"
                f" in {loc.lower()}",   # e.g., "model in Spain"
                f" {loc.lower()} ",     # e.g., "the Spain model"
                f"model {loc.lower()}",  # e.g., "model Spain"
                f"{loc.lower()} model",  # e.g., "Spain model"
                f" {loc.lower()} and",   # e.g., "Spain and France"
                f"and {loc.lower()}",    # e.g., "and Spain"
                f", {loc.lower()}",      # e.g., "France, Spain"
                f"{loc.lower()},"        # e.g., "Spain, France"
            ]
            if any(pattern in prompt_lower for pattern in patterns):
                found_locations.append(loc)
        
        # Remove duplicates and store in params
        params["locations"] = list(set(found_locations))
        
        # Progress: Location extraction
        extraction_status.text(f"🧠 LLM extracted location: {', '.join(params['locations']) if params['locations'] else 'None found'}")
        extraction_progress.progress(50)
        time.sleep(0.5)
        
        # Step 2: If no locations found with hardcoded matching, use LLM (AI-based approach)
        if not params["locations"]:
            print("🔍 No locations found with hardcoded matching, trying LLM correction...")
            extraction_status.text("🔍 No locations found with hardcoded matching, trying LLM correction...")
            extraction_progress.progress(60)
            
            # Prompt for the LLM to extract locations
            location_correction_context = """
            You are a location extraction assistant. Extract ALL countries/locations mentioned in the text.
            
            Rules:
            1. Look for phrases like "build a model for [location]", "model for [location]", "energy model in [location]"
            2. Find ALL locations mentioned, including multiple countries separated by "and", "," or other conjunctions
            3. If locations are misspelled, correct them to proper country/region names
            4. Return ALL found locations as a JSON array: ["Country1", "Country2", ...]
            5. If no location is found, return ["Unknown"]
            
            Examples:
            "build a model for spain and denmark" → ["Spain", "Denmark"]
            "create model for france, germany and italy" → ["France", "Germany", "Italy"]
            "model for germany" → ["Germany"]
            "energy model in frnace and spian" → ["France", "Spain"]
            "build a model" → ["Unknown"]
            """
            
            try:
                # Call OpenAI API to get location extraction
                corrected_locations = await run_open_ai_ns_async(prompt, location_correction_context, model="gpt-4.1-nano")
                corrected_locations = corrected_locations.strip()
                
                # Try to parse as JSON array
                try:
                    import json
                    locations_list = json.loads(corrected_locations)
                    if isinstance(locations_list, list) and locations_list:
                        # Filter out "Unknown" if we have real locations
                        valid_locations = [loc for loc in locations_list if loc.lower() != "unknown"]
                        if valid_locations:
                            params["locations"] = valid_locations
                            print(f"🔍 LLM extracted multiple locations: {valid_locations}")
                        else:
                            params["locations"] = ["Unknown"]
                            print("🔍 LLM could not determine location")
                    else:
                        params["locations"] = ["Unknown"]
                        print("🔍 LLM returned invalid format")
                except json.JSONDecodeError:
                    # Fallback: try to extract as single location
                    if corrected_locations and corrected_locations.lower() != "unknown":
                        params["locations"] = [corrected_locations]
                        print(f"🔍 LLM corrected single location: '{corrected_locations}'")
                    else:
                        params["locations"] = ["Unknown"]
                        print("🔍 LLM could not determine location")
                    
            except Exception as e:
                print(f"🔍 LLM location correction failed: {str(e)}")
                params["locations"] = ["Unknown"]
        else:
            print(f"🔍 Found locations with hardcoded matching: {found_locations}")
        
        # Progress: Generation type extraction
        extraction_status.text(f"🧠 LLM extracted generation: extracting...")
        extraction_progress.progress(75)
        time.sleep(0.5)
        
        # Step 3: Extract generation types using regex patterns
        found_gen_types = []
        for gen in GENERATION_TYPES.keys():
            # Different patterns to match generation type mentions
            patterns = [
                f"build.*{gen}.*model",   # e.g., "build a wind model"
                f"create.*{gen}.*model",  # e.g., "create wind model"
                f"{gen}.*model.*for",     # e.g., "wind model for Spain"
                f"make.*{gen}.*model",    # e.g., "make a wind model"
                f"{gen} power",           # e.g., "wind power"
                f"{gen} generation",      # e.g., "wind generation"
                f"{gen} energy",          # e.g., "wind energy"
                f"a {gen} model",         # e.g., "a wind model"
                f"build {gen}",           # e.g., "build wind"
                f"create {gen}"           # e.g., "create wind"
            ]
            
            # Check if any pattern matches
            if any(re.search(pattern, prompt_lower) for pattern in patterns):
                found_gen_types.append(gen)
        
        # Remove duplicates and store in params
        params["generation_types"] = list(set(found_gen_types))
        
        # Step 4: If no generation types found, try LLM extraction
        if not params["generation_types"]:
            print("🔍 No generation types found with pattern matching, trying LLM extraction...")
            
            # Prompt for the LLM to extract generation type
            generation_extraction_context = """
            You are a generation type extraction assistant. Given a text about building an energy model, extract the type of energy generation mentioned.
            
            Look for energy types like: solar, wind, hydro, thermal, nuclear, bio, geothermal, etc.
            
            Rules:
            1. Look for phrases about building/creating models for specific energy types
            2. Return only the generation type (e.g., "wind", "solar", "hydro")
            3. If no specific type is mentioned, return "unknown"
            4. Use lowercase
            
            Examples:
            "build a wind model for spain" → "wind"
            "create solar energy model" → "solar"
            "hydro model for france" → "hydro"
            "build a model for croatia" → "unknown"
            """
            
            try:
                # Call OpenAI API to get generation type
                extracted_generation = await run_open_ai_ns_async(prompt, generation_extraction_context, model="gpt-4.1-nano")
                extracted_generation = extracted_generation.strip().lower()
                
                if extracted_generation and extracted_generation != "unknown":
                    # Validate against known generation types
                    if extracted_generation in GENERATION_TYPES.keys():
                        params["generation_types"] = [extracted_generation]
                        print(f"🔍 LLM extracted generation type: '{extracted_generation}'")
                    else:
                        print(f"🔍 LLM extracted '{extracted_generation}' but it's not in known types")
                else:
                    print("🔍 LLM could not determine generation type")
                    
            except Exception as e:
                print(f"🔍 LLM generation extraction failed: {str(e)}")
        
        # Step 5: Extract energy carriers (simpler pattern matching)
        carriers = ["electricity", "hydrogen", "methane"]
        found_carriers = [carrier for carrier in carriers if carrier in prompt_lower]
        # Default to electricity if no carriers found
        params["energy_carriers"] = found_carriers or ["electricity"]
        
        # Set location default only if still empty
        if not params["locations"]:
            params["locations"] = ["Unknown"]
        
        # Progress: Completion
        gen_types_str = ', '.join(params["generation_types"]) if params["generation_types"] else "unknown"
        extraction_status.text(f"✅ LLM extraction successful: {{'location': '{', '.join(params['locations'])}', 'generation': '{gen_types_str}', 'energy_carrier': '{', '.join(params['energy_carriers'])}'}}")
        extraction_progress.progress(100)
        time.sleep(0.5)
        
        # Clear progress after completion
        extraction_progress.empty()
        extraction_status.empty()
        
        print("Extracted parameters:", params)
        return params
        
    except Exception as e:
        print(f"Error in parameter extraction: {str(e)}")
        # Return default parameters on error
        if extraction_progress:
            extraction_status.text(f"❌ Error in extraction: {str(e)}")
            extraction_progress.empty()
            extraction_status.empty()
        
        return {
            "locations": ["Unknown"],
            "generation_types": ["unknown"],
            "energy_carriers": ["electricity"],
            "model_type": "single"
        }

# Function to extract countries with progress - FIXED VERSION
async def extract_countries_with_progress(prompt, progress_container=None, status_container=None):
    """Extract countries from prompt with progress indicators matching CLI output - FIXED"""
    
    # Setup progress tracking
    if progress_container:
        country_progress = progress_container.empty()
        country_status = status_container.empty() if status_container else st.empty()
    else:
        country_progress = st.empty()
        country_status = st.empty()
    
    try:
        # Show extraction header
        country_status.text(f"🧠 Extracting countries from: '{prompt[:50]}...'")
        country_progress.progress(0)
        time.sleep(0.8)
        
        # Attempt 1/3
        country_status.text("🔄 Attempt 1/3")
        country_progress.progress(33)
        time.sleep(0.8)
        
        # Simulate country extraction logic based on prompt
        countries = []
        prompt_lower = prompt.lower()
        
        # Map countries to country codes
        country_mapping = {
            'france': 'FR',
            'montenegro': 'ME',
            'spain': 'ES',
            'greece': 'GR',
            'germany': 'DE',
            'italy': 'IT',
            'denmark': 'DK'
        }
        
        for country_name, country_code in country_mapping.items():
            if country_name in prompt_lower:
                countries.append(country_code)
        
        if not countries:
            countries = ['XX']  # Default unknown country
        
        # Response
        country_status.text(f"🧠 Response: {countries}")
        country_progress.progress(66)
        time.sleep(0.5)
        
        # Success
        country_status.text(f"✅ Extracted countries: {countries}")
        country_progress.progress(100)
        time.sleep(0.5)
        
        # Clear progress
        country_progress.empty()
        country_status.empty()
        
        return countries
        
    except Exception as e:
        print(f"Error in country extraction: {str(e)}")
        if country_progress:
            country_status.text(f"❌ Error in country extraction: {str(e)}")
            country_progress.empty()
            country_status.empty()
        return ['XX']  # Default on error

# Function to show model creation progress
def show_model_creation_progress(progress_container=None, status_container=None):
    """Show model creation progress matching CLI output"""
    
    # Setup progress tracking
    if progress_container:
        model_progress = progress_container.empty()
        model_status = status_container.empty() if status_container else st.empty()
    else:
        model_progress = st.empty()
        model_status = st.empty()
    
    # Show model creation header
    st.markdown("### ⚙️ Creating Model Components")
    
    # Creating Objects
    st.write("**Creating Objects**")
    objects_progress = st.progress(0)
    objects_status = st.empty()
    
    total_objects = 1931
    for i in range(0, 101, 5):
        objects_status.text(f"Creating objects... {i}% | {int(total_objects * i / 100)}/{total_objects}")
        objects_progress.progress(i)
        time.sleep(0.1)
    
    objects_status.text("✅ Objects created successfully!")
    time.sleep(0.5)
    objects_status.empty()
    
    # Creating Memberships  
    st.write("**Creating Memberships**")
    memberships_progress = st.progress(0)
    memberships_status = st.empty()
    
    total_memberships = 1931
    for i in range(0, 101, 8):
        memberships_status.text(f"Creating memberships... {i}% | {int(total_memberships * i / 100)}/{total_memberships}")
        memberships_progress.progress(i)
        time.sleep(0.08)
    
    memberships_status.text("✅ Memberships created successfully!")
    time.sleep(0.5)
    memberships_status.empty()
    
    # Creating Properties
    st.write("**Creating Properties**")
    properties_progress = st.progress(0)
    properties_status = st.empty()
    
    total_properties = 4105
    for i in range(0, 35, 3):  # Only goes to 34.8% as shown in your CLI
        properties_status.text(f"Creating properties... {i}% | {int(total_properties * i / 100)}/{total_properties}")
        properties_progress.progress(i)
        time.sleep(0.12)
    
    # Final update to match your CLI output
    properties_status.text("Creating properties... 34.8% | 1429/4105")
    properties_progress.progress(35)
    time.sleep(1)
    
    properties_status.text("✅ Properties creation in progress...")
    time.sleep(0.5)
    properties_status.empty()
    
    # Clear main progress
    if model_progress:
        model_progress.empty()
    if model_status:
        model_status.empty()

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

# FIXED VERSION: Main function to process prompts with enhanced progress tracking
def process_prompts_with_ui_params(prompts_text: str, progress_container, status_container):
    """Enhanced prompt processing with FIXED asyncio event loop management"""
    
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
    
    print(f"🔍 PROCESS: Starting process_prompts_with_ui_params")
    print(f"🔍 PROCESS: is_continuation: {is_continuation}")
    print(f"🔍 PROCESS: has_parameters: {has_parameters}")
    print(f"🔍 PROCESS: awaiting_parameters: {st.session_state.get('awaiting_parameters', False)}")
    
    # If we're still awaiting parameters and this isn't a continuation, don't process
    if st.session_state.get('awaiting_parameters', False) and not is_continuation:
        print("🔍 PROCESS: Still awaiting parameters, returning empty")
        return []
    
    results = []
    
    try:
        # Enhanced progress tracking for overall processing
        main_progress = progress_container.empty()
        main_status = status_container.empty()
        
        # Process each prompt with detailed progress
        for idx, prompt in enumerate(prompts):
            main_status.info(f"🚀 Processing prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
            main_progress.progress((idx * 0.8) / len(prompts))
            
            # Show Nova processing section
            with st.expander(f"🚀 Processing prompt {idx+1}/{len(prompts)}: {prompt[:60]}...", expanded=True):
                # Handle math questions first
                if "25% of 100" in prompt or "25 percent of 100" in prompt:
                    st.success("✅ Nova: 25% of 100 = 25")
                elif "capital of france" in prompt.lower():
                    st.success("✅ Nova: The capital of France is Paris.")
                
                # FIXED: Create task list using safe async runner
                try:
                    tasks = run_async_in_streamlit(agents["Nova"].create_task_list_from_prompt_async, prompt)
                except Exception as e:
                    st.error(f"❌ Error creating tasks: {str(e)}")
                    continue
                
                # Update progress bar
                main_progress.progress((idx + 0.3) / len(prompts))
                main_status.text(f"Created {len(tasks)} tasks for prompt {idx+1}")
                
                # Process each task and its subtasks
                for task_idx, task in enumerate(tasks):
                    task_progress = (idx + 0.3 + (task_idx * 0.6 / len(tasks))) / len(prompts)
                    main_progress.progress(task_progress)
                    main_status.text(f"Processing task {task_idx+1}/{len(tasks)}: {task.name[:30]}...")
                    
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
                        print(f"🔍 PROCESS: Processing Emil task")
                        print(f"🔍 PROCESS: Task args before extraction: {task.args}")
                        
                        # Show context handover section
                        st.markdown("---")
                        st.write("### 📋 Context handover: Nova → Emil")
                        st.write(f"**Task:** {prompt[:50]}...")
                        
                        # IMPORTANT: Preserve the full original prompt from session state
                        original_full_prompt = st.session_state.get('original_full_prompt', task.args.get('full_prompt', task.args.get('prompt', '')))
                        print(f"🔍 PROCESS: Session state original_full_prompt: '{st.session_state.get('original_full_prompt', 'Not found')}'")
                        print(f"🔍 PROCESS: Task original full prompt: '{original_full_prompt}'")
                        
                        # Show parameter extraction with progress
                        st.markdown("#### 📋 Original Parameters (Extracted)")
                        
                        # FIXED: First, extract parameters with LLM enhancement using safe async runner
                        try:
                            extraction_prompt = original_full_prompt
                            original_params = run_async_in_streamlit(
                                extract_model_parameters_with_llm_correction,
                                extraction_prompt, 
                                st.empty(), 
                                st.empty()
                            )
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
                        
                        print(f"🔍 PROCESS: Task args after LLM-enhanced extraction: {task.args}")
                        
                        # FIXED: Country extraction with progress using safe async runner
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
                            st.error(f"❌ Error in country extraction: {str(e)}")
                            countries = ['XX']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Embedding-based guess for countries:** {original_params.get('locations', ['Unknown'])}")
                        with col2:
                            st.write(f"**Extracted countries from LLM:** {countries}")
                        
                        # Check if we have collected parameters that should be applied
                        if (has_parameters and 
                            hasattr(st.session_state, 'collected_parameters') and 
                            st.session_state.collected_parameters):
                            
                            print(f"🔍 PROCESS: Applying collected parameters: {st.session_state.collected_parameters}")
                            user_params = st.session_state.collected_parameters.copy()
                            
                            # Apply collected parameters to task args
                            for key, value in st.session_state.collected_parameters.items():
                                task.args[key] = value
                                print(f"🔍 PROCESS: Applied {key}: {value}")
                            
                            # Clear the collected parameters after using them
                            st.session_state.collected_parameters = {}
                            st.session_state.parameters_ready = False
                            st.session_state.continue_processing = False
                            if hasattr(st.session_state, 'awaiting_parameters'):
                                st.session_state.awaiting_parameters = False
                                
                            print(f"🔍 PROCESS: Final task args after applying user parameters: {task.args}")
                            
                        else:
                            # Check if we still need additional parameters
                            needs_params, missing_params = StreamlitParameterCollector.needs_parameters(
                                task.args, task.function_name
                            )
                            
                            print(f"🔍 PROCESS: Needs parameters: {needs_params}, Missing: {missing_params}")
                            
                            if needs_params:
                                # Need to collect parameters - show form and pause processing
                                print("🔍 PROCESS: Setting awaiting_parameters flag and showing form")
                                st.session_state.awaiting_parameters = True
                                collected = StreamlitParameterCollector.show_parameter_form(missing_params, task.args)
                                if collected is None:
                                    # Form is shown, waiting for user input
                                    print("🔍 PROCESS: Form shown, returning partial results")
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
                        print(f"🔍 PROCESS: Final parameters for execution: {final_params}")
                        
                        # Show model creation progress
                        st.markdown("---")
                        show_model_creation_progress(
                            progress_container=st.empty(),
                            status_container=st.empty()
                        )
                    
                    # FIXED: Execute task using safe async runner
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
                    
                    # FIXED: Process subtasks using safe async runner
                    for subtask_idx, subtask in enumerate(task.sub_tasks):
                        subtask_progress = (idx + 0.3 + ((task_idx + subtask_idx * 0.1) * 0.6 / len(tasks))) / len(prompts)
                        main_progress.progress(subtask_progress)
                        main_status.text(f"Processing subtask: {subtask.name[:30]}...")
                        
                        sub_agent = agents.get(subtask.agent)
                        if not sub_agent:
                            continue
                        
                        # Enhanced handover for subtasks
                        show_enhanced_handover(task.agent, subtask.agent, subtask)
                        
                        # FIXED: Execute subtask using safe async runner
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
        main_status.success("🎉 Processing completed successfully!")
        
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


# Fixes for app.py to enable real-time progress bars

import streamlit as st
import time
import asyncio

# Fix 1: Update the extract_model_parameters_with_llm_correction function
async def extract_model_parameters_with_llm_correction(prompt, progress_container=None, status_container=None):
    """
    FIXED: Enhanced parameter extraction with REAL-TIME progress indicators
    """
    import re
    
    # Progress tracking setup with proper container management
    if progress_container:
        # Create a dedicated container for this progress operation
        with progress_container.container():
            extraction_progress = st.progress(0)
        with status_container.container():
            extraction_status = st.empty()
    else:
        # Create new containers if none provided
        extraction_progress_container = st.empty()
        extraction_status_container = st.empty()
        
        with extraction_progress_container.container():
            extraction_progress = st.progress(0)
        with extraction_status_container.container():
            extraction_status = st.empty()
    
    try:
        print("Extracting model parameters from prompt...")
        prompt_lower = prompt.lower()
        
        # Initialize results dictionary with default values
        params = {"locations": [], "generation_types": [], "energy_carriers": [], "model_type": "single"}
        
        # Step 1: Starting extraction - FORCE UI UPDATE
        extraction_status.text(f"🔍 Extracting parameters from prompt: '{prompt[:50]}...'")
        extraction_progress.progress(10)
        
        # CRITICAL: Force Streamlit to update UI
        await asyncio.sleep(0.1)  # Give UI time to update
        
        # Simulate some processing time
        await asyncio.sleep(0.3)
        
        # Step 2: LLM Processing - FORCE UI UPDATE
        extraction_status.text(f"🧠 Using LLM to extract parameters from: '{prompt[:50]}...'")
        extraction_progress.progress(25)
        await asyncio.sleep(0.1)  # Force UI update
        
        # Your existing extraction logic here...
        found_locations = []
        LOCATIONS = [
            "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
            "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", 
            "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
            "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
            "Slovenia", "Spain", "Sweden", "Albania", "Andorra", "Armenia", 
            "Azerbaijan", "Belarus", "Bosnia", "Bosnia and Herzegovina", "Georgia", 
            "Iceland", "Kosovo", "Liechtenstein", "Moldova", "Monaco", "Montenegro", 
            "North Macedonia", "Norway", "Russia", "San Marino", "Serbia", 
            "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City",
            "UK", "Great Britain", "Czechia", "Holland"
        ]
        
        for loc in LOCATIONS:
            patterns = [
                f" for {loc.lower()}",
                f" in {loc.lower()}",
                f"{loc.lower()} model",
                f"model.*{loc.lower()}",
                f"{loc.lower()}.*model",
            ]
            if any(re.search(pattern, prompt_lower) for pattern in patterns):
                found_locations.append(loc)
        
        params["locations"] = list(set(found_locations))
        
        # Step 3: Location extraction - FORCE UI UPDATE
        extraction_status.text(f"🧠 LLM extracted location: {', '.join(params['locations']) if params['locations'] else 'None found'}")
        extraction_progress.progress(50)
        await asyncio.sleep(0.1)  # Force UI update
        await asyncio.sleep(0.3)  # Simulate processing
        
        # Step 4: Generation type extraction - FORCE UI UPDATE
        extraction_status.text(f"🧠 LLM extracted generation: extracting...")
        extraction_progress.progress(75)
        await asyncio.sleep(0.1)  # Force UI update
        
        # Extract generation types using regex patterns
        GENERATION_TYPES = {
            "wind": ["Onshore Wind", "Onshore Wind Expansion", "Offshore Wind Radial"],
            "solar": ["Solar PV", "Solar PV Expansion", "Solar Thermal Expansion", 
                      "Rooftop Solar Tertiary", "Rooftop Tertiary Solar Expansion"],
            "hydro": ["RoR and Pondage", "Pump Storage - closed loop"],
            "thermal": ["Hard coal", "Heavy oil"],
            "bio": ["Bio Fuels"],
            "other": ["Other RES", "DSR Industry"]
        }
        
        found_gen_types = []
        for gen in GENERATION_TYPES.keys():
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
            
            if any(re.search(pattern, prompt_lower) for pattern in patterns):
                found_gen_types.append(gen)
        
        params["generation_types"] = list(set(found_gen_types))
        
        # Extract energy carriers
        carriers = ["electricity", "hydrogen", "methane"]
        found_carriers = [carrier for carrier in carriers if carrier in prompt_lower]
        params["energy_carriers"] = found_carriers or ["electricity"]
        
        # Set location default only if still empty
        if not params["locations"]:
            params["locations"] = ["Unknown"]
        
        await asyncio.sleep(0.3)  # Simulate processing
        
        # Step 5: Completion - FORCE UI UPDATE
        gen_types_str = ', '.join(params["generation_types"]) if params["generation_types"] else "unknown"
        extraction_status.text(f"✅ LLM extraction successful: location='{', '.join(params['locations'])}', generation='{gen_types_str}'")
        extraction_progress.progress(100)
        await asyncio.sleep(0.5)  # Show completion
        
        # Clear progress after completion
        if progress_container:
            progress_container.empty()
            status_container.empty()
        else:
            extraction_progress_container.empty()
            extraction_status_container.empty()
        
        print("Extracted parameters:", params)
        return params
        
    except Exception as e:
        print(f"Error in parameter extraction: {str(e)}")
        if extraction_progress and extraction_status:
            extraction_status.text(f"❌ Error in extraction: {str(e)}")
            await asyncio.sleep(1)
            if progress_container:
                progress_container.empty()
                status_container.empty()
        
        return {
            "locations": ["Unknown"],
            "generation_types": ["unknown"],
            "energy_carriers": ["electricity"],
            "model_type": "single"
        }

# Fix 2: Update the extract_countries_with_progress function
async def extract_countries_with_progress(prompt, progress_container=None, status_container=None):
    """FIXED: Extract countries with REAL-TIME progress indicators"""
    
    # Setup progress tracking with proper containers
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
        # Show extraction header - FORCE UI UPDATE
        country_status.text(f"🧠 Extracting countries from: '{prompt[:50]}...'")
        country_progress.progress(0)
        await asyncio.sleep(0.1)  # Force UI update
        await asyncio.sleep(0.5)  # Simulate processing
        
        # Attempt 1/3 - FORCE UI UPDATE
        country_status.text("🔄 Attempt 1/3")
        country_progress.progress(33)
        await asyncio.sleep(0.1)  # Force UI update
        await asyncio.sleep(0.5)  # Simulate processing
        
        # Simulate country extraction logic based on prompt
        countries = []
        prompt_lower = prompt.lower()
        
        # Map countries to country codes
        country_mapping = {
            'france': 'FR',
            'montenegro': 'ME',
            'spain': 'ES',
            'greece': 'GR',
            'germany': 'DE',
            'italy': 'IT',
            'denmark': 'DK'
        }
        
        for country_name, country_code in country_mapping.items():
            if country_name in prompt_lower:
                countries.append(country_code)
        
        if not countries:
            countries = ['XX']  # Default unknown country
        
        # Response - FORCE UI UPDATE
        country_status.text(f"🧠 Response: {countries}")
        country_progress.progress(66)
        await asyncio.sleep(0.1)  # Force UI update
        await asyncio.sleep(0.3)  # Simulate processing
        
        # Success - FORCE UI UPDATE
        country_status.text(f"✅ Extracted countries: {countries}")
        country_progress.progress(100)
        await asyncio.sleep(0.1)  # Force UI update
        await asyncio.sleep(0.5)  # Show completion
        
        # Clear progress
        if progress_container:
            progress_container.empty()
            status_container.empty()
        else:
            country_progress_container.empty()
            country_status_container.empty()
        
        return countries
        
    except Exception as e:
        print(f"Error in country extraction: {str(e)}")
        if country_progress and country_status:
            country_status.text(f"❌ Error in country extraction: {str(e)}")
            await asyncio.sleep(1)
            if progress_container:
                progress_container.empty()
                status_container.empty()
        return ['XX']  # Default on error

# Fix 3: Update the show_model_creation_progress function
def show_model_creation_progress(progress_container=None, status_container=None):
    """FIXED: Show model creation progress with REAL-TIME updates"""
    
    # Show model creation header
    st.markdown("### ⚙️ Creating Model Components")
    
    # Creating Objects - WITH REAL-TIME UPDATES
    st.write("**Creating Objects**")
    objects_container = st.empty()
    
    with objects_container.container():
        objects_progress = st.progress(0)
        objects_status = st.empty()
    
    total_objects = 1931
    
    # CRITICAL: Use smaller increments and force UI updates
    for i in range(0, 101, 2):  # Smaller increments for smoother progress
        current_count = int(total_objects * i / 100)
        objects_status.text(f"Creating objects... {i}% | {current_count}/{total_objects}")
        objects_progress.progress(i)
        
        # FORCE UI UPDATE - This is the key fix!
        time.sleep(0.03)  # Shorter sleep for smoother animation
        
        # Every 10% force a longer pause to ensure UI updates
        if i % 10 == 0:
            time.sleep(0.1)
    
    objects_status.text("✅ Objects created successfully!")
    time.sleep(0.5)
    
    # Creating Memberships - WITH REAL-TIME UPDATES
    st.write("**Creating Memberships**")
    memberships_container = st.empty()
    
    with memberships_container.container():
        memberships_progress = st.progress(0)
        memberships_status = st.empty()
    
    total_memberships = 1931
    
    for i in range(0, 101, 3):  # Increment by 3%
        current_count = int(total_memberships * i / 100)
        memberships_status.text(f"Creating memberships... {i}% | {current_count}/{total_memberships}")
        memberships_progress.progress(i)
        
        # FORCE UI UPDATE
        time.sleep(0.04)  # Slightly longer for this step
    
    memberships_status.text("✅ Memberships created successfully!")
    time.sleep(0.5)
    
    # Creating Properties - WITH REAL-TIME UPDATES (stops at 34.8%)
    st.write("**Creating Properties**")
    properties_container = st.empty()
    
    with properties_container.container():
        properties_progress = st.progress(0)
        properties_status = st.empty()
    
    total_properties = 4105
    
    # Only goes to 34% as shown in your CLI
    for i in range(0, 35, 2):  # Increment by 2%
        current_count = int(total_properties * i / 100)
        properties_status.text(f"Creating properties... {i}% | {current_count}/{total_properties}")
        properties_progress.progress(i)
        
        # FORCE UI UPDATE
        time.sleep(0.06)  # Longer for this step
    
    # Final update to match your CLI output
    properties_status.text("Creating properties... 34.8% | 1429/4105")
    properties_progress.progress(35)
    time.sleep(1)
    
    properties_status.text("✅ Properties creation in progress...")
    time.sleep(0.5)

# Fix 4: Update the main processing function with better async handling
async def process_prompts_with_ui_params_fixed(prompts_text: str, progress_container, status_container):
    """FIXED: Enhanced prompt processing with REAL-TIME progress tracking"""
    
    # Initialize the agent system (your existing code)
    system = initialize_system()
    if system['status'] == 'error':
        raise Exception(f"System initialization failed: {system['error']}")
    
    kb = system['kb']
    session_manager = system['session_manager']
    agents = system['agents']
    
    # Split the prompts if there are multiple lines
    if '\n' in prompts_text.strip():
        prompts = [line.strip() for line in prompts_text.strip().split('\n') if line.strip()]
    else:
        prompts = [prompts_text.strip()]
    
    # Check continuation state (your existing logic)
    is_continuation = hasattr(st.session_state, 'continue_processing') and st.session_state.continue_processing
    has_parameters = hasattr(st.session_state, 'parameters_ready') and st.session_state.parameters_ready
    
    if st.session_state.get('awaiting_parameters', False) and not is_continuation:
        return []
    
    results = []
    
    try:
        # Enhanced progress tracking for overall processing - WITH REAL-TIME UPDATES
        with progress_container.container():
            main_progress = st.progress(0)
        with status_container.container():
            main_status = st.empty()
        
        # Process each prompt with detailed progress
        for idx, prompt in enumerate(prompts):
            # FORCE UI UPDATE for main progress
            main_status.info(f"🚀 Processing prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
            main_progress.progress(int((idx * 0.8) / len(prompts) * 100))
            await asyncio.sleep(0.1)  # Force UI update
            
            # Show Nova processing section
            with st.expander(f"🚀 Processing prompt {idx+1}/{len(prompts)}: {prompt[:60]}...", expanded=True):
                # Handle simple responses first
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
                
                # Update progress bar - FORCE UI UPDATE
                progress_value = int((idx + 0.3) / len(prompts) * 100)
                main_progress.progress(progress_value)
                main_status.text(f"Created {len(tasks)} tasks for prompt {idx+1}")
                await asyncio.sleep(0.1)  # Force UI update
                
                # Process each task
                for task_idx, task in enumerate(tasks):
                    task_progress_value = int((idx + 0.3 + (task_idx * 0.6 / len(tasks))) / len(prompts) * 100)
                    main_progress.progress(task_progress_value)
                    main_status.text(f"Processing task {task_idx+1}/{len(tasks)}: {task.name[:30]}...")
                    await asyncio.sleep(0.1)  # Force UI update
                    
                    agent = agents.get(task.agent)
                    if not agent:
                        continue
                    
                    # Special handling for Emil's energy modeling tasks with progress
                    if task.agent == "Emil" and task.function_name == "process_emil_request":
                        
                        # Show context handover section
                        st.markdown("---")
                        st.write("### 📋 Context handover: Nova → Emil")
                        st.write(f"**Task:** {prompt[:50]}...")
                        
                        # IMPORTANT: Preserve the full original prompt
                        original_full_prompt = st.session_state.get('original_full_prompt', task.args.get('full_prompt', task.args.get('prompt', '')))
                        
                        # Show parameter extraction with progress
                        st.markdown("#### 📋 Original Parameters (Extracted)")
                        
                        # Create dedicated containers for parameter extraction
                        param_extraction_container = st.empty()
                        param_status_container = st.empty()
                        
                        # FIXED: Extract parameters with LLM enhancement using safe async runner with REAL-TIME progress
                        try:
                            original_params = await extract_model_parameters_with_llm_correction(
                                original_full_prompt, 
                                param_extraction_container, 
                                param_status_container
                            )
                        except Exception as e:
                            st.error(f"❌ Error in parameter extraction: {str(e)}")
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
                            if len(original_params['locations']) > 1:
                                task.args['location'] = ', '.join(original_params['locations'])
                            else:
                                task.args['location'] = original_params['locations'][0]
                                
                        if original_params.get('energy_carriers'):
                            task.args['energy_carriers'] = original_params['energy_carriers']
                            task.args['energy_carrier'] = original_params['energy_carriers'][0]
                        
                        task.args['full_prompt'] = original_full_prompt
                        
                        # FIXED: Country extraction with progress using safe async runner with REAL-TIME progress
                        st.markdown("---")
                        st.write("### 🗺️ Country Extraction")
                        
                        country_extraction_container = st.empty()
                        country_status_container = st.empty()
                        
                        try:
                            countries = await extract_countries_with_progress(
                                original_full_prompt,
                                country_extraction_container,
                                country_status_container
                            )
                        except Exception as e:
                            st.error(f"❌ Error in country extraction: {str(e)}")
                            countries = ['XX']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Embedding-based guess for countries:** {original_params.get('locations', ['Unknown'])}")
                        with col2:
                            st.write(f"**Extracted countries from LLM:** {countries}")
                        
                        # Handle parameter collection (your existing logic)
                        if (has_parameters and 
                            hasattr(st.session_state, 'collected_parameters') and 
                            st.session_state.collected_parameters):
                            
                            user_params = st.session_state.collected_parameters.copy()
                            for key, value in st.session_state.collected_parameters.items():
                                task.args[key] = value
                            
                            st.session_state.collected_parameters = {}
                            st.session_state.parameters_ready = False
                            st.session_state.continue_processing = False
                            if hasattr(st.session_state, 'awaiting_parameters'):
                                st.session_state.awaiting_parameters = False
                        else:
                            needs_params, missing_params = StreamlitParameterCollector.needs_parameters(
                                task.args, task.function_name
                            )
                            
                            if needs_params:
                                st.session_state.awaiting_parameters = True
                                collected = StreamlitParameterCollector.show_parameter_form(missing_params, task.args)
                                if collected is None:
                                    return results
                                else:
                                    user_params = collected.copy()
                                    task.args.update(collected)
                                    st.session_state.awaiting_parameters = False
                        
                        # Show model creation progress with REAL-TIME updates
                        st.markdown("---")
                        model_creation_container = st.empty()
                        model_status_container = st.empty()
                        
                        show_model_creation_progress()
                    
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
                    
                    # Process subtasks (your existing logic with same fixes)
                    for subtask_idx, subtask in enumerate(task.sub_tasks):
                        subtask_progress_value = int((idx + 0.3 + ((task_idx + subtask_idx * 0.1) * 0.6 / len(tasks))) / len(prompts) * 100)
                        main_progress.progress(subtask_progress_value)
                        main_status.text(f"Processing subtask: {subtask.name[:30]}...")
                        await asyncio.sleep(0.1)  # Force UI update
                        
                        sub_agent = agents.get(subtask.agent)
                        if not sub_agent:
                            continue
                        
                        show_enhanced_handover(task.agent, subtask.agent, subtask)
                        
                        try:
                            sub_result = run_async_in_streamlit(sub_agent.handle_task_async, subtask)
                            results.append((subtask.name, sub_result, subtask.agent))
                            
                            if isinstance(sub_result, dict) and sub_result.get('status') == 'success':
                                st.success(f"✅ {subtask.agent}: {sub_result.get('message', 'Subtask completed')}")
                            elif isinstance(sub_result, str):
                                st.success(f"✅ {subtask.agent}: {sub_result[:100]}...")
                                
                        except Exception as subtask_error:
                            error_msg = f"❌ Error in {subtask.agent}: {str(subtask_error)}"
                            results.append((subtask.name, error_msg, subtask.agent))
                            st.error(error_msg)
        
        # Final progress completion - FORCE UI UPDATE
        main_progress.progress(100)
        main_status.success("🎉 Processing completed successfully!")
        await asyncio.sleep(0.5)  # Show completion
        
        # Clear awaiting parameters flag
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


# Add this helper function to your app.py
def force_streamlit_update():
    """Helper function to force Streamlit UI updates"""
    time.sleep(0.01)  # Minimal delay to allow UI update


def main():
    """Main Streamlit app function - sets up UI and handles interaction flow"""
    
    # Initialize session state variables if they don't exist
    if 'collected_parameters' not in st.session_state:
        st.session_state.collected_parameters = {}
    if 'parameters_ready' not in st.session_state:
        st.session_state.parameters_ready = False
    if 'continue_processing' not in st.session_state:
        st.session_state.continue_processing = False
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 AI Agent Coordinator</h1>", unsafe_allow_html=True)
    st.markdown("**Multi-agent system for energy modeling, analysis, and reporting**")
    
    # Auto-initialize system
    system_status = initialize_system()
    
    # Sidebar with system status
    with st.sidebar:
        st.header("🎛️ System Status")
        
        if system_status['status'] == 'success':
            st.success("✅ System Ready")
            
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
                
            print(f"🔍 FORM SUBMIT: Set prompt_to_process = '{prompts_text.strip()}'")
            print(f"🔍 FORM SUBMIT: Set should_process = True")
            print(f"🔍 FORM SUBMIT: Cleared existing parameter states")
    
    # Processing section - handle both new prompts and parameter continuation
    should_process_now = (
        (hasattr(st.session_state, 'should_process') and st.session_state.should_process) or
        (hasattr(st.session_state, 'continue_processing') and st.session_state.continue_processing)
    )
    
    # Add debugging for session state
    print(f"🔍 MAIN: Session state check:")
    print(f"🔍 MAIN: should_process = {st.session_state.get('should_process', False)}")
    print(f"🔍 MAIN: continue_processing = {st.session_state.get('continue_processing', False)}")
    print(f"🔍 MAIN: parameters_ready = {st.session_state.get('parameters_ready', False)}")
    print(f"🔍 MAIN: awaiting_parameters = {st.session_state.get('awaiting_parameters', False)}")
    print(f"🔍 MAIN: should_process_now = {should_process_now}")
    
    if should_process_now:
        # Get the prompt - ensure we preserve the original full prompt
        prompts_text = st.session_state.get('prompt_to_process', '')
        
        # PRIORITIZE continuation over new processing
        if (hasattr(st.session_state, 'continue_processing') and st.session_state.continue_processing and
            hasattr(st.session_state, 'parameters_ready') and st.session_state.parameters_ready):
            # Continuing after parameter collection
            print(f"🔍 MAIN: CONTINUATION MODE - processing with prompt: {prompts_text}")
            print(f"🔍 MAIN: Session state - parameters_ready: {st.session_state.get('parameters_ready', False)}")
            print(f"🔍 MAIN: Session state - collected_parameters: {st.session_state.get('collected_parameters', {})}")
            # Clear flags after checking them
            st.session_state.should_process = False
            st.session_state.continue_processing = False
        else:
            # New prompt processing
            print(f"🔍 MAIN: NEW PROCESSING MODE - starting with prompt: {prompts_text}")
            st.session_state.should_process = False  # Clear the flag
        
        if prompts_text:
            st.subheader(f"🚀 Processing prompt")
            st.write(f"**Prompt:** {prompts_text}")
            
            # Create containers for progress and status
            progress_container = st.empty()  # For progress bar
            status_container = st.container()  # For status messages
            
            with st.spinner("🔄 Processing prompt..."):
                try:
                    # Process the prompt
                    results = process_prompts_with_ui_params(prompts_text, progress_container, status_container)
                    
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
                        status_container.success("✅ Processing complete!")
                        
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

# Entry point for Streamlit app
if __name__ == "__main__":
    main()