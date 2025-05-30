import streamlit as st
import asyncio
import os
import json
import datetime
import sys
from typing import Dict, Any, List

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import your existing modules
from agents import Nova, Emil, Ivan, Lola
from core.knowledge_base import KnowledgeBase
from core.session_manager import SessionManager
from core.functions_registery import *
from utils.csv_function_mapper import FunctionMapLoader
from utils.do_maths import do_maths
from utils.general_knowledge import answer_general_question
from utils.open_ai_utils import (
    ai_chat_session,
    ai_spoken_chat_session,
    run_open_ai_ns_async,
    open_ai_categorisation_async
)

# Page config
st.set_page_config(
    page_title="AI Agent Coordinator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Import necessary constants for enhanced parameter extraction
GENERATION_TYPES = {
    "wind": ["Onshore Wind", "Onshore Wind Expansion", "Offshore Wind Radial"],
    "solar": ["Solar PV", "Solar PV Expansion", "Solar Thermal Expansion", 
              "Rooftop Solar Tertiary", "Rooftop Tertiary Solar Expansion"],
    "hydro": ["RoR and Pondage", "Pump Storage - closed loop"],
    "thermal": ["Hard coal", "Heavy oil"],
    "bio": ["Bio Fuels"],
    "other": ["Other RES", "DSR Industry"]
}

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

# Enhanced parameter extraction functions
async def extract_model_parameters_with_llm_correction(prompt):
    """
    Enhanced parameter extraction that uses LLM to correct misspelled locations
    and find parameters that hardcoded lists might miss.
    """
    import re
    print("Extracting model parameters from prompt...")
    prompt_lower = prompt.lower()
    params = {"locations": [], "generation_types": [], "energy_carriers": [], "model_type": "single"}
    
    # Step 1: Try hardcoded location matching first
    found_locations = []
    for loc in LOCATIONS:
        patterns = [
            f" for {loc.lower()}",
            f" in {loc.lower()}", 
            f" {loc.lower()} ",
            f"model {loc.lower()}",
            f"{loc.lower()} model",
            f" {loc.lower()} and",
            f"and {loc.lower()}",
            f", {loc.lower()}",
            f"{loc.lower()},"
        ]
        if any(pattern in prompt_lower for pattern in patterns):
            found_locations.append(loc)
    
    params["locations"] = list(set(found_locations))
    
    # Step 2: If no locations found with hardcoded matching, use LLM
    if not params["locations"]:
        print("🔍 No locations found with hardcoded matching, trying LLM correction...")
        
        # Enhanced LLM extraction for multiple locations
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
    
    # Step 3: Extract generation types (enhanced patterns)
    found_gen_types = []
    for gen in GENERATION_TYPES.keys():
        patterns = [
            f"build.*{gen}.*model",
            f"create.*{gen}.*model", 
            f"{gen}.*model.*for",
            f"make.*{gen}.*model",
            f"{gen} power",
            f"{gen} generation",
            f"{gen} energy",
            f"a {gen} model",
            f"build {gen}",
            f"create {gen}"
        ]
        
        if any(re.search(pattern, prompt_lower) for pattern in patterns):
            found_gen_types.append(gen)
    
    params["generation_types"] = list(set(found_gen_types))
    
    # Step 4: If no generation types found, try LLM extraction
    if not params["generation_types"]:
        print("🔍 No generation types found with pattern matching, trying LLM extraction...")
        
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
    
    # Step 5: Extract energy carriers (existing logic)
    carriers = ["electricity", "hydrogen", "methane"]
    found_carriers = [carrier for carrier in carriers if carrier in prompt_lower]
    params["energy_carriers"] = found_carriers or ["electricity"]
    
    # Set location default only if still empty
    if not params["locations"]:
        params["locations"] = ["Unknown"]
    
    print("Extracted parameters:", params)
    return params

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
        
        kb = KnowledgeBase(storage_path=kb_path, use_persistence=True)
        session_manager = SessionManager(base_path=sessions_path)
        
        # Check for existing active session (exactly like main.py)
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
                else:
                    session_id, session_file = session_manager.create_session()
                    kb.set_item("current_session", session_id)
                    kb.set_item("current_session_file", session_file)
                    
            except Exception as e:
                session_id, session_file = session_manager.create_session()
                kb.set_item("current_session", session_id)
                kb.set_item("current_session_file", session_file)
        else:
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

        # Initialize function loader and agents (exactly like main.py)
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

        nova_functions = function_loader.load_function_map("Nova") or {}
        nova_functions.setdefault("answer_general_question", answer_general_question)
        nova_functions.setdefault("do_maths", do_maths)
        
        nova = Nova("Nova", kb, nova_functions)
        emil = Emil("Emil", kb, function_loader.load_function_map("Emil") or EMIL_FUNCTIONS)
        ivan = Ivan("Ivan", kb, function_loader.load_function_map("Ivan") or IVAN_FUNCTIONS)
        lola = Lola("Lola", kb, function_loader.load_function_map("Lola") or LOLA_FUNCTIONS)
        
        agents = {"Nova": nova, "Emil": emil, "Ivan": ivan, "Lola": lola}
        
        return {
            'kb': kb,
            'session_manager': session_manager,
            'agents': agents,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

class StreamlitParameterCollector:
    """Custom parameter collector for Streamlit that works with your existing agent system"""
    
    @staticmethod
    def needs_parameters(task_args, function_name):
        """Check if Emil needs additional parameters"""
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
        """Show parameter collection form in Streamlit"""
        st.info("🤖 **I need some additional information to complete your request:**")
        
        collected_params = {}
        
        # Use a unique key with timestamp to avoid form collision
        form_key = f"parameter_collection_form_{id(task_args)}"
        
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
                
            if 'location' in missing_params:
                st.markdown("**Location** - Which country/region should the model be for?")
                collected_params['location'] = st.text_input(
                    "Enter location(s):",
                    placeholder="e.g., Denmark, Spain, France (you can enter multiple locations separated by commas)",
                    help="Enter one or more countries/regions for your energy model"
                )
                    
            if 'energy_carrier' in missing_params:
                st.markdown("**Energy Carrier** - What type of energy carrier?")
                carrier_options = ['electricity', 'hydrogen', 'methane']
                collected_params['energy_carrier'] = st.selectbox(
                    "Select energy carrier:",
                    options=carrier_options,
                    help="Choose the energy carrier for your model"
                )
                
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
                    st.rerun()
                    
        # If we get here, either the form hasn't been submitted yet or the rerun failed
        return None




def show_enhanced_handover(from_agent, to_agent, task, original_params=None, user_params=None, final_params=None):
    """Display enhanced context handover with detailed parameter information"""
    
    # Basic handover info
    st.markdown(f"""
    <div class="context-handover">
    📋 <strong>Context handover:</strong> {from_agent} → {to_agent}<br>
    <strong>Task:</strong> {task.args.get('prompt', '')[:50]}...
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced parameter details
    if to_agent == "Emil" and (original_params or user_params or final_params):
        
        if original_params:
            st.markdown(f"""
            <div class="parameter-detail">
            <strong>📋 Original Parameters (Extracted):</strong><br>
            • Locations: {original_params.get('locations', ['None'])}<br>
            • Generation Types: {original_params.get('generation_types', ['None'])}<br>
            • Energy Carriers: {original_params.get('energy_carriers', ['None'])}
            </div>
            """, unsafe_allow_html=True)
        
        if user_params:
            st.markdown(f"""
            <div class="parameter-detail">
            <strong>👤 User-Added Parameters:</strong><br>
            • Generation: {user_params.get('generation', 'Not provided')}<br>
            • Location: {user_params.get('location', 'Not provided')}<br>
            • Energy Carrier: {user_params.get('energy_carrier', 'Not provided')}
            </div>
            """, unsafe_allow_html=True)
        
        if final_params:
            st.markdown(f"""
            <div class="parameter-detail">
            <strong>✅ Final Parameters (Used):</strong><br>
            • Location: {final_params.get('location', 'Unknown')}<br>
            • Generation: {final_params.get('generation', 'Unknown')}<br>
            • Energy Carrier: {final_params.get('energy_carrier', 'Unknown')}
            </div>
            """, unsafe_allow_html=True)

def process_prompts_with_ui_params(prompts_text: str, progress_container, status_container):
    """Enhanced prompt processing with better task detection for reports and detailed handovers"""
    system = initialize_system()
    if system['status'] == 'error':
        raise Exception(f"System initialization failed: {system['error']}")
    
    kb = system['kb']
    session_manager = system['session_manager']
    agents = system['agents']
    
    # Split the prompts
    if '\n' in prompts_text.strip():
        prompts = [line.strip() for line in prompts_text.strip().split('\n') if line.strip()]
    else:
        prompts = [prompts_text.strip()]
    
    # Check if we're in a continuation state
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
        # Process synchronously to avoid async issues with Streamlit
        for idx, prompt in enumerate(prompts):
            status_container.info(f"🚀 Processing prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
            
        # Process synchronously to avoid async issues with Streamlit
        for idx, prompt in enumerate(prompts):
            status_container.info(f"🚀 Processing prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Create task list using asyncio.run for each prompt
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Let Nova handle task creation with enhanced LLM prompting
                tasks = loop.run_until_complete(agents["Nova"].create_task_list_from_prompt_async(prompt))
            finally:
                loop.close()
            
            progress_container.progress((idx + 0.3) / len(prompts), f"Created {len(tasks)} tasks for prompt {idx+1}")
            
            # Process each task and its subtasks
            for task_idx, task in enumerate(tasks):
                progress_container.progress(
                    (idx + 0.3 + (task_idx * 0.6 / len(tasks))) / len(prompts), 
                    f"Processing task {task_idx+1}/{len(tasks)}: {task.name[:30]}..."
                )
                
                agent = agents.get(task.agent)
                if not agent:
                    continue
                
                # Parameter handling for Emil tasks
                original_params = None
                user_params = None
                final_params = None
                
                if task.agent == "Emil" and task.function_name == "process_emil_request":
                    print(f"🔍 PROCESS: Processing Emil task")
                    print(f"🔍 PROCESS: Task args before extraction: {task.args}")
                    
                    # IMPORTANT: Preserve the full original prompt from session state
                    original_full_prompt = st.session_state.get('original_full_prompt', task.args.get('full_prompt', task.args.get('prompt', '')))
                    print(f"🔍 PROCESS: Session state original_full_prompt: '{st.session_state.get('original_full_prompt', 'Not found')}'")
                    print(f"🔍 PROCESS: Task original full prompt: '{original_full_prompt}'")
                    
                    # First, extract parameters with LLM enhancement using the FULL ORIGINAL prompt
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # Use the full original prompt from session state for parameter extraction
                        extraction_prompt = original_full_prompt
                        original_params = loop.run_until_complete(
                            extract_model_parameters_with_llm_correction(extraction_prompt)
                        )
                    finally:
                        loop.close()
                    
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
                
                # Show enhanced context handover with detailed parameters
                if task.agent != "Nova":
                    show_enhanced_handover("Nova", task.agent, task, original_params, user_params, final_params)
                
                # Execute task
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(agent.handle_task_async(task))
                    finally:
                        loop.close()
                        
                    results.append((task.name, result, task.agent))
                    
                    # Show success
                    if isinstance(result, dict) and result.get('status') == 'success':
                        status_container.success(f"✅ {task.agent}: {result.get('message', 'Task completed')}")
                    elif isinstance(result, str):
                        status_container.success(f"✅ {task.agent}: {result[:100]}...")
                        
                except Exception as task_error:
                    error_msg = f"❌ Error in {task.agent}: {str(task_error)}"
                    results.append((task.name, error_msg, task.agent))
                    status_container.error(error_msg)
                
                
                # **CRITICAL FIX**: Process subtasks (this is where reports are handled!)
                for subtask_idx, subtask in enumerate(task.sub_tasks):
                    progress_container.progress(
                        (idx + 0.3 + ((task_idx + subtask_idx * 0.1) * 0.6 / len(tasks))) / len(prompts), 
                        f"Processing subtask: {subtask.name[:30]}..."
                    )
                    
                    sub_agent = agents.get(subtask.agent)
                    if not sub_agent:
                        continue
                    
                    # Enhanced handover for subtasks
                    show_enhanced_handover(task.agent, subtask.agent, subtask)
                    
                    # Execute subtask
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            sub_result = loop.run_until_complete(sub_agent.handle_task_async(subtask))
                        finally:
                            loop.close()
                            
                        results.append((subtask.name, sub_result, subtask.agent))
                        
                        # Show success for subtask
                        if isinstance(sub_result, dict) and sub_result.get('status') == 'success':
                            status_container.success(f"✅ {subtask.agent}: {sub_result.get('message', 'Subtask completed')}")
                        elif isinstance(sub_result, str):
                            status_container.success(f"✅ {subtask.agent}: {sub_result[:100]}...")
                            
                    except Exception as subtask_error:
                        error_msg = f"❌ Error in {subtask.agent}: {str(subtask_error)}"
                        results.append((subtask.name, error_msg, subtask.agent))
                        status_container.error(error_msg)
        
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




def display_results(results: List[tuple]):
    """Display results in the same format as main.py"""
    st.subheader("Results:")
    st.markdown("********************")
    
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
    """Main Streamlit app"""
    
    # Initialize session state
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
            progress_container = st.empty()
            status_container = st.container()
            
            with st.spinner("🔄 Processing prompt..."):
                try:
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

if __name__ == "__main__":
    main()