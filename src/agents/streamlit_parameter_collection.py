# src/agents/streamlit_parameter_collection.py
import streamlit as st
import asyncio
from utils.function_logger import log_function_call

@log_function_call
async def get_missing_parameters_streamlit_async(function_name: str, missing_params: list, initial_args: dict = None) -> dict:
    """
    Streamlit-specific parameter collection that properly integrates with the UI.
    This version uses Streamlit session state to handle parameter collection.

    Parameters:
        function_name (str): The function needing parameters
        missing_params (list): Missing parameter names
        initial_args (dict, optional): Any pre-filled args

    Returns:
        dict: Collected parameters or special flags
    """
    collected_args = initial_args.copy() if initial_args else {}

    print(f"Nova needs input for {function_name}...")

    # Parameter descriptions and examples for UI display
    param_descriptions = {
        "location": "The geographic location for the energy model (e.g., UK, France, Spain, etc.)",
        "generation": "The generation type (e.g., solar, wind, hydro, thermal, bio)",
        "energy_carrier": "Energy carrier to model (e.g., electricity, hydrogen, methane)",
        "prompt": "Detailed prompt describing the task",
        "scenario_name": "The scenario name",
        "analysis_type": "Type of analysis: basic, detailed, or comprehensive",
        "style": "Report style (executive_summary, technical_report, etc.)"
    }

    param_examples = {
        "location": "UK, France, Germany, or 'all'",
        "generation": "solar, wind, hydro, etc.",
        "energy_carrier": "electricity (default), hydrogen, methane",
        "prompt": "e.g. build an energy model for Spain",
        "scenario_name": "baseline_2025, high_RE_2030",
        "analysis_type": "basic, detailed, comprehensive",
        "style": "executive_summary, presentation"
    }

    # Store what parameters we need to collect in session state
    if 'pending_parameters' not in st.session_state:
        st.session_state.pending_parameters = {}

    # Check if we already have collected parameters for this request
    if 'collected_parameters' in st.session_state and st.session_state.collected_parameters:
        # Use the collected parameters from UI
        for param in missing_params:
            if param in st.session_state.collected_parameters:
                collected_args[param] = st.session_state.collected_parameters[param]
        
        # Clear the collected parameters to avoid reusing them
        st.session_state.collected_parameters = {}
        
        # If all parameters are collected, return them
        if all(param in collected_args for param in missing_params):
            return collected_args

    # Set up pending parameters for the UI to display
    st.session_state.pending_parameters = {
        'function': function_name,
        'missing': missing_params,
        'descriptions': {param: param_descriptions.get(param, f"The {param} input required") for param in missing_params},
        'examples': {param: param_examples.get(param, "No examples available") for param in missing_params},
        'initial_args': collected_args
    }

    # Signal to the main app that we need parameter collection
    st.session_state.parameter_collection_needed = True

    # Return a special flag to indicate parameters are being collected via UI
    return {"__STREAMLIT_PARAMETER_COLLECTION__": True}


# Also need to update the main parameter_collection.py to use this for Streamlit
# src/agents/parameter_collection.py - UPDATED VERSION

import asyncio
import importlib.util
from utils.function_logger import log_function_call

# Check if streamlit is available
streamlit_available = False
try:
    import streamlit as st
    streamlit_available = True
except ImportError:
    streamlit_available = False

# Import the CLI version which has no external dependencies
from .cli_parameter_collection import get_missing_parameters_cli_async

@log_function_call
async def get_missing_parameters_async(function_name: str, missing_params: list, initial_args: dict = None, force_cli: bool = False) -> dict:
    """
    Unified parameter collection function that chooses the appropriate implementation
    based on environment capabilities.
    
    Parameters:
        function_name (str): The function needing parameters
        missing_params (list): Missing parameter names
        initial_args (dict, optional): Any pre-filled args
        force_cli (bool): Force using CLI version even if Streamlit is available
        
    Returns:
        dict: Collected parameters
    """
    # Check if we're in a Streamlit environment
    if streamlit_available and not force_cli:
        try:
            # Import the Streamlit version
            from .streamlit_parameter_collection import get_missing_parameters_streamlit_async
            print("Using Streamlit for parameter collection")
            return await get_missing_parameters_streamlit_async(function_name, missing_params, initial_args)
        except Exception as e:
            print(f"Streamlit parameter collection failed: {e}, falling back to CLI")
            return await get_missing_parameters_cli_async(function_name, missing_params, initial_args)
    else:
        print("Using CLI for parameter collection")
        return await get_missing_parameters_cli_async(function_name, missing_params, initial_args)