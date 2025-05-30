import asyncio
import streamlit as st
from utils.function_logger import log_function_call

@log_function_call
async def get_missing_parameters_simple_async(function_name: str, missing_params: list, initial_args: dict = None) -> dict:
    """
    Streamlit-compatible version of a simplified parameter collection routine.
    Uses Streamlit session state to store and retrieve pending parameters.

    Parameters:
        function_name (str): The function needing parameters
        missing_params (list): Missing parameter names
        initial_args (dict, optional): Any pre-filled args

    Returns:
        dict: Collected parameters or empty dict to signal parameter collection is pending
    """
    collected_args = initial_args.copy() if initial_args else {}

    print(f"Nova needs input for {function_name}...")

    # Define parameter descriptions and examples for UI display
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

    # Check if we have collected parameters in session state
    if 'collected_parameters' in st.session_state and st.session_state.collected_parameters:
        # Use the collected parameters from previous UI interaction
        for param in missing_params:
            if param in st.session_state.collected_parameters:
                collected_args[param] = st.session_state.collected_parameters[param]
        
        # Clear the collected parameters to avoid reusing them
        st.session_state.collected_parameters = {}
        
        # If all parameters are collected, return them
        if all(param in collected_args for param in missing_params):
            return collected_args
    
    # Store the pending parameters in session state for UI to display
    if 'pending_parameters' not in st.session_state:
        st.session_state.pending_parameters = {}
    
    # Store what parameters we need to collect
    st.session_state.pending_parameters = {
        'function': function_name,
        'missing': missing_params,
        'descriptions': {param: param_descriptions.get(param, f"The {param} input required") for param in missing_params},
        'examples': {param: param_examples.get(param, "No examples available") for param in missing_params},
        'initial_args': collected_args
    }
    
    # For debugging to console
    for param in missing_params:
        description = param_descriptions.get(param, f"The {param} input required")
        examples = param_examples.get(param, "No examples available")
        print(f"\nNova: I need the '{param}' for this task.")
        print(f"Description: {description}")
        print(f"Examples: {examples}")
    
    # Return empty dict to signal that parameters are pending collection via UI
    # The main app flow should detect this and pause for parameter collection
    return {}