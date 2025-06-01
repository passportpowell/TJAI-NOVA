import asyncio
from utils.function_logger import log_function_call

@log_function_call
async def get_missing_parameters_cli_async(function_name: str, missing_params: list, initial_args: dict = None) -> dict:
    """
    Command-line version of parameter collection routine.
    No UI dependencies - works in any terminal environment.

    Parameters:
        function_name (str): The function needing parameters
        missing_params (list): Missing parameter names
        initial_args (dict, optional): Any pre-filled args

    Returns:
        dict: Collected parameters
    """
    collected_args = initial_args.copy() if initial_args else {}

    print(f"\n===== Nova needs input for {function_name} =====")

    # Define parameter descriptions and examples
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

    # Collect each missing parameter from the user via CLI
    for param in missing_params:
        description = param_descriptions.get(param, f"The {param} input required")
        examples = param_examples.get(param, "No examples available")
        
        print(f"\nNova: I need the '{param}' for this task.")
        print(f"Description: {description}")
        print(f"Examples: {examples}")
        
        # Get user input
        value = input(f"Please enter {param}: ")
        collected_args[param] = value.strip()
    
    print("\n===== Parameters collected successfully =====")
    return collected_args


# agents/simplified_parameter_collection.py
import streamlit as st
from utils.function_logger import log_function_call # Assuming this path is correct from your project
import asyncio

@log_function_call
async def get_missing_parameters_simple_async(function_name: str, missing_params: list, initial_args: dict = None) -> dict:
    """
    Streamlit version of parameter collection routine.
    Uses Streamlit widgets for user input.
    """
    if 'collected_params_form_data' not in st.session_state:
        st.session_state.collected_params_form_data = {}
    
    # Use a unique key for the form based on the function and missing params
    form_key = f"form_{function_name}_{'_'.join(sorted(missing_params))}"

    with st.form(key=form_key):
        st.subheader(f"Additional Information Needed for: {function_name}")
        st.write(f"The task '{function_name}' requires the following details:")

        current_values = initial_args.copy() if initial_args else {}

        # Descriptions and examples (enhance as needed)
        param_details = {
            "location": ("Location for the model (e.g., Spain, Greece, UK)", "Spain, Greece"),
            "generation": ("Generation type (e.g., solar, wind, hydro)", "solar"),
            "energy_carrier": ("Energy carrier (e.g., electricity, hydrogen)", "electricity"),
            "prompt": ("Detailed prompt describing the task", "build an energy model for Spain"),
            # Add more known parameters here
        }

        for param in missing_params:
            if param not in current_values or not current_values[param]: # Only ask if not already filled
                default_desc, default_ex = f"Enter {param}", ""
                description, example = param_details.get(param, (default_desc, default_ex))
                help_text = f"{description}. Example: '{example}'" if example else description
                
                # Use a unique key for each text_input
                input_key = f"input_{function_name}_{param}"
                current_values[param] = st.text_input(
                    f"{param.replace('_', ' ').capitalize()}:",
                    value=current_values.get(param, ""),
                    help=help_text,
                    key=input_key
                )
        
        submit_button = st.form_submit_button("Submit Parameters")

    if submit_button:
        # Store submitted values in session state to be picked up by the backend
        st.session_state.collected_params_form_data = {k: v for k, v in current_values.items() if k in missing_params and v}
        # Clear the flag that indicates we are waiting for these specific parameters
        if 'waiting_for_user_params' in st.session_state and \
           st.session_state.waiting_for_user_params == (function_name, missing_params):
            del st.session_state['waiting_for_user_params']
        st.rerun() # Rerun to allow the main logic to pick up the parameters
        return {} # Return empty, actual values are in session_state

    # If the form is displayed, we are waiting for input. Halt further backend execution.
    # This tells Streamlit to stop the script here and wait for form submission.
    st.stop()
    return {} # Should not be reached due to st.stop()

