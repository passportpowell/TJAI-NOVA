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