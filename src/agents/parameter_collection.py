import asyncio
import importlib.util
from utils.function_logger import log_function_call

# Check if streamlit is available
streamlit_available = importlib.util.find_spec("streamlit") is not None

# Import the appropriate module based on availability
if streamlit_available:
    try:
        from .simplified_parameter_collection import get_missing_parameters_simple_async as st_get_params
    except ImportError:
        streamlit_available = False

# Import the CLI version which has no external dependencies
from .cli_parameter_collection import get_missing_parameters_cli_async


@log_function_call
async def get_missing_parameters_async(function_name: str, missing_params: list, initial_args: dict = None, force_cli: bool = True) -> dict:
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
    import os
    param_mode = os.environ.get("NOVA_PARAM_MODE", "auto").lower()

    print("\n🔧 PARAMETER COLLECTION MODE:")

    if param_mode == "cli":
        force_cli = True
        print("🖥️  CLI Mode Active (from environment variable)")
    elif param_mode == "streamlit":
        force_cli = False
        print("🌐 Streamlit Mode Active (from environment variable)")
    else:
        print(f"🔄 Auto-detection Mode (will use {'CLI' if force_cli else 'Streamlit if available, otherwise CLI'})")

    if streamlit_available and not force_cli:
        print("Using Streamlit for parameter collection")
        return await st_get_params(function_name, missing_params, initial_args)
    else:
        print("Using CLI for parameter collection")
        return await get_missing_parameters_cli_async(function_name, missing_params, initial_args)
