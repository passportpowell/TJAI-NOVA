
# import sys, os
# def get_api_key(key_name):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     if key_name == "openai":
#         api_key_path = os.path.abspath(os.path.join(current_dir, "../external_resources/api_keys/openai"))

#     if key_name == "gemini":
#         api_key_path = os.path.abspath(os.path.join(current_dir, "../external_resources/api_keys/gemini"))

#     if key_name == "claude":
#         api_key_path = os.path.abspath(os.path.join(current_dir, "../external_resources/api_keys/claude"))

#     if key_name == "groq":
#         api_key_path = os.path.abspath(os.path.join(current_dir, "../external_resources/api_keys/groq"))

#     try:
#         with open(api_key_path, "r") as file:
#             api_key = file.read().strip()
#             if not api_key:
#                 raise ValueError("API key file is empty")
#             return api_key
#     except FileNotFoundError:
#         print(f"API key file not found: {api_key_path}")
#         return None
#     except ValueError as ve:
#         print(ve)
#         return None
    
# def get_api_key(service_name: str) -> str:
#     """
#     Reads and returns the API key for a given service from a file
#     located in the same directory as this module.
    
#     Parameters:
#       - service_name (str): The name of the service (e.g., 'openai', 'groq').
    
#     Returns:
#       - str: The API key as a string, or None if the file is not found.
#     """
#     # Build the path to the API key file in the current utils folder.
#     key_path = os.path.join(os.path.dirname(__file__), service_name)
    
#     if os.path.exists(key_path):
#         with open(key_path, "r") as f:
#             return f.read().strip()
#     else:
#         print(f"API key file not found: {key_path}")
#         return None    


import os
import sys

def get_api_key(service_name: str) -> str:
    """
    Reads and returns the API key for a given service from a file.
    
    Parameters:
      - service_name (str): The name of the service (e.g., 'openai', 'groq').
    
    Returns:
      - str: The API key as a string, or None if the file is not found.
    """
    # Check multiple potential locations for the API key
    potential_paths = [
        os.path.join(os.path.dirname(__file__), service_name),  # In the utils directory
        os.path.join(os.path.dirname(__file__), '..', 'external_resources', 'api_keys', service_name),  # In external resources
        os.path.join(os.path.dirname(__file__), '..', service_name),  # In parent directory
        os.path.expanduser(f'~/.{service_name}_api_key')  # User's home directory
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    api_key = f.read().strip()
                    if api_key:
                        # Set environment variable for OpenAI
                        if service_name == 'openai':
                            os.environ['OPENAI_API_KEY'] = api_key
                        return api_key
            except Exception as e:
                print(f"Error reading API key from {path}: {e}")
    
    print(f"No API key found for {service_name}")
    return None

# Configure OpenAI at import time (optional, but can help)
try:
    import openai
    api_key = get_api_key('openai')
    if api_key:
        openai.api_key = api_key
except ImportError:
    print("OpenAI library not installed")