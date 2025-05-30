
import os
import sys

# src/utils/get_api_keys.py
import os

def get_api_key(service_name: str) -> str:
    """
    Get API key from environment variable first, then fall back to file
    """
    # Check environment variables first (uppercase convention)
    env_var_name = f"{service_name.upper()}_API_KEY"
    api_key = os.environ.get(env_var_name)
    
    if api_key:
        return api_key
        
    # Then check files as backup
    potential_paths = [
        os.path.join(os.path.dirname(__file__), service_name),
        os.path.join(os.path.dirname(__file__), '..', 'external_resources', 'api_keys', service_name),
        os.path.expanduser(f'~/.{service_name}_api_key')
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    api_key = f.read().strip()
                    if api_key:
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



import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Function to get API key from multiple sources
def get_api_key(service_name):
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if service_name.lower() in st.secrets.api_keys:
            return st.secrets.api_keys[service_name.lower()]
    except:
        pass
        
    # Then try environment variables (from .env file)
    env_var_name = f"{service_name.upper()}_API_KEY"
    return os.environ.get(env_var_name)
    
# Example usage
openai_key = get_api_key("openai")