import sys, os

def get_api_key(key_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if key_name == "openai":
        api_key_path = os.path.abspath(os.path.join(current_dir, "openai"))

    if key_name == "gemini":
        api_key_path = os.path.abspath(os.path.join(current_dir, "gemini"))

    if key_name == "claude":
        api_key_path = os.path.abspath(os.path.join(current_dir, "claude"))

    if key_name == "groq":
        api_key_path = os.path.abspath(os.path.join(current_dir, "groq"))

    try:
        with open(api_key_path, "r") as file:
            api_key = file.read().strip()
            if not api_key:
                raise ValueError("API key file is empty")
            return api_key
    except FileNotFoundError:
        print(f"API key file not found: {api_key_path}")
        return None
    except ValueError as ve:
        print(ve)
        return None
    


"""
Fix for loading API keys from .env file for PLEXOS functions
Replace the content of src/agents/PLEXOS_functions/get_api_keys.py with this code
"""

import sys, os
from dotenv import load_dotenv

# Try to load .env file from different possible locations
possible_env_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')),  # Main project root
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env')),  # src folder
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')),  # agents folder
    os.path.abspath('D:\\Tera-joule\\Terajoule - AI Architecture\\AI Assistants\\Nova - AI Coordinator v2\\.env')  # Hardcoded path
]

# Try each possible path
for env_path in possible_env_paths:
    if os.path.exists(env_path):
        print(f"Loading environment variables from: {env_path}")
        load_dotenv(env_path)
        break

def get_api_key(key_name):
    """
    Get API key from environment variables first, then try file fallback.
    """
    # First try environment variables (uppercase convention)
    env_var_name = f"{key_name.upper()}_API_KEY"
    api_key = os.environ.get(env_var_name)
    
    if api_key:
        print(f"Using {key_name} API key from environment variables")
        return api_key
        
    # Fallback to file-based approach
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if key_name == "openai":
        api_key_path = os.path.abspath(os.path.join(current_dir, "openai"))
    elif key_name == "gemini":
        api_key_path = os.path.abspath(os.path.join(current_dir, "gemini"))
    elif key_name == "claude":
        api_key_path = os.path.abspath(os.path.join(current_dir, "claude"))
    elif key_name == "groq":
        api_key_path = os.path.abspath(os.path.join(current_dir, "groq"))
    else:
        api_key_path = os.path.abspath(os.path.join(current_dir, key_name))

    try:
        with open(api_key_path, "r") as file:
            api_key = file.read().strip()
            if not api_key:
                raise ValueError(f"{key_name} API key file is empty")
            print(f"Using {key_name} API key from file: {api_key_path}")
            return api_key
    except FileNotFoundError:
        print(f"{key_name} API key not found in environment variables or file: {api_key_path}")
        
        # Last resort: check if .env values were stored directly in env vars
        # (this is for legacy compatibility)
        if key_name == "openai" and "OPENAI_API_KEY" in os.environ:
            return os.environ["OPENAI_API_KEY"]
        elif key_name == "groq" and "GROQ_API_KEY" in os.environ:
            return os.environ["GROQ_API_KEY"]
            
        return None
    except ValueError as ve:
        print(ve)
        return None
    
import os

def get_api_key(service_name: str) -> str:
    """
    Get API key from Streamlit secrets first, then environment variables
    """
    
    # Method 1: Try Streamlit secrets (PRIMARY for Streamlit Cloud)
    try:
        import streamlit as st
        
        # Try st.secrets.api_keys.service_name
        if hasattr(st.secrets, 'api_keys') and service_name.lower() in st.secrets.api_keys:
            key = st.secrets.api_keys[service_name.lower()]
            if key:
                print(f"✅ Found {service_name} key in st.secrets.api_keys")
                return key
                
        # Try st.secrets.SERVICE_NAME_API_KEY
        env_var_name = f"{service_name.upper()}_API_KEY"
        if hasattr(st.secrets, env_var_name):
            key = getattr(st.secrets, env_var_name)
            if key:
                print(f"✅ Found {service_name} key in st.secrets.{env_var_name}")
                return key
                
    except Exception as e:
        print(f"⚠️ Streamlit secrets not available: {e}")
    
    # Method 2: Try environment variables (FALLBACK)
    env_var_name = f"{service_name.upper()}_API_KEY"
    api_key = os.environ.get(env_var_name)
    if api_key:
        print(f"✅ Found {service_name} key in environment: {env_var_name}")
        return api_key
    
    # Method 3: Try other environment variable patterns
    other_patterns = [
        f"{service_name.upper()}",
        f"{service_name.lower()}_api_key",
        f"{service_name.lower()}"
    ]
    
    for pattern in other_patterns:
        key = os.environ.get(pattern)
        if key:
            print(f"✅ Found {service_name} key in environment: {pattern}")
            return key
    
    print(f"❌ No API key found for {service_name}")
    return None

# Configure OpenAI at import time
try:
    import openai
    api_key = get_api_key('openai')
    if api_key:
        openai.api_key = api_key
        print(f"✅ Configured openai.api_key")
except ImportError:
    pass    