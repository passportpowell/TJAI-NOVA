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