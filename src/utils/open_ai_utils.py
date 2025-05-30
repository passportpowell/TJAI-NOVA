# Add these at the top of open_ai_utils.py
from utils.function_logger import log_function_call

import openai
import pandas as pd
import threading
import os, sys
import base64
from groq import Groq
import aiohttp
import asyncio
import json
import time
import re
import urllib.parse
from openai import OpenAI
from pydantic import BaseModel

# Import API key handling
from utils.get_api_keys import get_api_key

# Make sure the original synchronous function is defined
@log_function_call
def ai_chat_session(kb, prompt=None):
    """
    Handles a general AI chat session.
    
    Parameters:
        kb (KnowledgeBase): The knowledge base
        prompt (str, optional): Initial prompt to start the conversation
        
    Returns:
        str: The result of the chat session
    """
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    print(f"Starting chat session with prompt: {prompt}")
    
    history = [
        {"role": "system", "content": "You are a friendly, intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. Keep your responses concise and to the point."},
        {"role": "user", "content": f"Hello, you have been requested. Here is the prompt: {prompt}"},
    ]
    
    try:
        # Single response mode since we're in a script context
        completion = client.chat.completions.create(
            model="bartowski/Phi-3-medium-128k-instruct-GGUF",
            messages=history,
            temperature=0.7
        )
        
        response = completion.choices[0].message.content
        
        # Store the result in the knowledge base
        kb.set_item("chat_result", response)
        kb.set_item("final_report", response)  # This will be displayed at the end
        
        return response
    except Exception as e:
        error_msg = f"Error in chat session: {str(e)}"
        print(error_msg)
        return error_msg


@log_function_call
async def ai_chat_session_async(kb, prompt=None):
    """
    Asynchronous version of ai_chat_session.
    
    Parameters:
        kb (KnowledgeBase): The knowledge base
        prompt (str, optional): Initial prompt to start the conversation
        
    Returns:
        str: The result of the chat session
    """
    # We'll run the synchronous version in a thread pool
    # This is because the OpenAI client doesn't have async support
    return await asyncio.to_thread(ai_chat_session, kb, prompt)


@log_function_call
def ai_spoken_chat_session(kb, prompt=None):
    """
    Handles a spoken AI chat session (simulated).
    
    Parameters:
        kb (KnowledgeBase): The knowledge base
        prompt (str, optional): Initial prompt to start the conversation
        
    Returns:
        str: The result of the chat session
    """
    # Since real text-to-speech and speech-to-text might not be available,
    # we'll simulate the spoken aspects
    print("Starting spoken chat session (text-based simulation)")
    
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    history = [
        {"role": "system", "content": """You are an friendly, intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. 
                                         I just called you and asked you to have a chat. Respond to let me know you have heard me and we will start                                 
                                        Keep your responses concise and to the point. If there is no prompt given just respond simply"""},
        {"role": "user", "content": f"{prompt if prompt else 'Hello'}"},
    ]
    
    try:
        # Single response mode for a script context
        completion = client.chat.completions.create(
            model="bartowski/Phi-3-medium-128k-instruct-GGUF",
            messages=history,
            temperature=0.7
        )
        
        response = completion.choices[0].message.content
        
        # Simulate speech output
        print(f"\nAssistant (spoken): {response}")
        
        # Store the result in the knowledge base
        kb.set_item("spoken_chat_result", response)
        kb.set_item("final_report", response)  # This will be displayed at the end
        
        return response
    except Exception as e:
        error_msg = f"Error in spoken chat session: {str(e)}"
        print(error_msg)
        return error_msg


@log_function_call
async def ai_spoken_chat_session_async(kb, prompt=None):
    """
    Asynchronous version of ai_spoken_chat_session.
    
    Parameters:
        kb (KnowledgeBase): The knowledge base
        prompt (str, optional): Initial prompt to start the conversation
        
    Returns:
        str: The result of the chat session
    """
    # We'll run the synchronous version in a thread pool
    return await asyncio.to_thread(ai_spoken_chat_session, kb, prompt)    


@log_function_call
def run_open_ai_ns(message, context, temperature=0.7, top_p=1.0, model="gpt-4.1-nano", max_tokens=500, verbose=False):
    """
    Call OpenAI API to generate a response based on the input message and context.
    This is the most complete implementation of the function with reduced verbosity.
    
    Parameters:
    - message (str): The message to send to the API
    - context (str): System context/instructions
    - temperature (float): Controls randomness in generation
    - top_p (float): Top-p sampling parameter
    - model (str): The model to use
    - max_tokens (int): Maximum number of tokens to generate
    - verbose (bool): Whether to print debug information
    
    Returns:
    - str: The API response content
    """
    import time
    
    # Only print debug info if verbose is True
    if verbose:
        print(f"Starting OpenAI API call to model: {model}")
        print(f"Context length: {len(context)} chars")
        print(f"Message length: {len(message)} chars")
    
    start_time = time.time()
    try:
        if model == "gpt-4.1-nano" or model == "o3-mini":
            # For gpt-4.1-nano or o3-mini 
            # Note: gpt-4.1-nano supports temperature parameter unlike o3-mini
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ],
                "top_p": top_p,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
            
            # Add temperature for gpt-4.1-nano but not for o3-mini
            if model == "gpt-4.1-nano":
                params["temperature"] = temperature
                
            response = openai.chat.completions.create(**params)
            AI_response = response.choices[0].message.content
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"API call completed in {elapsed:.2f} seconds")
            
            return AI_response

        elif 'gpt' in model.lower():
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ],
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            AI_response = response.choices[0].message.content
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"API call completed in {elapsed:.2f} seconds")
            
            return AI_response

        elif 'o1' in model:
            # Special case for O1 model which requires different formatting
            message_clean = message.replace('\n', '')
            context_clean = context.replace('\n', '')
            
            # For O1, combine context and message
            combined_prompt = f"{context_clean}\n\nThe user's request is: {message_clean}"
            
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": combined_prompt}
                ]
            )
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"API call completed in {elapsed:.2f} seconds")
                
            return response.choices[0].message.content

        elif 'studio' in model:
            # For local LM Studio models
            from openai import OpenAI
            lms_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            
            history = [
                {"role": "system", "content": context},
                {"role": "user", "content": message},
            ]
            
            completion = lms_client.chat.completions.create(
                model=model,
                messages=history,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            AI_response = completion.choices[0].message.content
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"API call completed in {elapsed:.2f} seconds")
                
            return AI_response

        elif 'groq' in model:
            # For Groq API
            from groq import Groq
            client = Groq()
            
            # Default to a standard model if not specified
            groq_model = "llama-3.1-70b-versatile"
            
            completion = client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False
            )
            
            return completion.choices[0].message.content

        elif 'deepseek' in model:
            # For DeepSeek API
            deepseek_api_key = get_api_key('deepseek') or "sk-bba69b9e3f4c40529602d89878f7b6fa"
            client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": message},
                ],
                stream=False
            )
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"API call completed in {elapsed:.2f} seconds")
                
            return response.choices[0].message.content

        elif 'sonar' in model:
            # For Perplexity API
            perplexity_key = get_api_key('perplexity') or "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"
            perplexity_client = OpenAI(api_key=perplexity_key, base_url="https://api.perplexity.ai")
            
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": message},
            ]
            
            response = perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
            )
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"API call completed in {elapsed:.2f} seconds")
                
            return response.choices[0].message.content

        elif 'test' in model:
            return 'This is a test call'
        
        else:
            # Default fallback - use standard OpenAI API with gpt-4.1-nano
            response = openai.chat.completions.create(
                model="gpt-4.1-nano",  # Changed from o3-mini
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ],
                temperature=temperature,
                top_p=top_p
            )
            
            return response.choices[0].message.content

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ API call failed after {elapsed:.2f} seconds: {str(e)}")
        
        # Fallback: If an unsupported_parameter error occurs, retry with minimal parameters
        if "unsupported_parameter" in str(e).lower():
            if verbose:
                print("Retrying with minimal parameters...")
            
            minimal_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ]
            }
            
            try:
                fallback_start_time = time.time()
                response = openai.chat.completions.create(**minimal_params)
                AI_response = response.choices[0].message.content
                
                if verbose:
                    fallback_elapsed = time.time() - fallback_start_time
                    print(f"Fallback successful in {fallback_elapsed:.2f} seconds")
                    
                return AI_response
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {str(fallback_error)}")
                return "I apologize, but I'm having trouble processing your request right now."
                
        return "I apologize, but I'm having trouble processing your request right now."


@log_function_call
async def run_open_ai_ns_async(message, context, temperature=0.7, top_p=1.0, model="gpt-4.1-nano", max_tokens=500, verbose=False):
    """
    Asynchronous version of run_open_ai_ns.
    Uses aiohttp for non-blocking API calls.
    
    Parameters:
        message (str): The message to send to the API
        context (str): System context/instructions
        temperature (float): Controls randomness in generation
        top_p (float): Top-p sampling parameter
        model (str): The model to use
        max_tokens (int): Maximum number of tokens to generate
        verbose (bool): Whether to print debug information
        
    Returns:
        str: The API response content
    """
    import time
    
    # Only print debug info if verbose is True
    if verbose:
        print(f"Starting async OpenAI API call to model: {model}")
        print(f"Context length: {len(context)} chars")
        print(f"Message length: {len(message)} chars")
    
    start_time = time.time()
    
    # Create the API request parameters
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": message}
        ]
    }
    
    # Add model-specific parameters
    # gpt-4.1-nano supports temperature (unlike o3-mini)
    if model != "o3-mini":
        params["temperature"] = temperature
    
    # Add top_p if provided
    if top_p is not None:
        params["top_p"] = top_p
    
    # Other parameters for certain models
    if 'gpt' in model.lower() or 'claude' in model.lower():
        params["frequency_penalty"] = 0.0
        params["presence_penalty"] = 0.0
    
    # Headers for the API request
    API_KEY = get_api_key('openai')
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        # Make the API call using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    response_json = await response.json()
                    AI_response = response_json["choices"][0]["message"]["content"]
                    
                    if verbose:
                        elapsed = time.time() - start_time
                        print(f"API call completed in {elapsed:.2f} seconds")
                    
                    return AI_response
                else:
                    error_text = await response.text()
                    print(f"API error: {response.status} - {error_text}")
                    
                    # Try fallback with minimal parameters
                    if "unsupported_parameter" in error_text.lower():
                        if verbose:
                            print("Retrying with minimal parameters...")
                        
                        # Create minimal params
                        minimal_params = {
                            "model": model,
                            "messages": params["messages"]
                        }
                        
                        # Try again with minimal params
                        async with session.post(
                            "https://api.openai.com/v1/chat/completions",
                            json=minimal_params,
                            headers=headers
                        ) as fallback_response:
                            if fallback_response.status == 200:
                                fallback_json = await fallback_response.json()
                                return fallback_json["choices"][0]["message"]["content"]
                            else:
                                return '{}'  # Return empty result on error
                    return '{}'  # Return empty result on error
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ API call failed after {elapsed:.2f} seconds: {str(e)}")
        return '{}'  # Return empty result on error


@log_function_call
def open_ai_categorisation(question, function_map, level=None):
    """
    Categorize user questions into specific functions using OpenAI.
    
    Parameters:
    - question (str or object): The user's question or prompt
    - function_map (str): Path to the function map CSV file
    - level (str, optional): Task level ('task list' or None)
    
    Returns:
    - str: The categorized function key
    """
    if not isinstance(question, str):
        question = str(question)
        
    # ENHANCED: Special handling for session history queries
    # If this is a question about a past session, direct it to general_question
    # which has special handling for history retrieval
    session_history_pattern = re.compile(r'(what|tell|show|about).*?session\s+\d+', re.IGNORECASE)
    if "session" in question.lower() and session_history_pattern.search(question.lower()):
        print(f"⚠️ Pre-check identified session history query: '{question}'")
        return "general_question"
        
    categories = load_category_descriptions(function_map)
    
    if level == 'task list' and 'Create task list' in categories:
        categories.pop('Create task list')

    # Enhanced instruction for energy modeling tasks with more explicit guidance
    additional_instruction = (
        "IMPORTANT: If the prompt contains ANY of the following: "
        "- Any form of 'run a model', 'create model', 'execute model' "
        "- Any reference to energy modeling "
        "- Words such as 'model', 'solar', 'energy', 'generation', 'run model', 'electricity' "
        "- Any request to simulate or analyze energy systems "
        "THEN you must categorize it as 'Energy Model'. "
        "The Energy Model category takes priority over Data analysis for any model-related queries."
    )
    
    category_info = ", ".join([f"'{key}': {desc}" for key, desc in categories.items()])
    
    system_msg = (
        f"I am an assistant trained to categorize questions into specific functions. "
        f"Here are the available categories with descriptions: {category_info}. "
        f"{additional_instruction} "
        f"If none of the categories are appropriate, categorize as 'Uncategorized'. "
        f"Please respond with only the category from the list given, with no additional text."
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",  # Changed from o3-mini
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": [{"type": "text", "text": question}]}
            ],
            temperature=0.7,  # Added temperature, supported by gpt-4.1-nano
            top_p=1.0
        )
        category = response.choices[0].message.content.strip()
        
        # Enhanced fallback logic to catch misclassifications
        model_related_keywords = ['model', 'run', 'solar', 'energy', 'generation', 'electricity']
        model_phrases = ['run a model', 'create model', 'execute model', 'run model']
        
        # Check for model-related phrases or keywords
        is_model_related = any(phrase in question.lower() for phrase in model_phrases) or \
                          (any(word in question.lower() for word in model_related_keywords) and 'model' in question.lower())
        
        # Override if it's clearly model-related but was misclassified
        if is_model_related and category != "Energy Model":
            print(f"open_ai_call.py: \n ⚠️ Overriding '{category}' to 'Energy Model' - detected model-related query")
            category = "Energy Model"
        # Also keep the original fallback for Uncategorized
        elif category.lower() == 'uncategorized' and any(word in question.lower() for word in model_related_keywords):
            category = "Energy Model"
            
        print(f"open_ai_call.py: \n ✅ OpenAI Response (open_ai_calls.py): {category}")
        return category
    except Exception as e:
        print(f"open_ai_call.py: \n ❌ Error in OpenAI categorization: {str(e)}")
        return 'Uncategorized'


@log_function_call
async def open_ai_categorisation_async(question, function_map, level=None):
    """
    Asynchronous version of open_ai_categorisation.
    
    Parameters:
        question (str or object): The user's question or prompt
        function_map (str): Path to the function map CSV file
        level (str, optional): Task level ('task list' or None)
        
    Returns:
        str: The categorized function key
    """
    if not isinstance(question, str):
        question = str(question)
    
    # ENHANCED: Special handling for session history queries
    # If this is a question about a past session, direct it to general_question
    # which has special handling for history retrieval
    import re
    session_history_pattern = re.compile(r'(what|tell|show|about).*?session\s+\d+', re.IGNORECASE)
    if "session" in question.lower() and session_history_pattern.search(question.lower()):
        print(f"⚠️ Pre-check identified session history query: '{question}'")
        return "general_question"
    
    # Load category descriptions
    import pandas as pd
    
    try:
        df = pd.read_csv(function_map)
        categories = dict(zip(df['Key'], df['Description']))
    except Exception as e:
        print(f"Error loading categories from {function_map}: {str(e)}")
        categories = {}
    
    # Add general_question to categories if not present
    if "general_question" not in categories:
        categories["general_question"] = "General knowledge questions, basic arithmetic, or factual queries"
    
    if level == 'task list' and 'Create task list' in categories:
        categories.pop('Create task list')
    
    # Energy model keywords for pre-check
    energy_model_keywords = [
        'energy model', 'model energy', 'build model', 
        'create model', 'design model', 'energy system', 
        'power system', 'renewable model', 'electricity model',
        'build a model'
    ]
    
    # Pre-check for energy model
    if any(keyword in question.lower() for keyword in energy_model_keywords):
        print(f"⚠️ Pre-check identified energy modeling request: '{question}'")
        return "Energy Model"
    
    # System message for the API
    category_info = ", ".join([f"'{key}': {desc}" for key, desc in categories.items()])
    
    system_msg = (
        f"You are an assistant trained to categorize questions into specific functions. "
        f"Here are the available categories with descriptions: {category_info}. "
        f"If none of the categories are appropriate, categorize as 'general_question'. "
        f"Please respond with only the category from the list given, with no additional text."
    )
    
    # Import math patterns for pre-check
    import re
    math_patterns = [
        r'\d+\s*[\+\-\*\/]\s*\d+',  # Simple operations like 2+2
        r'what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+',  # "What is 2+2"
        r'calculate\s+\d+',  # "Calculate 25"
    ]
    
    # Pre-check for math
    if any(re.search(pattern, question.lower()) for pattern in math_patterns):
        print(f"⚠️ Pre-check identified math problem: '{question}'")
        return "do_maths"
    
    # Check for website patterns
    website_patterns = [
        r'open\s+.*website',
        r'go\s+to\s+.*site',
        r'visit\s+.*page',
        r'browse\s+to',
        r'open\s+.*page'
    ]
    
    if any(re.search(pattern, question.lower()) for pattern in website_patterns):
        print(f"⚠️ Pre-check identified website opening: '{question}'")
        return "Open a website"
    
    try:
        # Call OpenAI API using async version
        response = await run_open_ai_ns_async(question, system_msg, model="gpt-4.1-nano", temperature=0.7)  # Updated model
        category = response.strip()
        
        # Clean up response
        category = category.replace("'", "").replace("\"", "").strip()
        
        # Final check for energy model
        if category.lower() in ['uncategorized', 'general_question'] and \
           any(keyword in question.lower() for keyword in energy_model_keywords):
            category = "Energy Model"
        
        print(f"✅ OpenAI categorization: {category}")
        return category
    except Exception as e:
        print(f"❌ Error in OpenAI categorization: {str(e)}")
        
        # Fallback based on keywords in question
        lower_question = question.lower()
        if any(re.search(pattern, lower_question) for pattern in website_patterns):
            return "Open a website"
        elif any(re.search(pattern, lower_question) for pattern in math_patterns):
            return "do_maths"
        elif any(keyword in lower_question for keyword in energy_model_keywords):
            return "Energy Model"
        return 'general_question'


def run_open_ai(message, context, temperature=0.7, top_p=1.0):
    global sound_playing
    chat_log = []
    sound_path = play_chime('speech_dis')  # play_chime returns a valid path or None
    sound_thread = threading.Thread(target=lambda: None)  # Initialize thread with a dummy function to handle cases where sound_path is None
    if sound_path:
        sound_thread = threading.Thread(target=play_sound_on_off, args=(sound_path,))
        sound_thread.start()
    response = openai.ChatCompletions.create(
        model="gpt-4.1-nano",  # Changed from o3-mini
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": [{"type": "text", "text": message}]},
        ],
        temperature=temperature,  # gpt-4.1-nano supports temperature
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    AI_response = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': AI_response.strip('\n').strip()})
    if sound_thread.is_alive():
        sound_thread.join()
    return AI_response


def run_open_ai_json(message, context, temperature=None, top_p=1.0, model="gpt-4.1-nano"):
    """
    Call OpenAI API to get a JSON-formatted response.
    Modified to handle models that don't support certain parameters.
    
    Parameters:
    - message (str): The message to send to the API
    - context (str): System context/instructions
    - temperature (float, optional): Temperature parameter for models that support it
    - top_p (float): Top-p sampling parameter
    - model (str): The model to use
    
    Returns:
    - str: The JSON response content
    """
    chat_log = []
    
    # Base params that should work with all models
    params = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": [{"type": "text", "text": message}]},
        ],
    }
    
    # Conditionally add parameters based on model
    if 'o3' not in model.lower():  # o3-mini doesn't support temperature, but gpt-4.1-nano does
        if temperature is not None:
            params["temperature"] = temperature
    
    if top_p is not None:
        params["top_p"] = top_p
    
    # These parameters may not be supported by all models, add conditionally
    if 'gpt' in model.lower() or 'claude' in model.lower():
        params["frequency_penalty"] = 0.0
        params["presence_penalty"] = 0.0
    
    try:
        response = openai.chat.completions.create(**params)
        AI_response = response.choices[0].message.content
        chat_log.append({'role': 'assistant', 'content': AI_response.strip('\n').strip()})
        return AI_response
    except Exception as e:
        print(f"open_ai_call.py: \n Error in OpenAI JSON call: {str(e)}")
        
        # Fallback: Try again with minimal parameters if we got parameter errors
        if "unsupported_parameter" in str(e).lower():
            print("Retrying with minimal parameters...")
            minimal_params = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": params["messages"]
            }
            
            try:
                response = openai.chat.completions.create(**minimal_params)
                AI_response = response.choices[0].message.content
                return AI_response
            except Exception as fallback_error:
                print(f"open_ai_call.py: \n Fallback also failed: {str(fallback_error)}")
                return '{}'  # Return empty JSON object as last resort
        
        return '{}'  # Return empty JSON object on error








