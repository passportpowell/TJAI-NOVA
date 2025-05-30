# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:38:54 2024

@author: ENTSOE
"""

import openai
import pandas as pd
import threading
import os, sys
import base64
from groq import Groq

sys.path.append(os.getenv('SECONDARY_PATH', ''))
# Point to the local server

# import time
# import simpleaudio as sa
from openai import OpenAI
# import requests
# import base64
from pydantic import BaseModel
sys.path.append('utils')
# from play_chime import play_chime

# from tts import tts
# from stt import stt
lms_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Set the API_KEY variable at the module level
from .get_api_keys import get_api_key

API_KEY = get_api_key('openai')
if API_KEY:
    os.environ['OPENAI_API_KEY'] = API_KEY
else:
    print("Failed to load API key.")

GROQ_API_KEY = get_api_key('groq')
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
perplexity_key = "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"

#Taken from run_open_ai
def play_sound_on_off(sound_path):
    if sound_path:
        wave_obj = sa.WaveObject.from_wave_file(sound_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()


def run_open_ai(message, context, temperature=0.7, top_p=1.0):
    global sound_playing
    chat_log = []
    sound_path = play_chime('speech_dis')  # play_chime returns a valid path or None
    sound_thread = threading.Thread(target=lambda: None)  # Initialize thread with a dummy function to handle cases where sound_path is None
    if sound_path:
        sound_thread = threading.Thread(target=play_sound_on_off, args=(sound_path,))
        sound_thread.start()
    response = openai.ChatCompletions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": [{"type": "text", "text": message}]},
        ],
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    AI_response = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': AI_response.strip('\n').strip()})
    if sound_thread.is_alive():
        sound_thread.join()
    return AI_response


def run_open_ai_json(message, context, temperature=None, top_p=1.0, model="o3-mini"):
    """
    Call OpenAI API to get a JSON-formatted response.
    Modified to handle models that don't support certain parameters.
    Enhanced with detailed logging to verify API calls.
    
    Parameters:
    - message (str): The message to send to the API
    - context (str): System context/instructions
    - temperature (float, optional): Temperature parameter for models that support it
    - top_p (float): Top-p sampling parameter
    - model (str): The model to use
    
    Returns:
    - str: The JSON response content
    """
    import json
    import time
    import uuid
    import logging
    
    # Generate a unique request ID
    request_id = str(uuid.uuid4())[:8]
    
    print(f"open_ai_call.py: \n \n🔄 [Request {request_id}] Preparing OpenAI API call to model: {model}")
    print(f"open_ai_call.py: \n 📋 Context length: {len(context)} chars")
    print(f"open_ai_call.py: \n 📋 Message length: {len(message)} chars")
    
    # Base params that should work with all models
    params = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": [{"type": "text", "text": message}]},
        ],
    }
    
    # Log which parameters we're using
    used_params = ["model", "response_format", "messages"]
    
    # Conditionally add parameters based on model
    if 'o3' not in model.lower():  # o3-mini doesn't support temperature
        if temperature is not None:
            params["temperature"] = temperature
            used_params.append("temperature")
    
    if top_p is not None:
        params["top_p"] = top_p
        used_params.append("top_p")
    
    # These parameters may not be supported by all models, add conditionally
    if 'gpt' in model.lower() or 'claude' in model.lower():
        params["frequency_penalty"] = 0.0
        params["presence_penalty"] = 0.0
        used_params.extend(["frequency_penalty", "presence_penalty"])
    
    # print(f"open_ai_call.py: \n 🔧 Using parameters: {', '.join(used_params)}")
    
    # Start timer
    start_time = time.time()
    
    try:
        # print(f"open_ai_call.py: \n 📡 [Request {request_id}] Sending request to OpenAI API...")
        
        # Actually make the API call
        response = openai.chat.completions.create(**params)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Extract response
        AI_response = response.choices[0].message.content
        
        # Log success
        print(f"open_ai_call.py: \n ✅ [Request {request_id}] OpenAI API call successful in {elapsed_time:.2f} seconds")
        # print(f"open_ai_call.py: \n 📊 Response length: {len(AI_response)} chars")
        
        # Try to validate it's actually JSON
        try:
            json.loads(AI_response)
            # print(f"open_ai_call.py: \n ✓ Response is valid JSON")
        except json.JSONDecodeError:
            print(f"open_ai_call.py: \n ⚠️ Warning: Response is not valid JSON")
        
        return AI_response
    except Exception as e:
        # Calculate elapsed time even for failures
        elapsed_time = time.time() - start_time
        
        print(f"open_ai_call.py: \n ❌ [Request {request_id}] Error in OpenAI API call after {elapsed_time:.2f} seconds")
        print(f"open_ai_call.py: \n ❌ Error type: {type(e).__name__}")
        print(f"open_ai_call.py: \n ❌ Error details: {str(e)}")
        
        # Fallback: Try again with minimal parameters if we got parameter errors
        if "unsupported_parameter" in str(e).lower():
            print(f"open_ai_call.py: \n 🔄 [Request {request_id}] Retrying with minimal parameters...")
            
            minimal_params = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": params["messages"]
            }
            
            # Reset timer for fallback attempt
            fallback_start_time = time.time()
            
            try:
                print(f"open_ai_call.py: \n 📡 [Request {request_id}] Sending fallback request...")
                
                response = openai.chat.completions.create(**minimal_params)
                
                # Calculate elapsed time for fallback
                fallback_elapsed_time = time.time() - fallback_start_time
                
                AI_response = response.choices[0].message.content
                
                print(f"open_ai_call.py: \n ✅ [Request {request_id}] Fallback successful in {fallback_elapsed_time:.2f} seconds")
                print(f"open_ai_call.py: \n 📊 Response length: {len(AI_response)} chars")
                
                return AI_response
            except Exception as fallback_error:
                # Calculate elapsed time for failed fallback
                fallback_elapsed_time = time.time() - fallback_start_time
                
                print(f"open_ai_call.py: \n ❌ [Request {request_id}] Fallback also failed after {fallback_elapsed_time:.2f} seconds")
                print(f"open_ai_call.py: \n ❌ Fallback error: {str(fallback_error)}")
                
                return '{}'  # Return empty JSON object as last resort
        
        # Return empty JSON object on error
        return '{}'


def run_open_ai_o1(message, context, model="o1-mini"):
    message = message.replace('\n', '')
    context = context.replace('\n', '')
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f'{context}. {message}'}
        ]
    )
    return response.choices[0].message.content


def ai_gap_filler(prompt):
    user_message = """Please make an assessment of whether this prompt will take more than 3 seconds to from the OpenAI API o3-mini to give a response. 
                If so please stall while the request is being processed. The output of the call will be text-to-speech, therefore the response must be very short, very roughly in the 20 word range.
                Summarize the request, reassure the user you know what the request is and that it is being processed. E.g. 'Hey, I got your request for xyz. I'll get back to you in a sec.' Don't include timings.
                Here is your prompt: {prompt}. 
                Return either the stall_response or no_stall_required. Remember your response will be directly converted to TTS, so ensure no additional text.
                """
    chat_log = []
    chat_log.append({"role": "user", "content": user_message})
    response = openai.chat.completions.create(
        model="o3-mini",
        messages=chat_log
    )
    AI_response = response.choices[0].message.content
    return AI_response


def open_ai_copywriter(purpose, target_audience, tone_voice, format_, length, example, position, prompt):
    print(prompt)
    response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": f"You are an experienced copywriter tasked with writing compelling content. \
                Please use the following inputs: Purpose = {purpose}, Target Audience = {target_audience}, Tone and Voice = {tone_voice}, \
                Format = {format_}, Length = {length}, Position on topic = {position}, Examples of content = {example}. \
                Ensure the content is engaging, adheres to the provided specifications, and reflects the intended tone and style. \
                Please respond in the format. Amount of Input tokens used:, Amount of Output tokens used:, Final Copy:"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].message.content


def load_category_descriptions(file_path):
    df = pd.read_csv(file_path)
    return dict(zip(df['Key'], df['Description']))


def ai_chat_session(prompt=None):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    history = [
        {"role": "system", "content": "You are a friendly, intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. Keep your responses concise and to the point."},
        {"role": "user", "content": f"Hello, you have been requested. Here is the prompt: {prompt}"},
    ]
    while True:
        completion = client.chat.completions.create(
            model="bartowski/Phi-3-medium-128k-instruct-GGUF",
            messages=history,
            temperature=0.7,
            stream=True,
        )
        new_message = {"role": "assistant", "content": ""}
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content
        history.append(new_message)
        print()
        history.append({"role": "user", "content": input("> ")})


def ai_spoken_chat_session(prompt=None):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    history = [
        {"role": "system", "content": """You are an friendly, intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. 
                                         I just called you and asked you to have a chat. Respond to let me know you have heard me and we will start                                 
                                        Keep your responses concise and to the point. If there is no prompt given just respond simply"""},
        {"role": "user", "content": "{prompt}"},
    ]
    while True:
        completion = client.chat.completions.create(
            model="bartowski/Phi-3-medium-128k-instruct-GGUF",
            messages=history,
            temperature=0.7,
            stream=True,
        )
        new_message = {"role": "assistant", "content": ""}
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content
        tts(new_message["content"])
        history.append(new_message)
        print()
        my_prompt = stt()
        history.append({"role": "user", "content": my_prompt})
        if my_prompt.lower() in ["bye", "goodbye", "end", "stop", "see you later"]:
            break


def openai_vision(image_path, context):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "o3-mini",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": context},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]}
        ],
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return (response.json()['choices'][0]['message']['content'])


def openai_cot(prompt, context, model="gpt-4o-2024-08-06"):
    class Step(BaseModel):
        explanation: str
        output: str
    class Reasoning(BaseModel):
        steps: list[Step]
        final_answer: str
    completion = openai.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ],
        response_format=Reasoning,
    )
    response = completion.choices[0].message.parsed
    return response.final_answer


def continue_conversation(user_input, context, instruction, model="o3-mini"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": context}
        ],
        stream=True
    )
    nova_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="")
            nova_response += content
    return nova_response


if __name__ == '__main__':
    context = (
    "You are Nova, the central coordination agent in our multi-assistant system. "
    "Your role is to parse user requests, break them down into actionable sub-tasks, "
    "and delegate these tasks to the appropriate specialized agents (e.g., Emil, Lola, etc.)."
    )
    prompt = "what is the latest us news?"
    print(run_open_ai_ns(prompt, context, temperature=0.7, top_p=1.0, model="sonar", max_tokens=500))
