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
sys.path.append(os.path.join(os.path.dirname(__file__)))

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
from get_api_keys import get_api_key

# Load all API keys using the get_api_key function
API_KEY = get_api_key('openai')
if API_KEY: 
    os.environ['OPENAI_API_KEY'] = API_KEY
else: 
    print("Failed to load OpenAI API key.")

GROQ_API_KEY = get_api_key('groq')
if GROQ_API_KEY:
    os.environ['GROQ_API_KEY'] = GROQ_API_KEY
else:
    print("Failed to load Groq API key.")

# Load Perplexity API key
PERPLEXITY_API_KEY = get_api_key('perplexity')
if not PERPLEXITY_API_KEY:
    print("Failed to load Perplexity API key.")

# Load Deepseek API key
DEEPSEEK_API_KEY = get_api_key('deepseek')
if not DEEPSEEK_API_KEY:
    print("Failed to load Deepseek API key.")

#Taken from run_open_ai
def play_sound_on_off(sound_path, duration_on=1, duration_off=1, repeat=5,  play_flag=0):
    wave_obj = sa.WaveObject.from_wave_file(sound_path)
    while play_flag[0]:
        play_obj = wave_obj.play()# Play sound
        time.sleep(duration_on) # Wait for the duration the sound should be on
        play_obj.stop() # Stop the sound (if it's still playing)
        time.sleep(duration_off)  # Wait for the off duration

def play_sound_on_off(sound_path):
    if sound_path:
        wave_obj = sa.WaveObject.from_wave_file(sound_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

def run_open_ai(message, context, temperature = 0.7, top_p = 1.0):
    global sound_playing
    chat_log = []
    sound_path = play_chime('speech_dis')# play_chime returns a valid path or None
    sound_thread = threading.Thread(target=lambda: None)# Initialize thread with a dummy function to handle cases where sound_path is None
    if sound_path:
        sound_thread = threading.Thread(target=play_sound_on_off, args=(sound_path,))
        sound_thread.start()
    response = openai.ChatCompletions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content":context},
            {"role": "user", "content": [{"type": "text", "text": message}]}, ],
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,)
    AI_response = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': AI_response.strip('\n').strip()})
    if sound_thread.is_alive():
        sound_thread.join()
    return AI_response

def run_open_ai_ns(message, context, temperature = 0.7, top_p = 1.0, model = "gpt-4.1-mini", max_tokens = 500):
    if 'gpt' in model:
        chat_log = []
        response = openai.chat.completions.create(
            model=model,       
            messages=[
                {"role": "system", "content":context},
                {"role": "user", "content": [{"type": "text", "text": message}]}, ],
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,)
        AI_response = response.choices[0].message.content
        chat_log.append({'role': 'assistant', 'content': AI_response.strip('\n').strip()})
        return AI_response
    
    if 'o1' in model:
        # Remove newline characters from context and message
        message = message.replace('\n', '')
        context = context.replace('\n', '')

        # Create the completion with the correct structure
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user", 
                    "content": f'{context}. {message}'
                }
            ]
        )
        return response.choices[0].message.content

    if 'studio' in model:
        history = [
            {"role": "system", "content": context},
            {"role": "user", "content": [{"type": "text", "text": message}]},
        ]
        completion = lms_client.chat.completions.create(
            model = model,
            messages = history,
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens
        )
        AI_response = (completion.choices[0].message.content)
        return AI_response
    
    if 'groq' in model:
        client = Groq()
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")
    
    if 'deepseek' in model:
        # Use the API key from environment variable
        if not DEEPSEEK_API_KEY:
            return "Error: Deepseek API key not available"
        
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model= model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
            stream=False
        )

        return response.choices[0].message.content

    if 'sonar' in model:
        # Use the API key from environment variable
        if not PERPLEXITY_API_KEY:
            return "Error: Perplexity API key not available"
        
        perplexity_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
        messages = [
            {
                "role": "system",
                "content": f"{context}",
            },
            {   
                "role": "user",
                "content": f'{message}',
            },
        ]

        response = perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=messages,
        )
        # chat completion with streaming
        # response_stream = perplexity_client.chat.completions.create(
        #     model= model,
        #     messages=messages,
        #         stream=False
        #     )
        # for response in response_stream:
        #     print(response)
        return response.choices[0].message.content

    if 'test' in model:
        return 'This is a test call'

def run_open_ai_json(message, context, temperature = 0.7, top_p = 1.0, model = "gpt-4.1-mini"):
    chat_log = []
    response = openai.chat.completions.create(
        model=model,
        response_format = {"type": "json_object"},
        
        messages=[
            {"role": "system", "content":context},
            {"role": "user", "content": [{"type": "text", "text": message}]}, ],
        temperature=temperature,
        
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,)
    AI_response = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': AI_response.strip('\n').strip()})
    return AI_response

def run_open_ai_o1(message, context, model="o1-mini"):
    # Remove newline characters from context and message
    message = message.replace('\n', '')
    context = context.replace('\n', '')

    # Create the completion with the correct structure
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user", 
                "content": f'{context}. {message}'
            }
        ]
    )
    return response.choices[0].message.content

def ai_gap_filler(prompt):
    user_message = """Please make an assessment of whether this prompt will take more than 3 seconds to from the openAI API gpt-4o to give a response. 
                If so please stall while the request is being processed. The output of the call will be text-to-speech, therefore the response must be very short, very roughly in the 20 word range.\
                Sumamrize the request reassure the user you know what the request is and that it is being processed. E.g. 'Hey, i got your request for xyz. I'll get back to you in a sec' Don't include timings.
                Here is your prompt: {prompt}. 
                Return either the stall_response or no_stall_required. Remember your response will be directly converted to tts, so ensure no additional text.
                """
    chat_log = []
    chat_log.append({"role": "user", "content": user_message})
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages = chat_log    )
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
            {"role": "user", "content": prompt}],
        temperature=.5,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,)
    return response.choices[0].message.content

def load_category_descriptions(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    # Create a dictionary with categories as keys and descriptions as values
    return dict(zip(df['Key'], df['Description']))

def open_ai_categorisation(question, function_map, level=None):
    """
    Categorize a question using OpenAI's API.
    
    Parameters:
    question (str): The question or prompt to categorize
    function_map (str): Path to CSV file containing function categories and descriptions
    level (str, optional): Task processing level ('task list' or None)
    
    Returns:
    str: The determined category
    """
    # Ensure question is a string
    if not isinstance(question, str):
        question = str(question)
        
    categories = load_category_descriptions(function_map)  # Load category descriptions
    
    # If level is 'task list', remove 'Create task list' from the categories
    if level == 'task list' and 'Create task list' in categories:
        categories.pop('Create task list')

    # Prepare category descriptions
    category_info = ", ".join([f"'{key}': {desc}" for key, desc in categories.items()])
    
    system_msg = (
        f"I am an assistant trained to categorize questions into specific functions. "
        f"Here are the categories with descriptions: {category_info}. "
        f"If none of the categories are appropriate, categorize as 'Uncategorized'. "
        f"Please respond with only the category from the list given, no additional text. "
        f"Ensure names are taken from the list provided. Do not add additional punctuation or text."
    )
    
    try:
        # Make sure we construct the messages in a consistent format for gpt-4o
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": [{"type": "text", "text": question}]}
            ],
            temperature=0.0,  # Ensure deterministic output
            max_tokens=50,    # Limit unnecessary token usage
            top_p=1.0
        )

        category = response.choices[0].message.content.strip()
        print(f"✅ OpenAI Response (open_ai_calls.py): {category}")  # Debugging output
        return category
        
    except Exception as e:
        print(f"❌ Error in OpenAI categorization: {str(e)}")
        # Fallback to 'Uncategorized' in case of API errors
        return 'Uncategorized'




def ai_chat_session(prompt = None):
    # Point to the local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    history = [
        {"role": "system", "content": "You are an friendly, intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. Keep your responses concise and to the point."},
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
        
        # Uncomment to see chat history
        # import json
        # gray_color = "\033[90m"
        # reset_color = "\033[0m"
        # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
        # print(json.dumps(history, indent=2))
        # print(f"\n{'-'*55}\n{reset_color}")

        print()
        history.append({"role": "user", "content": input("> ")})

def ai_spoken_chat_session(prompt = None):
    # Point to the local server
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
        
        # Uncomment to see chat history
        # import json
        # gray_color = "\033[90m"
        # reset_color = "\033[0m"
        # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
        # print(json.dumps(history, indent=2))
        # print(f"\n{'-'*55}\n{reset_color}")

        print()
        my_prompt = stt()
        history.append({"role": "user", "content": my_prompt})

        # Check if the user's phrase indicates the end of the conversation
        if my_prompt.lower() in ["bye", "goodbye", "end", "stop", "see you later"]:
            break

def openai_vision(image_path, context ):
    # Load the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Call the OpenAI Vision API
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
                }

    payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": context
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": 1000
                }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return (response.json()['choices'][0]['message']['content'])

def openai_cot(prompt, context, model = "gpt-4o-2024-08-06"):
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

# Function to continue the conversation with updated context
def continue_conversation(user_input, context, instruction, model = "gpt-4.1-mini"):
    # Combine logs for context
    # context = "\n".join(log)
    
    # Call GPT-4 with context-aware prompt
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": context}
        ],
        stream=True  # Enable streaming if available
    )

    nova_response = ""
    # Collect and return the response
    for chunk in response:
        # Assuming `chunk` is an instance of `ChatCompletionChunk` and has attributes `choices` and `delta`
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="")
            nova_response += content
    
    return nova_response

if __name__ == '__main__':
    # prompt = 'Give 10 ways to loose weight, with a workout regime and meal plan, considering my weight is 100 kg and height is 180 meters and i am 26'
    # image_path = os.getenv('SECONDARY_PATH', '')
    # openai_vision(image_path)
    context = "You are a helpful assistant."
    prompt = "what is the latest us news?"
    # print(run_open_ai_o1(prompt, context))
    # response = openai_cot(prompt, context)
    # print(response.final_answer)
    print(run_open_ai_ns(prompt, context, temperature = 0.7, top_p = 1.0, model = "sonar", max_tokens = 500))