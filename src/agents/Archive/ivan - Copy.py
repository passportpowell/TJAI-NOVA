from .base_agent import BaseAgent
from core.functions_registery import *
from core.task_manager import Task
from utils.function_logger import log_function_call
import asyncio

class Ivan(BaseAgent):
    @log_function_call
    def handle_task(self, task: Task):
        """
        Synchronous version of handle_task.
        
        Parameters:
            task (Task): The task to handle
            
        Returns:
            The result of executing the function or None
        """
        if task.function_name in self.function_map:
            func = self.function_map[task.function_name]
            missing = []
            for param in func.__code__.co_varnames[1:]:
                if param not in task.args:
                    missing.append(param)
            if missing:
                new_args = self.ask_user_for_missing_args(missing)
                task.args.update(new_args)
            
            # Call the function with the args
            result = func(self.kb, **task.args)
            task.result = result
            return result
        else:
            print(f"Ivan doesn't recognize function {task.function_name}")
            task.result = None
            return None
    
    @log_function_call
    async def handle_task_async(self, task: Task):
        """
        Asynchronous version of handle_task.
        """
        print(f"Ivan handling task asynchronously: {task.name}")
        
        # Check for image generation requests
        if task.function_name == "generate_image":
            result = await asyncio.to_thread(self.generate_image, self.kb, **task.args)
            task.result = result
            return result
            
        # Regular task processing
        if task.function_name in self.function_map:
            func = self.function_map[task.function_name]
            missing = []
            for param in func.__code__.co_varnames[1:]:
                if param not in task.args:
                    missing.append(param)
            if missing:
                new_args = await self.ask_user_for_missing_args_async(missing)
                task.args.update(new_args)
            
            # Run the synchronous function in a thread pool
            result = await asyncio.to_thread(func, self.kb, **task.args)
            task.result = result
            return result
        else:
            print(f"Ivan doesn't recognize function {task.function_name}")
            task.result = None
            return None
            
    @log_function_call
    def generate_image(self, kb, prompt):
        """
        Generate an image based on the provided prompt.
        Currently returns ASCII art and text suggestions.
        
        Parameters:
            kb (KnowledgeBase): The knowledge base
            prompt (str): The image description prompt
            
        Returns:
            str: A text description or ASCII representation
        """
        print(f"Ivan handling image generation request: {prompt}")
        
        # Extract the main subject from the prompt
        subject = prompt.lower()
        for remove_word in ["create", "generate", "make", "an", "a", "image", "picture", "of", "about"]:
            subject = subject.replace(remove_word, "")
        subject = subject.strip()
        
        # Simple ASCII art templates based on subjects
        if "wind" in subject.lower():
            ascii_art = """
       ~~~~~         ~~~~~
   ~~~~~~~~~~~~~   ~~~~~~~~~~~~~
 ~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~
   ~~~~~~~~~~~~~   ~~~~~~~~~~~~~
       ~~~~~         ~~~~~
            """
        elif "sun" in subject.lower() or "solar" in subject.lower():
            ascii_art = """
          \\   |   /
           \\  |  /
       -----( @ )-----
           /  |  \\
          /   |   \\
            """
        elif "water" in subject.lower() or "ocean" in subject.lower() or "sea" in subject.lower():
            ascii_art = """
      ~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~
      ~~~~~~~~~~~~~~~~~~~~~
            """
        else:
            ascii_art = """
      _____       _____
     /     \\     /     \\
    /       \\___/       \\
    \\                   /
     \\       ^         /
      \\     / \\       /
       \\___/   \\     /
                \\___/
            """
        
        # Prepare a detailed response
        result = f"""
# Image Generation for: {prompt}

Since I can't generate actual images, here are two alternatives:

## ASCII Art Representation:
{ascii_art}

## Text-to-Image Prompt:
You can use the following prompt with an AI image generator like DALL-E, Midjourney or Stable Diffusion:

"A highly detailed, professional {subject} visualization with dynamic composition, dramatic lighting, and photorealistic details. The image should have vibrant colors, proper perspective, and a balanced composition."

To create this image, you would need to:
1. Visit an image generation service
2. Enter the prompt above
3. Adjust parameters like style, aspect ratio, and detail level
4. Generate and download your image
        """
        
        # Store the result in the knowledge base
        kb.set_item("image_result", result)
        kb.set_item("final_report", result)
        
        return result
    

    