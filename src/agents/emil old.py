import asyncio
import inspect
import datetime
import json
from .base_agent import BaseAgent
from core.functions_registery import *
from core.task_manager import Task
from utils.function_logger import log_function_call
from utils.open_ai_utils import run_open_ai_ns_async
from .parameter_collection import get_missing_parameters_async


async def extract_energy_parameters_from_prompt(prompt: str) -> dict:
    system_msg = """
You are an expert assistant for extracting energy model configuration parameters from prompts.

Given a user's prompt, return a JSON object with the following fields:
{
  "location": "Country or region (e.g. France, Germany, etc)",
  "generation_type": "solar, wind, hydro, nuclear, etc.",
  "energy_carrier": "electricity, hydrogen, methane, etc."
}

If a field is not mentioned, set it to null. Only return the JSON object.
"""
    try:
        response = await run_open_ai_ns_async(prompt, system_msg)
        parsed = json.loads(response)
        return {k: v for k, v in parsed.items() if v}
    except Exception as e:
        print(f"❌ Failed to extract energy parameters: {e}")
        return {}


class Emil(BaseAgent):
    def __init__(self, name, kb, function_map, verbose=False):
        super().__init__(name, kb, function_map)
        self.verbose = verbose

    @log_function_call
    async def verify_parameters_async(self, function_name: str, task_args: dict) -> dict:
        """
        Verify that all required parameters are present for a function call.
        Enhanced to specifically check for energy modeling parameters.
        
        Parameters:
            function_name (str): The name of the function to verify parameters for
            task_args (dict): The arguments provided for the function
            
        Returns:
            dict: Verification result with success status, missing parameters, and message
        """
        # Special case for basic Emil requests with just a prompt
        if function_name == 'process_emil_request' and task_args.get('prompt'):
            # Check for critical model parameters
            extracted_params = {}
            
            # For energy modeling, check if location is missing or "Unknown"
            location = task_args.get('location')
            if not location or location == "Unknown":
                if self.verbose:
                    print("Location parameter missing or Unknown for energy model")
                return {
                    "success": False,
                    "missing": ["location"],
                    "message": "Please specify a location (country or region) for the energy model."
                }
                
            # Check if generation type is missing
            generation = task_args.get('generation') or task_args.get('generation_type')
            if not generation:
                if self.verbose:
                    print("Generation type parameter missing for energy model")
                return {
                    "success": False,
                    "missing": ["generation"],
                    "message": "Please specify a generation type (solar, wind, hydro, etc.)."
                }
                
            # Check if energy carrier is missing (less critical, can default to electricity)
            energy_carrier = task_args.get('energy_carrier')
            if not energy_carrier:
                if self.verbose:
                    print("Energy carrier parameter missing, will default to electricity")
                # This is not critical enough to fail verification
                # task_args['energy_carrier'] = 'electricity'  # Set default
                
            return {"success": True, "missing": [], "message": "Prompt and essential parameters provided for Emil request"}
        
        # Analysis tasks are handled separately
        if function_name == 'analyze_results':
            return {"success": True, "missing": [], "message": "Analysis tasks don't require explicit parameters"}
        
        # Check if function exists in function map
        if function_name not in self.function_map:
            return {"success": False, "missing": [], "message": f"Function {function_name} not found in Emil's function map"}

        # Use function signature to check for required parameters
        func = self.function_map[function_name]
        sig = inspect.signature(func)
        required_params = [
            param.name for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty and param.name not in ('self', 'kb')
        ]
        missing = [param for param in required_params if param not in task_args]
        
        # Return appropriate result based on missing parameters
        if missing:
            return {"success": False, "missing": missing, "message": f"Missing required parameters: {', '.join(missing)}"}
        
        return {"success": True, "missing": [], "message": "All required parameters are present"}    
    
    @log_function_call
    async def handle_task_async(self, task: Task):
        if self.verbose:
            print(f"Emil handling task asynchronously: {task.name}")
        self.kb.log_interaction(task.name, "Starting execution", agent="Emil", function=task.function_name)

        session_context = task.session_context or {}
        if "emil" not in session_context:
            session_context["emil"] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "tasks_processed": []
            }

        # 🔍 Dynamically extract parameters from prompt if not provided
        if task.function_name == "process_emil_request" and task.args.get("prompt"):
            extracted = await extract_energy_parameters_from_prompt(task.args["prompt"])
            for k, v in extracted.items():
                task.args.setdefault(k, v)

        if task.function_name in self.function_map:
            func = self.function_map[task.function_name]
            validation = await self.verify_parameters_async(task.function_name, task.args)

            if not validation["success"]:
                # IMPROVED: Continue parameter collection until all required parameters are collected
                while not validation["success"] and validation.get("missing"):
                    print(f"Emil needs additional information for {task.name}")
                    
                    # Handle missing parameters interactively
                    # Using our new parameter collection module
                    collected_params = await get_missing_parameters_async(
                        function_name=task.function_name,
                        missing_params=validation["missing"],
                        initial_args=task.args
                    )
                    
                    # Update task arguments with collected parameters
                    for k, v in collected_params.items():
                        task.args[k] = v
                    
                    # Re-verify with the new parameters
                    validation = await self.verify_parameters_async(task.function_name, task.args)
                
                # After the loop, check if validation is still failing for other reasons
                if not validation["success"]:
                    msg = validation["message"]
                    self.kb.log_interaction(task.name, msg, agent="Emil", function=task.function_name)
                    await self.kb.set_item_async("emil_error", msg)
                    task.result = msg
                    return msg

            try:
                result = await asyncio.to_thread(func, self.kb, **task.args)
                task.result = result

                if task.function_name == "process_emil_request":
                    await self.kb.set_item_async("emil_result", result, category="energy_models")
                    if isinstance(result, dict):
                        await self.kb.set_item_async("latest_model_file", result.get("file"))
                        await self.kb.set_item_async("latest_model_details", result, category="models")
                        for key in ["location", "generation", "generation_type", "energy_carrier"]:
                            if key in result:
                                await self.kb.set_item_async(f"latest_model_{key}", result[key])
                else:
                    await self.kb.set_item_async(f"emil_{task.function_name}_result", result, category=task.function_name)
                    final_msg = result.get("message") if isinstance(result, dict) else str(result)
                    await self.kb.set_item_async("final_report", final_msg)

                return result
            except Exception as e:
                msg = f"Error executing {task.function_name}: {str(e)}"
                self.kb.log_interaction(task.name, msg, agent="Emil", function=task.function_name)
                task.result = msg
                return msg

        msg = f"Emil has no function for task: {task.name}"
        self.kb.log_interaction(task.name, msg, agent="Emil", function=task.function_name)
        task.result = msg
        return msg