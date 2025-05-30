import asyncio
import datetime
import json
import os
import inspect
from typing import Any, Dict, List

from .base_agent import BaseAgent
from core.knowledge_base import KnowledgeBase
from core.task_manager import Task
from utils.function_logger import log_function_call
from utils.open_ai_utils import run_open_ai_ns_async

# Import parameter collection
try:
    from .parameter_collection import get_missing_parameters_async
except ImportError:
    from .cli_parameter_collection import get_missing_parameters_cli_async as get_missing_parameters_async


async def extract_energy_parameters_from_prompt(prompt: str) -> dict:
    """Extract energy model parameters from prompts."""
    system_msg = """
You are an expert assistant for extracting energy model configuration parameters from prompts.

Return this JSON:
{
  "location": "Country or region",
  "generation_type": "solar, wind, hydro, nuclear, etc.",
  "energy_carrier": "electricity, hydrogen, methane, etc."
}

Use null if any value is missing.
"""
    try:
        response = await run_open_ai_ns_async(prompt, system_msg)
        return {k: v for k, v in json.loads(response).items() if v}
    except Exception as e:
        print(f"❌ Failed to extract energy parameters: {e}")
        return {}


# In src/agents/emil.py, replace the extract_energy_parameters_from_prompt function:

async def extract_energy_parameters_from_prompt(prompt: str) -> dict:
    """Extract energy model parameters from prompts with enhanced detection."""
    
    # First try keyword-based extraction for common patterns
    prompt_lower = prompt.lower()
    extracted = {}
    
    # Extract generation types using keyword matching
    generation_keywords = {
        "wind": ["wind", "wind power", "wind energy", "wind generation", "wind model"],
        "solar": ["solar", "solar power", "solar energy", "solar pv", "photovoltaic", "solar model"],
        "hydro": ["hydro", "hydroelectric", "hydropower", "hydro power", "hydro model"],
        "nuclear": ["nuclear", "nuclear power", "nuclear energy", "nuclear model"],
        "thermal": ["thermal", "coal", "gas", "natural gas", "thermal model"],
        "bio": ["bio", "biomass", "biofuel", "biogas", "bio model"]
    }
    
    # Check for generation type
    for gen_type, keywords in generation_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            extracted["generation"] = gen_type
            print(f"🔍 Extracted generation type via keywords: {gen_type}")
            break
    
    # Extract countries using keyword matching
    country_keywords = {
        "spain": ["spain", "spanish", "es"],
        "greece": ["greece", "greek", "gr"],
        "denmark": ["denmark", "danish", "dk"],
        "france": ["france", "french", "fr"],
        "germany": ["germany", "german", "de"],
        "italy": ["italy", "italian", "it"],
        "uk": ["uk", "united kingdom", "britain", "england", "gb"],
        "netherlands": ["netherlands", "dutch", "holland", "nl"],
        "belgium": ["belgium", "belgian", "be"],
        "portugal": ["portugal", "portuguese", "pt"],
        "norway": ["norway", "norwegian", "no"],
        "sweden": ["sweden", "swedish", "se"],
        "finland": ["finland", "finnish", "fi"]
    }
    
    found_countries = []
    for country, keywords in country_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            found_countries.append(country.capitalize())
    
    if found_countries:
        if len(found_countries) == 1:
            extracted["location"] = found_countries[0]
        else:
            extracted["location"] = ", ".join(found_countries)
        print(f"🔍 Extracted countries via keywords: {found_countries}")
    
    # Extract energy carrier
    if any(word in prompt_lower for word in ["hydrogen", "h2"]):
        extracted["energy_carrier"] = "hydrogen"
    elif any(word in prompt_lower for word in ["methane", "ch4", "gas"]):
        extracted["energy_carrier"] = "methane"
    else:
        extracted["energy_carrier"] = "electricity"  # Default
    
    # If keyword extraction worked, return early
    if extracted.get("generation") and extracted.get("location"):
        print(f"✅ Successfully extracted via keywords: {extracted}")
        return extracted
    
    # Fallback to LLM extraction if keywords didn't work
    system_msg = """
You are an expert assistant for extracting energy model configuration parameters from prompts.

Extract ALL countries mentioned in the prompt, not just one.
For generation types, look for: wind, solar, hydro, nuclear, thermal, bio, etc.

Return this JSON:
{
  "location": "All countries mentioned, comma-separated",
  "generation": "wind, solar, hydro, nuclear, thermal, bio, etc.",
  "energy_carrier": "electricity, hydrogen, methane, etc."
}

Use null if any value is missing.
IMPORTANT: If multiple countries are mentioned (like "spain, greece and denmark"), include ALL of them.
"""
    try:
        response = await run_open_ai_ns_async(prompt, system_msg, model="gpt-4.1-nano")
        llm_result = json.loads(response)
        
        # Merge keyword results with LLM results, preferring keyword results
        final_result = {}
        for key in ["location", "generation", "energy_carrier"]:
            if extracted.get(key):
                final_result[key] = extracted[key]
            elif llm_result.get(key):
                final_result[key] = llm_result[key]
        
        print(f"✅ Final extracted parameters: {final_result}")
        return {k: v for k, v in final_result.items() if v}
    except Exception as e:
        print(f"❌ Failed to extract energy parameters via LLM: {e}")
        return extracted  # Return keyword results as fallback


async def extract_energy_parameters_from_prompt(prompt: str) -> dict:
    """Extract energy model parameters from prompts with accurate country detection."""
    
    prompt_lower = prompt.lower()
    extracted = {}
    
    # Extract generation types using keyword matching
    generation_keywords = {
        "wind": ["wind", "wind power", "wind energy", "wind generation", "wind model"],
        "solar": ["solar", "solar power", "solar energy", "solar pv", "photovoltaic", "solar model"],
        "hydro": ["hydro", "hydroelectric", "hydropower", "hydro power", "hydro model"],
        "nuclear": ["nuclear", "nuclear power", "nuclear energy", "nuclear model"],
        "thermal": ["thermal", "coal", "gas", "natural gas", "thermal model"],
        "bio": ["bio", "biomass", "biofuel", "biogas", "bio model"]
    }
    
    # Check for generation type
    for gen_type, keywords in generation_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            extracted["generation"] = gen_type
            print(f"🔍 Extracted generation type via keywords: {gen_type}")
            break
    
    # FIXED: More precise country extraction using word boundaries and specific patterns
    import re
    
    # Define country patterns with word boundaries to avoid false matches
    country_patterns = {
        "spain": r'\b(?:spain|spanish)\b',
        "greece": r'\b(?:greece|greek|hellenic)\b', 
        "denmark": r'\b(?:denmark|danish)\b',
        "france": r'\b(?:france|french)\b',
        "germany": r'\b(?:germany|german|deutschland)\b',
        "italy": r'\b(?:italy|italian|italia)\b',
        "uk": r'\b(?:uk|united kingdom|britain|england|great britain)\b',
        "netherlands": r'\b(?:netherlands|dutch|holland)\b',
        "belgium": r'\b(?:belgium|belgian|belgie)\b',
        "portugal": r'\b(?:portugal|portuguese)\b',
        "norway": r'\b(?:norway|norwegian|norge)\b',
        "sweden": r'\b(?:sweden|swedish|sverige)\b',
        "finland": r'\b(?:finland|finnish|suomi)\b',
        "poland": r'\b(?:poland|polish|polska)\b',
        "austria": r'\b(?:austria|austrian|osterreich)\b'
    }
    
    found_countries = []
    for country, pattern in country_patterns.items():
        if re.search(pattern, prompt_lower):
            found_countries.append(country.capitalize())
            print(f"🔍 Found country: {country.capitalize()}")
    
    if found_countries:
        if len(found_countries) == 1:
            extracted["location"] = found_countries[0]
        else:
            extracted["location"] = ", ".join(found_countries)
        print(f"🔍 Extracted countries via keywords: {found_countries}")
    
    # Extract energy carrier
    if any(word in prompt_lower for word in ["hydrogen", "h2"]):
        extracted["energy_carrier"] = "hydrogen"
    elif any(word in prompt_lower for word in ["methane", "ch4", "gas"]):
        extracted["energy_carrier"] = "methane"
    else:
        extracted["energy_carrier"] = "electricity"  # Default
    
    # If keyword extraction worked, return early
    if extracted.get("generation") and extracted.get("location"):
        print(f"✅ Successfully extracted via keywords: {extracted}")
        return extracted
    
    # Fallback to LLM extraction if keywords didn't work completely
    system_msg = """
You are an expert assistant for extracting energy model configuration parameters from prompts.

Extract ALL countries mentioned in the prompt, not just one.
For generation types, look for: wind, solar, hydro, nuclear, thermal, bio, etc.

Return this JSON:
{
  "location": "All countries mentioned, comma-separated",
  "generation": "wind, solar, hydro, nuclear, thermal, bio, etc.",
  "energy_carrier": "electricity, hydrogen, methane, etc."
}

Use null if any value is missing.
IMPORTANT: If multiple countries are mentioned (like "spain, greece and denmark"), include ALL of them.
"""
    try:
        response = await run_open_ai_ns_async(prompt, system_msg, model="gpt-4.1-nano")
        llm_result = json.loads(response)
        
        # Merge keyword results with LLM results, preferring keyword results
        final_result = {}
        for key in ["location", "generation", "energy_carrier"]:
            if extracted.get(key):
                final_result[key] = extracted[key]
            elif llm_result.get(key):
                final_result[key] = llm_result[key]
        
        print(f"✅ Final extracted parameters: {final_result}")
        return {k: v for k, v in final_result.items() if v}
    except Exception as e:
        print(f"❌ Failed to extract energy parameters via LLM: {e}")
        return extracted


async def extract_energy_parameters_from_prompt(prompt: str) -> dict:
    """Extract energy model parameters using LLM intelligence."""
    
    print(f"🧠 Using LLM to extract parameters from: '{prompt}'")
    
    # Use LLM with intelligent context understanding
    system_msg = """You are an expert at extracting energy model parameters from natural language.

TASK: Extract these parameters from the user's request:
- location: Countries mentioned (comma-separated if multiple)
- generation: Type of energy generation (wind, solar, hydro, nuclear, thermal, bio)
- energy_carrier: Energy carrier type (electricity, hydrogen, methane)

EXAMPLES:
Input: "Build a wind model for Spain, Greece and Denmark"
Output: {"location": "Spain, Greece, Denmark", "generation": "wind", "energy_carrier": "electricity"}

Input: "Create a solar model for France"
Output: {"location": "France", "generation": "solar", "energy_carrier": "electricity"}

Input: "Generate a hydrogen model for Germany and Italy"
Output: {"location": "Germany, Italy", "generation": "solar", "energy_carrier": "hydrogen"}

INSTRUCTIONS:
- Extract ALL countries mentioned, not just the first one
- Use null for missing values
- Return valid JSON only
"""
    
    try:
        # Use LLM for intelligent extraction
        response = await run_open_ai_ns_async(prompt, system_msg, model="gpt-4.1-nano")
        
        try:
            import json
            extracted = json.loads(response)
            
            # Validate and clean the extracted data
            result = {}
            if extracted.get("location"):
                result["location"] = extracted["location"]
                print(f"🧠 LLM extracted location: {result['location']}")
                
            if extracted.get("generation"):
                result["generation"] = extracted["generation"]
                print(f"🧠 LLM extracted generation: {result['generation']}")
                
            if extracted.get("energy_carrier"):
                result["energy_carrier"] = extracted["energy_carrier"]
            else:
                result["energy_carrier"] = "electricity"  # Default
                
            print(f"✅ LLM extraction successful: {result}")
            return result
            
        except json.JSONDecodeError:
            print("🔄 JSON parsing failed, using keyword fallback...")
            
    except Exception as e:
        print(f"❌ LLM extraction failed: {str(e)}")
    
    # Simple keyword fallback only if LLM fails
    prompt_lower = prompt.lower()
    fallback = {}
    
    # Generation type fallback
    if "wind" in prompt_lower:
        fallback["generation"] = "wind"
    elif "solar" in prompt_lower:
        fallback["generation"] = "solar"
    elif "hydro" in prompt_lower:
        fallback["generation"] = "hydro"
        
    # Simple country fallback
    countries = []
    if "spain" in prompt_lower:
        countries.append("Spain")
    if "greece" in prompt_lower:
        countries.append("Greece")
    if "denmark" in prompt_lower:
        countries.append("Denmark")
    if "france" in prompt_lower:
        countries.append("France")
    if "germany" in prompt_lower:
        countries.append("Germany")
        
    if countries:
        fallback["location"] = ", ".join(countries)
        
    fallback["energy_carrier"] = "electricity"
    
    print(f"🔄 Fallback extraction: {fallback}")
    return fallback


class Emil(BaseAgent):
    def __init__(self, name, kb, function_map, verbose=False):
        super().__init__(name, kb, function_map)
        self.verbose = verbose

        for d in ["PLEXOS_models", "PLEXOS_inputs", "PLEXOS_functions"]:
            os.makedirs(os.path.join(os.path.dirname(__file__), d), exist_ok=True)

    async def verify_parameters_async(self, function_name: str, task_args: dict) -> dict:
        if function_name == 'process_emil_request':
            if not task_args.get('location'):
                return {"success": False, "missing": ["location"], "message": "Please specify a location."}
            if not task_args.get('generation'):
                return {"success": False, "missing": ["generation"], "message": "Please specify a generation type."}
            return {"success": True, "missing": [], "message": "Valid parameters"}

        if function_name == 'analyze_results':
            return {"success": True, "missing": [], "message": "No parameters required"}

        if function_name not in self.function_map:
            return {"success": False, "missing": [], "message": f"Function {function_name} not found"}

        func = self.function_map[function_name]
        sig = inspect.signature(func)
        required = [p.name for p in sig.parameters.values() if p.default == inspect.Parameter.empty and p.name not in ('self', 'kb')]
        missing = [p for p in required if p not in task_args]

        if missing:
            return {"success": False, "missing": missing, "message": f"Missing: {', '.join(missing)}"}

        return {"success": True, "missing": [], "message": "All parameters present"}

    
    @log_function_call
    async def handle_task_async(self, task: Task):
        if self.verbose:
            print(f"Emil handling task: {task.name}")
        self.kb.log_interaction(task.name, "Starting execution", agent="Emil", function=task.function_name)

        # Extract missing parameters from prompt
        if task.function_name == "process_emil_request" and task.args.get("prompt"):
            extracted = await extract_energy_parameters_from_prompt(task.args["prompt"])
            for k, v in extracted.items():
                task.args.setdefault(k, v)

        if task.function_name in self.function_map:
            func = self.function_map[task.function_name]
            validation = await self.verify_parameters_async(task.function_name, task.args)

            while not validation["success"] and validation.get("missing"):
                print(f"🧩 Emil needs: {validation['missing']}")
                collected = await get_missing_parameters_async(task.function_name, validation["missing"], task.args)
                task.args.update(collected)
                validation = await self.verify_parameters_async(task.function_name, task.args)

            if not validation["success"]:
                msg = validation["message"]
                self.kb.log_interaction(task.name, msg, agent="Emil", function=task.function_name)
                await self.kb.set_item_async("emil_error", msg)
                task.result = msg
                return msg

            try:
                if task.function_name == "process_emil_request":
                    location = task.args["location"]
                    generation = task.args["generation"]
                    energy_carrier = task.args.get("energy_carrier", "electricity")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = f"{location}_{generation}_{energy_carrier}_{timestamp}.xml"
                    model_path = os.path.join(os.path.dirname(__file__), "PLEXOS_models", model_name)

                    if build_plexos_model_with_base(location, generation, energy_carrier, model_path):
                        result = {
                            "status": "success",
                            "message": f"Created {generation} {energy_carrier} model for {location}",
                            "file": model_path,
                            "location": location,
                            "generation_type": generation,
                            "energy_carrier": energy_carrier,
                            "model_type": "comprehensive_plexos"
                        }
                    else:
                        from core.functions_registery import create_simple_xml
                        result = await asyncio.to_thread(create_simple_xml, location, generation, energy_carrier, model_path)

                    task.result = result
                    await self.kb.set_item_async("emil_result", result, category="energy_models")
                    for key in ["location", "generation", "generation_type", "energy_carrier"]:
                        if key in result:
                            await self.kb.set_item_async(f"latest_model_{key}", result[key])
                    return result

                else:
                    result = await asyncio.to_thread(func, self.kb, **task.args)
                    task.result = result
                    await self.kb.set_item_async(f"emil_{task.function_name}_result", result)
                    return result

            except Exception as e:
                msg = f"❌ Error in {task.function_name}: {str(e)}"
                self.kb.log_interaction(task.name, msg, agent="Emil", function=task.function_name)
                task.result = msg
                return msg

        msg = f"Emil has no function for task: {task.name}"
        self.kb.log_interaction(task.name, msg, agent="Emil", function=task.function_name)
        task.result = msg
        return msg


    async def verify_parameters_async(self, function_name: str, task_args: dict) -> dict:
        """Enhanced parameter verification with better extraction."""
        
        if function_name == 'process_emil_request':
            # First extract parameters from prompt if available
            if task_args.get('prompt') and not task_args.get('location'):
                extracted = await extract_energy_parameters_from_prompt(task_args['prompt'])
                # Update task_args with extracted parameters
                for key, value in extracted.items():
                    if key not in task_args or not task_args[key]:
                        task_args[key] = value
                        print(f"🔧 Auto-filled {key}: {value}")
            
            # Check for location
            if not task_args.get('location'):
                return {"success": False, "missing": ["location"], "message": "Please specify a location."}
            
            # Check for generation - be more flexible in key names
            generation_keys = ['generation', 'generation_type', 'gen_type']
            has_generation = any(task_args.get(key) for key in generation_keys)
            
            if not has_generation:
                return {"success": False, "missing": ["generation"], "message": "Please specify a generation type."}
            
            # Standardize the generation key name
            for key in generation_keys:
                if task_args.get(key):
                    task_args['generation'] = task_args[key]
                    break
                    
            return {"success": True, "missing": [], "message": "Valid parameters"}

        if function_name == 'analyze_results':
            return {"success": True, "missing": [], "message": "No parameters required"}

        if function_name not in self.function_map:
            return {"success": False, "missing": [], "message": f"Function {function_name} not found"}

        func = self.function_map[function_name]
        sig = inspect.signature(func)
        required = [p.name for p in sig.parameters.values() if p.default == inspect.Parameter.empty and p.name not in ('self', 'kb')]
        missing = [p for p in required if p not in task_args]

        if missing:
            return {"success": False, "missing": missing, "message": f"Missing: {', '.join(missing)}"}

        return {"success": True, "missing": [], "message": "All parameters present"}


    # Also update the handle_task_async method to better handle parameter extraction:

    @log_function_call
    async def handle_task_async(self, task: Task):
        if self.verbose:
            print(f"Emil handling task: {task.name}")
        self.kb.log_interaction(task.name, "Starting execution", agent="Emil", function=task.function_name)

        # ENHANCED: Extract parameters from prompt BEFORE validation
        if task.function_name == "process_emil_request" and task.args.get("prompt"):
            print(f"🔍 Extracting parameters from prompt: {task.args['prompt']}")
            extracted = await extract_energy_parameters_from_prompt(task.args["prompt"])
            
            # Update task args with extracted parameters (don't override existing values)
            for key, value in extracted.items():
                if key not in task.args or not task.args[key]:
                    task.args[key] = value
                    print(f"✅ Auto-filled parameter {key}: {value}")

        if task.function_name in self.function_map:
            func = self.function_map[task.function_name]
            validation = await self.verify_parameters_async(task.function_name, task.args)

            # Only ask for missing parameters if validation fails
            while not validation["success"] and validation.get("missing"):
                print(f"🧩 Emil needs: {validation['missing']}")
                collected = await get_missing_parameters_async(task.function_name, validation["missing"], task.args)
                task.args.update(collected)
                validation = await self.verify_parameters_async(task.function_name, task.args)

            if not validation["success"]:
                msg = validation["message"]
                self.kb.log_interaction(task.name, msg, agent="Emil", function=task.function_name)
                await self.kb.set_item_async("emil_error", msg)
                task.result = msg
                return msg

            try:
                if task.function_name == "process_emil_request":
                    # Use the standardized parameter names
                    location = task.args["location"]
                    generation = task.args["generation"]  # Should be standardized by now
                    energy_carrier = task.args.get("energy_carrier", "electricity")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = f"{location}_{generation}_{energy_carrier}_{timestamp}.xml"
                    model_path = os.path.join(os.path.dirname(__file__), "PLEXOS_models", model_name)

                    if build_plexos_model_with_base(location, generation, energy_carrier, model_path):
                        result = {
                            "status": "success",
                            "message": f"Created {generation} {energy_carrier} model for {location}",
                            "file": model_path,
                            "location": location,
                            "generation_type": generation,
                            "energy_carrier": energy_carrier,
                            "model_type": "comprehensive_plexos"
                        }
                    else:
                        from core.functions_registery import create_simple_xml
                        result = await asyncio.to_thread(create_simple_xml, location, generation, energy_carrier, model_path)

                    task.result = result
                    await self.kb.set_item_async("emil_result", result, category="energy_models")
                    for key in ["location", "generation", "generation_type", "energy_carrier"]:
                        if key in result:
                            await self.kb.set_item_async(f"latest_model_{key}", result[key])
                    return result

                else:
                    result = await asyncio.to_thread(func, self.kb, **task.args)
                    task.result = result
                    await self.kb.set_item_async(f"emil_{task.function_name}_result", result)
                    return result

            except Exception as e:
                msg = f"❌ Error in {task.function_name}: {str(e)}"
                self.kb.log_interaction(task.name, msg, agent="Emil", function=task.function_name)
                task.result = msg
                return msg

        msg = f"Emil has no function for task: {task.name}"
        self.kb.log_interaction(task.name, msg, agent="Emil", function=task.function_name)
        task.result = msg
        return msg





def build_plexos_model_with_base(location, generation, energy_carrier, model_file):
    try:
        from .plexos_base_model_final import process_base_model_task
        from .PLEXOS_functions.plexos_build_functions_final import load_plexos_xml
        import pandas as pd

        inputs_path = os.path.join(os.path.dirname(__file__), "PLEXOS_inputs", "PLEXOS_Model_Builder_v2.xlsx")
        if not os.path.exists(inputs_path):
            print(f"❌ Missing input Excel: {inputs_path}")
            return False

        plexos_prompt_sheet = pd.read_excel(inputs_path, sheet_name=None)
        prompt = f"build a {generation} model for {location}"
        db = load_plexos_xml(blank=True, source_file=model_file)
        process_base_model_task(db, plexos_prompt_sheet, prompt)
        return True

    except Exception as e:
        print(f"❌ Failed PLEXOS model build: {e}")
        return False
