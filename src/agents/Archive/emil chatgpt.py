import os
import sys
import asyncio
import datetime
import inspect

from .plexos_base_model_final import (
    initiate_file,
    filter_data,
    process_base_model_task
)


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Emil(BaseAgent):
    def __init__(self, name, kb, function_map, verbose=False):
        super().__init__(name, kb, function_map)
        self.verbose = verbose

        base_dir = os.path.dirname(os.path.abspath(__file__))
        plexos_models_dir = os.path.join(base_dir, "PLEXOS_models")
        plexos_inputs_dir = os.path.join(base_dir, "PLEXOS_inputs")

        os.makedirs(plexos_models_dir, exist_ok=True)
        os.makedirs(plexos_inputs_dir, exist_ok=True)

    async def verify_parameters_async(self, function_name: str, task_args: dict) -> dict:
        if function_name == 'process_emil_request' and task_args.get('prompt'):
            location = task_args.get('location')
            if not location or location == "Unknown":
                return {
                    "success": False,
                    "missing": ["location"],
                    "message": "Please specify a location (country or region) for the energy model."
                }

            generation = task_args.get('generation') or task_args.get('generation_type')
            if not generation:
                return {
                    "success": False,
                    "missing": ["generation"],
                    "message": "Please specify a generation type (solar, wind, hydro, etc.)."
                }

            return {"success": True, "missing": [], "message": "Prompt and essential parameters provided for Emil request"}

        if function_name == 'analyze_results':
            return {"success": True, "missing": [], "message": "Analysis tasks don't require explicit parameters"}

        if function_name not in self.function_map:
            return {"success": False, "missing": [], "message": f"Function {function_name} not found in Emil's function map"}

        func = self.function_map[function_name]
        sig = inspect.signature(func)
        required_params = [
            param.name for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty and param.name not in ('self', 'kb')
        ]
        missing = [param for param in required_params if param not in task_args]

        if missing:
            return {"success": False, "missing": missing, "message": f"Missing required parameters: {', '.join(missing)}"}

        return {"success": True, "missing": [], "message": "All required parameters are present"}

    async def handle_task_async(self, task):
        if self.verbose:
            print(f"Emil handling task asynchronously: {task.name}")

        session_context = task.session_context or {}
        if "emil" not in session_context:
            session_context["emil"] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "tasks_processed": []
            }

        if task.function_name == "process_emil_request" and task.args.get("prompt"):
            loop = asyncio.get_event_loop()
            extracted_params = await loop.run_in_executor(
                None,
                extract_model_parameters_with_llm_correction,
                task.args.get("prompt", "")
            )

            if extracted_params.get('locations'):
                task.args['locations'] = extracted_params['locations']
                task.args['location'] = extracted_params['locations'][0]

            if extracted_params.get('generation_types'):
                task.args['generation_types'] = extracted_params['generation_types']
                task.args['generation'] = extracted_params['generation_types'][0]

            if extracted_params.get('energy_carriers'):
                task.args['energy_carriers'] = extracted_params['energy_carriers']
                task.args['energy_carrier'] = extracted_params['energy_carriers'][0]

        if task.function_name in self.function_map:
            validation = await self.verify_parameters_async(task.function_name, task.args)

            if not validation["success"]:
                while not validation["success"] and validation.get("missing"):
                    collected_params = await get_missing_parameters_async(
                        function_name=task.function_name,
                        missing_params=validation["missing"],
                        initial_args=task.args
                    )
                    task.args.update(collected_params)
                    validation = await self.verify_parameters_async(task.function_name, task.args)

                if not validation["success"]:
                    task.result = validation["message"]
                    return validation["message"]

            try:
                if task.function_name == "process_emil_request":
                    location = task.args.get('location')
                    generation = task.args.get('generation')
                    energy_carrier = task.args.get('energy_carrier', 'electricity')
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = f"{location}_{generation}_{energy_carrier}_{timestamp}.xml"

                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    models_dir = os.path.join(script_dir, "PLEXOS_models")
                    os.makedirs(models_dir, exist_ok=True)
                    model_file = os.path.join(models_dir, model_name)

                    try:
                        import pandas as pd
                        plexos_sheet_path = os.path.join(script_dir, "PLEXOS_inputs", "PLEXOS_Model_Builder_v2.xlsx")
                        if not os.path.exists(plexos_sheet_path):
                            raise FileNotFoundError("PLEXOS configuration file not found")

                        plexos_prompt_sheet = pd.read_excel(plexos_sheet_path, sheet_name=None)
                        high_level_prompt = f"build a {generation} model for {location}"

                        from PLEXOS_functions.plexos_build_functions_final import load_plexos_xml
                        db = load_plexos_xml(blank=True, source_file=model_file)
                        process_base_model_task(db, plexos_prompt_sheet, high_level_prompt)

                        result = {
                            "status": "success",
                            "message": f"Created {generation} {energy_carrier} model for {location}",
                            "file": model_file,
                            "location": location,
                            "generation_type": generation,
                            "energy_carrier": energy_carrier
                        }
                    except Exception as e:
                        from core.functions_registery import create_simple_xml
                        result = create_simple_xml(location, generation, energy_carrier, model_file)

                    task.result = result
                    return result
                else:
                    func = self.function_map[task.function_name]
                    result = await asyncio.to_thread(func, self.kb, **task.args)
                    task.result = result
                    return result

            except Exception as e:
                task.result = f"Error executing {task.function_name}: {str(e)}"
                return task.result

        task.result = f"Emil has no function for task: {task.name}"
        return task.result
