from .base_agent import BaseAgent
import os
import json
import time
import re
import asyncio
from typing import List, Dict, Any
from core.task_manager import Task
from utils.function_logger import log_function_call
from utils.open_ai_utils import open_ai_categorisation_async, run_open_ai_ns_async

class Nova(BaseAgent):
    def __init__(self, name, kb, function_map, verbose=False):
        super().__init__(name, kb, function_map)
        self.verbose = verbose

    @log_function_call
    async def handle_task_async(self, task: Task):
        if self.verbose:
            print(f"[{time.strftime('%X')}] Nova handling task: {task.name} with function: {task.function_name}")
        start_time = time.time()

        self.kb.log_interaction(f"Task: {task.name}", "Starting execution", agent="Nova", function=task.function_name)

        func = self.function_map.get(task.function_name)
        async_func = self.async_function_map.get(task.function_name)

        if func or async_func:
            try:
                result = await async_func(self.kb, **task.args) if async_func else await asyncio.to_thread(func, self.kb, **task.args)
                task.result = result

                await self.kb.set_item_async(f"{task.function_name}_result_{id(task)}", result, category=task.function_name.replace("_", ""))

                if task.function_name == "answer_general_question":
                    await self.kb.set_item_async("general_answer", result, category="general_knowledge")
                if task.function_name == "do_maths":
                    await self.kb.set_item_async("math_result", result, category="math_calculations")

                self.kb.log_interaction(f"Task: {task.name}", "Completed successfully", agent="Nova", function=task.function_name)
                if self.verbose:
                    print(f"[{time.strftime('%X')}] Task completed in {time.time() - start_time:.2f} seconds")
                return result

            except Exception as e:
                error_msg = f"Error executing {task.function_name}: {str(e)}"
                print(error_msg)
                self.kb.log_interaction(f"Task: {task.name}", error_msg, agent="Nova", function=task.function_name)
                task.result = error_msg
                return error_msg

        else:
            error_msg = f"Nova has no function mapped for: {task.name} ({task.function_name})"
            print(error_msg)
            self.kb.log_interaction(f"Task: {task.name}", "Function not found", agent="Nova", function=task.function_name)
            task.result = error_msg
            return error_msg

    @log_function_call
    async def create_task_list_from_prompt_async(self, prompt: str) -> List[Task]:
        tasks = []
        self.kb.log_interaction(prompt, "Creating task list", agent="Nova", function="create_task_list_from_prompt_async")

        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, "Nova_function_map_enhanced.csv")
        
        # ENHANCED: Identify multiple intents with better detection
        intents = await self.identify_multiple_intents_async(prompt)

        if self.verbose:
            print(f"\nIdentified {len(intents)} intent(s):")
            for i, intent in enumerate(intents, 1):
                print(f"  Intent {i}: {intent['intent']}")

        # IMPROVED: Better detection of model, analyze, and report intents
        model_intents = []
        analyze_intents = []
        report_intents = []
        other_intents = []
        
        # More robust intent classification
        for intent in intents:
            intent_text = intent["intent"].lower()
            
            # Model detection - look for various model-related terms
            model_keywords = ['model', 'build', 'create', 'generate', 'design', 'construct']
            energy_keywords = ['energy', 'solar', 'wind', 'hydro', 'electricity', 'power', 'generation']
            
            if (any(mk in intent_text for mk in model_keywords) and 
                any(ek in intent_text for ek in energy_keywords)):
                model_intents.append(intent)
            # Analysis detection
            elif any(word in intent_text for word in ['analyz', 'analyse', 'study', 'examine', 'evaluate']):
                analyze_intents.append(intent)
            # Report detection - more comprehensive
            elif any(word in intent_text for word in ['report', 'summary', 'document', 'write', 'generate report', 'create report']):
                report_intents.append(intent)
            else:
                other_intents.append(intent)

        if self.verbose:
            print(f"Classified intents:")
            print(f"  Model intents: {len(model_intents)}")
            print(f"  Analyze intents: {len(analyze_intents)}")
            print(f"  Report intents: {len(report_intents)}")
            print(f"  Other intents: {len(other_intents)}")

        # Process other intents first (math, general questions, etc.)
        for intent in other_intents:
            task = await self._create_task_with_category(intent["intent"], csv_path)
            tasks.append(task)

        # CRITICAL FIX: Handle sequential workflows more reliably
        if model_intents:
            if self.verbose:
                print("DETECTED: Model creation task(s)")
            
            # Create the main model task
            model_intent = model_intents[0]  # Take the first model intent
            model_task = await self._create_task_with_category(
                model_intent["intent"], csv_path, "Emil", "process_emil_request"
            )
            
            # ENHANCED: Add subtasks for analysis and/or reports
            if analyze_intents:
                if self.verbose:
                    print("DETECTED: Adding analysis subtask")
                analyze_intent = analyze_intents[0]
                analyze_task = await self._create_task_with_category(
                    analyze_intent["intent"], csv_path, "Emil", "analyze_results"
                )
                model_task.sub_tasks.append(analyze_task)
                
                # If we have both analysis and report, report should be subtask of analysis
                if report_intents:
                    if self.verbose:
                        print("DETECTED: Adding report subtask to analysis")
                    report_intent = report_intents[0]
                    report_task = await self._create_task_with_category(
                        report_intent["intent"], csv_path, "Lola", "write_report"
                    )
                    analyze_task.sub_tasks.append(report_task)
            
            elif report_intents:
                # Direct model → report (no analysis)
                if self.verbose:
                    print("DETECTED: Adding direct report subtask")
                report_intent = report_intents[0]
                report_task = await self._create_task_with_category(
                    report_intent["intent"], csv_path, "Lola", "write_report"
                )
                model_task.sub_tasks.append(report_task)
            
            tasks.append(model_task)
            
        else:
            # Handle standalone analysis and report intents
            for intent in analyze_intents:
                task = await self._create_task_with_category(intent["intent"], csv_path)
                tasks.append(task)
                
            for intent in report_intents:
                task = await self._create_task_with_category(intent["intent"], csv_path)
                tasks.append(task)

        if self.verbose:
            print(f"\nFinal task structure:")
            for i, task in enumerate(tasks, 1):
                print(f"  Task {i}: {task.name} (Agent: {task.agent})")
                for j, subtask in enumerate(task.sub_tasks, 1):
                    print(f"    Subtask {i}.{j}: {subtask.name} (Agent: {subtask.agent})")

        return tasks

    
    async def create_task_list_from_prompt_async(self, prompt: str) -> List[Task]:
        tasks = []
        self.kb.log_interaction(prompt, "Creating task list", agent="Nova", function="create_task_list_from_prompt_async")

        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, "Nova_function_map_enhanced.csv")
        
        # ENHANCED: Identify multiple intents with better detection
        intents = await self.identify_multiple_intents_async(prompt)

        if self.verbose:
            print(f"\nIdentified {len(intents)} intent(s):")
            for i, intent in enumerate(intents, 1):
                print(f"  Intent {i}: {intent['intent']}")

        # Process intents and create tasks
        for intent in intents:
            intent_text = intent["intent"]
            
            # FIXED: Create task with FULL intent text, no truncation
            task = await self._create_task_with_category(intent_text, csv_path)
            
            # CRITICAL: Ensure the full original prompt is preserved
            task.args["original_prompt"] = prompt  # Store the original full prompt
            task.args["prompt"] = intent_text      # Store the intent text
            task.args["full_prompt"] = intent_text # Also for compatibility
            
            tasks.append(task)

        if self.verbose:
            print(f"\nFinal task structure:")
            for i, task in enumerate(tasks, 1):
                print(f"  Task {i}: {task.name} (Agent: {task.agent})")
                print(f"    Full prompt: {task.args.get('prompt', 'No prompt')}")

        return tasks    
        
    
    
    
    async def _create_task_with_category(self, intent_text: str, csv_path: str, agent: str = None, function: str = None) -> Task:
        # First check if this is a math query using regex
        import re
        math_patterns = [
            r'\d+\s*[\+\-\*\/]\s*\d+',  # Simple operations like 2+2
            r'what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+',  # "What is 2+2"
            r'calculate\s+\d+',  # "Calculate 25"
        ]
        
        # If it matches a math pattern, directly assign the function
        is_math = any(re.search(pattern, intent_text.lower()) for pattern in math_patterns)
        
        if is_math:
            if self.verbose:
                print(f"⚠️ Assigning do_maths function for: '{intent_text}'")
            function = "do_maths"
            agent = "Nova"
            category = "math and logic"
        else:
            # Normal categorization for non-math queries
            category = await open_ai_categorisation_async(intent_text, csv_path)
            if self.verbose:
                print(f"Intent '{intent_text}' categorized as: {category}")

            if not function:
                function = {
                    "copywriting and proofreading": "write_report",
                    "energy model": "process_emil_request",
                    "math and logic": "do_maths",
                    "general knowledge": "answer_general_question",
                    "general_question": "answer_general_question"
                }.get(category.lower(), None)

            # Determine the appropriate agent based on function
            if not agent:
                agent = {
                    "write_report": "Lola",
                    "process_emil_request": "Emil",
                    "do_maths": "Nova",
                    "answer_general_question": "Nova"
                }.get(function, "Nova")

        return Task(
            name=f"Handle Intent: {intent_text[:30]}...",
            description=f"Process intent categorized as {category}",
            agent=agent,
            function_name=function,
            args={
                "prompt": intent_text,
                "full_prompt": intent_text
            }
        )
    
    
    async def _create_task_with_category(self, intent_text: str, csv_path: str, agent: str = None, function: str = None) -> Task:
        # First check if this is a math query using regex
        import re
        math_patterns = [
            r'\d+\s*[\+\-\*\/]\s*\d+',  # Simple operations like 2+2
            r'what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+',  # "What is 2+2"
            r'calculate\s+\d+',  # "Calculate 25"
        ]
        
        # If it matches a math pattern, directly assign the function
        is_math = any(re.search(pattern, intent_text.lower()) for pattern in math_patterns)
        
        if is_math:
            if self.verbose:
                print(f"⚠️ Assigning do_maths function for: '{intent_text}'")
            function = "do_maths"
            agent = "Nova"
            category = "math and logic"
        else:
            # Normal categorization for non-math queries
            category = await open_ai_categorisation_async(intent_text, csv_path)
            if self.verbose:
                print(f"Intent '{intent_text}' categorized as: {category}")

            if not function:
                function = {
                    "copywriting and proofreading": "write_report",
                    "energy model": "process_emil_request",
                    "math and logic": "do_maths",
                    "general knowledge": "answer_general_question",
                    "general_question": "answer_general_question"
                }.get(category.lower(), None)

            # Determine the appropriate agent based on function
            if not agent:
                agent = {
                    "write_report": "Lola",
                    "process_emil_request": "Emil",
                    "do_maths": "Nova",
                    "answer_general_question": "Nova"
                }.get(function, "Nova")

        # FIXED: Don't truncate the task name, and pass the full prompt
        return Task(
            name=f"Handle Intent: {intent_text}",  # REMOVED truncation [:30]
            description=f"Process intent categorized as {category}",
            agent=agent,
            function_name=function,
            args={
                "prompt": intent_text,        # Full prompt, not truncated
                "full_prompt": intent_text    # Also store in full_prompt for compatibility
            }
        )

    
    
    
    async def identify_multiple_intents_async(self, prompt: str) -> List[Dict[str, str]]:
        # ENHANCED: Better intent detection with more robust parsing
        context = """
        Extract individual tasks or intents from the user's input. Each intent should be a distinct actionable request.
        
        Examples:
        Input: 'Build a model for France. Write a report. What is the capital of France?'
        Output: { "intents": [ {"intent": "Build a model for France"}, {"intent": "Write a report"}, {"intent": "What is the capital of France"} ] }
        
        Input: 'build a wind model for spain and write a report'
        Output: { "intents": [ {"intent": "build a wind model for spain"}, {"intent": "write a report"} ] }
        
        Input: 'Calculate 2+2 and what is the population of Germany'
        Output: { "intents": [ {"intent": "Calculate 2+2"}, {"intent": "what is the population of Germany"} ] }
        
        Return only valid JSON with the "intents" array. Each intent should be a separate actionable task.
        """
        
        try:
            result = await run_open_ai_ns_async(prompt, context, model="gpt-4.1-nano")
            # Find JSON part if present
            if "{" in result:
                json_str = result[result.index("{"):]
                try:
                    parsed = json.loads(json_str)
                    if "intents" in parsed and isinstance(parsed["intents"], list):
                        # Ensure each item is properly formatted
                        formatted_intents = []
                        for item in parsed["intents"]:
                            if isinstance(item, dict) and "intent" in item:
                                formatted_intents.append({"intent": item["intent"]})
                            elif isinstance(item, str):
                                formatted_intents.append({"intent": item})
                        
                        if self.verbose:
                            print(f"Successfully parsed {len(formatted_intents)} intents from LLM")
                        
                        return formatted_intents
                except json.JSONDecodeError as e:
                    if self.verbose:
                        print(f"JSON parsing failed: {e}")
            
            # Fallback: Split by common separators and conjunctions
            if self.verbose:
                print("Using fallback intent detection")
            
            # Enhanced fallback splitting
            separators = [
                r'\s+and\s+(?:then\s+)?(?:also\s+)?',  # "and", "and then", "and also"
                r'\s*[.!?]+\s*',  # Punctuation
                r'\s*,\s*(?:and\s+)?(?:then\s+)?',  # Commas with optional "and" or "then"
                r'\s+then\s+',  # "then"
                r'\s+also\s+',  # "also"
            ]
            
            intents = [prompt]  # Start with the whole prompt
            
            for separator in separators:
                new_intents = []
                for intent in intents:
                    parts = re.split(separator, intent, flags=re.IGNORECASE)
                    new_intents.extend([part.strip() for part in parts if part.strip()])
                intents = new_intents
            
            # Filter out very short intents (likely artifacts)
            intents = [intent for intent in intents if len(intent.split()) >= 2]
            
            # If we end up with just one intent that's the same as the original, 
            # but it clearly has multiple tasks, try a simpler split
            if len(intents) == 1 and intents[0] == prompt:
                # Look for obvious task boundaries
                if " and " in prompt.lower():
                    simple_split = [part.strip() for part in prompt.split(" and ") if part.strip()]
                    if len(simple_split) > 1:
                        intents = simple_split
            
            formatted_intents = [{"intent": intent} for intent in intents]
            
            if self.verbose:
                print(f"Fallback detected {len(formatted_intents)} intents")
                
            return formatted_intents
            
        except Exception as e:
            if self.verbose:
                print(f"Error in identify_multiple_intents_async: {e}")
            return [{"intent": prompt}]  # Return whole prompt as single intent
        



    async def identify_multiple_intents_async(self, prompt: str) -> List[Dict[str, str]]:
        # ENHANCED: Better intent detection with more robust parsing
        context = """
        Extract individual tasks or intents from the user's input. Each intent should be a distinct actionable request.
        
        IMPORTANT: If the input is a single coherent request (like "build a wind model for spain, greece and denmark"), 
        treat it as ONE intent, not multiple intents.
        
        Only split into multiple intents if there are clearly separate, distinct tasks.
        
        Examples:
        Input: 'Build a model for France. Write a report. What is the capital of France?'
        Output: { "intents": [ {"intent": "Build a model for France"}, {"intent": "Write a report"}, {"intent": "What is the capital of France"} ] }
        
        Input: 'build a wind model for spain, greece and denmark'
        Output: { "intents": [ {"intent": "build a wind model for spain, greece and denmark"} ] }
        
        Input: 'Create a solar model for Germany and also calculate 2+2'
        Output: { "intents": [ {"intent": "Create a solar model for Germany"}, {"intent": "calculate 2+2"} ] }
        
        Return only valid JSON with the "intents" array. Do NOT split single coherent requests into multiple intents.
        """
        
        try:
            # Use LLM to analyze if this is a multi-intent query
            result = await run_open_ai_ns_async(prompt, context, model="gpt-4.1-nano")
            
            # Find JSON part if present
            if "{" in result:
                json_str = result[result.index("{"):]
                try:
                    parsed = json.loads(json_str)
                    if "intents" in parsed and isinstance(parsed["intents"], list):
                        # Ensure each item is properly formatted
                        formatted_intents = []
                        for item in parsed["intents"]:
                            if isinstance(item, dict) and "intent" in item:
                                formatted_intents.append({"intent": item["intent"]})
                            elif isinstance(item, str):
                                formatted_intents.append({"intent": item})
                        
                        if self.verbose:
                            print(f"Successfully parsed {len(formatted_intents)} intents from LLM")
                        
                        return formatted_intents
                except json.JSONDecodeError as e:
                    if self.verbose:
                        print(f"JSON parsing failed: {e}")
            
            # Fallback: Check if this looks like a single coherent request
            if self.verbose:
                print("Using fallback intent detection")
            
            # ENHANCED fallback: Be more conservative about splitting
            # Only split if there are clear separators AND multiple distinct task types
            
            # Check for clear task separators
            strong_separators = [
                r'\s*[.!?]+\s+(?:and\s+)?(?:then\s+)?(?:also\s+)?',  # Punctuation followed by connectors
                r'\s+then\s+',  # "then"
                r'\s+also\s+',  # "also" (but only if followed by a verb)
            ]
            
            # Check for multiple distinct task types
            task_indicators = {
                'model': ['build', 'create', 'make', 'generate', 'design'],
                'calculate': ['calculate', 'compute', 'solve', 'what is'],
                'report': ['write', 'create report', 'generate report'],
                'question': ['what', 'how', 'why', 'when', 'where']
            }
            
            # Count how many different task types are present
            task_types_found = set()
            prompt_lower = prompt.lower()
            
            for task_type, indicators in task_indicators.items():
                if any(indicator in prompt_lower for indicator in indicators):
                    task_types_found.add(task_type)
            
            # Only split if we have multiple task types AND clear separators
            has_multiple_tasks = len(task_types_found) > 1
            has_separators = any(re.search(sep, prompt) for sep in strong_separators)
            
            if has_multiple_tasks and has_separators:
                # Try to split carefully
                intents = [prompt]  # Start with the whole prompt
                
                for separator in strong_separators:
                    new_intents = []
                    for intent in intents:
                        parts = re.split(separator, intent, flags=re.IGNORECASE)
                        new_intents.extend([part.strip() for part in parts if part.strip()])
                    intents = new_intents
                
                # Filter out very short intents (likely artifacts)
                intents = [intent for intent in intents if len(intent.split()) >= 3]
                
                # If splitting resulted in meaningful parts, use them
                if len(intents) > 1:
                    formatted_intents = [{"intent": intent} for intent in intents]
                    if self.verbose:
                        print(f"Fallback split into {len(formatted_intents)} intents")
                    return formatted_intents
            
            # Default: treat as single intent (most common case)
            if self.verbose:
                print("Treating as single intent (default)")
            
            return [{"intent": prompt}]
            
        except Exception as e:
            if self.verbose:
                print(f"Error in identify_multiple_intents_async: {e}")
            return [{"intent": prompt}]  # Return whole prompt as single intent        
        

