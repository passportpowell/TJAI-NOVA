import asyncio
import os


import json
import warnings
import datetime
from agents import Nova, Emil, Ivan, Lola
from core.knowledge_base import KnowledgeBase
from core.session_manager import *
from core.functions_registery import *
from utils.csv_function_mapper import FunctionMapLoader
from utils.function_logger import log_function_call
from utils.do_maths import do_maths
from utils.general_knowledge import answer_general_question
from utils.open_ai_utils import (
    ai_chat_session,
    ai_spoken_chat_session,
    run_open_ai_ns_async,
    open_ai_categorisation_async
)

# In src/main.py, find the imports section (around line 10-15) and make sure it includes:

from core.functions_registery import *
from core.functions_registery import (
    build_plexos_model,
    run_plexos_model,
    analyze_results,
    write_report,
    generate_python_script,
    extract_model_parameters,
    create_single_location_model,
    create_simple_xml,
    create_multi_location_model,
    create_simple_multi_location_xml,
    create_comprehensive_model,
    create_simple_comprehensive_xml,
    process_emil_request_enhanced,  # ✅ IMPORT THE ENHANCED FUNCTION
    NOVA_FUNCTIONS,
    EMIL_FUNCTIONS,
    IVAN_FUNCTIONS,
    LOLA_FUNCTIONS
)

# In src/main.py
import atexit

# Register function to save session state on exit
def save_session_state():
    # Get the current KB and session manager
    kb = KnowledgeBase(storage_path="knowledge_db", use_persistence=True)
    current_session = kb.get_item("current_session")
    current_file = kb.get_item("current_session_file")
    
    if current_session and current_file:
        try:
            # Make sure any pending changes are saved
            with open(current_file, 'r') as f:
                session_data = json.load(f)
                
            session_data["metadata"]["last_modified"] = datetime.datetime.now().isoformat()
            
            with open(current_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            print(f"\nSession state saved: {current_session}")
        except Exception as e:
            print(f"Error saving session state: {str(e)}")

# Register the exit handler
atexit.register(save_session_state)


@log_function_call
async def interactive_async_main():
    print("\n***************\nAsyncronous Mode\n****************")
    
    # Initialize KB and session manager
    kb = KnowledgeBase(storage_path="knowledge_db", use_persistence=True)
    session_manager = SessionManager(base_path="sessions")
    
    # Check for existing active session
    existing_session = kb.get_item("current_session")
    existing_file = kb.get_item("current_session_file")
    
    if existing_session and existing_file and os.path.exists(existing_file):
        # Try to load the existing session
        try:
            with open(existing_file, 'r') as f:
                session_data = json.load(f)
                
            # Check if session is still active
            if session_data["metadata"].get("session_active", False):
                print(f"Continuing existing session: {existing_session}")
                session_manager.current_session_id = existing_session
                session_manager.current_session_file = existing_file
                session_manager.session_data = session_data
            else:
                # Create new session if existing one is inactive
                session_id, session_file = session_manager.create_session()
                kb.set_item("current_session", session_id)
                kb.set_item("current_session_file", session_file)
                
        except Exception as e:
            print(f"Error loading existing session: {str(e)}")
            # Create new session as fallback
            session_id, session_file = session_manager.create_session()
            kb.set_item("current_session", session_id)
            kb.set_item("current_session_file", session_file)
    else:
        # No existing session, create a new one
        session_id, session_file = session_manager.create_session()
        kb.set_item("current_session", session_id)
        kb.set_item("current_session_file", session_file)

    # Clear previous session values
    kb.set_item("latest_model_file", None)
    kb.set_item("latest_model_details", None)
    kb.set_item("latest_analysis_results", None)
    kb.set_item("latest_model_location", None)
    kb.set_item("latest_model_generation_type", None)
    kb.set_item("latest_model_energy_carrier", None)

    function_loader = FunctionMapLoader(verbose=False)
    function_loader.register_functions({
        "build_plexos_model": build_plexos_model,
        "run_plexos_model": run_plexos_model,
        "analyze_results": analyze_results,
        "write_report": write_report,
        "generate_python_script": generate_python_script,
        "extract_model_parameters": extract_model_parameters,
        "create_single_location_model": create_single_location_model,
        "create_simple_xml": create_simple_xml,
        "create_multi_location_model": create_multi_location_model,
        "create_simple_multi_location_xml": create_simple_multi_location_xml,
        "create_comprehensive_model": create_comprehensive_model,
        "create_simple_comprehensive_xml": create_simple_comprehensive_xml,
        "process_emil_request": process_emil_request_enhanced,  # ✅ CORRECT FUNCTION NAME
        "do_maths": do_maths,
        "answer_general_question": answer_general_question,
        "ai_chat_session": ai_chat_session,
        "ai_spoken_chat_session": ai_spoken_chat_session,
    })

    function_loader = FunctionMapLoader(verbose=False)
    function_loader.register_functions({
        "build_plexos_model": build_plexos_model,
        "run_plexos_model": run_plexos_model,
        "analyze_results": analyze_results,
        "write_report": write_report,
        "generate_python_script": generate_python_script,
        "extract_model_parameters": extract_model_parameters,
        "create_single_location_model": create_single_location_model,
        "create_simple_xml": create_simple_xml,
        "create_multi_location_model": create_multi_location_model,
        "create_simple_multi_location_xml": create_simple_multi_location_xml,
        "create_comprehensive_model": create_comprehensive_model,
        "create_simple_comprehensive_xml": create_simple_comprehensive_xml,
        "process_emil_request": process_emil_request_enhanced,  # ⭐ Use enhanced version
        "do_maths": do_maths,
        "answer_general_question": answer_general_question,
        "ai_chat_session": ai_chat_session,
        "ai_spoken_chat_session": ai_spoken_chat_session,
    })

    # Initialize agents with proper function maps
    nova_functions = function_loader.load_function_map("Nova") or {}
    nova_functions.setdefault("answer_general_question", answer_general_question)
    nova_functions.setdefault("do_maths", do_maths)
    nova = Nova("Nova", kb, nova_functions)
    # WITH THIS ENHANCED VERSION:
    emil_functions = function_loader.load_function_map("Emil") or {}
    # Ensure the enhanced function is available
    emil_functions.setdefault("process_emil_request", process_emil_request_enhanced)
    emil_functions.setdefault("analyze_results", analyze_results)
    emil_functions.setdefault("build_plexos_model", build_plexos_model)
    emil_functions.setdefault("run_plexos_model", run_plexos_model)
    emil_functions.setdefault("extract_model_parameters", extract_model_parameters)
    emil_functions.setdefault("create_single_location_model", create_single_location_model)
    emil_functions.setdefault("create_simple_xml", create_simple_xml)
    emil_functions.setdefault("create_multi_location_model", create_multi_location_model)
    emil_functions.setdefault("create_simple_multi_location_xml", create_simple_multi_location_xml)
    emil_functions.setdefault("create_comprehensive_model", create_comprehensive_model)
    emil_functions.setdefault("create_simple_comprehensive_xml", create_simple_comprehensive_xml)
    
    emil = Emil("Emil", kb, emil_functions)
    
    # Initialize other agents (these stay the same)
    ivan = Ivan("Ivan", kb, function_loader.load_function_map("Ivan") or IVAN_FUNCTIONS)
    lola = Lola("Lola", kb, function_loader.load_function_map("Lola") or LOLA_FUNCTIONS)
    
    # Create agents dictionary
    agents = {"Nova": nova, "Emil": emil, "Ivan": ivan, "Lola": lola}

    def simplify_context(ctx):
        return {
            "file": ctx.get("latest_model_file") or ctx.get("file"),
            "location": ctx.get("location") or ctx.get("latest_model_location"),
            "generation_type": ctx.get("generation_type") or ctx.get("latest_model_generation_type"),
            "energy_carrier": ctx.get("energy_carrier") or ctx.get("latest_model_energy_carrier"),
        }

    # Main continuous loop - keeps the program running until explicitly terminated
    running = True
    while running:
        # Get prompts from user with enhanced commands
        prompts = []
        print("\nEnter prompts (type 'done' to process, 'exit' to quit, 'close' to end session):")
        while True:
            user_input = input("> ").strip()
            if user_input.lower() == "done":
                break
            elif user_input.lower() == "exit":
                # Exit without closing the session
                print("Exiting without closing session. Context will be preserved.")
                running = False
                return
            elif user_input.lower() == "close":
                # Close the session and exit
                print("Closing session and exiting.")
                session_manager.close_session_context()
                running = False
                return
            prompts.append(user_input)

        if not prompts:
            print("No prompts entered. Ready for new input.")
            continue  # Return to prompt entry

        # Add new prompts to existing session data
        if not session_manager.session_data.get("prompts"):
            session_manager.session_data["prompts"] = []
        session_manager.session_data["prompts"].extend(prompts)
        session_manager._save_current_session()
        
        task_lists = await asyncio.gather(*(nova.create_task_list_from_prompt_async(p) for p in prompts))

        results = []
        all_parameters = []

        for idx, (prompt, tasks) in enumerate(zip(prompts, task_lists)):
            print(f"\n------------------------\nProcessing prompt {idx+1}/{len(prompts)}: {prompt}")

            for task in tasks:
                agent = agents.get(task.agent)
                if not agent:
                    continue

                task.session_context.update({
                    "latest_model_file": kb.get_item("latest_model_file"),
                    "latest_model_details": kb.get_item("latest_model_details"),
                    "latest_analysis_results": kb.get_item("latest_analysis_results"),
                    "location": kb.get_item("latest_model_location"),
                    "generation_type": kb.get_item("latest_model_generation_type"),
                    "energy_carrier": kb.get_item("latest_model_energy_carrier"),
                })

                context = {**simplify_context(task.session_context), "prompt": task.args.get("full_prompt", task.args.get("prompt", ""))}
                session_manager.add_context_handover("Nova", task.agent, context)

                # Add detailed handover notification if the agent is not Nova
                if task.agent != "Nova":
                    print(f"\n📋 Context handover: Nova → {task.agent}")
                    print(f"   Task: {task.args.get('prompt', '')}")
                    
                    # Print relevant context information
                    if "location" in task.session_context and task.session_context.get("location"):
                        print(f"   Location: {task.session_context.get('location')}")
                    if "generation_type" in task.session_context and task.session_context.get("generation_type"):
                        print(f"   Generation type: {task.session_context.get('generation_type')}")
                    if "energy_carrier" in task.session_context and task.session_context.get("energy_carrier"):
                        print(f"   Energy carrier: {task.session_context.get('energy_carrier')}")
                    if "latest_model_file" in task.session_context and task.session_context.get("latest_model_file"):
                        # Extract just the filename, not the full path
                        model_file = os.path.basename(task.session_context.get("latest_model_file", ""))
                        if model_file:
                            print(f"   Model file: {model_file}")
                    print()

                result = await agent.handle_task_async(task)
                results.append((task.name, result, task.agent))
                all_parameters.append(task.args)

                for subtask in task.sub_tasks:
                    sub_agent = agents.get(subtask.agent)
                    if not sub_agent:
                        continue

                    subtask.session_context.update({
                        "latest_model_file": kb.get_item("latest_model_file"),
                        "latest_model_details": kb.get_item("latest_model_details"),
                        "latest_analysis_results": kb.get_item("latest_analysis_results"),
                        "location": kb.get_item("latest_model_location"),
                        "generation_type": kb.get_item("latest_model_generation_type"),
                        "energy_carrier": kb.get_item("latest_model_energy_carrier"),
                    })

                    sub_context = {**simplify_context(subtask.session_context), "prompt": subtask.args.get("full_prompt", subtask.args.get("prompt", ""))}
                    session_manager.add_context_handover(task.agent, subtask.agent, sub_context)

                    # Add detailed handover notification for subtasks between different agents
                    if subtask.agent != task.agent:
                        print(f"\n📋 Context handover: {task.agent} → {subtask.agent}")
                        print(f"   Task: {subtask.args.get('prompt', '')[:40]}...")
                        
                        # Print relevant context information
                        if "location" in subtask.session_context and subtask.session_context.get("location"):
                            print(f"   Location: {subtask.session_context.get('location')}")
                        if "generation_type" in subtask.session_context and subtask.session_context.get("generation_type"):
                            print(f"   Generation type: {subtask.session_context.get('generation_type')}")
                        if "energy_carrier" in subtask.session_context and subtask.session_context.get("energy_carrier"):
                            print(f"   Energy carrier: {subtask.session_context.get('energy_carrier')}")
                        if "latest_model_file" in subtask.session_context and subtask.session_context.get("latest_model_file"):
                            # Extract just the filename, not the full path
                            model_file = os.path.basename(subtask.session_context.get("latest_model_file", ""))
                            if model_file:
                                print(f"   Model file: {model_file}")
                        print()

                    sub_result = await sub_agent.handle_task_async(subtask)
                    results.append((subtask.name, sub_result, subtask.agent))
                    all_parameters.append(subtask.args)

        clean_results = []
        for name, result, agent in results:
            if isinstance(result, str) and len(result) > 300:
                result = result[:300] + "..."
            clean_results.append({
                "task": name,
                "agent": agent,
                "result": result
            })

        # Update session data with new results but keep context open
        session_manager.session_data = {
            "id": session_manager.current_session_id,
            "metadata": {
                "timestamp": session_manager.session_data["metadata"]["timestamp"],
                "created_at": session_manager.session_data["metadata"]["created_at"],
                "last_modified": datetime.datetime.now().isoformat(),
                "finalized_at": None,  # Not finalizing - keeping context open
                "context_open": True,  # Keep context open
                "session_active": True  # Session remains active
            },
            "prompts": session_manager.session_data.get("prompts", []),
            "parameters": session_manager.session_data.get("parameters", []) + all_parameters,
            "context_handovers": session_manager.session_data.get("context_handovers", []),
            "results": session_manager.session_data.get("results", []) + clean_results,
            "summary": "\n".join(
                [f"Processed: {p}" for p in prompts] +
                [f"✓ {agent}: \"{task_name.replace('Handle Intent: ', '')}\" → {result.get('message') if isinstance(result, dict) else result}" for task_name, result, agent in results]
            )
        }

        # Save the session
        session_manager._save_current_session()
        print(f"\n🗂️  Session updated and remains open: {session_manager.current_session_file}")
        print(f"Ready for next set of prompts...")

        # Display formatted results
        print("\n\n********\nResults\n********")
        print(" ------------------------")
        for task_name, result, agent in results:
            # Extract the full prompt from the task name
            task_prompt = task_name.replace("Handle Intent: ", "")
            print("-------------------\n")
            print(f"Task: {task_prompt}")  # Display the full prompt
            print(f"Agent: {agent}")
            
            # Format dictionary results more elegantly
            if isinstance(result, dict):
                # For model creation results
                if 'status' in result and result.get('status') == 'success':
                    print(f"  ✅ {result.get('message', 'Operation successful')}")
                    
                    # Include key parameters but not full paths
                    if 'location' in result:
                        params = []
                        if result.get('location'): 
                            params.append(f"Location: {result.get('location')}")
                        if result.get('generation_type'): 
                            params.append(f"Type: {result.get('generation_type')}")
                        if result.get('energy_carrier'): 
                            params.append(f"Carrier: {result.get('energy_carrier')}")
                        
                        print(f"  Parameters: {', '.join(params)}")
                        
                    # Just show filename, not full path
                    if 'file' in result:
                        filename = os.path.basename(result['file'])
                        print(f"  File: {filename}")
                else:
                    # For other dictionary results, show key fields nicely formatted
                    for key, value in result.items():
                        if key != 'file' and not isinstance(value, dict):  # Skip file paths and nested dicts
                            print(f"  {key.capitalize()}: {value}")
            else:
                # For string or other types of results
                print(f"  Result: {result}")
            
            print()  # Add blank line between results


if __name__ == "__main__":
    asyncio.run(interactive_async_main())







