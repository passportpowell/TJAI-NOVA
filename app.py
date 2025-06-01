import streamlit as st
import asyncio
import os
import json
import time
import tempfile
import datetime
from pathlib import Path
import sys

# Adjust system path to import project modules
# This assumes streamlit_app.py is in the root,
# and your project modules (agents, core, etc.) are WITHIN the 'src' subdirectory.
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Attempt to import necessary components from your project
try:
    from agents import Nova, Emil, Ivan, Lola
    from core.knowledge_base import KnowledgeBase
    from core.session_manager import SessionManager
    from core.functions_registery import (
        build_plexos_model, run_plexos_model, analyze_results, write_report,
        generate_python_script, extract_model_parameters, create_single_location_model,
        create_simple_xml, create_multi_location_model, create_simple_multi_location_xml,
        create_comprehensive_model, create_simple_comprehensive_xml,
        process_emil_request_enhanced, # Make sure this is correctly named and imported
        NOVA_FUNCTIONS, EMIL_FUNCTIONS, IVAN_FUNCTIONS, LOLA_FUNCTIONS
    )
    from utils.csv_function_mapper import FunctionMapLoader
    from utils.do_maths import do_maths
    from utils.general_knowledge import answer_general_question
    from agents.emil import extract_energy_parameters_from_prompt # Specific import for clarity
except ImportError as e:
    st.error(f"Failed to import project modules from '{SRC_DIR}'. Please ensure paths are correct and __init__.py files exist. Error: {e}")
    st.error(f"Current sys.path: {sys.path}")
    st.stop()

# --- Configuration ---
# This tells parameter_collection.py to attempt using the Streamlit version
os.environ["NOVA_PARAM_MODE"] = "streamlit"
PROGRESS_FILE = os.path.join(tempfile.gettempdir(), "plexos_progress.json")

# --- Helper Functions ---
def initialize_services():
    """Initializes KnowledgeBase, SessionManager, and Agents."""
    if 'kb' not in st.session_state:
        st.session_state.kb = KnowledgeBase(storage_path="knowledge_db", use_persistence=True)
        # Mimic CLI: "Loaded 204 items from persistent storage"
        st.session_state.processing_log.append(f"Loaded {len(st.session_state.kb.storage)} items from persistent storage.") #


    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager(base_path="sessions")

    if 'agents' not in st.session_state:
        kb = st.session_state.kb
        function_loader = FunctionMapLoader(verbose=False) #
        function_map_dict = {
            "build_plexos_model": build_plexos_model, "run_plexos_model": run_plexos_model, #
            "analyze_results": analyze_results, "write_report": write_report, #
            "generate_python_script": generate_python_script, "extract_model_parameters": extract_model_parameters, #
            "create_single_location_model": create_single_location_model, "create_simple_xml": create_simple_xml, #
            "create_multi_location_model": create_multi_location_model, "create_simple_multi_location_xml": create_simple_multi_location_xml, #
            "create_comprehensive_model": create_comprehensive_model, "create_simple_comprehensive_xml": create_simple_comprehensive_xml, #
            "process_emil_request": process_emil_request_enhanced, #
            "do_maths": do_maths, "answer_general_question": answer_general_question #
        }
        function_loader.register_functions(function_map_dict)

        nova_functions = function_loader.load_function_map("Nova") or {} #
        nova_functions.setdefault("answer_general_question", answer_general_question) #
        nova_functions.setdefault("do_maths", do_maths) #
        nova = Nova("Nova", kb, nova_functions, verbose=True) #

        emil_functions = function_loader.load_function_map("Emil") or {} #
        emil_functions.setdefault("process_emil_request", process_emil_request_enhanced) #
        emil_functions.setdefault("analyze_results", analyze_results) #
        emil_functions.setdefault("build_plexos_model", build_plexos_model) #
        # ... Add other Emil functions as in main.py ...
        emil = Emil("Emil", kb, emil_functions, verbose=True) #

        ivan = Ivan("Ivan", kb, function_loader.load_function_map("Ivan") or IVAN_FUNCTIONS) #
        lola = Lola("Lola", kb, function_loader.load_function_map("Lola") or LOLA_FUNCTIONS) #

        st.session_state.agents = {"Nova": nova, "Emil": emil, "Ivan": ivan, "Lola": lola} #
        st.session_state.nova = nova

    if not st.session_state.session_manager.current_session_id: #
        existing_session = st.session_state.kb.get_item("current_session") #
        existing_file = st.session_state.kb.get_item("current_session_file") #
        loaded_existing = False
        if existing_session and existing_file and os.path.exists(existing_file): #
            try:
                with open(existing_file, 'r') as f: #
                    session_data = json.load(f) #
                if session_data["metadata"].get("session_active", False): #
                    st.session_state.session_manager.current_session_id = existing_session #
                    st.session_state.session_manager.current_session_file = existing_file #
                    st.session_state.session_manager.session_data = session_data #
                    st.session_state.processing_log.append(f"Continuing existing session: {existing_session}") #
                    loaded_existing = True
            except Exception as e:
                st.warning(f"Could not load existing session: {e}") #

        if not loaded_existing:
            session_id, session_file = st.session_state.session_manager.create_session() #
            st.session_state.kb.set_item("current_session", session_id) #
            st.session_state.kb.set_item("current_session_file", session_file) #
            st.session_state.processing_log.append(f"Started new session: {session_id}")

        # Clear previous run KB items
        st.session_state.kb.set_item("latest_model_file", None)
        st.session_state.kb.set_item("latest_model_details", None)
        st.session_state.kb.set_item("latest_analysis_results", None)
        st.session_state.kb.set_item("latest_model_location", None)
        st.session_state.kb.set_item("latest_model_generation_type", None)
        st.session_state.kb.set_item("latest_model_energy_carrier", None)


def simplify_context(ctx): # From main.py
    return {
        "file": ctx.get("latest_model_file") or ctx.get("file"), #
        "location": ctx.get("location") or ctx.get("latest_model_location"), #
        "generation_type": ctx.get("generation_type") or ctx.get("latest_model_generation_type"), #
        "energy_carrier": ctx.get("energy_carrier") or ctx.get("latest_model_energy_carrier"), #
    }

async def process_single_prompt_batch(prompts_list):
    kb = st.session_state.kb
    session_manager = st.session_state.session_manager
    agents = st.session_state.agents
    nova = st.session_state.nova

    def log_to_streamlit(message):
        cleaned_message = str(message).strip()
        # Add to set first to ensure it's tracked even if not appended to visible log immediately
        if 'logged_messages_set' not in st.session_state:
            st.session_state.logged_messages_set = set()

        if cleaned_message and cleaned_message not in st.session_state.logged_messages_set:
            st.session_state.processing_log.append(cleaned_message)
            st.session_state.logged_messages_set.add(cleaned_message)


    if not session_manager.session_data.get("prompts"): #
        session_manager.session_data["prompts"] = [] #
    session_manager.session_data["prompts"].extend(prompts_list) #
    session_manager._save_current_session()

    try:
        task_lists = await asyncio.gather(*(nova.create_task_list_from_prompt_async(p) for p in prompts_list)) #
    except Exception as e:
        log_to_streamlit(f"Error creating task list: {e}")
        st.error(f"Error during task creation: {e}")
        st.session_state.processing_complete = True
        st.session_state.backend_task_active = False
        return

    results_accumulator = [] #
    all_parameters_accumulator = [] #

    # Resume with collected parameters if they exist
    if st.session_state.get('collected_user_parameters'):
        # This assumes we are resuming for a specific task.
        # The logic to associate these params with the correct task needs to be robust.
        # For simplicity, we'll assume it's for the current task being processed if this flag is set.
        # This part is tricky and might need more context about which task was pending.
        # Let's assume process_single_prompt_batch is re-entrant or the state is managed carefully.
        # The current design has the form directly in app_main_flow, which then sets collected_user_parameters.
        # The backend needs to pick this up.
        log_to_streamlit(f"Resuming with collected parameters: {st.session_state.collected_user_parameters}")
        # Find the task that was waiting for parameters and update its args.
        # This is simplified; a more robust system would tag the pending task.
        # For now, if we have collected_user_parameters, we assume they are for the next Emil task.
        # This is a potential point of failure if multiple Emil tasks are in a sequence.

    for idx, (prompt, tasks) in enumerate(zip(prompts_list, task_lists)): #
        log_to_streamlit(f"------------------------\nProcessing prompt {idx+1}/{len(prompts_list)}: {prompt}") #

        if "model" in prompt.lower() and ("solar" in prompt.lower() or "wind" in prompt.lower() or "hydro" in prompt.lower()): #
             log_to_streamlit(f"⚠️ Pre-check identified energy modeling request: '{prompt}'") #

        for task_idx, task in enumerate(tasks): #
            agent = agents.get(task.agent) #
            if not agent:
                log_to_streamlit(f"Agent {task.agent} not found for task {task.name}. Skipping.")
                continue

            # Parameter extraction and verification for Emil's process_emil_request
            if agent.name == "Emil" and task.function_name == "process_emil_request":
                # If parameters were just collected via the form, apply them
                if st.session_state.get('collected_user_parameters'):
                    task.args.update(st.session_state.collected_user_parameters)
                    log_to_streamlit(f"Applied collected parameters to task {task.name}: {st.session_state.collected_user_parameters}")
                    st.session_state.collected_user_parameters = None # Clear after use

                if task.args.get("prompt"): #
                    log_to_streamlit(f"🔍 Extracting parameters from prompt: {task.args['prompt']}") #
                    extracted_params = await extract_energy_parameters_from_prompt(task.args["prompt"]) #
                    for k, v in extracted_params.items(): #
                        if k not in task.args or not task.args[k]: #
                           task.args[k] = v #
                           log_to_streamlit(f"✅ Auto-filled parameter {k}: {v}") #

                validation = await agent.verify_parameters_async(task.function_name, task.args) #
                if not validation["success"] and validation.get("missing"): #
                    log_to_streamlit(f"🧩 Emil needs: {validation['missing']} for {task.function_name}") #
                    st.session_state.waiting_for_user_params_info = {
                        "function_name": task.function_name,
                        "missing_params": validation["missing"],
                        "initial_args": task.args.copy()
                    }
                    st.session_state.param_collection_pending = True
                    # Do not proceed further in this backend call; Streamlit will handle UI and rerun.
                    st.session_state.backend_task_active = False # Pause backend activity
                    return # Exit this function to allow Streamlit to take over for param collection

            # Update task context (as in main.py)
            task.session_context.update({ #
                "latest_model_file": kb.get_item("latest_model_file"), #
                "latest_model_details": kb.get_item("latest_model_details"), #
                "latest_analysis_results": kb.get_item("latest_analysis_results"), #
                "location": kb.get_item("latest_model_location"), #
                "generation_type": kb.get_item("latest_model_generation_type"), #
                "energy_carrier": kb.get_item("latest_model_energy_carrier"), #
            })

            context_for_handover = {**simplify_context(task.session_context), "prompt": task.args.get("full_prompt", task.args.get("prompt", ""))} #
            session_manager.add_context_handover("Nova", task.agent, context_for_handover) #

            if task.agent != "Nova": #
                log_to_streamlit(f"\n📋 Context handover: Nova → {task.agent}") #
                log_to_streamlit(f"   Task: {task.args.get('prompt', '')}") #
                if "location" in task.session_context and task.session_context.get("location"): #
                    log_to_streamlit(f"   Location: {task.session_context.get('location')}") #
                if "generation_type" in task.session_context and task.session_context.get("generation_type"): #
                    log_to_streamlit(f"   Generation type: {task.session_context.get('generation_type')}") #
                if "energy_carrier" in task.session_context and task.session_context.get("energy_carrier"): #
                    log_to_streamlit(f"   Energy carrier: {task.session_context.get('energy_carrier')}") #
                if "latest_model_file" in task.session_context and task.session_context.get("latest_model_file"): #
                    model_file_path = task.session_context.get("latest_model_file", "") #
                    if model_file_path: #
                        log_to_streamlit(f"   Model file: {os.path.basename(model_file_path)}") #


            try:
                result = await agent.handle_task_async(task) #
                results_accumulator.append((task.name, result, task.agent)) #
                all_parameters_accumulator.append(task.args) #

                if isinstance(result, dict): #
                    if result.get('file'): kb.set_item("latest_model_file", result['file']) #
                    if result.get('location'): kb.set_item("latest_model_location", result['location']) #
                    if result.get('generation_type'): kb.set_item("latest_model_generation_type", result['generation_type']) #
                    if result.get('energy_carrier'): kb.set_item("latest_model_energy_carrier", result['energy_carrier']) #

            except Exception as e:
                log_message = f"Error during task execution {task.name} by {task.agent}: {e}"
                log_to_streamlit(log_message)
                results_accumulator.append((task.name, {"status": "error", "message": str(e)}, task.agent))

            for subtask in task.sub_tasks: #
                log_to_streamlit(f"  Subtask: {subtask.name} for agent {subtask.agent}") #
                sub_agent = agents.get(subtask.agent) #
                if not sub_agent: continue #
                subtask.session_context.update({ #
                    "latest_model_file": kb.get_item("latest_model_file"), #
                    "latest_model_details": kb.get_item("latest_model_details"), #
                    "latest_analysis_results": kb.get_item("latest_analysis_results"), #
                    "location": kb.get_item("latest_model_location"), #
                    "generation_type": kb.get_item("latest_model_generation_type"), #
                    "energy_carrier": kb.get_item("latest_model_energy_carrier"), #
                })
                sub_context_for_handover = {**simplify_context(subtask.session_context), "prompt": subtask.args.get("full_prompt", subtask.args.get("prompt", ""))} #
                session_manager.add_context_handover(task.agent, subtask.agent, sub_context_for_handover) #
                if subtask.agent != task.agent: #
                    log_to_streamlit(f"\n📋 Context handover: {task.agent} → {subtask.agent}") #
                    log_to_streamlit(f"   Task: {subtask.args.get('prompt', '')[:40]}...") #
                try:
                    sub_result = await sub_agent.handle_task_async(subtask) #
                    results_accumulator.append((subtask.name, sub_result, subtask.agent)) #
                    all_parameters_accumulator.append(subtask.args) #
                except Exception as e:
                    log_to_streamlit(f"Error during sub_task execution {subtask.name} by {subtask.agent}: {e}")
                    results_accumulator.append((subtask.name, {"status": "error", "message": str(e)}, subtask.agent))

    clean_results_summary = [] #
    for name, res, agent_name in results_accumulator: #
        if isinstance(res, str) and len(res) > 100: res_display = res[:100] + "..." #
        else: res_display = res
        clean_results_summary.append({"task": name, "agent": agent_name, "result": res_display}) #

    st.session_state.current_results_full = results_accumulator

    session_manager.session_data.update({ #
        "parameters": session_manager.session_data.get("parameters", []) + all_parameters_accumulator, #
        "results": session_manager.session_data.get("results", []) + clean_results_summary, #
        "last_modified": datetime.datetime.now().isoformat(), #
        "context_open": True, #
        "session_active": True #
    })
    session_manager._save_current_session() #
    log_to_streamlit(f"🗂️ Session updated and remains open: {session_manager.current_session_file}") #

    st.session_state.processing_complete = True
    st.session_state.backend_task_active = False


# --- Streamlit UI and Main Logic ---
st.set_page_config(layout="wide", page_title="AI Model Builder")
st.title("💡 AI Energy Model Builder Interface")

# Initialize session state variables
if 'services_initialized' not in st.session_state: st.session_state.services_initialized = False
if 'processing_started' not in st.session_state: st.session_state.processing_started = False
if 'processing_complete' not in st.session_state: st.session_state.processing_complete = True # Start as complete
if 'param_collection_pending' not in st.session_state: st.session_state.param_collection_pending = False
if 'waiting_for_user_params_info' not in st.session_state: st.session_state.waiting_for_user_params_info = None
if 'collected_user_parameters' not in st.session_state: st.session_state.collected_user_parameters = None
if 'current_results_full' not in st.session_state: st.session_state.current_results_full = []
if 'processing_log' not in st.session_state: st.session_state.processing_log = []
if 'logged_messages_set' not in st.session_state: st.session_state.logged_messages_set = set()
if 'backend_task_active' not in st.session_state: st.session_state.backend_task_active = False


async def app_main_flow():
    global PROGRESS_FILE

    if not st.session_state.services_initialized:
        with st.spinner("Initializing services..."):
            initialize_services()
        st.session_state.services_initialized = True
        st.rerun()

    # Display collected logs (condensed to one area)
    log_display_area = st.expander("Backend Processing Log", expanded=False)
    with log_display_area:
        if st.session_state.processing_log:
            st.text("\n".join(st.session_state.processing_log))
        else:
            st.caption("No log messages yet.")

    # Handle parameter collection UI if pending
    if st.session_state.param_collection_pending:
        param_info = st.session_state.waiting_for_user_params_info
        if param_info:
            with st.form(key=f"form_dyn_{param_info['function_name']}_{'_'.join(param_info['missing_params'])}"):
                st.subheader(f"Input Needed: {param_info['function_name']}")
                current_params_form = param_info['initial_args'].copy()
                for p_name in param_info['missing_params']:
                    # Use a unique key for each input field within the dynamic form
                    input_field_key = f"form_input_dyn_{param_info['function_name']}_{p_name}"
                    current_params_form[p_name] = st.text_input(
                        f"Enter {p_name.replace('_', ' ').capitalize()}:",
                        value=current_params_form.get(p_name, ""),
                        key=input_field_key
                    )
                submitted = st.form_submit_button("Submit These Parameters")
                if submitted:
                    st.session_state.collected_user_parameters = {
                        p_name: current_params_form[p_name] for p_name in param_info['missing_params'] if current_params_form[p_name]
                    }
                    st.session_state.param_collection_pending = False
                    st.session_state.waiting_for_user_params_info = None
                    st.session_state.backend_task_active = True # Resume backend processing
                    st.rerun()
        return # Stop further rendering in this run, wait for form submission.

    # --- Input and Main Processing Trigger ---
    if not st.session_state.backend_task_active and st.session_state.processing_complete: # Only show input if not already processing
        user_prompts_str = st.text_area("Enter your prompt(s) (one per line):", height=100, key="prompt_input_main")
        process_button = st.button("🚀 Process Prompts", key="process_button_main")

        if process_button and user_prompts_str:
            st.session_state.prompts_to_process = [p.strip() for p in user_prompts_str.split('\n') if p.strip()]
            if st.session_state.prompts_to_process:
                st.session_state.processing_started = True
                st.session_state.processing_complete = False
                st.session_state.backend_task_active = True
                st.session_state.current_results_full = []
                st.session_state.processing_log = ["Backend processing initiated by user..."]
                st.session_state.logged_messages_set.clear()

                if os.path.exists(PROGRESS_FILE):
                    try: os.remove(PROGRESS_FILE)
                    except Exception: pass
                st.rerun()
            else:
                st.warning("Please enter at least one prompt.")

    # --- Active Processing Loop ---
    if st.session_state.backend_task_active: # If task is active and not waiting for params
        with st.spinner("🤖 AI agents are working... this may take a few minutes."):
            await process_single_prompt_batch(st.session_state.prompts_to_process)
            # process_single_prompt_batch sets backend_task_active to False when done or if params are needed.
            if st.session_state.param_collection_pending or st.session_state.processing_complete:
                 st.rerun() # Rerun to show param form or results


    # --- Progress Display (runs on each rerun if processing is active and not complete) ---
    if st.session_state.processing_started and not st.session_state.processing_complete and not st.session_state.param_collection_pending:
        progress_updated_in_this_run = False
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    progress_data = json.load(f)

                if progress_data: # Only show header if there's data
                    st.subheader("--- Task Progress ---")
                for task_name_progress, p_info in progress_data.items():
                    percent = int(p_info.get('percent', 0) * 100)
                    message = p_info.get('message', task_name_progress)
                    display_message = (message[:100] + '...') if len(message) > 100 else message #
                    st.text(f"{display_message}") #
                    st.progress(percent) #
                    progress_updated_in_this_run = True
            except (json.JSONDecodeError, FileNotFoundError):
                st.caption("Waiting for progress updates...")
            except Exception as e:
                st.warning(f"Error reading progress: {e}")

        if progress_updated_in_this_run or (not os.path.exists(PROGRESS_FILE) and st.session_state.backend_task_active):
             # Only rerun if progress was updated OR if task is active but no progress file yet (still initializing)
            time.sleep(0.3) # Short sleep for UI to catch up
            st.rerun()


    # --- Result Display ---
    if st.session_state.processing_complete and not st.session_state.backend_task_active:
        if not st.session_state.param_collection_pending : # Ensure not to show this if we are about to show param form
            st.success("✅ Processing Complete!")

            results_to_display = st.session_state.get('current_results_full', [])
            if results_to_display:
                st.subheader("📊 Final Results") #
                for task_name_result, result_data, agent_name in results_to_display: #
                    st.markdown(f"---") #
                    st.markdown(f"**Task:** `{task_name_result.replace('Handle Intent: ', '')}`") #
                    st.markdown(f"**Agent:** `{agent_name}`") #
                    if isinstance(result_data, dict): #
                        if result_data.get('status') == 'success' and 'file' in result_data: #
                            st.success(f"  ✅ {result_data.get('message', 'Operation successful')}") #
                            params_display = []
                            if result_data.get('location'): params_display.append(f"Location: {result_data.get('location')}") #
                            if result_data.get('generation_type'): params_display.append(f"Type: {result_data.get('generation_type')}") #
                            if result_data.get('energy_carrier'): params_display.append(f"Carrier: {result_data.get('energy_carrier')}") #
                            if params_display: st.info(f"  Parameters: {', '.join(params_display)}") #

                            file_path = result_data.get('file') #
                            if file_path and os.path.exists(file_path): #
                                st.markdown(f"  📄 File: `{os.path.basename(file_path)}`") #
                                try:
                                    if file_path.endswith(".xml"):
                                        with open(file_path, "rb") as fp:
                                            st.download_button(
                                                label="Download XML Model",
                                                data=fp,
                                                file_name=os.path.basename(file_path),
                                                mime="application/xml"
                                            )
                                except Exception as e:
                                    st.warning(f"Could not offer file for download: {e}")
                            elif file_path:
                                 st.warning(f"  ⚠️ File not found for download: {os.path.basename(file_path)}")

                        elif result_data.get('status') == 'error': #
                            st.error(f"  ❌ Error: {result_data.get('message', 'Unknown error')}") #
                        else:
                            st.json(result_data) #
                    else:
                        st.text_area("Result Data:", str(result_data), height=100, disabled=True) #
            else:
                if st.session_state.processing_started : # only show if processing was actually attempted
                    st.info("No specific results were generated from this run, or an error occurred early.")

            # Clean up session state for the next full run
            keys_to_reset_on_completion = [
                'processing_started', #'processing_complete' is handled by initial state for next run
                'prompts_to_process',
                'backend_task_active', 'param_collection_pending',
                'waiting_for_user_params_info', 'collected_user_parameters',
                'processing_log', 'logged_messages_set' # Keep current_results_full until next processing starts
            ]
            for key in keys_to_reset_on_completion:
                if key in st.session_state:
                    del st.session_state[key]
            if os.path.exists(PROGRESS_FILE):
                try: os.remove(PROGRESS_FILE)
                except: pass
            # Set complete to true so the input form shows again.
            st.session_state.processing_complete = True


if __name__ == "__main__":
    asyncio.run(app_main_flow())