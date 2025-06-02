import streamlit as st
import asyncio
import os
import json
import time
import tempfile
import datetime
from pathlib import Path
import sys

# Set Streamlit page config immediately
st.set_page_config(layout="wide", page_title="AI Model Builder")

def init_session_state():
    """Initialize all session state variables with default values"""
    defaults = {
        'services_initialized': False,
        'processing_started': False,
        'processing_complete': True,
        'param_collection_pending': False,
        'waiting_for_user_params_info': None,
        'collected_user_parameters': None,
        'current_results_full': [],
        'processing_log': [],
        'logged_messages_set': set(),
        'backend_task_active': False,
        'cli_output_displayed': [],
        'last_cli_length': 0
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Fix import path for modules (assuming src/ is next to this file)
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import your project modules
try:
    from agents import Nova, Emil, Ivan, Lola
    from core.knowledge_base import KnowledgeBase
    from core.session_manager import SessionManager
    from core.functions_registery import (
        build_plexos_model, run_plexos_model, analyze_results, write_report,
        generate_python_script, extract_model_parameters, create_single_location_model,
        create_simple_xml, create_multi_location_model, create_simple_multi_location_xml,
        create_comprehensive_model, create_simple_comprehensive_xml,
        process_emil_request_enhanced,
        NOVA_FUNCTIONS, EMIL_FUNCTIONS, IVAN_FUNCTIONS, LOLA_FUNCTIONS
    )
    from utils.csv_function_mapper import FunctionMapLoader
    from utils.do_maths import do_maths
    from utils.general_knowledge import answer_general_question
    from agents.emil import extract_energy_parameters_from_prompt
except ImportError as e:
    st.error(f"Failed to import project modules from '{SRC_DIR}'. Please ensure paths are correct and __init__.py files exist. Error: {e}")
    st.error(f"Current sys.path: {sys.path}")
    st.stop()

# --- Progress file location ---
PROGRESS_FILE = os.path.join(tempfile.gettempdir(), "plexos_progress.json")

# --- CLI Output Capture Setup ---
original_stdout = sys.stdout
original_stderr = sys.stderr
captured_cli_output = []

class CLICapture:
    def write(self, text):
        captured_cli_output.append(text)
        original_stdout.write(text)
        original_stdout.flush()
    def flush(self):
        original_stdout.flush()

def start_cli_capture():
    global captured_cli_output
    captured_cli_output.clear()
    sys.stdout = CLICapture()
    sys.stderr = CLICapture()

def stop_cli_capture():
    sys.stdout = original_stdout
    sys.stderr = original_stderr

def get_new_cli_output():
    if 'last_cli_length' not in st.session_state:
        st.session_state.last_cli_length = 0
    new_output = captured_cli_output[st.session_state.last_cli_length:]
    st.session_state.last_cli_length = len(captured_cli_output)
    return new_output

# --- Helper: Service Initialization ---
def initialize_services():
    if 'kb' not in st.session_state:
        st.session_state.kb = KnowledgeBase(storage_path="knowledge_db", use_persistence=True)
        st.session_state.processing_log.append(f"Loaded {len(st.session_state.kb.storage)} items from persistent storage.")

    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager(base_path="sessions")

    if 'agents' not in st.session_state:
        kb = st.session_state.kb
        function_loader = FunctionMapLoader(verbose=False)
        function_map_dict = {
            "build_plexos_model": build_plexos_model, "run_plexos_model": run_plexos_model,
            "analyze_results": analyze_results, "write_report": write_report,
            "generate_python_script": generate_python_script, "extract_model_parameters": extract_model_parameters,
            "create_single_location_model": create_single_location_model, "create_simple_xml": create_simple_xml,
            "create_multi_location_model": create_multi_location_model, "create_simple_multi_location_xml": create_simple_multi_location_xml,
            "create_comprehensive_model": create_comprehensive_model, "create_simple_comprehensive_xml": create_simple_comprehensive_xml,
            "process_emil_request": process_emil_request_enhanced,
            "do_maths": do_maths, "answer_general_question": answer_general_question
        }
        function_loader.register_functions(function_map_dict)
        nova_functions = function_loader.load_function_map("Nova") or {}
        nova_functions.setdefault("answer_general_question", answer_general_question)
        nova_functions.setdefault("do_maths", do_maths)
        nova = Nova("Nova", kb, nova_functions, verbose=True)

        emil_functions = function_loader.load_function_map("Emil") or {}
        emil_functions.setdefault("process_emil_request", process_emil_request_enhanced)
        emil_functions.setdefault("analyze_results", analyze_results)
        emil_functions.setdefault("build_plexos_model", build_plexos_model)
        emil = Emil("Emil", kb, emil_functions, verbose=True)

        ivan = Ivan("Ivan", kb, function_loader.load_function_map("Ivan") or IVAN_FUNCTIONS)
        lola = Lola("Lola", kb, function_loader.load_function_map("Lola") or LOLA_FUNCTIONS)
        st.session_state.agents = {"Nova": nova, "Emil": emil, "Ivan": ivan, "Lola": lola}
        st.session_state.nova = nova

    if not st.session_state.session_manager.current_session_id:
        existing_session = st.session_state.kb.get_item("current_session")
        existing_file = st.session_state.kb.get_item("current_session_file")
        loaded_existing = False
        if existing_session and existing_file and os.path.exists(existing_file):
            try:
                with open(existing_file, 'r') as f:
                    session_data = json.load(f)
                if session_data["metadata"].get("session_active", False):
                    st.session_state.session_manager.current_session_id = existing_session
                    st.session_state.session_manager.current_session_file = existing_file
                    st.session_state.session_manager.session_data = session_data
                    st.session_state.processing_log.append(f"Continuing existing session: {existing_session}")
                    loaded_existing = True
            except Exception as e:
                st.warning(f"Could not load existing session: {e}")
        if not loaded_existing:
            session_id, session_file = st.session_state.session_manager.create_session()
            st.session_state.kb.set_item("current_session", session_id)
            st.session_state.kb.set_item("current_session_file", session_file)
            st.session_state.processing_log.append(f"Started new session: {session_id}")

        st.session_state.kb.set_item("latest_model_file", None)
        st.session_state.kb.set_item("latest_model_details", None)
        st.session_state.kb.set_item("latest_analysis_results", None)
        st.session_state.kb.set_item("latest_model_location", None)
        st.session_state.kb.set_item("latest_model_generation_type", None)
        st.session_state.kb.set_item("latest_model_energy_carrier", None)

def simplify_context(ctx):
    return {
        "file": ctx.get("latest_model_file") or ctx.get("file"),
        "location": ctx.get("location") or ctx.get("latest_model_location"),
        "generation_type": ctx.get("generation_type") or ctx.get("latest_model_generation_type"),
        "energy_carrier": ctx.get("energy_carrier") or ctx.get("latest_model_energy_carrier"),
    }

async def process_single_prompt_batch(prompts_list):
    kb = st.session_state.kb
    session_manager = st.session_state.session_manager
    agents = st.session_state.agents
    nova = st.session_state.nova

    def log_to_streamlit(message):
        cleaned_message = str(message).strip()
        if 'logged_messages_set' not in st.session_state:
            st.session_state.logged_messages_set = set()
        if cleaned_message and cleaned_message not in st.session_state.logged_messages_set:
            st.session_state.processing_log.append(cleaned_message)
            st.session_state.logged_messages_set.add(cleaned_message)

    # Start CLI capture
    start_cli_capture()

    if not session_manager.session_data.get("prompts"):
        session_manager.session_data["prompts"] = []
    session_manager.session_data["prompts"].extend(prompts_list)
    session_manager._save_current_session()

    try:
        task_lists = await asyncio.gather(*(nova.create_task_list_from_prompt_async(p) for p in prompts_list))
    except Exception as e:
        log_to_streamlit(f"Error creating task list: {e}")
        st.error(f"Error during task creation: {e}")
        st.session_state.processing_complete = True
        st.session_state.backend_task_active = False
        stop_cli_capture()
        return

    results_accumulator = []
    all_parameters_accumulator = []

    for idx, (prompt, tasks) in enumerate(zip(prompts_list, task_lists)):
        log_to_streamlit(f"------------------------\nProcessing prompt {idx+1}/{len(prompts_list)}: {prompt}")
        for task_idx, task in enumerate(tasks):
            agent = agents.get(task.agent)
            if not agent:
                log_to_streamlit(f"Agent {task.agent} not found for task {task.name}. Skipping.")
                continue
            try:
                result = await agent.handle_task_async(task)
                results_accumulator.append((task.name, result, task.agent))
                all_parameters_accumulator.append(task.args)
            except Exception as e:
                log_message = f"Error during task execution {task.name} by {task.agent}: {e}"
                log_to_streamlit(log_message)
                results_accumulator.append((task.name, {"status": "error", "message": str(e)}, task.agent))
    clean_results_summary = []
    for name, res, agent_name in results_accumulator:
        if isinstance(res, str) and len(res) > 100: res_display = res[:100] + "..."
        else: res_display = res
        clean_results_summary.append({"task": name, "agent": agent_name, "result": res_display})
    st.session_state.current_results_full = results_accumulator

    session_manager.session_data.update({
        "parameters": session_manager.session_data.get("parameters", []) + all_parameters_accumulator,
        "results": session_manager.session_data.get("results", []) + clean_results_summary,
        "last_modified": datetime.datetime.now().isoformat(),
        "context_open": True,
        "session_active": True
    })
    session_manager._save_current_session()
    log_to_streamlit(f"🗂️ Session updated and remains open: {session_manager.current_session_file}")

    # Stop CLI capture
    stop_cli_capture()
    st.session_state.processing_complete = True
    st.session_state.backend_task_active = False

# --- Streamlit UI Starts Here ---
st.title("💡 AI Energy Model Builder Interface")

if getattr(st.session_state, 'backend_task_active', False):
    st.info("🤖 **AI Agents are actively processing your request...** Progress updates below ⬇️")

if not st.session_state.services_initialized:
    with st.spinner("Initializing services..."):
        initialize_services()
    st.session_state.services_initialized = True
    st.rerun()

# --- Processing Log Display ---
log_display_area = st.expander("Backend Processing Log", expanded=False)
with log_display_area:
    if st.session_state.processing_log:
        st.text("\n".join(st.session_state.processing_log))
    else:
        st.caption("No log messages yet.")

# --- CLI Output Display (Real-time) ---
if getattr(st.session_state, 'backend_task_active', False):
    st.markdown("---")
    with st.expander("📟 Live Processing Details", expanded=True):
        new_cli = get_new_cli_output()
        if new_cli:
            st.session_state.cli_output_displayed.extend(new_cli)
        if st.session_state.cli_output_displayed:
            recent_lines = st.session_state.cli_output_displayed[-50:]
            cli_text = ''.join(recent_lines)
            st.code(cli_text, language=None)
            st.caption("🟢 Live output updating...")
        else:
            st.info("Waiting for processing output...")

# --- Progress Bar UI: Real-time file reading ---
if getattr(st.session_state, 'backend_task_active', False):
    st.markdown("---")
    st.subheader("🔄 Live Progress")
    progress_updated_in_this_run = False
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress_data = json.load(f)
            if progress_data:
                progress_container = st.container()
                with progress_container:
                    for task_name_progress, p_info in progress_data.items():
                        percent = p_info.get('percent', 0)
                        current = p_info.get('current', 0)
                        total = p_info.get('total', 1)
                        message = p_info.get('message', task_name_progress)
                        display_message = message
                        if len(display_message) > 80:
                            display_message = display_message[:77] + "..."
                        st.write(f"**{task_name_progress}:** {current}/{total}")
                        progress_text = f"{percent*100:.1f}% - {display_message}"
                        st.progress(percent, text=progress_text)
                        if message and message != task_name_progress:
                            st.caption(f"Currently processing: {display_message}")
                        st.write("")
                progress_updated_in_this_run = True
            else:
                st.info("⏳ Initializing background process...")
        except (json.JSONDecodeError, FileNotFoundError):
            st.info("⏳ Waiting for progress data...")
        except Exception as e:
            st.warning(f"Error reading progress: {e}")
    else:
        st.info("⏳ Starting background task...")

    # Refresh the page rapidly for near real-time updates
    time.sleep(0.3)
    st.rerun()

# --- Prompt Input ---
if not getattr(st.session_state, 'backend_task_active', False) and getattr(st.session_state, 'processing_complete', True):
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
            st.session_state.cli_output_displayed = []
            st.session_state.last_cli_length = 0
            if os.path.exists(PROGRESS_FILE):
                try: os.remove(PROGRESS_FILE)
                except Exception: pass
            st.rerun()
        else:
            st.warning("Please enter at least one prompt.")

# --- Async Processing Execution Loop ---
if getattr(st.session_state, 'backend_task_active', False):
    with st.spinner("🤖 AI agents are working... this may take a few minutes."):
        result = asyncio.run(process_single_prompt_batch(st.session_state.prompts_to_process))
        if st.session_state.processing_complete:
            st.rerun()

# --- Results Display ---
if (getattr(st.session_state, 'processing_complete', True) and 
    not getattr(st.session_state, 'backend_task_active', False)):
    st.success("✅ Processing Complete!")
    results_to_display = st.session_state.get('current_results_full', [])
    if results_to_display:
        st.subheader("📊 Final Results")
        for task_name_result, result_data, agent_name in results_to_display:
            st.markdown(f"---")
            st.markdown(f"**Task:** `{task_name_result.replace('Handle Intent: ', '')}`")
            st.markdown(f"**Agent:** `{agent_name}`")
            if isinstance(result_data, dict):
                if result_data.get('status') == 'success' and 'file' in result_data:
                    st.success(f"  ✅ {result_data.get('message', 'Operation successful')}")
                    params_display = []
                    if result_data.get('location'): params_display.append(f"Location: {result_data.get('location')}")
                    if result_data.get('generation_type'): params_display.append(f"Generation: {result_data.get('generation_type')}")
                    if result_data.get('energy_carrier'): params_display.append(f"Carrier: {result_data.get('energy_carrier')}")
                    if params_display:
                        st.write(", ".join(params_display))
                    st.write(f"Output file: `{result_data.get('file')}`")
                elif result_data.get('status') == 'error':
                    st.error(result_data.get('message'))
                else:
                    st.write(str(result_data))
            else:
                st.write(str(result_data))
