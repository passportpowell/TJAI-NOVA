import streamlit as st
import asyncio
import os
import json
import time
import tempfile
import datetime
from pathlib import Path
import sys
import threading
import queue
from typing import Dict, Any, List

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
        'last_cli_length': 0,
        'error_occurred': False,
        'error_message': '',
        'message_queue': queue.Queue(),
        'result_queue': queue.Queue(),
        'agents': None,
        'kb': None,
        'session_manager': None,
        'nova': None,
        'progress_history': [],  # Store completed progress steps
        'current_progress': {},  # Store current active progress
        'created_files': [],     # Store created files for download
        'task_progress': {},     # NEW: Track individual task progress
        'total_tasks': 0,        # NEW: Total number of tasks
        'completed_tasks': 0     # NEW: Number of completed tasks
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

# Import your project modules with error handling
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
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    st.error(f"❌ Failed to import project modules: {e}")
    IMPORTS_SUCCESSFUL = False

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

# --- Helper: Service Initialization with Error Handling ---
def initialize_services():
    if not IMPORTS_SUCCESSFUL:
        st.error("Cannot initialize services due to import failures.")
        return False
        
    try:
        if st.session_state.kb is None:
            st.session_state.kb = KnowledgeBase(storage_path="knowledge_db", use_persistence=True)
            st.session_state.processing_log.append(f"✅ Loaded {len(st.session_state.kb.storage)} items from persistent storage.")

        if st.session_state.session_manager is None:
            st.session_state.session_manager = SessionManager(base_path="sessions")

        if st.session_state.agents is None:
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
            
            # Initialize agents with error handling
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

        # Session management
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
                        st.session_state.processing_log.append(f"✅ Continuing existing session: {existing_session}")
                        loaded_existing = True
                except Exception as e:
                    st.warning(f"Could not load existing session: {e}")
            if not loaded_existing:
                session_id, session_file = st.session_state.session_manager.create_session()
                st.session_state.kb.set_item("current_session", session_id)
                st.session_state.kb.set_item("current_session_file", session_file)
                st.session_state.processing_log.append(f"✅ Started new session: {session_id}")

            # Clear previous model data
            st.session_state.kb.set_item("latest_model_file", None)
            st.session_state.kb.set_item("latest_model_details", None)
            st.session_state.kb.set_item("latest_analysis_results", None)
            st.session_state.kb.set_item("latest_model_location", None)
            st.session_state.kb.set_item("latest_model_generation_type", None)
            st.session_state.kb.set_item("latest_model_energy_carrier", None)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error initializing services: {e}")
        st.session_state.processing_log.append(f"❌ Service initialization error: {e}")
        return False

def create_task_progress_entry(task_name, status="starting"):
    """NEW: Create progress entry for individual tasks"""
    return {
        'name': task_name,
        'status': status,
        'start_time': datetime.datetime.now().isoformat(),
        'progress': 0.0,
        'message': f"{status.capitalize()} {task_name}..."
    }

def update_task_progress(message_queue, task_name, progress, message, status="in_progress"):
    """NEW: Update task progress"""
    try:
        message_queue.put(("task_progress", {
            'task_name': task_name,
            'progress': progress,
            'message': message,
            'status': status,
            'timestamp': datetime.datetime.now().isoformat()
        }))
    except:
        pass

# ENHANCED: Thread-safe processing function with task progress tracking
async def process_single_prompt_batch_async(prompts_list, kb, session_manager, agents, nova, message_queue):
    """Async processing function that communicates via queue instead of session state"""
    try:
        def log_message(message):
            """Thread-safe logging via message queue"""
            try:
                message_queue.put(("log", str(message).strip()))
            except:
                pass  # Ignore queue errors

        # Start CLI capture
        start_cli_capture()

        if not session_manager.session_data.get("prompts"):
            session_manager.session_data["prompts"] = []
        session_manager.session_data["prompts"].extend(prompts_list)
        session_manager._save_current_session()

        log_message("🚀 Creating task lists from prompts...")
        update_task_progress(message_queue, "Task Creation", 0.1, "Analyzing prompts and creating task lists")
        
        try:
            task_lists = await asyncio.gather(*(nova.create_task_list_from_prompt_async(p) for p in prompts_list))
            log_message(f"✅ Created {len(task_lists)} task lists")
            update_task_progress(message_queue, "Task Creation", 1.0, f"Created {len(task_lists)} task lists", "completed")
        except Exception as e:
            log_message(f"❌ Error creating task list: {e}")
            message_queue.put(("error", f"Error creating task list: {e}"))
            return

        # Count total tasks for progress tracking
        total_tasks = sum(len(tasks) for tasks in task_lists)
        message_queue.put(("total_tasks", total_tasks))

        results_accumulator = []
        all_parameters_accumulator = []
        completed_task_count = 0

        for idx, (prompt, tasks) in enumerate(zip(prompts_list, task_lists)):
            log_message(f"📝 Processing prompt {idx+1}/{len(prompts_list)}: {prompt[:50]}...")
            
            for task_idx, task in enumerate(tasks):
                task_name = task.name.replace('Handle Intent: ', '')
                log_message(f"🔧 Executing task: {task_name} (Agent: {task.agent})")
                
                # Create task progress entry
                update_task_progress(message_queue, task_name, 0.0, f"Starting {task_name}", "starting")
                
                agent = agents.get(task.agent)
                if not agent:
                    log_message(f"❌ Agent {task.agent} not found for task {task_name}")
                    update_task_progress(message_queue, task_name, 1.0, f"Agent {task.agent} not found", "error")
                    continue
                    
                try:
                    # Log handover parameters for Emil tasks
                    if task.agent == "Emil" and task.args:
                        handover_info = "🔄 Emil Handover Parameters:"
                        for key, value in task.args.items():
                            if key in ['location', 'generation', 'energy_carrier', 'prompt']:
                                handover_info += f"\n   • {key}: {value}"
                        log_message(handover_info)
                    
                    update_task_progress(message_queue, task_name, 0.3, f"Executing {task_name}", "in_progress")
                    
                    result = await agent.handle_task_async(task)
                    results_accumulator.append((task.name, result, task.agent))
                    all_parameters_accumulator.append(task.args)
                    completed_task_count += 1
                    
                    log_message(f"✅ Task completed: {task_name}")
                    update_task_progress(message_queue, task_name, 1.0, f"Completed {task_name}", "completed")
                    
                    # Update overall progress
                    overall_progress = completed_task_count / total_tasks
                    message_queue.put(("overall_progress", {
                        'completed': completed_task_count,
                        'total': total_tasks,
                        'progress': overall_progress
                    }))
                    
                    # Track created files for download
                    if isinstance(result, dict) and result.get('file'):
                        file_info = {
                            'path': result['file'],
                            'name': os.path.basename(result['file']),
                            'task': task.name,
                            'agent': task.agent,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'details': result
                        }
                        message_queue.put(("file_created", file_info))
                        
                except Exception as e:
                    log_message(f"❌ Error during task execution {task_name} by {task.agent}: {e}")
                    update_task_progress(message_queue, task_name, 1.0, f"Error: {str(e)[:50]}...", "error")
                    results_accumulator.append((task.name, {"status": "error", "message": str(e)}, task.agent))

        # Process results
        clean_results_summary = []
        for name, res, agent_name in results_accumulator:
            if isinstance(res, str) and len(res) > 100: 
                res_display = res[:100] + "..."
            else: 
                res_display = res
            clean_results_summary.append({"task": name, "agent": agent_name, "result": res_display})

        # Update session
        session_manager.session_data.update({
            "parameters": session_manager.session_data.get("parameters", []) + all_parameters_accumulator,
            "results": session_manager.session_data.get("results", []) + clean_results_summary,
            "last_modified": datetime.datetime.now().isoformat(),
            "context_open": True,
            "session_active": True
        })
        session_manager._save_current_session()
        log_message(f"💾 Session updated: {session_manager.current_session_file}")

        # Stop CLI capture
        stop_cli_capture()
        
        # Send results via queue
        message_queue.put(("results", results_accumulator))
        message_queue.put(("complete", True))
        
        return results_accumulator
        
    except Exception as e:
        log_message(f"❌ Critical error in processing: {e}")
        message_queue.put(("error", f"Critical processing error: {e}"))
        stop_cli_capture()
        return []

def run_async_processing(prompts_list):
    """Run async processing in a thread-safe way with queue communication"""
    try:
        # Get references to session state objects before starting thread
        kb = st.session_state.kb
        session_manager = st.session_state.session_manager
        agents = st.session_state.agents
        nova = st.session_state.nova
        message_queue = st.session_state.message_queue
        
        if not all([kb, session_manager, agents, nova]):
            st.error("❌ Services not properly initialized")
            return False
        
        # Clear progress history for new run
        st.session_state.progress_history = []
        st.session_state.current_progress = {}
        st.session_state.created_files = []
        st.session_state.task_progress = {}
        st.session_state.total_tasks = 0
        st.session_state.completed_tasks = 0
        
        def run_in_thread():
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the async function with passed objects
                result = loop.run_until_complete(
                    process_single_prompt_batch_async(prompts_list, kb, session_manager, agents, nova, message_queue)
                )
                
                loop.close()
                return result
                
            except Exception as e:
                try:
                    message_queue.put(("error", f"Thread processing error: {e}"))
                except:
                    pass  # Queue might be full or closed
        
        # Start processing in background thread
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        
        return True
        
    except Exception as e:
        st.session_state.processing_log.append(f"❌ Failed to start async processing: {e}")
        st.session_state.error_occurred = True
        st.session_state.error_message = f"Failed to start processing: {e}"
        return False

def process_message_queue():
    """Process messages from the background thread"""
    try:
        while not st.session_state.message_queue.empty():
            try:
                message_type, content = st.session_state.message_queue.get_nowait()
                
                if message_type == "log":
                    if content not in st.session_state.logged_messages_set:
                        st.session_state.processing_log.append(content)
                        st.session_state.logged_messages_set.add(content)
                        
                elif message_type == "error":
                    st.session_state.error_occurred = True
                    st.session_state.error_message = content
                    st.session_state.processing_complete = True
                    st.session_state.backend_task_active = False
                    
                elif message_type == "results":
                    st.session_state.current_results_full = content
                    
                elif message_type == "file_created":
                    st.session_state.created_files.append(content)
                    
                elif message_type == "total_tasks":
                    st.session_state.total_tasks = content
                    
                elif message_type == "task_progress":
                    task_name = content['task_name']
                    st.session_state.task_progress[task_name] = content
                    
                elif message_type == "overall_progress":
                    st.session_state.completed_tasks = content['completed']
                    
                elif message_type == "complete":
                    st.session_state.processing_complete = True
                    st.session_state.backend_task_active = False
                    
            except queue.Empty:
                break
            except Exception as e:
                st.error(f"Error processing message: {e}")
                break
    except Exception as e:
        st.error(f"Error in message processing: {e}")

def display_progress_with_history():
    """Display progress with history preservation"""
    current_progress_data = {}
    
    # Read current progress from PLEXOS progress file
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                current_progress_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Update current progress in session state
    st.session_state.current_progress = current_progress_data
    
    # Check for completed steps (100% progress)
    for task_name, progress_info in current_progress_data.items():
        if progress_info.get('percent', 0) >= 1.0:
            # Check if this step is already in history
            if not any(h['task_name'] == task_name for h in st.session_state.progress_history):
                st.session_state.progress_history.append({
                    'task_name': task_name,
                    'progress_info': progress_info,
                    'completed_at': datetime.datetime.now().isoformat()
                })
    
    # Display completed steps from PLEXOS
    if st.session_state.progress_history:
        st.write("**✅ Completed PLEXOS Steps:**")
        for completed_step in st.session_state.progress_history:
            task_name = completed_step['task_name']
            progress_info = completed_step['progress_info']
            total = progress_info.get('total', 1)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"✅ {task_name}: {total}/{total}")
            with col2:
                st.text("100%")
        
        st.markdown("---")
    
    # NEW: Display completed task progress
    completed_tasks = {k: v for k, v in st.session_state.task_progress.items() 
                      if v.get('status') == 'completed'}
    
    if completed_tasks:
        st.write("**✅ Completed Tasks:**")
        for task_name, task_info in completed_tasks.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"✅ {task_name}")
            with col2:
                st.text("100%")
        
        if current_progress_data or any(v.get('status') != 'completed' for v in st.session_state.task_progress.values()):
            st.markdown("---")
    
    # Display current active progress from PLEXOS
    active_progress = {k: v for k, v in current_progress_data.items() 
                      if v.get('percent', 0) < 1.0}
    
    if active_progress:
        st.write("**🔄 Current PLEXOS Progress:**")
        for task_name, p_info in active_progress.items():
            percent = p_info.get('percent', 0)
            current = p_info.get('current', 0)
            total = p_info.get('total', 1)
            message = p_info.get('message', task_name)
            
            display_message = message
            if len(display_message) > 80:
                display_message = display_message[:77] + "..."
                
            st.write(f"**{task_name}:** {current}/{total}")
            progress_text = f"{percent*100:.1f}% - {display_message}"
            st.progress(percent, text=progress_text)
            
            if message and message != task_name:
                st.caption(f"Currently processing: {display_message}")
            st.write("")
    
    # NEW: Display current active task progress
    active_tasks = {k: v for k, v in st.session_state.task_progress.items() 
                   if v.get('status') in ['starting', 'in_progress']}
    
    if active_tasks:
        if active_progress:
            st.markdown("---")
        st.write("**🔄 Current Task Progress:**")
        for task_name, task_info in active_tasks.items():
            progress = task_info.get('progress', 0)
            message = task_info.get('message', task_name)
            status = task_info.get('status', 'in_progress')
            
            status_icon = "🔄" if status == "in_progress" else "⏳"
            st.write(f"**{status_icon} {task_name}**")
            
            progress_text = f"{progress*100:.1f}% - {message}"
            st.progress(progress, text=progress_text)
            st.write("")
    
    # NEW: Display overall progress if tasks are running
    if st.session_state.total_tasks > 0:
        if active_progress or active_tasks:
            st.markdown("---")
        st.write("**📊 Overall Progress:**")
        overall_progress = st.session_state.completed_tasks / st.session_state.total_tasks
        st.progress(overall_progress, text=f"Completed {st.session_state.completed_tasks}/{st.session_state.total_tasks} tasks ({overall_progress*100:.1f}%)")

def create_download_section():
    """Create download section for created files"""
    if st.session_state.created_files:
        st.markdown("---")
        st.subheader("📥 Download Created Files")
        
        for i, file_info in enumerate(st.session_state.created_files):
            file_path = file_info['path']
            file_name = file_info['name']
            task_name = file_info['task']
            agent_name = file_info['agent']
            
            if os.path.exists(file_path):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"📁 **{file_name}**")
                    st.caption(f"Created by {agent_name} - {task_name.replace('Handle Intent: ', '')}")
                    
                with col2:
                    # File size
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    st.caption(f"Size: {size_mb:.2f} MB")
                    
                with col3:
                    # Download button
                    try:
                        with open(file_path, 'rb') as f:
                            file_bytes = f.read()
                        
                        st.download_button(
                            label="⬇️ Download",
                            data=file_bytes,
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"download_{i}_{file_name}"
                        )
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            else:
                st.warning(f"⚠️ File not found: {file_name}")

# --- Streamlit UI Starts Here ---
st.title("💡 AI Energy Model Builder Interface")

# Process any messages from background thread
if st.session_state.backend_task_active:
    process_message_queue()

# Error Display
if getattr(st.session_state, 'error_occurred', False):
    st.error(f"❌ Error: {st.session_state.error_message}")
    if st.button("🔄 Reset and Try Again"):
        st.session_state.error_occurred = False
        st.session_state.error_message = ''
        st.session_state.backend_task_active = False
        st.session_state.processing_complete = True
        # Clear the queue
        while not st.session_state.message_queue.empty():
            try:
                st.session_state.message_queue.get_nowait()
            except queue.Empty:
                break
        st.rerun()

# Status Display
if getattr(st.session_state, 'backend_task_active', False):
    st.info("🤖 **AI Agents are actively processing your request...** Progress updates below ⬇️")

# Service Initialization
if not st.session_state.services_initialized:
    with st.spinner("⚙️ Initializing services..."):
        if initialize_services():
            st.session_state.services_initialized = True
            st.success("✅ Services initialized successfully!")
        else:
            st.error("❌ Failed to initialize services. Please check the error messages above.")
            st.stop()
    st.rerun()

# --- EXPANDED Processing Log Display ---
log_display_area = st.expander("📋 Backend Processing Log", expanded=True)  # CHANGED: Now expanded by default
with log_display_area:
    if st.session_state.processing_log:
        # Show last 30 log entries to show more detail
        recent_logs = st.session_state.processing_log[-30:]
        for log_entry in recent_logs:
            st.text(log_entry)
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
            if cli_text.strip():
                st.code(cli_text, language=None)
            st.caption("🟢 Live output updating...")
        else:
            st.info("⏳ Waiting for processing output...")

# --- ENHANCED Progress Display with History ---
if getattr(st.session_state, 'backend_task_active', False):
    st.markdown("---")
    st.subheader("🔄 Live Progress")
    display_progress_with_history()
    
    # Auto-refresh every 2 seconds during processing
    time.sleep(2)
    st.rerun()

# --- Prompt Input ---
if (not getattr(st.session_state, 'backend_task_active', False) and 
    getattr(st.session_state, 'processing_complete', True) and
    not getattr(st.session_state, 'error_occurred', False)):
    
    st.markdown("---")
    st.subheader("🎯 Enter Your Request")
    
    # REMOVED: Example prompts section as requested
    
    user_prompts_str = st.text_area(
        "Enter your prompt(s) (one per line):", 
        height=100, 
        key="prompt_input_main",
        placeholder="e.g., Build a wind model for Spain, Greece and Denmark"
    )
    
    process_button = st.button("🚀 Process Prompts", key="process_button_main", type="primary")
    
    if process_button and user_prompts_str:
        st.session_state.prompts_to_process = [p.strip() for p in user_prompts_str.split('\n') if p.strip()]
        if st.session_state.prompts_to_process:
            st.session_state.processing_started = True
            st.session_state.processing_complete = False
            st.session_state.backend_task_active = True
            st.session_state.current_results_full = []
            st.session_state.processing_log = ["🚀 Backend processing initiated by user..."]
            st.session_state.logged_messages_set.clear()
            st.session_state.cli_output_displayed = []
            st.session_state.last_cli_length = 0
            st.session_state.error_occurred = False
            st.session_state.error_message = ''
            
            # Clear message queue
            while not st.session_state.message_queue.empty():
                try:
                    st.session_state.message_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Clean up progress file
            if os.path.exists(PROGRESS_FILE):
                try: 
                    os.remove(PROGRESS_FILE)
                except Exception: 
                    pass
            
            # Start async processing
            if run_async_processing(st.session_state.prompts_to_process):
                st.success("✅ Processing started successfully!")
            else:
                st.error("❌ Failed to start processing.")
                st.session_state.backend_task_active = False
                st.session_state.processing_complete = True
            
            st.rerun()
        else:
            st.warning("⚠️ Please enter at least one prompt.")
    elif process_button and not user_prompts_str:
        st.warning("⚠️ Please enter a prompt before processing.")

# --- Results Display ---
if (getattr(st.session_state, 'processing_complete', True) and 
    not getattr(st.session_state, 'backend_task_active', False) and
    not getattr(st.session_state, 'error_occurred', False)):
    
    results_to_display = st.session_state.get('current_results_full', [])
    if results_to_display:
        st.markdown("---")
        st.success("✅ Processing Complete!")
        st.subheader("📊 Results")
        
        for task_name_result, result_data, agent_name in results_to_display:
            with st.expander(f"📋 {task_name_result.replace('Handle Intent: ', '')} (by {agent_name})", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if isinstance(result_data, dict):
                        if result_data.get("status") == "success":
                            st.success(f"✅ {result_data.get('message', 'Operation successful')}")
                            
                            # Display model parameters if available
                            params_display = []
                            if result_data.get('location'): 
                                params_display.append(f"📍 **Location:** {result_data.get('location')}")
                            if result_data.get('generation_type'): 
                                params_display.append(f"⚡ **Generation:** {result_data.get('generation_type')}")
                            if result_data.get('energy_carrier'): 
                                params_display.append(f"🔋 **Carrier:** {result_data.get('energy_carrier')}")
                            
                            if params_display:
                                for param in params_display:
                                    st.markdown(param)
                            
                            if result_data.get('file'):
                                st.info(f"📁 **Output file:** `{os.path.basename(result_data.get('file'))}`")
                                
                        elif result_data.get("status") == "error":
                            st.error(f"❌ {result_data.get('message')}")
                        else:
                            st.write(str(result_data))
                    else:
                        # Handle string results (like math calculations, general questions)
                        if isinstance(result_data, str):
                            if len(result_data) > 500:
                                st.text_area("Result:", value=result_data, height=200, disabled=True)
                            else:
                                st.info(result_data)
                        else:
                            st.write(str(result_data))
                
                with col2:
                    st.caption(f"**Agent:** {agent_name}")
                    st.caption(f"**Status:** Complete")
        
        # Download Section
        create_download_section()
        
        # NEW: Live Progress at the end (collapsed)
        st.markdown("---")
        with st.expander("🔄 Final Progress Summary", expanded=False):  # CHANGED: Collapsed by default
            display_progress_with_history()
        
        # Option to start new processing
        st.markdown("---")
        if st.button("🔄 Process New Request", key="new_request_button"):
            st.session_state.current_results_full = []
            st.session_state.processing_log = []
            st.session_state.logged_messages_set.clear()
            st.session_state.task_progress = {}
            st.session_state.progress_history = []
            st.session_state.completed_tasks = 0
            st.session_state.total_tasks = 0
            st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("💡 AI Energy Model Builder v2.0 | Powered by Nova, Emil, Ivan & Lola")
