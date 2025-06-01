import streamlit as st
import time
import os
import json
import sys

# ---- THIS MUST BE FIRST ----
st.set_page_config(page_title="AI-Driven PLEXOS Model Builder", layout="centered")

# ---- Use a project-local progress file ----
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "plexos_progress.json")
st.write(f"Progress file location: {PROGRESS_FILE}")  # For debug: see where we are reading/writing

# ---- Flexible main.py import (update as needed for your structure) ----
main_imported = False
main_locations = [
    ".",                        # Same folder as app.py
    "./src",                    # src/main.py
    "./agents",                 # agents/main.py
    os.path.dirname(__file__),  # current file's dir
]
for loc in main_locations:
    path = os.path.abspath(loc)
    if os.path.exists(os.path.join(path, "main.py")):
        sys.path.insert(0, path)
        try:
            from main import interactive_async_main
            main_imported = True
            break
        except ImportError:
            pass
if not main_imported:
    st.error("Could not import 'interactive_async_main' from main.py. Check your folder and import paths.")
    st.stop()

def read_progress(task_name="Progress"):
    """Reads the current progress dictionary for the given task (returns None if not found)."""
    if not os.path.exists(PROGRESS_FILE):
        return None
    try:
        with open(PROGRESS_FILE, "r") as f:
            progress_data = json.load(f)
        return progress_data.get(task_name) or next(iter(progress_data.values()), None)
    except Exception:
        return None

def streamlit_prompt_runner(prompt):
    """
    Wrapper to run your CLI async main with the prompt via monkey-patching input().
    """
    import asyncio
    import builtins

    async def run_once():
        # Patch input() to simulate CLI: returns prompt, then 'done'
        inputs = iter([prompt, "done"])
        orig_input = builtins.input
        builtins.input = lambda _: next(inputs)
        try:
            await interactive_async_main()
        finally:
            builtins.input = orig_input

    asyncio.run(run_once())

# ---- Streamlit UI ----

st.title("AI-Driven PLEXOS Model Builder (Beta)")

st.markdown("""
Enter your model/task prompt below.  
Progress updates will appear while your model builds.
""")

user_prompt = st.text_area("Your prompt", height=100, key="prompt_input")
run_button = st.button("Run", type="primary")

if run_button and user_prompt.strip():
    # Clear progress file before starting
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    st.info("Model building started. Progress updates will appear below…")
    status_area = st.empty()
    progress_bar = st.progress(0.0)

    # Capture results (including stdout log)
    from io import StringIO
    result_text = []

    def process_and_capture():
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            streamlit_prompt_runner(user_prompt)
            result_text.append(mystdout.getvalue())
        finally:
            sys.stdout = old_stdout

    import threading
    thread = threading.Thread(target=process_and_capture)
    thread.start()

    last_progress = 0.0
    last_message = ""
    while thread.is_alive():
        progress = read_progress()
        if progress:
            percent = progress.get("percent", 0)
            message = progress.get("message", "")
            progress_bar.progress(percent)
            status_area.markdown(f"**{message}**  \n{percent*100:.1f}%")
            last_progress = percent
            last_message = message
        else:
            progress_bar.progress(last_progress)
            status_area.markdown(f"**{last_message}**  \n{last_progress*100:.1f}%")
        time.sleep(0.5)

    # One last update after done
    thread.join()
    progress = read_progress()
    if progress:
        progress_bar.progress(progress.get("percent", 1.0))
        status_area.markdown(f"**{progress.get('message', 'Done!')}**")
    else:
        progress_bar.progress(1.0)
        status_area.markdown("**Done!**")

    st.success("Task completed.")
    st.markdown("---")
    st.subheader("Results / Log")
    st.code("".join(result_text), language="text")

elif run_button:
    st.warning("Please enter a prompt before running.")
