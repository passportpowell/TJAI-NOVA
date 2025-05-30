# -*- coding: utf-8 -*-
"""
Modified loading_bar.py with progress file output

This version writes progress updates to a file that Streamlit can read.
"""
import os
import json
import tempfile

# Create a progress file path if it doesn't exist
PROGRESS_FILE = os.path.join(tempfile.gettempdir(), "plexos_progress.json")

def save_progress(task_name, current, total, message=""):
    """Write progress data to the progress file"""
    try:
        # Make sure the progress file exists
        if not os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({}, f)
        
        # Read current progress data
        with open(PROGRESS_FILE, 'r') as f:
            try:
                progress_data = json.load(f)
            except json.JSONDecodeError:
                progress_data = {}
        
        # Update with new progress
        progress_data[task_name] = {
            'current': current,
            'total': total,
            'message': message,
            'percent': min(1.0, current / max(1, float(total))),
            'timestamp': __import__('time').time()
        }
        
        # Write back to file
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
            
    except Exception as e:
        # Fail silently to avoid disrupting the main process
        print(f"Error saving progress: {str(e)}")

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=20, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    
    # Also save progress to file for Streamlit to read
    save_progress(
        task_name=prefix or "Progress", 
        current=iteration,
        total=total,
        message=f"{prefix} {suffix}"
    )
    
    # Print a newline when complete
    if iteration == total: 
        print()