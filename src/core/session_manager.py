import os
import json
import datetime
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple

class SessionManager:
    """
    Manages creation, updating, and storage of user sessions.
    Enhanced with improved naming convention and index files.
    """
    
    def __init__(self, base_path="sessions"):
        """
        Initialize the session manager.
        
        Parameters:
            base_path (str): Base directory for storing sessions
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.current_session_file = None
        self.current_session_id = None
        self.session_data = None


    def create_session(self, prompts=None) -> Tuple[str, str]:
        """
        Create a new session with extended metadata for persistent context.
        
        Parameters:
            prompts (list, optional): Initial prompts for the session
                
        Returns:
            tuple: (session_id, session_file)
        """
        now = datetime.datetime.now()
        
        # Generate short UUID (6 characters) for uniqueness
        short_uuid = uuid.uuid4().hex[:6]
        
        # Format timestamp in readable format with separators
        readable_timestamp = now.strftime("%Y_%m_%d_T%H_%M_%S")
        
        # Create session ID with timestamp and short UUID
        session_id = f"session_{readable_timestamp}_{short_uuid}"
        
        # Create folder structure
        folder_path = os.path.join(
            self.base_path,
            str(now.year),
            f"{now.month:02d}",
            f"{now.day:02d}"
        )
        os.makedirs(folder_path, exist_ok=True)
        
        # Full path to session file
        session_file = os.path.join(folder_path, f"{session_id}.json")
        
        # Initialize session data with extended metadata
        self.session_data = {
            "id": session_id,
            "metadata": {
                "timestamp": now.isoformat(),
                "created_at": now.isoformat(),
                "last_modified": now.isoformat(),
                "finalized_at": None,
                "context_open": True,             # New field: indicates session is active
                "session_active": True            # Flag to keep context window open
            },
            "prompts": prompts or [],
            "parameters": [],
            "context_handovers": [],
            "results": [],
            "summary": None
        }
        
        # Save initial session data
        with open(session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        # Update the daily index
        self._update_daily_index(folder_path, session_id, session_file, now, prompts)
        
        self.current_session_file = session_file
        self.current_session_id = session_id
        
        print(f"Created new session: {session_id}")
        print(f"Session file: {session_file}")
        
        return session_id, session_file   
   
    def add_context_handover(self, from_agent: str, to_agent: str, context: Dict[str, Any]):
        """
        Record a context handover between agents.
        
        Parameters:
            from_agent (str): Source agent
            to_agent (str): Destination agent
            context (dict): Context data being passed
        """
        if not self.session_data:
            return

        handover = {
            "timestamp": datetime.datetime.now().isoformat(),
            "from": from_agent,
            "to": to_agent,
            "context": dict(context)  # force copy
        }

        if "prompt" in context:
            handover["context"]["prompt"] = context["prompt"]

        self.session_data["context_handovers"].append(handover)
        self._save_current_session()

    def update_session(self, updates: Dict[str, Any]):
        """
        Update session data with new information.
        
        Parameters:
            updates (dict): New data to add to the session
        """
        if not self.session_data or not self.current_session_file:
            print("Warning: No active session to update")
            return

        for key, value in updates.items():
            if key == "context":
                self.session_data["context"].update(value)
            elif key in ["prompts", "tasks", "results", "agents_involved"]:
                if isinstance(value, list):
                    self.session_data.setdefault(key, []).extend(value)
                else:
                    self.session_data.setdefault(key, []).append(value)
            else:
                self.session_data[key] = value

        self.session_data["metadata"]["last_modified"] = datetime.datetime.now().isoformat()
        self._save_current_session()

    def finalize_session(self, success=True):
        """
        Finalize the session with results and metrics.
        
        Parameters:
            success (bool): Whether the session completed successfully
        """
        if not self.session_data or not self.current_session_file:
            print("Warning: No active session to finalize")
            return
        
        # Calculate duration
        created_at = datetime.datetime.fromisoformat(self.session_data["metadata"]["created_at"])
        now = datetime.datetime.now()
        duration = (now - created_at).total_seconds()
        
        # Update session metadata
        self.session_data["metadata"]["finalized_at"] = now.isoformat()
        self.session_data["metadata"]["duration_seconds"] = duration
        self.session_data["metadata"]["success"] = success
        
        # Get list of unique agents used
        agents_used = set()
        for handover in self.session_data.get("context_handovers", []):
            agents_used.add(handover.get("from"))
            agents_used.add(handover.get("to"))
        agents_used.discard(None)
        
        # Tasks completed count
        tasks_completed = len(self.session_data.get("results", []))
        
        # Save updated session
        self._save_current_session()
        
        # Update the index with final metrics
        folder_path = os.path.dirname(self.current_session_file)
        index_file = os.path.join(folder_path, "index.json")
        
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Find this session in the index
            for session in index_data["sessions"]:
                if session["id"] == self.current_session_id:
                    # Update with final metrics
                    session["agents_used"] = list(agents_used)
                    session["tasks_completed"] = tasks_completed
                    session["duration_seconds"] = duration
                    session["success"] = success
                    break
            
            # Save updated index
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
        
        print(f"Session {self.current_session_id} finalized successfully")
        return self.current_session_id

    def _save_current_session(self):
        """Save the current session data to disk."""
        if self.current_session_file and self.session_data:
            try:
                with open(self.current_session_file, 'w') as f:
                    json.dump(self.session_data, f, indent=2, default=str)
            except Exception as e:
                print(f"Error saving session: {e}")

    def _update_daily_index(self, folder_path, session_id, session_file, timestamp, prompts=None):
        """
        Update the daily index file with the new session.
        
        Parameters:
            folder_path (str): Path to the day's folder
            session_id (str): ID of the new session
            session_file (str): Path to the session file
            timestamp (datetime): Session creation timestamp
            prompts (list): Initial prompts for the session
        """
        index_file = os.path.join(folder_path, "index.json")
        
        # Initial structure for a new index
        if not os.path.exists(index_file):
            index_data = {
                "date": timestamp.strftime("%Y-%m-%d"),
                "sessions_count": 0,
                "sessions": []
            }
        else:
            # Load existing index
            with open(index_file, 'r') as f:
                index_data = json.load(f)
        
        # Create session summary for the index
        prompt_summary = ", ".join(prompts[:2]) if prompts else "No prompts"
        if len(prompt_summary) > 50:
            prompt_summary = prompt_summary[:47] + "..."
        
        session_summary = {
            "id": session_id,
            "timestamp": timestamp.isoformat(),
            "prompt_summary": prompt_summary,
            "agents_used": [],  # Will be populated during execution
            "tasks_completed": 0,  # Will be updated during finalization
            "duration_seconds": 0,  # Will be calculated on finalization
            "file": os.path.basename(session_file)
        }
        
        # Add to index
        index_data["sessions"].append(session_summary)
        index_data["sessions_count"] = len(index_data["sessions"])
        
        # Save index
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2, default=str)

    def _update_index(self, session_id: str, session_file: str, metadata: Dict[str, Any]):
        """
        Update the global index file with a new session.
        Legacy method maintained for backward compatibility.
        
        Parameters:
            session_id (str): ID of the session
            session_file (str): Path to the session file
            metadata (dict): Session metadata
        """
        index_file = os.path.join(self.base_path, "index.json")

        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {"sessions": [], "total_sessions": 0}

        existing = False
        for i, session in enumerate(index["sessions"]):
            if session["id"] == session_id:
                index["sessions"][i] = {
                    "id": session_id,
                    "file": session_file,
                    "created": metadata["timestamp"]
                }
                existing = True
                break

        if not existing:
            index["sessions"].append({
                "id": session_id,
                "file": session_file,
                "created": metadata["timestamp"]
            })

        index["total_sessions"] = len(index["sessions"])
        index["last_updated"] = datetime.datetime.now().isoformat()

        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)


    def update_session_results(self, success=True, close_context=False):
        """
        Update session with results but keep context open by default.
        
        Parameters:
            success (bool): Whether the task completed successfully
            close_context (bool): Whether to close the context window (defaults to False)
            
        Returns:
            str: Session ID
        """
        if not self.session_data or not self.current_session_file:
            print("Warning: No active session to update")
            return
        
        now = datetime.datetime.now()
        
        # Update session metadata
        self.session_data["metadata"]["last_modified"] = now.isoformat()
        
        # Only set finalized timestamp if explicitly closing the context
        if close_context:
            self.session_data["metadata"]["finalized_at"] = now.isoformat()
            self.session_data["metadata"]["context_open"] = False
            self.session_data["metadata"]["session_active"] = False
        
        # Calculate duration (for metrics only)
        created_at = datetime.datetime.fromisoformat(self.session_data["metadata"]["created_at"])
        duration = (now - created_at).total_seconds()
        self.session_data["metadata"]["duration_seconds"] = duration
        self.session_data["metadata"]["success"] = success
        
        # Get list of unique agents used (for metrics)
        agents_used = set()
        for handover in self.session_data.get("context_handovers", []):
            agents_used.add(handover.get("from"))
            agents_used.add(handover.get("to"))
        agents_used.discard(None)
        
        # Tasks completed count
        tasks_completed = len(self.session_data.get("results", []))
        
        # Save updated session
        self._save_current_session()
        
        # Update the index with metrics
        folder_path = os.path.dirname(self.current_session_file)
        index_file = os.path.join(folder_path, "index.json")
        
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Find this session in the index
            for session in index_data["sessions"]:
                if session["id"] == self.current_session_id:
                    # Update with metrics
                    session["agents_used"] = list(agents_used)
                    session["tasks_completed"] = tasks_completed
                    session["duration_seconds"] = duration
                    session["success"] = success
                    session["context_open"] = not close_context  # Keep context open unless explicitly closed
                    break
            
            # Save updated index
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
        
        print(f"Session {self.current_session_id} updated" + (" and closed" if close_context else " (context remains open)"))
        return self.current_session_id



    def close_session_context(self):
        """
        Explicitly close the session context when the user is done with it.
        
        Returns:
            bool: Success status
        """
        return self.update_session_results(success=True, close_context=True) is not None



