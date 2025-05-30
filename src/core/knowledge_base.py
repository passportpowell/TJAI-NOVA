# knowledge_base.py
import asyncio
import os
import json
import time
import datetime
from typing import Any, Dict, List, Optional

class KnowledgeBase:
    """
    A central repository for storing context, results, and configuration data.
    Enhanced with persistent storage, categorization, and session tracking.
    """
    def __init__(self, storage_path="knowledge_db", use_persistence=True):
        self.storage = {}
        self.storage_path = storage_path
        self.use_persistence = use_persistence
        # Create a lock for thread safety in async operations
        self.lock = asyncio.Lock()
        
        # Create storage directory if persistence is enabled
        if use_persistence:
            os.makedirs(storage_path, exist_ok=True)
            self.load_from_disk()

    
    def set_item(self, key: str, value: any, category=None):
        """
        Synchronous set with optional category tagging.
        
        Parameters:
            key (str): The key to store the value under
            value (any): The value to store
            category (str, optional): Category to tag this item with
        """
        # Store in categorized structure if category is provided
        if category:
            if "__categories__" not in self.storage:
                self.storage["__categories__"] = {}
            
            if category not in self.storage["__categories__"]:
                self.storage["__categories__"][category] = []
                
            # Add key to category if not already there
            if key not in self.storage["__categories__"][category]:
                self.storage["__categories__"][category].append(key)
        
        # Store the actual data
        self.storage[key] = value
        
        # Save changes to disk if persistence is enabled
        if self.use_persistence:
            self.save_to_disk()

    
    def get_item(self, key: str) -> any:
        """Synchronous get - original method preserved"""
        return self.storage.get(key)

    
    def get_item(self, key: str, default=None) -> any:
        """Synchronous get with default value support"""
        return self.storage.get(key, default)
    
    
    def update(self, data: dict):
        """
        Synchronous update with optional persistence
        
        Parameters:
            data (dict): Dictionary of key-value pairs to update
        """
        self.storage.update(data)
        
        # Save changes to disk if persistence is enabled
        if self.use_persistence:
            self.save_to_disk()

    
    def __repr__(self):
        return f"KnowledgeBase({self.storage})"
    
    
    # Async methods
    async def set_item_async(self, key: str, value: any, category=None):
        """Thread-safe asynchronous set with optional category"""
        async with self.lock:
            self.set_item(key, value, category)

    
    async def get_item_async(self, key: str) -> any:
        """Thread-safe asynchronous get"""
        async with self.lock:
            return self.storage.get(key)

    
    async def update_async(self, data: dict):
        """Thread-safe asynchronous update"""
        async with self.lock:
            self.storage.update(data)
            if self.use_persistence:
                await asyncio.to_thread(self.save_to_disk)
    
    
    # Persistence methods
    def load_from_disk(self):
        """Load knowledge data from persistent storage"""
        try:
            main_db_path = os.path.join(self.storage_path, "main_db.json")
            if os.path.exists(main_db_path):
                with open(main_db_path, 'r') as f:
                    stored_data = json.load(f)
                    self.storage.update(stored_data)
                print(f"Loaded {len(stored_data)} items from persistent storage")
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
    
    
    def save_to_disk(self):
        """Save current knowledge to persistent storage"""
        if not self.use_persistence:
            return
            
        try:
            # Ensure directory exists
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Save main database
            main_db_path = os.path.join(self.storage_path, "main_db.json")
            with open(main_db_path, 'w') as f:
                json.dump(self.storage, f, indent=2, default=str)
                
            # Create periodic backup (once per hour)
            hour_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
            backup_path = os.path.join(self.storage_path, f"backup_{hour_timestamp}.json")
            
            # Only create if doesn't exist for this hour
            if not os.path.exists(backup_path):
                with open(backup_path, 'w') as f:
                    json.dump(self.storage, f, default=str)
                
                # Limit number of backups
                self._cleanup_old_backups()
        except Exception as e:
            print(f"Error saving knowledge base: {str(e)}")
    
    
    def _cleanup_old_backups(self, max_backups=10):
        """Maintain only a limited number of backups"""
        try:
            backups = [f for f in os.listdir(self.storage_path) if f.startswith("backup_")]
            if len(backups) > max_backups:
                backups.sort()  # Sort by timestamp
                for old_backup in backups[:-max_backups]:
                    os.remove(os.path.join(self.storage_path, old_backup))
        except Exception as e:
            print(f"Error cleaning up backups: {str(e)}")
    
    
    # Category methods
    def get_items_by_category(self, category):
        """
        Retrieve all items in a specific category
        
        Parameters:
            category (str): The category to retrieve items from
            
        Returns:
            dict: Dictionary of items in the category
        """
        if "__categories__" not in self.storage or category not in self.storage["__categories__"]:
            return {}
            
        return {key: self.storage.get(key) 
                for key in self.storage["__categories__"][category]}
    
    
    # Session management
    def create_session(self, session_id=None):
        """
        Create a new session for tracking interactions
        
        Parameters:
            session_id (str, optional): Custom session ID
            
        Returns:
            str: The session ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
            
        session_data = {
            "id": session_id,
            "start_time": datetime.datetime.now().isoformat(),
            "interactions": [],
            "models_created": [],
            "analyses_performed": [],
            "reports_generated": []
        }
        
        self.set_item(f"session_{session_id}", session_data, category="sessions")
        self.set_item("current_session", session_id)
        return session_id
    
    
    def log_interaction(self, prompt, response, agent="Nova", function=None):
        current_session = self.get_item("current_session")
        if not current_session:
            current_session = self.create_session()

        session_data = self.get_item(f"session_{current_session}")
        if session_data:
            # ✅ Ensure 'interactions' exists
            if "interactions" not in session_data:
                session_data["interactions"] = []

            interaction = {
                "timestamp": datetime.datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "agent": agent,
                "function": function
            }

            session_data["interactions"].append(interaction)
            self.set_item(f"session_{current_session}", session_data, category="sessions")
    
    
    # Context retrieval
    def get_context_for_agent(self, agent_name, context_depth=5):
        """
        Get relevant context information for a specific agent
        
        Parameters:
            agent_name (str): Name of the agent (Nova, Emil, Ivan, Lola)
            context_depth (int): Number of recent interactions to include
            
        Returns:
            dict: Context information relevant to the agent
        """
        current_session = self.get_item("current_session")
        if not current_session:
            return {}
            
        # Get current session data
        session_data = self.get_item(f"session_{current_session}")
        if not session_data:
            return {}
            
        # Get recent interactions
        recent_interactions = []
        for interaction in reversed(session_data.get("interactions", [])):
            if len(recent_interactions) >= context_depth:
                break
            recent_interactions.append(interaction)
            
        # Get agent-specific context based on agent role
        agent_context = {"recent_interactions": recent_interactions}
        
        if agent_name == "Emil":
            # Energy modeling context
            agent_context["latest_model"] = self.get_item("latest_model_details")
            agent_context["energy_models"] = session_data.get("models_created", [])
            
        elif agent_name == "Lola":
            # Report writing context
            agent_context["latest_model"] = self.get_item("latest_model_details")
            agent_context["latest_analysis"] = self.get_item("latest_analysis_results")
            agent_context["reports"] = session_data.get("reports_generated", [])
            
        elif agent_name == "Ivan":
            # Code and image generation context
            agent_context["latest_image"] = self.get_item("last_dalle_prompt")
            
        return agent_context
    
    
    # Data archival
    def archive_old_sessions(self, days_threshold=30):
        """
        Archive sessions older than the threshold
        
        Parameters:
            days_threshold (int): Age in days after which to archive
        """
        if "__categories__" not in self.storage or "sessions" not in self.storage["__categories__"]:
            return
            
        current_time = datetime.datetime.now()
        sessions_to_archive = []
        
        for session_key in self.storage["__categories__"]["sessions"]:
            session_data = self.storage.get(session_key)
            if not session_data or "start_time" not in session_data:
                continue
                
            try:
                start_time = datetime.datetime.fromisoformat(session_data["start_time"])
                age_days = (current_time - start_time).days
                
                if age_days > days_threshold:
                    sessions_to_archive.append(session_key)
            except Exception:
                continue
        
        # Archive the old sessions
        if sessions_to_archive:
            archive_data = {"archived_sessions": {}}
            
            for session_key in sessions_to_archive:
                # Move to archive
                archive_data["archived_sessions"][session_key] = self.storage[session_key]
                
                # Remove from main storage
                del self.storage[session_key]
                self.storage["__categories__"]["sessions"].remove(session_key)
            
            # Save archive
            archive_path = os.path.join(self.storage_path, f"archive_{int(time.time())}.json")
            with open(archive_path, 'w') as f:
                json.dump(archive_data, f, indent=2, default=str)
                
            print(f"Archived {len(sessions_to_archive)} old sessions")
            
            # Save main database after archiving
            if self.use_persistence:
                self.save_to_disk()
    
    
    # Search capability
    def search_knowledge_base(self, query, categories=None):
        """
        Search the knowledge base for relevant information
        
        Parameters:
            query (str): Search query
            categories (list, optional): List of categories to search in
            
        Returns:
            list: List of matching items
        """
        results = []
        
        if categories is None:
            # Search all items
            for key, value in self.storage.items():
                if key == "__categories__":
                    continue
                    
                # Try to match in key or values
                if self._matches_query(key, query) or self._matches_query(value, query):
                    results.append({"key": key, "value": value})
        else:
            # Search only in specified categories
            for category in categories:
                if "__categories__" in self.storage and category in self.storage["__categories__"]:
                    for key in self.storage["__categories__"][category]:
                        value = self.storage.get(key)
                        if self._matches_query(key, query) or self._matches_query(value, query):
                            results.append({"key": key, "value": value, "category": category})
        
        return results
    
    
    def _matches_query(self, item, query):
        """Check if an item matches the search query"""
        if isinstance(item, str):
            return query.lower() in item.lower()
        elif isinstance(item, dict):
            return any(self._matches_query(v, query) for v in item.values())
        elif isinstance(item, list):
            return any(self._matches_query(v, query) for v in item)
        return False
    

    def store_report(self, report, prompt=None, model_details=None):
        """
        Store a report in both the session history and current session data.
        
        Parameters:
            report (str): The generated report
            prompt (str, optional): The original prompt
            model_details (dict, optional): Details about the model
        """
        # Get current session
        current_session = self.get_item("current_session")
        if not current_session:
            current_session = self.create_session()
        
        # Get session data
        session_data = self.get_item(f"session_{current_session}")
        
        if session_data:
            # Create report entry
            report_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "report": report,
                "prompt": prompt or "No prompt provided"
            }
            
            # Add to reports_generated list in session data
            if "reports_generated" not in session_data:
                session_data["reports_generated"] = []
            
            session_data["reports_generated"].append(report_entry)
            
            # Update session data
            self.set_item(f"session_{current_session}", session_data, category="sessions")
            
            # Now update the main session history
            history = self.get_item("session_history") or {}
            
            # Ensure reports list exists
            if "reports" not in history:
                history["reports"] = []
            
            # Get model details if not provided
            if model_details is None:
                model_details = self.get_item("latest_model_details") or {}
            
            # Create history entry
            history_entry = {
                "session_id": session_data.get("id", current_session),
                "prompt": prompt or "No prompt provided",
                "result": report,
                "model_type": model_details.get("generation_type", "Unknown"),
                "location": model_details.get("location", "Unknown"),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Add to reports list in history
            history["reports"].append(history_entry)
            
            # Update history
            self.set_item("session_history", history)
            
            print(f"Report stored in session {current_session} and session history")
            print(f"Session now has {len(session_data['reports_generated'])} reports")
            print(f"History now has {len(history['reports'])} reports")
            
            return True
        
        return False



        """
        Extension to add the missing get_session_conversation method to KnowledgeBase
        This should be added to the KnowledgeBase class in src/core/knowledge_base.py
        """

        
        def get_session_conversation(self, session_id):
            """
            Retrieve the full conversation details for a specific session.
            
            Parameters:
                session_id (int or str): The ID of the session to retrieve
                
            Returns:
                dict: Conversation details with prompts and results
            """
            session_details = self.get_session_details(session_id)
            
            if session_details and 'prompts' in session_details and 'results' in session_details:
                conversation = []
                for prompt, result in zip(session_details['prompts'], session_details['results']):
                    conversation.append({
                        "prompt": prompt,
                        "result": result
                    })
                
                return {
                    "session_id": session_details.get('id'),
                    "timestamp": session_details.get('timestamp'),
                    "conversation": conversation
                }
            
            return None

        def get_session_details(self, session_id):
            """
            Retrieve detailed information about a specific session.
            
            Parameters:
                session_id (int or str): The ID of the session to retrieve
                
            Returns:
                dict: Detailed session information or None if not found
            """
            # First, try to get session details from the session history
            history = self.storage.get("session_history", {})
            sessions = history.get("sessions", [])
            
            # Convert session_id to integer or handle string representations
            try:
                # If it's a string like 'session_X', extract the number
                if isinstance(session_id, str):
                    session_id = int(session_id.replace('session_', ''))
            except ValueError:
                print(f"Invalid session ID format: {session_id}")
                return None
            
            # Look for the session in the list of sessions
            for session in sessions:
                if session.get('id') == session_id:
                    return session
            
            # If not found in session history, check alternative storage methods
            alternative_key = f"session_{session_id}"
            alternative_session = self.storage.get(alternative_key)
            if alternative_session:
                return alternative_session
            
            return None





def get_conversation_context(self, session_id=None):
    """Get the conversation context for the current or specified session"""
    if session_id is None:
        session_id = self.get_item("current_session")
        
    context_key = f"conversation_context_{session_id}"
    context = self.get_item(context_key)
    
    if not context:
        # Initialize new context if none exists
        context = {
            "turns": [],                # List of Q&A pairs
            "entities": {},             # Entities mentioned (name -> value)
            "current_subject": None,    # Current subject of conversation
            "last_question": "",
            "last_answer": ""
        }
        self.set_item(context_key, context)
        
    return context


def update_conversation_context(self, question, answer, session_id=None):
    """Update the conversation context with a new Q&A pair"""
    if session_id is None:
        session_id = self.get_item("current_session")
        
    context = self.get_conversation_context(session_id)
    
    # Update the context
    context["turns"].append({"question": question, "answer": answer})
    context["last_question"] = question
    context["last_answer"] = answer
    
    # Keep only the last 5 turns to prevent context from growing too large
    if len(context["turns"]) > 5:
        context["turns"] = context["turns"][-5:]
    
    # Store the updated context
    context_key = f"conversation_context_{session_id}"
    self.set_item(context_key, context)    


def create_session(self, session_id=None, force_new=True):
    """
    Create a new session for tracking interactions
    
    Parameters:
        session_id (str, optional): Custom session ID
        force_new (bool): If True, always create a new session even if one exists
        
    Returns:
        str: The session ID
    """
    # If force_new is True or no current session exists, create new one
    current_session = self.get_item("current_session") if not force_new else None
    
    if current_session and not force_new:
        # Return existing session
        return current_session
        
    # Create new session
    if session_id is None:
        # Use timestamp with microseconds for uniqueness
        import time
        session_id = f"session_{int(time.time() * 1000000)}"
        
    session_data = {
        "id": session_id,
        "start_time": datetime.datetime.now().isoformat(),
        "interactions": [],
        "models_created": [],
        "analyses_performed": [],
        "reports_generated": []
    }
    
    self.set_item(f"session_{session_id}", session_data, category="sessions")
    self.set_item("current_session", session_id)
    
    print(f"Created new session: {session_id}")
    return session_id




