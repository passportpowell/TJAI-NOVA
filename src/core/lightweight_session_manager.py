import os
import json
import datetime
import time
from typing import Dict, Any, List, Optional

class LightweightSessionManager:
    """
    Simplified session manager optimized for RAG systems.
    Focuses on essential information without redundancy.
    """
    def __init__(self, base_path="sessions"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.session_data = None
        self.current_file = None
        self.start_time = None
        
    def create_session(self, input_prompt: str) -> tuple:
        """Create a new lightweight session."""
        now = datetime.datetime.now()
        self.start_time = time.time()
        
        # Unique session ID
        session_id = f"session_{int(time.time() * 1000000)}"
        
        # Create folder structure
        folder_path = os.path.join(
            self.base_path,
            str(now.year),
            f"{now.month:02d}",
            f"{now.day:02d}"
        )
        os.makedirs(folder_path, exist_ok=True)
        
        # Session file
        filename = f"{session_id}.json"
        self.current_file = os.path.join(folder_path, filename)
        
        # Initialize lightweight session data
        self.session_data = {
            "session_id": session_id,
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "input": input_prompt,
            "workflow": [],
            "summary": {
                "tasks_completed": 0,
                "agents_used": [],
                "outputs": {},
                "success": False
            }
        }
        
        return session_id, self.current_file
        
    def add_workflow_step(self, agent: str, action: str, input_text: str, 
                         output: Any, context_change: str = None):
        """Add a workflow step with minimal information."""
        if not self.session_data:
            return
            
        step = {
            "step": len(self.session_data["workflow"]) + 1,
            "agent": agent,
            "action": action,
            "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
            "output": self._summarize_output(output),
        }
        
        # Only add context changes if something actually changed
        if context_change:
            step["context_change"] = context_change
            
        self.session_data["workflow"].append(step)
        
        # Update summary
        if agent not in self.session_data["summary"]["agents_used"]:
            self.session_data["summary"]["agents_used"].append(agent)
        self.session_data["summary"]["tasks_completed"] += 1
        
        # Save after each step
        self._save()
        
    def _summarize_output(self, output: Any) -> str:
        """Create a concise summary of the output."""
        if isinstance(output, dict):
            if output.get("status") == "success":
                # For model creation
                if "file" in output:
                    filename = os.path.basename(output["file"])
                    return f"Created: {filename}"
                return output.get("message", "Success")
            return output.get("message", str(output)[:50] + "...")
        elif isinstance(output, str):
            # For text outputs, keep it short
            if len(output) > 100:
                # Extract key information
                if "Executive Summary" in output:
                    return "Executive Summary generated"
                elif "capital" in output.lower():
                    return output[:50]
                return output[:100] + "..."
            return output
        else:
            return str(output)[:50] + "..."
            
    def set_outputs(self, outputs: Dict[str, str]):
        """Set the final outputs summary."""
        if self.session_data:
            self.session_data["summary"]["outputs"] = outputs
            self._save()
            
    def finalize(self, success: bool = True):
        """Finalize the session."""
        if self.session_data:
            # Calculate duration
            duration = int(time.time() - self.start_time) if self.start_time else 0
            self.session_data["duration_seconds"] = duration
            self.session_data["summary"]["success"] = success
            
            self._save()
            
    def _save(self):
        """Save the current session data."""
        if self.current_file and self.session_data:
            with open(self.current_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
                
    def get_rag_summary(self) -> str:
        """Generate a text summary optimized for RAG embedding."""
        if not self.session_data:
            return ""
            
        summary_parts = [
            f"Session {self.session_data['session_id']}",
            f"Input: {self.session_data['input']}",
            f"Agents: {', '.join(self.session_data['summary']['agents_used'])}",
        ]
        
        # Add key outputs
        outputs = self.session_data['summary']['outputs']
        if outputs:
            summary_parts.append("Outputs:")
            for key, value in outputs.items():
                summary_parts.append(f"  - {key}: {value}")
                
        # Add workflow summary
        summary_parts.append("Workflow:")
        for step in self.session_data['workflow']:
            summary_parts.append(
                f"  {step['step']}. {step['agent']}: {step['action']} → {step['output']}"
            )
            
        return "\n".join(summary_parts)