# Modified Task Manager to support hierarchical task structures

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import json


class Task(BaseModel):
    """
    Represents a task or subtask for the agent system.
    Enhanced to support hierarchical task structures and status tracking.
    """
    name: str
    description: str
    agent: str                  # Name of the agent responsible (e.g., "Nova", "Emil", "Ivan", "Lola")
    function_name: Optional[str] = None  # Name of the function to call, if applicable
    args: Dict[str, Any] = {}   # Arguments for the function
    sub_tasks: List["Task"] = []  # List of subtasks
    status: str = "pending"     # Status of the task: pending, in_progress, completed, failed
    result: Any = None          # Result of the task execution

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True
        
    def add_subtask(self, task: "Task"):
        """Add a subtask to this task"""
        self.sub_tasks.append(task)
        return task
        
    def mark_in_progress(self):
        """Mark this task as in progress"""
        self.status = "in_progress"
        
    def mark_completed(self, result=None):
        """Mark this task as completed with an optional result"""
        self.status = "completed"
        if result is not None:
            self.result = result
            
    def mark_failed(self, error=None):
        """Mark this task as failed with an optional error message"""
        self.status = "failed"
        if error is not None:
            self.result = error
            
    def update_args(self, new_args: Dict[str, Any]):
        """Update the arguments for this task"""
        self.args.update(new_args)
        
    def to_dict(self):
        """Convert the task to a dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "agent": self.agent,
            "function_name": self.function_name,
            "args": self.args,
            "sub_tasks": [task.to_dict() for task in self.sub_tasks],
            "status": self.status,
            "result": str(self.result) if self.result is not None else None
        }
        
    def pretty_print(self, indent=0):
        """Pretty print the task hierarchy"""
        indent_str = "  " * indent
        print(f"{indent_str}Task: {self.name} ({self.status})")
        print(f"{indent_str}  Agent: {self.agent}")
        print(f"{indent_str}  Function: {self.function_name}")
        print(f"{indent_str}  Args: {json.dumps(self.args, indent=2)}")
        if self.result:
            print(f"{indent_str}  Result: {self.result}")
        
        for subtask in self.sub_tasks:
            subtask.pretty_print(indent + 1)

# Allow recursive models
Task.update_forward_refs()


class TaskManager:
    """
    Manages task creation, execution, and tracking.
    """
    def __init__(self, kb):
        """
        Initialize the task manager.
        
        Parameters:
            kb: The knowledge base to use for storing task results
        """
        self.kb = kb
        self.tasks = []
        
    def create_task(self, name, description, agent, function_name=None, args=None):
        """
        Create a new task.
        
        Parameters:
            name (str): Name of the task
            description (str): Description of the task
            agent (str): Name of the agent responsible
            function_name (str, optional): Name of the function to call
            args (dict, optional): Arguments for the function
            
        Returns:
            Task: The created task
        """
        task = Task(
            name=name,
            description=description,
            agent=agent,
            function_name=function_name,
            args=args or {}
        )
        self.tasks.append(task)
        return task
        
    def execute_task(self, task, agents):
        """
        Execute a task using the appropriate agent.
        
        Parameters:
            task (Task): The task to execute
            agents (dict): Dictionary mapping agent names to agent instances
            
        Returns:
            Any: The result of the task execution
        """
        # Mark the task as in progress
        task.mark_in_progress()
        
        try:
            # Get the appropriate agent
            agent = agents.get(task.agent)
            if not agent:
                error = f"No agent found for name {task.agent}."
                task.mark_failed(error)
                return error
                
            # Execute the task
            result = agent.handle_task(task)
            
            # Mark the task as completed
            task.mark_completed(result)
            
            # Return the result
            return result
        except Exception as e:
            # Mark the task as failed
            error = f"Error executing task: {str(e)}"
            task.mark_failed(error)
            return error
            
    def execute_task_hierarchy(self, task, agents):
        """
        Execute a task and all its subtasks.
        
        Parameters:
            task (Task): The root task to execute
            agents (dict): Dictionary mapping agent names to agent instances
            
        Returns:
            Any: The result of the root task execution
        """
        # Execute all subtasks first
        for subtask in task.sub_tasks:
            self.execute_task_hierarchy(subtask, agents)
            
        # Execute the main task
        return self.execute_task(task, agents)
    

    async def execute_task_hierarchy_async(self, task, agents):
        """
        Execute a task and all its subtasks asynchronously, ensuring proper sequencing.
        
        Parameters:
            task (Task): The root task to execute
            agents (dict): Dictionary mapping agent names to agent instances
            
        Returns:
            Any: The result of the root task execution
        """
        # Mark the task as in progress
        task.mark_in_progress()
        
        try:
            # Get the appropriate agent
            agent = agents.get(task.agent)
            if not agent:
                error = f"No agent found for name {task.agent}."
                task.mark_failed(error)
                return error
                
            # Execute the task
            print(f"Executing task: {task.name} (Agent: {task.agent})")
            result = await agent.handle_task_async(task)
            
            # Store the result in knowledge base for chaining
            if task.function_name == "process_emil_request":
                # This is a model creation task - make sure data is in KB
                if isinstance(result, dict):
                    self.kb.set_item("latest_model_file", result.get('file'))
                    self.kb.set_item("latest_model_details", result)
                    print(f"Stored model details in KB: {result.get('file')}")
            
            elif task.function_name == "analyze_results":
                # This is an analysis task - make sure data is in KB
                self.kb.set_item("latest_analysis_results", result)
                print(f"Stored analysis results in KB")
            
            # Mark the task as completed
            task.mark_completed(result)
            
            # Execute all subtasks in sequence
            for subtask in task.sub_tasks:
                await self.execute_task_hierarchy_async(subtask, agents)
                
            # Return the result
            return result
        except Exception as e:
            # Mark the task as failed
            error = f"Error executing task: {str(e)}"
            task.mark_failed(error)
            return error


from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import json

class Task(BaseModel):
    """
    Represents a task or subtask for the agent system.
    Enhanced to support hierarchical task structures, status tracking, and session context.
    """
    name: str
    description: str
    agent: str                  # Name of the agent responsible (e.g., "Nova", "Emil", "Ivan", "Lola")
    function_name: Optional[str] = None  # Name of the function to call, if applicable
    args: Dict[str, Any] = {}   # Arguments for the function
    sub_tasks: List["Task"] = []  # List of subtasks
    status: str = "pending"     # Status of the task: pending, in_progress, completed, failed
    result: Any = None          # Result of the task execution
    session_context: Dict[str, Any] = {}  # ADDED: Shared context dictionary for passing between agents

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True
        
    def add_subtask(self, task: "Task"):
        """Add a subtask to this task"""
        self.sub_tasks.append(task)
        return task
        
    def mark_in_progress(self):
        """Mark this task as in progress"""
        self.status = "in_progress"
        
    def mark_completed(self, result=None):
        """Mark this task as completed with an optional result"""
        self.status = "completed"
        if result is not None:
            self.result = result
            
    def mark_failed(self, error=None):
        """Mark this task as failed with an optional error message"""
        self.status = "failed"
        if error is not None:
            self.result = error
            
    def update_args(self, new_args: Dict[str, Any]):
        """Update the arguments for this task"""
        self.args.update(new_args)
        
    def update_context(self, context_updates: Dict[str, Any]):
        """Update the session context for this task"""
        self.session_context.update(context_updates)
        
    def to_dict(self):
        """Convert the task to a dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "agent": self.agent,
            "function_name": self.function_name,
            "args": self.args,
            "sub_tasks": [task.to_dict() for task in self.sub_tasks],
            "status": self.status,
            "result": str(self.result) if self.result is not None else None,
            "session_context": self.session_context  # ADDED
        }
        
    def pretty_print(self, indent=0):
        """Pretty print the task hierarchy"""
        indent_str = "  " * indent
        print(f"{indent_str}Task: {self.name} ({self.status})")
        print(f"{indent_str}  Agent: {self.agent}")
        print(f"{indent_str}  Function: {self.function_name}")
        print(f"{indent_str}  Args: {json.dumps(self.args, indent=2)}")
        if self.session_context:
            print(f"{indent_str}  Context Keys: {list(self.session_context.keys())}")
        if self.result:
            print(f"{indent_str}  Result: {self.result}")
        
        for subtask in self.sub_tasks:
            subtask.pretty_print(indent + 1)

# Allow recursive models
Task.update_forward_refs()



