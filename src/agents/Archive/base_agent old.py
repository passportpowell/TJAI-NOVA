import asyncio
from typing import Any, Dict, List
from core.task_manager import Task
from core.functions_registery import KnowledgeBase
from utils.function_logger import log_function_call

class BaseAgent:
    def __init__(self, name: str, knowledge_base: KnowledgeBase, function_map: Dict[str, Any]):
        self.name = name
        self.kb = knowledge_base
        self.function_map = function_map
        self.async_function_map = {}

    @log_function_call
    async def handle_task_async(self, task: Task):
        """
        Asynchronous entry point for handling a task. Meant to be overridden.
        """
        raise NotImplementedError("handle_task_async must be implemented by subclasses")

    @log_function_call
    async def ask_user_for_missing_args_async(self, missing_args: List[str]) -> Dict[str, Any]:
        """
        Async method to collect missing arguments by prompting the user.
        """
        print(f"{self.name} is asking user for arguments: {missing_args}")
        collected = {}
        for arg in missing_args:
            val = await asyncio.to_thread(input, f"Please provide a value for {arg}: ")
            collected[arg] = val.strip()
        return collected
