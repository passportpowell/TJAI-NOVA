import asyncio
import datetime
from .base_agent import BaseAgent
from core.functions_registery import *
from core.task_manager import Task
from utils.function_logger import log_function_call

class Ivan(BaseAgent):
    def __init__(self, name, kb, function_map, verbose=False):
        super().__init__(name, kb, function_map)
        self.verbose = verbose

    @log_function_call
    async def handle_task_async(self, task: Task):
        if self.verbose:
            print(f"Ivan handling task asynchronously: {task.name}")

        self.kb.log_interaction(f"Task: {task.name}", "Starting execution", agent="Ivan", function=task.function_name)

        if task.function_name == "generate_image":
            return await self._handle_image_task(task)

        if task.function_name in self.function_map:
            func = self.function_map[task.function_name]

            # Collect missing arguments
            sig = func.__code__.co_varnames[1:]  # skip 'kb'
            missing = [p for p in sig if p not in task.args]
            if missing:
                self.kb.log_interaction(f"Task: {task.name}", f"Missing parameters: {missing}", agent="Ivan", function=task.function_name)
                new_args = await self.ask_user_for_missing_args_async(missing)
                task.args.update(new_args)

            try:
                result = await asyncio.to_thread(func, self.kb, **task.args)
                task.result = result

                key = f"ivan_{task.function_name}_result"
                cat = task.function_name.replace("_", "")
                await self.kb.set_item_async(key, result, category=cat)
                await self.kb.set_item_async("final_report", result)

                self.kb.log_interaction(f"Task: {task.name}", "Task completed successfully", agent="Ivan", function=task.function_name)
                return result

            except Exception as e:
                err = f"Error executing {task.function_name}: {str(e)}"
                self.kb.log_interaction(f"Task: {task.name}", err, agent="Ivan", function=task.function_name)
                task.result = err
                return err

        msg = f"Ivan doesn't recognize function {task.function_name}"
        self.kb.log_interaction(f"Task: {task.name}", msg, agent="Ivan", function=task.function_name)
        task.result = msg
        return msg

    async def _handle_image_task(self, task: Task):
        try:
            result = await asyncio.to_thread(self.generate_image, self.kb, **task.args)
            task.result = result

            await self.kb.set_item_async("image_result", result, category="image_generation")
            await self.kb.set_item_async("final_report", result)

            current_session = self.kb.get_item("current_session")
            if current_session:
                session_data = self.kb.get_item(f"session_{current_session}") or {}
                session_data.setdefault("images_generated", []).append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "prompt": task.args.get("prompt", "unknown"),
                    "enhanced_prompt": self.kb.get_item("last_dalle_prompt")
                })
                self.kb.set_item(f"session_{current_session}", session_data, category="sessions")

            self.kb.log_interaction(f"Task: {task.name}", "Image generated successfully", agent="Ivan", function="generate_image")
            return result

        except Exception as e:
            err = f"Error generating image: {str(e)}"
            self.kb.log_interaction(f"Task: {task.name}", err, agent="Ivan", function="generate_image")
            task.result = err
            return err
