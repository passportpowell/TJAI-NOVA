import asyncio
import datetime
import inspect
from .base_agent import BaseAgent
from core.functions_registery import *
from core.task_manager import Task
from utils.function_logger import log_function_call

class Lola(BaseAgent):
    def __init__(self, name, kb, function_map, verbose=False):
        super().__init__(name, kb, function_map)
        self.verbose = verbose

    @log_function_call
    async def verify_parameters_async(self, function_name: str, task_args: dict) -> dict:
        if function_name == "write_report":
            return {"success": True, "missing": [], "message": "Report tasks don't require explicit parameters"}

        if function_name not in self.function_map:
            return {"success": False, "missing": [], "message": f"Function {function_name} not found in Lola's function map"}

        sig = inspect.signature(self.function_map[function_name])
        required = [p.name for p in sig.parameters.values() if p.default == inspect.Parameter.empty and p.name not in ('self', 'kb')]
        missing = [param for param in required if param not in task_args]

        if missing:
            return {"success": False, "missing": missing, "message": f"Missing required parameters: {', '.join(missing)}"}
        return {"success": True, "missing": [], "message": "All required parameters are present"}

    @log_function_call
    async def handle_task_async(self, task: Task):
        if self.verbose:
            print(f"Lola handling task asynchronously: {task.name}")
        self.kb.log_interaction(task.name, "Starting execution", agent="Lola", function=task.function_name)

        session_context = task.session_context or {}
        session_context.setdefault("lola", {"timestamp": datetime.datetime.now().isoformat(), "tasks_processed": []})

        if task.function_name == "write_report":
            model_file = session_context.get("latest_model_file") or self.kb.get_item("latest_model_file")
            model_details = session_context.get("latest_model_details") or self.kb.get_item("latest_model_details")
            analysis_results = session_context.get("latest_analysis_results") or self.kb.get_item("latest_analysis_results")

            if "emil" in session_context:
                model_file = model_file or session_context["emil"].get("model_file")
                model_details = model_details or session_context["emil"].get("model_details")
                if not analysis_results and session_context["emil"].get("analysis_performed"):
                    analysis_results = {
                        "key_findings": session_context["emil"]["analysis_performed"].get("key_findings", []),
                        "analysis_type": session_context["emil"]["analysis_performed"].get("analysis_type", "basic")
                    }

            try:
                from core.functions_registery import write_report as global_write_report
                result = await asyncio.to_thread(
                    global_write_report,
                    self.kb,
                    style=task.args.get("style", "executive_summary"),
                    prompt=task.args.get("prompt", session_context.get("original_prompt", "")),
                    model_file=model_file,
                    model_details=model_details,
                    analysis_results=analysis_results
                )

                task.result = result
                session_context["lola"]["report_generated"] = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "style": task.args.get("style", "executive_summary"),
                    "model_file": model_file,
                    "has_analysis": bool(analysis_results)
                }

                await self.kb.set_item_async("latest_report", result, category="reports")
                await self.kb.set_item_async("final_report", result)

                current_session = self.kb.get_item("current_session")
                if current_session:
                    session_data = self.kb.get_item(f"session_{current_session}") or {}
                    session_data.setdefault("reports_generated", []).append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "style": task.args.get("style", "executive_summary"),
                        "prompt": task.args.get("prompt", "")
                    })
                    self.kb.set_item(f"session_{current_session}", session_data, category="sessions")

                self.kb.log_interaction(task.name, "Report generated successfully", agent="Lola", function="write_report")
                return result

            except Exception as e:
                err = f"Error writing report: {str(e)}"
                self.kb.log_interaction(task.name, err, agent="Lola", function="write_report")
                task.result = err
                return err

        if task.function_name in self.function_map:
            func = self.function_map[task.function_name]
            check = await self.verify_parameters_async(task.function_name, task.args)
            if not check["success"]:
                err = check["message"]
                await self.kb.set_item_async("lola_error", err, category="errors")
                task.result = err
                self.kb.log_interaction(task.name, err, agent="Lola", function=task.function_name)
                return err

            try:
                result = await asyncio.to_thread(func, self.kb, **task.args)
                task.result = result

                await self.kb.set_item_async(f"lola_{task.function_name}_result", result, category=task.function_name)
                await self.kb.set_item_async("final_report", result)
                self.kb.log_interaction(task.name, "Task completed successfully", agent="Lola", function=task.function_name)
                return result
            except Exception as e:
                err = f"Error executing {task.function_name}: {str(e)}"
                self.kb.log_interaction(task.name, err, agent="Lola", function=task.function_name)
                task.result = err
                return err

        err = f"Lola has no function for task: {task.name}"
        self.kb.log_interaction(task.name, err, agent="Lola", function=task.function_name)
        task.result = err
        return err
