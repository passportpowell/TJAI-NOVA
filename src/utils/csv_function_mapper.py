import os
import pandas as pd
from typing import Dict, Any, Callable
from utils.function_logger import log_function_call

class FunctionMapLoader:
    """
    Loads function maps from CSV files and maps them to actual Python functions.
    This class bridges the CSV definitions with callable Python functions.
    """
    
    def __init__(self, base_path=None, verbose=False):
        """
        Initialize the function map loader.
        
        Parameters:
            base_path (str, optional): Base directory where function map CSVs are stored.
            verbose (bool): Whether to print loading logs. Default is True.
        """
        self.verbose = verbose

        if base_path is None:
            self.base_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "function maps"
            )
        else:
            self.base_path = base_path

        os.makedirs(self.base_path, exist_ok=True)
        self.function_registry = {}

    @log_function_call
    def register_function(self, key: str, function: Callable):
        """Register a single function."""
        self.function_registry[key] = function

    @log_function_call
    def register_functions(self, functions_dict: Dict[str, Callable]):
        """Register multiple functions at once."""
        self.function_registry.update(functions_dict)

    @log_function_call
    def load_function_map(self, agent_name: str, enhanced=True) -> Dict[str, Callable]:
        """
        Load a function map CSV for a specific agent and map it to actual Python functions.
        """
        filename = f"{agent_name}_function_map{'_enhanced' if enhanced else ''}.csv"
        csv_path = os.path.join(self.base_path, filename)

        if not os.path.exists(csv_path):
            if self.verbose:
                print(f"Warning: Function map CSV not found: {csv_path}")
            # Try fallback location
            agent_dir_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "agents", 
                filename
            )
            if os.path.exists(agent_dir_path):
                csv_path = agent_dir_path
                if self.verbose:
                    print(f"Found function map in agents directory: {csv_path}")
            else:
                if self.verbose:
                    print(f"No function map found for agent {agent_name}")
                return {}

        try:
            df = pd.read_csv(csv_path)
            function_map = {}

            for _, row in df.iterrows():
                key = row['Key']
                function_name = row['Function'] if 'Function' in row else key

                if function_name in self.function_registry:
                    function_map[key] = self.function_registry[function_name]
                else:
                    if self.verbose:
                        print(f"Warning: Function '{function_name}' is not registered for key '{key}'")

            if self.verbose:
                print(f"Successfully loaded {len(function_map)} functions for agent {agent_name}")
            return function_map

        except Exception as e:
            if self.verbose:
                print(f"Error loading function map for {agent_name}: {str(e)}")
            return {}
