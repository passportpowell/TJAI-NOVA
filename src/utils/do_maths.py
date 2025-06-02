import re
from utils.function_logger import *
from core.knowledge_base import KnowledgeBase

import re
import math
from utils.function_logger import log_function_call
from utils.open_ai_utils import *
from core.knowledge_base import KnowledgeBase


@log_function_call
#def do_maths(kb: KnowledgeBase, prompt: str, full_prompt=None, input2: str = "-") -> str:
def do_maths(kb: KnowledgeBase, prompt: str, full_prompt=None, original_prompt=None, input2: str = "-") -> str:
    """
    Performs mathematical calculations based on the input prompt.
    Enhanced to handle a wide range of math expressions including percentages.
    Falls back to LLM for complex calculations.
    
    Parameters:
        kb (KnowledgeBase): The knowledge base instance
        prompt (str): The math expression or question to solve
        full_prompt (str, optional): The full original prompt
        input2 (str): Optional additional input (required by function mapping)
        
    Returns:
        str: The result of the calculation with explanation
    """
    print(f"Calculating mathematical expression: {prompt}")
    
    # Try to solve with local calculation logic
    result, answer = attempt_local_calculation(prompt)
    # Handle both parameter names for backward compatibility
    actual_full_prompt = full_prompt or original_prompt

    # If local calculation failed, use LLM as fallback
    if result is None:
        print(f"Local calculation failed, using LLM for: {prompt}")
        context = (
            "You are a math problem solver. Calculate the answer to the given problem. "
            "For simple arithmetic, just provide the answer. "
            "For complex calculations, show your work step by step."
        )
        llm_result = run_open_ai_ns(prompt, context)
        
        # Store the LLM result
        kb.set_item("math_result", llm_result)
        kb.set_item("math_answer", llm_result)
        kb.set_item("final_report", llm_result)
        
        print(f"LLM math calculation result: {llm_result}")
        return llm_result
    
    # Store the local calculation result in the knowledge base
    kb.set_item("math_result", result)
    kb.set_item("math_answer", answer)
    kb.set_item("final_report", answer)
    
    print(f"Math calculation result: {answer}")
    return answer


def attempt_local_calculation(prompt):
    """
    Attempts to perform a calculation locally using regex pattern matching.
    Supports various formats including percentages, fractions, and basic arithmetic.
    
    Parameters:
        prompt (str): The mathematical expression to evaluate
        
    Returns:
        tuple: (result, answer_text) if successful, (None, None) if failed
    """
    try:
        # Normalize the prompt
        #normalized_prompt = prompt.lower().replace('×', '*').replace('÷', '/')
        normalized_prompt = prompt.lower().replace('×', '*').replace('÷', '/').replace('x', '*')
        
        # Pattern for percentage calculations
        percentage_pattern = r'(?:what\s+is\s+)?(\d+\.?\d*)%\s+(?:of)\s+(\d+\.?\d*)'
        percentage_of_pattern = r'(?:what\s+is\s+)?(\d+\.?\d*)\s+(?:percent\s+of)\s+(\d+\.?\d*)'
        
        # Pattern for basic arithmetic operations with "what is" optional
        # FIXED: Enhanced basic arithmetic pattern with better spacing handling
        basic_math_pattern = r'(?:what\s+is\s+)?(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)'
        # Alternative pattern for "2 * 2" format
        simple_math_pattern = r'(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)'

        # Pattern for square root
        sqrt_pattern = r'(?:what\s+is\s+)?(?:the\s+)?(?:square\s+root\s+of)\s+(\d+\.?\d*)'
        
        # Pattern for powers 
        power_pattern = r'(?:what\s+is\s+)?(\d+\.?\d*)\s+(?:to\s+the\s+power\s+of|raised\s+to\s+the\s+power\s+of|\^)\s+(\d+\.?\d*)'
        
        # Check for percentage calculation
        percentage_match = re.search(percentage_pattern, normalized_prompt)
        percentage_of_match = re.search(percentage_of_pattern, normalized_prompt)
        if percentage_match:
            percentage = float(percentage_match.group(1))
            base_value = float(percentage_match.group(2))
            result = (percentage / 100) * base_value
            answer = f"{percentage}% of {base_value} is {result}"
            return result, answer
        elif percentage_of_match:
            percentage = float(percentage_of_match.group(1))
            base_value = float(percentage_of_match.group(2))
            result = (percentage / 100) * base_value
            answer = f"{percentage}% of {base_value} is {result}"
            return result, answer
        
        # Check for square root
        sqrt_match = re.search(sqrt_pattern, normalized_prompt)
        if sqrt_match:
            num = float(sqrt_match.group(1))
            if num < 0:
                return None, "Error: Cannot take square root of a negative number"
            result = math.sqrt(num)
            answer = f"The square root of {num} is {result}"
            return result, answer
        
        # Check for powers
        power_match = re.search(power_pattern, normalized_prompt)
        if power_match:
            base = float(power_match.group(1))
            exponent = float(power_match.group(2))
            result = math.pow(base, exponent)
            answer = f"{base} raised to the power of {exponent} is {result}"
            return result, answer
        
        # Check for basic arithmetic
        math_match = re.search(basic_math_pattern, normalized_prompt)
        # FIXED: Check for basic arithmetic - try both patterns
        math_match = re.search(basic_math_pattern, normalized_prompt) or re.search(simple_math_pattern, normalized_prompt)
        
        if math_match:
            num1 = float(math_match.group(1))
            op = math_match.group(2)
            num2 = float(math_match.group(3))
            
            if op == '+':
                result = num1 + num2
                answer = f"The result of {num1} + {num2} is {result}"
            elif op == '-':
                result = num1 - num2
                answer = f"The result of {num1} - {num2} is {result}"
            elif op == '*':
                result = num1 * num2
                answer = f"The result of {num1} * {num2} is {result}"
            elif op == '/':
                if num2 == 0:
                    return None, "Error: Division by zero"
                result = num1 / num2
                answer = f"The result of {num1} / {num2} is {result}"
            else:
                return None, None
            
            return result, answer
        
        # Handle simple numeric inputs
        if normalized_prompt.strip().replace(' ', '').isdigit():
            result = float(normalized_prompt.strip())
            answer = f"The number is {result}"
            return result, answer
        
        # No pattern matched, will fall back to LLM
        return None, None
        
    except Exception as e:
        print(f"Error in local calculation: {str(e)}")
        return None, None