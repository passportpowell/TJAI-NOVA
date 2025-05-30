"""
Enhanced evaluation system for Nova AI Coordinator with better debug output.
This module provides functions to evaluate answers and implement fallback strategies.
"""
from utils.function_logger import log_function_call
from utils.open_ai_utils import run_open_ai_ns, run_open_ai_ns_async
from core.knowledge_base import KnowledgeBase
import asyncio
import json

@log_function_call
def initialize_evaluation_system(kb, config=None):
    """
    Initialize the evaluation system with configuration.
    
    Parameters:
        kb (KnowledgeBase): The knowledge base
        config (dict, optional): Configuration parameters
        
    Returns:
        dict: The active configuration
    """
   
    default_config = {
        "evaluation_enabled": True,
        "quality_threshold": 0.7,
        "use_internet_search": True,
        "use_detailed_llm": True,
        "auto_improve_answers": True,
        "store_evaluations": True,
        "evaluation_model": "gpt-4.1-nano",
        "fallback_evaluation_model": "gpt-3.5-turbo",  # Add fallback model
        "max_retries": 2,  # Add retry count
        "debug_output": True
    }


    # Merge with provided config
    active_config = default_config.copy()
    if config:
        active_config.update(config)
    
    # Store in knowledge base
    kb.set_item("evaluation_config", active_config)
    kb.set_item("evaluation_enabled", active_config["evaluation_enabled"])
    kb.set_item("quality_threshold", active_config["quality_threshold"])
    
    print(f"🔍 Evaluation system initialized with quality threshold: {active_config['quality_threshold']}")
    print(f"🔍 Debug output: {'Enabled' if active_config['debug_output'] else 'Disabled'}")
    
    return active_config


@log_function_call
async def evaluate_answer_quality(kb, question, answer):
    """
    Evaluate the quality of an answer using LLM with improved reliability.
    """
    # Get evaluation config
    config = kb.get_item("evaluation_config") or {}
    model = config.get("evaluation_model", "gpt-4.1-nano")
    fallback_model = config.get("fallback_evaluation_model", "gpt-3.5-turbo")  # Add fallback model
    threshold = config.get("quality_threshold", 0.7)
    debug_output = config.get("debug_output", True)
    max_retries = 2  # Add retry mechanism
    
    if debug_output:
        print(f"\n🔍 EVALUATION: Evaluating answer quality with model: {model}")
        print(f"🔍 Question: {question}")
        print(f"🔍 Answer length: {len(answer)} chars")
        print(f"🔍 Threshold: {threshold}")
    
    # Simplify the evaluation prompt for better reliability
    evaluation_prompt = f"""
    Evaluate the quality of this answer to the given question.
    
    Question: "{question}"
    Answer: "{answer}"
    
    Rate from 0.0 to 1.0 where 1.0 is perfect.
    List 1-3 strengths and 1-3 weaknesses.
    Make 1-2 improvement suggestions.
    
    RETURN ONLY VALID JSON with this structure:
    {{
        "score": 0.X,
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "improvement_suggestions": ["suggestion1"],
        "passed": true/false
    }}
    Passed should be true if score >= {threshold}, otherwise false.
    """
    
    # Simpler context
    eval_context = """
    You are an evaluation assistant. Return ONLY valid JSON, no explanations.
    Always include all required fields: score, strengths, weaknesses, 
    improvement_suggestions, and passed.
    """
    
    # Try with retries
    retry_count = 0
    last_error = None
    eval_result = None
    
    while retry_count < max_retries:
        try:
            if debug_output:
                print(f"🔍 Evaluation attempt {retry_count + 1}/{max_retries}...")
            
            # Call LLM for evaluation
            eval_result_json = await run_open_ai_ns_async(
                evaluation_prompt, 
                eval_context,
                model=model,
                temperature=0.2
            )
            
            # Try to parse JSON response
            try:
                eval_result = json.loads(eval_result_json)
                
                # Validate required fields
                if "score" not in eval_result:
                    raise ValueError("Missing 'score' field")
                
                # Set passed field correctly
                eval_result["passed"] = eval_result.get("score", 0.0) >= threshold
                
                # Success - exit retry loop
                break
                
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract data using regex
                if debug_output:
                    print(f"🔍 Warning: JSON parsing failed. Trying regex extraction.")
                
                # Use the extract_evaluation_data function to try parsing non-JSON
                extracted = extract_evaluation_data(eval_result_json, threshold)
                
                if extracted.get("score") is not None and extracted.get("score") != 0.5:
                    # If extraction found a valid non-default score, use it
                    eval_result = extracted
                    break
            
            # If we got here, current attempt failed - retry
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Wait before retry
            
        except Exception as e:
            last_error = str(e)
            if debug_output:
                print(f"❌ Evaluation error: {last_error}")
            
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(1)
    
    # If primary model failed, try fallback model
    if eval_result is None and fallback_model and fallback_model != model:
        try:
            if debug_output:
                print(f"🔍 Trying fallback model: {fallback_model}")
            
            eval_result_json = await run_open_ai_ns_async(
                evaluation_prompt, 
                eval_context,
                model=fallback_model,
                temperature=0.3  # Slightly higher temperature for variety
            )
            
            # Try to parse JSON from fallback model
            try:
                eval_result = json.loads(eval_result_json)
                eval_result["passed"] = eval_result.get("score", 0.0) >= threshold
            except:
                # Try extraction here too
                extracted = extract_evaluation_data(eval_result_json, threshold)
                if extracted.get("score") is not None and extracted.get("score") != 0.5:
                    eval_result = extracted
                
        except Exception as e:
            last_error = f"Fallback model failed: {str(e)}"
    
    # If all attempts failed, use a more informative fallback
    if eval_result is None:
        eval_result = {
            "score": 0.5,
            "strengths": ["Answer contained some information"],
            "weaknesses": [f"Evaluation failed: {last_error[:50]}..."],
            "improvement_suggestions": ["Provide more specific information"],
            "passed": False,
            "error": last_error
        }
    
    # Store in KB and log results as before
    await kb.set_item_async("last_evaluation_result", eval_result)
    
    # Update conversation as before
    conversation = kb.get_item("current_conversation") or {}
    if "evaluations" not in conversation:
        conversation["evaluations"] = []
    
    conversation["evaluations"].append({
        "question": question,
        "score": eval_result.get("score", 0.0),
        "passed": eval_result.get("passed", False),
        "strengths": eval_result.get("strengths", []),
        "weaknesses": eval_result.get("weaknesses", [])
    })
    kb.set_item("current_conversation", conversation)
    
    if debug_output:
        print(f"🔍 Evaluation complete: Score {eval_result.get('score', 'N/A')}")
    
    return eval_result



@log_function_call
async def evaluate_answer_quality(kb, question, answer):
    """
    Evaluate the quality of an answer using LLM with improved reliability.
    """
    # Check if this is a direct session data response that should skip evaluation
    if answer and isinstance(answer, str) and answer.startswith("DIRECT_SESSION_DATA:"):
        print("🔍 Skipping evaluation for direct session data")
        # Remove the marker prefix before returning to user
        clean_answer = answer.replace("DIRECT_SESSION_DATA: ", "")
        
        # Store the clean answer back in KB
        await kb.set_item_async("general_answer", clean_answer)
        await kb.set_item_async("final_report", clean_answer)
        
        # Return perfect evaluation
        return {
            "score": 1.0,
            "strengths": ["Accurate session data reporting", "Direct information retrieval", "Factual content"],
            "weaknesses": [],
            "improvement_suggestions": [],
            "passed": True
        }
    
    # Get evaluation config
    config = kb.get_item("evaluation_config") or {}
    model = config.get("evaluation_model", "gpt-4.1-nano")
    fallback_model = config.get("fallback_evaluation_model", "gpt-3.5-turbo")  # Add fallback model
    threshold = config.get("quality_threshold", 0.7)
    debug_output = config.get("debug_output", True)
    max_retries = 2  # Add retry mechanism
    
    if debug_output:
        print(f"\n🔍 EVALUATION: Evaluating answer quality with model: {model}")
        print(f"🔍 Question: {question}")
        print(f"🔍 Answer length: {len(answer)} chars")
        print(f"🔍 Threshold: {threshold}")
    
    # Simplify the evaluation prompt for better reliability
    evaluation_prompt = f"""
    Evaluate the quality of this answer to the given question.
    
    Question: "{question}"
    Answer: "{answer}"
    
    Rate from 0.0 to 1.0 where 1.0 is perfect.
    List 1-3 strengths and 1-3 weaknesses.
    Make 1-2 improvement suggestions.
    
    RETURN ONLY VALID JSON with this structure:
    {{
        "score": 0.X,
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "improvement_suggestions": ["suggestion1"],
        "passed": true/false
    }}
    Passed should be true if score >= {threshold}, otherwise false.
    """
    
    # Simpler context
    eval_context = """
    You are an evaluation assistant. Return ONLY valid JSON, no explanations.
    Always include all required fields: score, strengths, weaknesses, 
    improvement_suggestions, and passed.
    """
    
    # Try with retries
    retry_count = 0
    last_error = None
    eval_result = None
    
    while retry_count < max_retries:
        try:
            if debug_output:
                print(f"🔍 Evaluation attempt {retry_count + 1}/{max_retries}...")
            
            # Call LLM for evaluation
            eval_result_json = await run_open_ai_ns_async(
                evaluation_prompt, 
                eval_context,
                model=model,
                temperature=0.2
            )
            
            # Try to parse JSON response
            try:
                eval_result = json.loads(eval_result_json)
                
                # Validate required fields
                if "score" not in eval_result:
                    raise ValueError("Missing 'score' field")
                
                # Set passed field correctly
                eval_result["passed"] = eval_result.get("score", 0.0) >= threshold
                
                # Success - exit retry loop
                break
                
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract data using regex
                if debug_output:
                    print(f"🔍 Warning: JSON parsing failed. Trying regex extraction.")
                
                # Use the extract_evaluation_data function to try parsing non-JSON
                extracted = extract_evaluation_data(eval_result_json, threshold)
                
                if extracted.get("score") is not None and extracted.get("score") != 0.5:
                    # If extraction found a valid non-default score, use it
                    eval_result = extracted
                    break
            
            # If we got here, current attempt failed - retry
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Wait before retry
            
        except Exception as e:
            last_error = str(e)
            if debug_output:
                print(f"❌ Evaluation error: {last_error}")
            
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(1)
    
    # If primary model failed, try fallback model
    if eval_result is None and fallback_model and fallback_model != model:
        try:
            if debug_output:
                print(f"🔍 Trying fallback model: {fallback_model}")
            
            eval_result_json = await run_open_ai_ns_async(
                evaluation_prompt, 
                eval_context,
                model=fallback_model,
                temperature=0.3  # Slightly higher temperature for variety
            )
            
            # Try to parse JSON from fallback model
            try:
                eval_result = json.loads(eval_result_json)
                eval_result["passed"] = eval_result.get("score", 0.0) >= threshold
            except:
                # Try extraction here too
                extracted = extract_evaluation_data(eval_result_json, threshold)
                if extracted.get("score") is not None and extracted.get("score") != 0.5:
                    eval_result = extracted
                
        except Exception as e:
            last_error = f"Fallback model failed: {str(e)}"
    
    # If all attempts failed, use a more informative fallback
    if eval_result is None:
        eval_result = {
            "score": 0.5,
            "strengths": ["Answer contained some information"],
            "weaknesses": [f"Evaluation failed: {last_error[:50]}..."],
            "improvement_suggestions": ["Provide more specific information"],
            "passed": False,
            "error": last_error
        }
    
    # Store in KB and log results as before
    await kb.set_item_async("last_evaluation_result", eval_result)
    
    # Update conversation as before
    conversation = kb.get_item("current_conversation") or {}
    if "evaluations" not in conversation:
        conversation["evaluations"] = []
    
    conversation["evaluations"].append({
        "question": question,
        "score": eval_result.get("score", 0.0),
        "passed": eval_result.get("passed", False),
        "strengths": eval_result.get("strengths", []),
        "weaknesses": eval_result.get("weaknesses", [])
    })
    kb.set_item("current_conversation", conversation)
    
    if debug_output:
        print(f"🔍 Evaluation complete: Score {eval_result.get('score', 'N/A')}")
    
    return eval_result




def extract_evaluation_data(text, threshold=0.7):
    """
    Extract evaluation data from text when JSON parsing fails.
    Enhanced to handle more formats and patterns.
    """
    import re
    
    # Default evaluation data
    eval_data = {
        "score": 0.5,
        "strengths": ["Partial information provided"],
        "weaknesses": ["Could not fully evaluate response"],
        "improvement_suggestions": ["Provide more specific information"],
        "passed": False
    }
    
    # First try to find any JSON-like structure
    json_pattern = r'\{.*\}'
    json_match = re.search(json_pattern, text, re.DOTALL)
    if json_match:
        try:
            import json
            json_str = json_match.group(0)
            data = json.loads(json_str)
            if "score" in data:
                return data
        except:
            pass
    
    # Try to extract score with multiple patterns
    score_patterns = [
        r'"score":\s*(0\.\d+|1\.0|1|0)',
        r'score.*?(\d+\.?\d*)\s*\/\s*1',
        r'(\d+\.?\d*)\s*\/\s*1',
        r'rated\s+(\d+\.?\d*)',
        r'score\s+of\s+(\d+\.?\d*)'
    ]
    
    for pattern in score_patterns:
        score_match = re.search(pattern, text, re.IGNORECASE)
        if score_match:
            try:
                eval_data["score"] = float(score_match.group(1))
                eval_data["passed"] = eval_data["score"] >= threshold
                break
            except:
                continue
    
    # Try to extract strengths
    strengths_match = re.search(r'"strengths":\s*\[(.*?)\]', text, re.DOTALL)
    if strengths_match:
        strengths_text = strengths_match.group(1)
        strengths = re.findall(r'"([^"]+)"', strengths_text)
        if strengths:
            eval_data["strengths"] = strengths
    
    # Try to extract weaknesses
    weaknesses_match = re.search(r'"weaknesses":\s*\[(.*?)\]', text, re.DOTALL)
    if weaknesses_match:
        weaknesses_text = weaknesses_match.group(1)
        weaknesses = re.findall(r'"([^"]+)"', weaknesses_text)
        if weaknesses:
            eval_data["weaknesses"] = weaknesses
    
    # Try to extract improvement suggestions
    suggestions_match = re.search(r'"improvement_suggestions":\s*\[(.*?)\]', text, re.DOTALL)
    if suggestions_match:
        suggestions_text = suggestions_match.group(1)
        suggestions = re.findall(r'"([^"]+)"', suggestions_text)
        if suggestions:
            eval_data["improvement_suggestions"] = suggestions
    
    return eval_data


@log_function_call
async def determine_fallback_strategy(kb, question, evaluation, available_alternatives):
    """
    Determine the best fallback strategy based on evaluation.
    
    Parameters:
        kb (KnowledgeBase): The knowledge base
        question (str): The original question
        evaluation (dict): Evaluation results
        available_alternatives (list): List of available fallback strategies
        
    Returns:
        dict: Recommended fallback strategy
    """
    # Get evaluation config
    config = kb.get_item("evaluation_config") or {}
    model = config.get("evaluation_model", "gpt-4.1-nano")
    debug_output = config.get("debug_output", True)
    
    if debug_output:
        print(f"\n🔍 FALLBACK: Determining fallback strategy for question: {question}")
        print(f"🔍 Evaluation score: {evaluation.get('score', 'N/A')}")
        print(f"🔍 Available strategies: {', '.join(available_alternatives)}")
    
    # Create strategy selection prompt
    strategy_prompt = f"""
    Determine the best fallback strategy for improving the following answer to a question.
    The answer has been evaluated and needs improvement.
    
    Question: "{question}"
    
    Evaluation:
    - Score: {evaluation.get('score', 'N/A')}
    - Strengths: {', '.join(evaluation.get('strengths', ['None']))}
    - Weaknesses: {', '.join(evaluation.get('weaknesses', ['None']))}
    
    Available strategies:
    {', '.join(available_alternatives)}
    
    Return your recommendation in JSON format:
    {{
        "recommended_strategy": "strategy_name",
        "reason": "brief explanation for this choice"
    }}
    
    Consider the nature of the question and the specific weaknesses identified.
    """
    
    # Set context for strategy selection
    strategy_context = """
    You are a strategy recommendation assistant. Your task is to analyze an 
    evaluation of an answer and determine the most effective fallback strategy 
    to improve it. Consider the nature of the question, the identified weaknesses, 
    and the available strategies. Provide your recommendation in JSON format.
    """
    
    try:
        # Call LLM for strategy selection
        if debug_output:
            print(f"🔍 Calling LLM to select fallback strategy...")
            
        strategy_json = await run_open_ai_ns_async(
            strategy_prompt, 
            strategy_context,
            model=model,
            temperature=0.3  # Low temperature for consistent recommendations
        )
        
        # Parse JSON response
        try:
            strategy = json.loads(strategy_json)
        except json.JSONDecodeError:
            # If JSON parsing fails, extract the strategy name
            import re
            recommended_match = re.search(r'"recommended_strategy":\s*"([^"]+)"', strategy_json)
            reason_match = re.search(r'"reason":\s*"([^"]+)"', strategy_json)
            
            strategy = {
                "recommended_strategy": recommended_match.group(1) if recommended_match else available_alternatives[0],
                "reason": reason_match.group(1) if reason_match else "Default selection"
            }
        
        # Ensure the recommended strategy is in the available alternatives
        if strategy.get("recommended_strategy") not in available_alternatives:
            strategy["recommended_strategy"] = available_alternatives[0]
            strategy["reason"] = "Default selection after validation failure"
        
        # Store strategy in KB
        await kb.set_item_async("fallback_strategy", strategy)
        
        # Add to conversation summary
        conversation = kb.get_item("current_conversation") or {}
        if "fallback_strategies" not in conversation:
            conversation["fallback_strategies"] = []
        
        conversation["fallback_strategies"].append({
            "question": question,
            "recommended_strategy": strategy.get("recommended_strategy"),
            "reason": strategy.get("reason")
        })
        kb.set_item("current_conversation", conversation)
        
        if debug_output:
            print(f"🔍 Fallback strategy selected: {strategy.get('recommended_strategy')}")
            print(f"🔍 Reason: {strategy.get('reason')}")
        
        return strategy
        
    except Exception as e:
        print(f"❌ Error selecting fallback strategy: {str(e)}")
        # Return a default strategy on error
        default_strategy = {
            "recommended_strategy": available_alternatives[0],
            "reason": "Default selection due to error"
        }
        
        await kb.set_item_async("fallback_strategy", default_strategy)
        
        # Add to conversation summary even on error
        conversation = kb.get_item("current_conversation") or {}
        if "fallback_strategies" not in conversation:
            conversation["fallback_strategies"] = []
        
        conversation["fallback_strategies"].append({
            "question": question,
            "recommended_strategy": default_strategy.get("recommended_strategy"),
            "reason": default_strategy.get("reason"),
            "error": str(e)
        })
        kb.set_item("current_conversation", conversation)
        
        return default_strategy


async def get_internet_search_results(kb, question):
    """
    Get internet search results for a question.
    
    Parameters:
        kb (KnowledgeBase): The knowledge base
        question (str): The question to search for
        
    Returns:
        dict: Search results
    """
    try:
        # Check if internet search function is available
        from utils.internet_search import internet_search
        
        # Perform internet search
        search_results = await internet_search(kb, question)
        return search_results
    except ImportError:
        print("Internet search function not available")
        return {"search_results": []}
    except Exception as e:
        print(f"Error in internet search: {str(e)}")
        return {"search_results": [], "error": str(e)}


async def generate_improved_answer(kb, question, original_answer, evaluation, search_results=None):
    """
    Generate an improved answer based on evaluation and optionally search results.
    
    Parameters:
        kb (KnowledgeBase): The knowledge base
        question (str): The original question
        original_answer (str): The original answer
        evaluation (dict): Evaluation results
        search_results (dict, optional): Internet search results
        
    Returns:
        str: Improved answer
    """
    # Get evaluation config
    config = kb.get_item("evaluation_config") or {}
    model = config.get("evaluation_model", "gpt-4.1-nano")
    debug_output = config.get("debug_output", True)
    
    if debug_output:
        print(f"\n🔍 IMPROVEMENT: Generating improved answer for question: {question}")
        print(f"🔍 Original score: {evaluation.get('score', 'N/A')}")
        print(f"🔍 Using search results: {bool(search_results)}")
    
    # Create improvement prompt
    improvement_prompt = f"""
    Improve the following answer to a question based on evaluation feedback
    and additional information (if provided).
    
    Question: "{question}"
    
    Original Answer: "{original_answer}"
    
    Evaluation:
    - Score: {evaluation.get('score', 'N/A')}
    - Strengths: {', '.join(evaluation.get('strengths', ['None']))}
    - Weaknesses: {', '.join(evaluation.get('weaknesses', ['None']))}
    - Improvement Suggestions: {', '.join(evaluation.get('improvement_suggestions', ['None']))}
    """
    
    # Add search results if available
    if search_results and search_results.get("search_results"):
        improvement_prompt += "\n\nAdditional Information from Search:\n"
        
        # Add featured snippet if available
        if search_results.get("featured_snippet"):
            improvement_prompt += f"Featured Answer: {search_results['featured_snippet']}\n\n"
        
        # Add search results (up to 3 for brevity)
        improvement_prompt += "Search Results:\n"
        for i, result in enumerate(search_results.get("search_results", [])[:3], 1):
            improvement_prompt += f"{i}. {result.get('title', 'No title')}\n"
            improvement_prompt += f"   {result.get('snippet', 'No snippet')}\n"
            improvement_prompt += f"   Source: {result.get('source', 'Unknown')}\n\n"
    
    improvement_prompt += """
    Please provide an improved answer that addresses the weaknesses identified 
    in the evaluation and incorporates any additional information provided.
    Focus on accuracy, completeness, clarity, and relevance.
    """
    
    # Set context for improvement
    improvement_context = """
    You are an answer improvement assistant. Your task is to enhance an 
    answer based on evaluation feedback and additional information. 
    Maintain the strengths of the original answer while addressing its 
    weaknesses. Incorporate new information appropriately, and ensure 
    the improved answer is accurate, complete, clear, and relevant.
    """
    
    try:
        # Call LLM for improvement
        if debug_output:
            print(f"🔍 Calling LLM to generate improved answer...")
            
        improved_answer = await run_open_ai_ns_async(
            improvement_prompt, 
            improvement_context,
            model=model,
            temperature=0.5  # Moderate temperature for creativity with control
        )
        
        # Store improved answer in KB
        await kb.set_item_async("improved_answer", improved_answer)
        
        # Add to conversation summary
        conversation = kb.get_item("current_conversation") or {}
        if "improved_answers" not in conversation:
            conversation["improved_answers"] = []
        
        conversation["improved_answers"].append({
            "question": question,
            "original_answer": original_answer,
            "improved_answer": improved_answer,
            "original_score": evaluation.get("score", 0.0)
        })
        kb.set_item("current_conversation", conversation)
        
        if debug_output:
            print(f"🔍 Generated improved answer")
            print(f"🔍 Original answer: {original_answer[:50]}...")
            print(f"🔍 Improved answer: {improved_answer[:50]}...")
        
        return improved_answer
        
    except Exception as e:
        print(f"❌ Error generating improved answer: {str(e)}")
        # Return slightly modified original answer on error
        error_note = f"\n\nNote: Attempted to improve this answer but encountered an error: {str(e)}"
        
        # Add to conversation summary even on error
        conversation = kb.get_item("current_conversation") or {}
        if "improved_answers" not in conversation:
            conversation["improved_answers"] = []
        
        conversation["improved_answers"].append({
            "question": question,
            "original_answer": original_answer,
            "improved_answer": original_answer + error_note,
            "original_score": evaluation.get("score", 0.0),
            "error": str(e)
        })
        kb.set_item("current_conversation", conversation)
        
        return original_answer + error_note