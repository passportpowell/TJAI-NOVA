"""
Internet search utility for when local knowledge is insufficient.
This module provides functions to search the internet for information.
"""
from utils.function_logger import log_function_call
from utils.open_ai_utils import run_open_ai_ns, run_open_ai_ns_async
from core.knowledge_base import KnowledgeBase
import aiohttp
import asyncio
import json
import re
import urllib.parse
from datetime import datetime


@log_function_call
async def internet_search(kb: KnowledgeBase, query: str):
    """
    Performs an internet search and returns the results.
    
    This is a placeholder implementation that would ideally connect to a real
    search API like Google, Bing, etc. For now, it simulates a search using LLM.
    
    Parameters:
        kb (KnowledgeBase): Knowledge base for storing results
        query (str): Search query
        
    Returns:
        dict: Search results with metadata
    """
    print(f"Performing internet search for query: {query}")
    
    # Record the search attempt
    await kb.set_item_async("last_search_query", query)
    await kb.set_item_async("search_timestamp", datetime.now().isoformat())
    
    try:
        # In a production environment, this would call a real search API
        # For now, simulate with an LLM
        search_prompt = f"""
        Simulate internet search results for the query: "{query}"
        
        Provide results in the following JSON format:
        {{
            "search_results": [
                {{
                    "title": "Result Title 1",
                    "url": "https://example.com/page1",
                    "snippet": "Brief excerpt from the page showing relevant content...",
                    "source": "example.com"
                }},
                // Additional results...
            ],
            "knowledge_graph": {{
                "title": "Main entity",
                "subtitle": "Description",
                "facts": [
                    "Fact 1",
                    "Fact 2",
                    // Additional facts...
                ]
            }},
            "featured_snippet": "A direct answer to the query if available",
            "related_queries": ["related query 1", "related query 2", "related query 3"]
        }}
        
        Make the results realistic and factually accurate. Include at least 3-5 search results.
        Only respond with valid JSON. No other text.
        """
        
        search_context = """
        You are simulating a search engine API. Provide realistic, factually correct search results
        for the given query. Focus on accuracy and relevance. Format the response exactly
        as specified in the JSON schema.
        """
        
        # Call LLM to simulate search
        search_response = await run_open_ai_ns_async(search_prompt, search_context, model="gpt-4.1-nano")
        
        # Parse the JSON response
        try:
            search_results = json.loads(search_response)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract structured data
            search_results = extract_structured_search_data(search_response)
        
        # Store in knowledge base
        await kb.set_item_async("internet_search_results", search_results)
        await kb.set_item_async("search_successful", True)
        
        print(f"Search completed with {len(search_results.get('search_results', []))} results")
        return search_results
        
    except Exception as e:
        error_results = {
            "search_results": [
                {
                    "title": "Search Error",
                    "url": "",
                    "snippet": f"Error performing search: {str(e)}",
                    "source": "system"
                }
            ],
            "error": str(e)
        }
        
        # Store error in knowledge base
        await kb.set_item_async("internet_search_results", error_results)
        await kb.set_item_async("search_successful", False)
        await kb.set_item_async("search_error", str(e))
        
        print(f"Search failed with error: {str(e)}")
        return error_results


def extract_structured_search_data(text):
    """
    Attempts to extract structured search data from text when JSON parsing fails.
    
    Parameters:
        text (str): The text to parse
        
    Returns:
        dict: Extracted search data or fallback structure
    """
    # Default structure
    search_data = {
        "search_results": [],
        "featured_snippet": "",
        "related_queries": []
    }
    
    # Try to extract search results
    result_pattern = re.compile(r'"title":\s*"([^"]+)".*?"snippet":\s*"([^"]+)".*?"source":\s*"([^"]+)"', re.DOTALL)
    results = result_pattern.findall(text)
    
    if results:
        for title, snippet, source in results:
            search_data["search_results"].append({
                "title": title,
                "url": f"https://{source}/",
                "snippet": snippet,
                "source": source
            })
    
    # Try to extract featured snippet
    snippet_match = re.search(r'"featured_snippet":\s*"([^"]+)"', text)
    if snippet_match:
        search_data["featured_snippet"] = snippet_match.group(1)
    
    # Try to extract related queries
    queries_match = re.search(r'"related_queries":\s*\[(.*?)\]', text, re.DOTALL)
    if queries_match:
        queries_text = queries_match.group(1)
        queries = re.findall(r'"([^"]+)"', queries_text)
        search_data["related_queries"] = queries
    
    return search_data


@log_function_call
async def search_and_summarize(kb: KnowledgeBase, query: str):
    """
    Performs an internet search and summarizes the results for a given query.
    
    Parameters:
        kb (KnowledgeBase): Knowledge base for storing results
        query (str): Search query
        
    Returns:
        str: Summarized search results
    """
    # First, perform the search
    search_results = await internet_search(kb, query)
    
    # Extract the relevant information
    search_text = ""
    
    # Add featured snippet if available
    if search_results.get("featured_snippet"):
        search_text += f"Featured answer: {search_results['featured_snippet']}\n\n"
    
    # Add knowledge graph if available
    if search_results.get("knowledge_graph"):
        kg = search_results["knowledge_graph"]
        search_text += f"About {kg.get('title', 'the topic')}:\n"
        search_text += f"{kg.get('subtitle', '')}\n"
        if kg.get("facts"):
            search_text += "Key facts:\n"
            for fact in kg["facts"]:
                search_text += f"- {fact}\n"
        search_text += "\n"
    
    # Add search results
    search_text += "Search Results:\n"
    for i, result in enumerate(search_results.get("search_results", [])[:5], 1):
        search_text += f"{i}. {result.get('title', 'No title')}\n"
        search_text += f"   {result.get('snippet', 'No snippet')}\n"
        search_text += f"   Source: {result.get('source', 'Unknown')}\n\n"
    
    # Now create a summary using an LLM
    summary_prompt = f"""
    Summarize the following search results for the query: "{query}"
    
    {search_text}
    
    Create a comprehensive, factual summary that directly answers the query based on the search results.
    Be specific and include the most relevant information. Cite sources where appropriate.
    """
    
    summary_context = """
    You are a search result summarizer. Your job is to synthesize information from 
    search results into a coherent, accurate summary that directly addresses the 
    original query. Focus on facts and relevance.
    """
    
    try:
        # Call LLM for summarization
        summary = await run_open_ai_ns_async(summary_prompt, summary_context, model="gpt-4.1-nano")
        
        # Store in knowledge base
        await kb.set_item_async("search_summary", summary)
        await kb.set_item_async("search_and_summarize_result", summary)
        
        print(f"Created summary for search query: {query}")
        return summary
    except Exception as e:
        error_message = f"Error creating search summary: {str(e)}"
        await kb.set_item_async("search_summary_error", error_message)
        
        # Return the raw search results as fallback
        return f"Could not summarize search results due to an error. Raw search data:\n\n{search_text}"