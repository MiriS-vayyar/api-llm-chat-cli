"""
Language model integration with Ollama.
"""

import json
from typing import Dict, List, Optional, Any
from ollama import Client as OllamaClient

from app.config import OLLAMA_HOST, DEFAULT_MODEL

# Initialize Ollama client
ollama_client = OllamaClient(host=OLLAMA_HOST)


def check_model_availability(model_name: str = DEFAULT_MODEL) -> bool:
    """
    Check if the specified model is available in Ollama.
    
    Args:
        model_name: Name of the Ollama model to check
        
    Returns:
        True if model is available, False otherwise
    """
    try:
        models = ollama_client.list()
        available_models = [model['name'] for model in models.get('models', [])]
        return model_name in available_models
    except Exception:
        return False


def get_api_call_from_query(query: str, api_context: str) -> Dict[str, Any]:
    """
    Send the query to the LLM and process the response to generate an API call.
    
    Args:
        query: User's query
        api_context: Retrieved API documentation
        
    Returns:
        Dictionary with response type and data
    """
    system_prompt = _format_system_prompt(api_context)
    
    # Send to Ollama LLM
    response = ollama_client.chat(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    
    response_content = response['message']['content']
    
    # Try to parse as JSON
    try:
        # Find JSON content (it might be wrapped in markdown code blocks)
        if "```json" in response_content:
            json_str = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_str = response_content.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_content.strip()
            
        api_call = json.loads(json_str)
        return {"type": "api_call", "data": api_call}
    except (json.JSONDecodeError, IndexError):
        # Not a valid JSON, treat as conversational response
        return {"type": "conversation", "data": response_content}


def summarize_api_response(api_response: Dict[str, Any], query: str) -> str:
    """
    Summarize the API response in a user-friendly way.
    
    Args:
        api_response: API call result
        query: Original user query
        
    Returns:
        Formatted, user-friendly response
    """
    if not api_response.get("success", False):
        error_msg = api_response.get("error", f"API call failed with status code: {api_response.get('status_code')}")
        return f"Error: {error_msg}"
    
    # Convert the API response to a string
    response_str = json.dumps(api_response.get("data", {}), indent=2)
    
    # Ask the LLM to summarize the response
    summary_prompt = f"""The user asked: "{query}"

The API returned this response:
{response_str}

Please summarize this response in a clear, concise, and user-friendly way. Focus on the key information that addresses the user's request.
"""
    
    response = ollama_client.chat(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes API responses in plain language."},
            {"role": "user", "content": summary_prompt}
        ]
    )
    
    return response['message']['content']


def _format_system_prompt(api_context: str) -> str:
    """
    Format the system prompt with relevant API context.
    
    Args:
        api_context: Retrieved API documentation
        
    Returns:
        Formatted system prompt
    """
    return f"""You are an intelligent assistant that helps users interact with an API using natural language. 
Your job is to understand the user's request and translate it into the appropriate API call.

Here is the relevant API documentation:

{api_context}

If you can determine the appropriate API call, respond ONLY with a JSON object in this exact format:
{{
  "method": "<HTTP_METHOD>",
  "url": "<ENDPOINT_PATH_WITH_QUERY_PARAMS>",
  "headers": {{<HEADERS_OBJECT>}},
  "body": {{<REQUEST_BODY_OBJECT_OR_NULL>}}
}}

If you cannot determine the API call, or need more information, respond conversationally to ask for clarification.
If the API documentation doesn't contain information relevant to the user's request, admit that you don't know how to perform that operation.
"""