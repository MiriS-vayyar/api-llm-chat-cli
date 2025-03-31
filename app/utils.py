"""
Utility functions for the API Chat CLI.
"""

import requests
import json
from typing import Dict, Any

from app.config import API_BASE_URL, API_KEY


def execute_api_call(api_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the API call determined by the LLM.
    
    Args:
        api_call: Dictionary containing API call details
        
    Returns:
        API response with success status and data
    """
    method = api_call.get("method", "GET").upper()
    url = f"{API_BASE_URL}{api_call.get('url', '')}"
    headers = api_call.get("headers", {})
    body = api_call.get("body")
    
    # Add API key to headers if available and not already included
    if API_KEY and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {API_KEY}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=body)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=body)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=body)
        else:
            return {"success": False, "error": f"Unsupported HTTP method: {method}"}
        
        # Check if the response is JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = response.text
        
        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "data": response_data
        }
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}