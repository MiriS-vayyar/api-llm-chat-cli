"""
Configuration settings for the API Chat CLI application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# LLM Configuration
DEFAULT_MODEL = "mistral"  # Default LLM model in Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Embedding Model Configuration
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"  # Default embedding model

# Vector Database Configuration
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "api_docs_db")

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.example.com")
API_KEY = os.getenv("API_KEY", "")

# Default API docs directory
API_DOCS_DIR = os.path.join(BASE_DIR, "api_docs")