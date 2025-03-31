"""
CLI interface for the API Chat application.
"""

import typer
import json
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

from app.config import API_DOCS_DIR
from app.llm import check_model_availability, get_api_call_from_query, summarize_api_response
from app.utils import execute_api_call
from models.sentence_transformer_model import EmbeddingModel

app = typer.Typer(help="API Assistant - A CLI that translates natural language to API calls")
console = Console()

# Initialize embedding model
embedding_model = EmbeddingModel()


@app.command("setup")
def setup(docs_dir: str = typer.Argument(
    API_DOCS_DIR, help="Directory containing API documentation files in Markdown format")):
    """Initialize the system by processing API documentation."""
    console.print(Markdown("# Setting up API Assistant"))
    console.print("\nThis will process your API documentation and create embeddings for search.")
    console.print("Make sure Ollama is installed and running with your chosen model.\n")
    
    # Check if Ollama is running and the model is available
    if not check_model_availability():
        console.print(f"[bold red]Error: Default model not found in Ollama.[/]")
        console.print("Please make sure Ollama is installed and running:")
        console.print("1. Install from https://ollama.com/")
        console.print("2. Run: ollama pull mistral")
        return
    
    # Initialize embeddings
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        console.print(f"[bold red]Error: Documentation directory '{docs_dir}' not found.[/]")
        return
    
    console.print("[bold blue]Initializing embeddings from API documentation...[/]")
    chunk_count = embedding_model.process_docs(docs_path)
    
    if chunk_count > 0:
        console.print(f"[bold green]Setup complete! Added {chunk_count} document chunks to the vector database.[/]")
        console.print("\nYou can now run the chat interface with:")
        console.print("python -m app.cli chat")
    else:
        console.print("[bold red]No documentation files were processed. Please check your docs directory.[/]")


@app.command("chat")
def chat():
    """Start an interactive chat session to query the API."""
    console.print(Markdown("# API Assistant Chat"))
    console.print("Type your queries in natural language, and I'll translate them to API calls.")
    console.print("Type 'exit' or 'quit' to end the session.\n")
    
    # Check if the vector database exists and has documents
    try:
        doc_count = embedding_model.get_doc_count()
        if doc_count == 0:
            console.print("[bold yellow]Warning: No API documentation found in the vector database.[/]")
            console.print("Please run setup first: python -m app.cli setup <docs_dir>")
            return
    except Exception as e:
        console.print(f"[bold red]Error accessing vector database: {str(e)}[/]")
        console.print("Please run setup first: python -m app.cli setup <docs_dir>")
        return
    
    # Main chat loop
    while True:
        # Get user input
        query = console.input("\n[bold blue]You:[/] ")
        
        if query.lower() in ("exit", "quit"):
            console.print("\nGoodbye!")
            break
        
        console.print("\n[bold green]Assistant:[/] Thinking...")
        
        # Retrieve relevant API documentation
        api_context = embedding_model.retrieve_relevant_docs(query)
        
        if not api_context:
            console.print("[italic]I don't have any relevant API documentation for that request.[/]")
            continue
        
        # Process the query with the LLM
        llm_response = get_api_call_from_query(query, api_context)
        
        if llm_response["type"] == "api_call":
            api_call = llm_response["data"]
            
            # Show the planned API call
            console.print("[dim]Planning to make this API call:[/]")
            console.print(json.dumps(api_call, indent=2))
            
            # Ask for confirmation
            confirm = console.input("\nExecute this API call? [Y/n]: ").lower()
            if confirm in ("", "y", "yes"):
                console.print("[dim]Executing API call...[/]")
                api_response = execute_api_call(api_call)
                
                # Format and display the response
                formatted_response = summarize_api_response(api_response, query)
                console.print("\n[bold green]Result:[/]")
                console.print(Markdown(formatted_response))
            else:
                console.print("[italic]API call cancelled.[/]")
        else:
            # Just display the conversational response
            console.print(Markdown(llm_response["data"]))


if __name__ == "__main__":
    app()