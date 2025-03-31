
# API Chat CLI

A conversational CLI assistant that translates user requests into API calls, executes them, and returns responses using Ollama, ChromaDB, and Typer. 🚀

## Features
- **Natural Language Interface**: Chat with the system in natural language and receive API call results.
- **API Call Generation**: Automatically translates user requests into API calls.
- **API Documentation Integration**: Fetches relevant information from API documentation using ChromaDB and Sentence Transformers.
- **Interactive CLI**: Built using Typer for a user-friendly experience.

## Architecture Overview

This project leverages **Ollama** for language processing, **ChromaDB** for storing API documentation embeddings, and **Sentence Transformers** for matching user queries to relevant API documentation. The system then translates these queries into API calls and executes them.

## Installation

### Prerequisites
- **Python** 3.8 or higher
- **Ollama** - Install from [here](https://ollama.com/download) and pull the `mistral` model.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MiriS-vayyar/api-llm-chat-cli.git
   cd api-chat-cli
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and configure the following:
  

4. Run the CLI:
   ```bash
   python app/cli.py
   ```

## Commands

### `chat`
Starts an interactive chat session where you can ask questions or make requests in natural language. The system will automatically generate the appropriate API calls.

### `setup`
Sets up the API documentation for use in the system. You will load your API documentation into the database to be used in processing queries.

## Project Structure

```
api-chat-cli/
├── api_docs/
│   └── README.md
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── cli.py
│   ├── llm.py
│   └── utils.py
├── data/
│   └── api_docs_db/
├── models/
│   └── sentence_transformer_model.py
├── requirements.txt
├── .env
└── README.md
```

## Dependencies
The project uses the following libraries:
- `ollama`: For natural language processing with Ollama models.
- `chromadb`: To store and manage API documentation embeddings.
- `sentence-transformers`: For embedding and comparing queries with API docs.
- `requests`: For making API calls.
- `typer`: For building the interactive CLI.
- `rich`: For enhanced terminal output.
- `python-dotenv`: To manage environment variables.

