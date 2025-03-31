"""
Sentence transformer model for creating and querying embeddings.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path

from app.config import DEFAULT_EMBED_MODEL, CHROMA_DB_PATH


class EmbeddingModel:
    """Class for handling embeddings and vector database operations."""
    
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, db_path: str = CHROMA_DB_PATH):
        """
        Initialize the embedding model and vector database.
        
        Args:
            model_name: Name of the sentence transformer model to use
            db_path: Path to store/load the vector database
        """
        self.model_name = model_name
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Try to get or create the collection
        try:
            self.collection = self.chroma_client.get_collection("api_docs")
        except:
            self.collection = self.chroma_client.create_collection("api_docs")
    
    def process_docs(self, docs_dir: Path) -> int:
        """
        Process API documentation files and create embeddings for the vector database.
        
        Args:
            docs_dir: Directory containing API documentation files
            
        Returns:
            Number of document chunks processed
        """
        # Get all markdown files in the docs directory
        doc_files = list(docs_dir.glob("**/*.md"))
        
        if not doc_files:
            return 0
        
        # Process each file
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        for doc_file in doc_files:
            # Read the file content
            content = doc_file.read_text(encoding="utf-8")
            
            # Split into chunks
            chunks = self._chunk_text(content, chunk_size=300, overlap=50)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{doc_file.stem}_{i}")
                all_metadatas.append({"source": str(doc_file.name), "index": i})
        
        # Create embeddings and add to the database
        if all_chunks:
            embeddings = self.model.encode(all_chunks).tolist()
            self.collection.add(
                documents=all_chunks,
                embeddings=embeddings,
                ids=all_ids,
                metadatas=all_metadatas
            )
            return len(all_chunks)
        
        return 0
    
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> str:
        """
        Retrieve the most relevant documentation chunks for a query.
        
        Args:
            query: User's query in natural language
            k: Number of relevant chunks to retrieve
            
        Returns:
            Combined text of the most relevant chunks
        """
        # Create embedding for the query
        query_embedding = self.model.encode(query).tolist()
        
        # Query the vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Combine the retrieved documents
        if results and results['documents']:
            context = "\n\n---\n\n".join(results['documents'][0])
            return context
        
        return ""
    
    def get_doc_count(self) -> int:
        """
        Get the count of documents in the collection.
        
        Returns:
            Number of documents in the collection
        """
        return self.collection.count()
    
    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split
            chunk_size: Target size of each chunk in words
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks