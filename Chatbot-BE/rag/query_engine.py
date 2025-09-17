#!/usr/bin/env python3
"""
RAG Query Engine for Aircraft Repair Manual Chatbot

This script implements a Retrieval-Augmented Generation (RAG) system that:
- Loads FAISS index and metadata
- Processes user queries
- Finds similar chunks using semantic search
- Generates grounded answers using Ollama

Features:
- Semantic similarity search with FAISS
- Context-aware prompt generation
- Integration with Ollama for local LLM inference
- Configurable retrieval parameters
"""

import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """RAG query engine for processing user queries and generating grounded answers."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "tinyllama",
                 top_k: int = 5):
        """
        Initialize the RAG query engine.
        
        Args:
            model_name: Sentence transformer model name
            ollama_url: Ollama server URL
            ollama_model: Ollama model name
            top_k: Number of similar chunks to retrieve
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.top_k = top_k
        
        # Initialize components
        self.embedding_model = None
        self.faiss_index = None
        self.metadata = None
        self.chunks = []
        
        # File paths
        self.index_file = Path("models/index.faiss")
        self.metadata_file = Path("models/chunk_texts.pkl")
        
        # System prompt template (includes strict fallback wording)
        self.system_prompt = """You are a helpful assistant for aircraft repair and maintenance. 
You have access to technical documentation and repair manuals. 
Always provide accurate, safety-focused answers based on the provided context.
If the context does not contain the information to answer the question safely, answer exactly: 'Not specified in the provided context.'
Use clear, technical language appropriate for aircraft maintenance professionals."""

    def load_models_and_data(self):
        """Load the embedding model, FAISS index, and metadata."""
        logger.info("Loading models and data...")
        
        try:
            # Load sentence transformer model
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Load FAISS index
            if not self.index_file.exists():
                raise FileNotFoundError(f"FAISS index not found: {self.index_file}")
            
            logger.info(f"Loading FAISS index from: {self.index_file}")
            self.faiss_index = faiss.read_index(str(self.index_file))
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            
            # Load metadata
            if not self.metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
            
            logger.info(f"Loading metadata from: {self.metadata_file}")
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.chunks = self.metadata.get('chunks', [])
            logger.info(f"Loaded {len(self.chunks)} chunks from metadata")
            
            # Verify consistency
            if len(self.chunks) != self.faiss_index.ntotal:
                raise ValueError(f"Chunk count mismatch: {len(self.chunks)} chunks vs {self.faiss_index.ntotal} vectors")
            
            logger.info("Models and data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models and data: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Create embedding for the user query.
        
        Args:
            query: User's question
            
        Returns:
            Query embedding as numpy array
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded. Call load_models_and_data() first.")
        
        logger.info(f"Creating embedding for query: {query[:50]}...")
        
        try:
            embedding = self.embedding_model.encode([query])
            return embedding.astype('float32')
            
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            raise
    
    def find_similar_chunks(self, query_embedding: np.ndarray) -> List[Tuple[int, float, Dict]]:
        """
        Find similar chunks using FAISS similarity search.
        
        Args:
            query_embedding: Query embedding
            
        Returns:
            List of tuples: (chunk_index, similarity_score, chunk_data)
        """
        if not self.faiss_index:
            raise RuntimeError("FAISS index not loaded. Call load_models_and_data() first.")
        
        logger.info(f"Searching for top-{self.top_k} similar chunks")
        
        try:
            # Perform similarity search
            distances, indices = self.faiss_index.search(query_embedding, self.top_k)
            
            # Convert distances to similarity scores (1 / (1 + distance))
            similarities = 1 / (1 + distances[0])
            
            # Get chunk data for each result
            results = []
            for i, (chunk_idx, similarity) in enumerate(zip(indices[0], similarities)):
                if chunk_idx < len(self.chunks):
                    chunk_data = self.chunks[chunk_idx].copy()
                    chunk_data['similarity_score'] = float(similarity)
                    chunk_data['rank'] = i + 1
                    results.append((chunk_idx, similarity, chunk_data))
                    logger.info(f"Rank {i+1}: Chunk {chunk_idx} (similarity: {similarity:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            raise
    
    def format_context_prompt(self, similar_chunks: List[Tuple[int, float, Dict]]) -> str:
        """
        Format retrieved chunks into a context prompt.
        
        Args:
            similar_chunks: List of similar chunks with scores
            
        Returns:
            Formatted context string
        """
        if not similar_chunks:
            return "No relevant context found."
        
        context_parts = ["Based on the following technical documentation:\n"]
        
        for chunk_idx, similarity, chunk_data in similar_chunks:
            title = chunk_data.get('title', f'Chunk {chunk_idx}')
            content = chunk_data.get('content', '')
            chunk_type = chunk_data.get('type', 'unknown')
            
            # Format each chunk
            chunk_text = f"\n--- {chunk_type.upper()}: {title} ---\n{content}\n"
            context_parts.append(chunk_text)
        
        return "\n".join(context_parts)
    
    def create_ollama_prompt(self, query: str, context: str) -> str:
        """
        Create the complete prompt for Ollama.
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Complete prompt string
        """
        prompt = f"""{self.system_prompt}

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def query_ollama(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Send prompt to Ollama and get response.
        
        Args:
            prompt: Complete prompt to send
            options: Optional dict to override generation settings
            
        Returns:
            Ollama's response
        """
        logger.info("Sending query to Ollama...")
        
        try:
            # Default generation options tuned for this laptop
            gen_options: Dict[str, Any] = {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 1000,
            }
            if options:
                gen_options.update(options)

            # Prepare request payload
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": gen_options,
            }
            
            # Send request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # Increased timeout to 2 minutes
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            # Parse response
            result = response.json()
            answer = result.get('response', '')
            
            if not answer:
                raise Exception("Empty response from Ollama")
            
            logger.info("Received response from Ollama")
            return answer.strip()
            
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.ollama_url}. Make sure Ollama is running.")
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return a grounded answer.
        
        Args:
            query: User's question
            
        Returns:
            Dictionary containing answer and metadata
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Create query embedding
            query_embedding = self.embed_query(query)
            
            # Find similar chunks
            similar_chunks = self.find_similar_chunks(query_embedding)
            
            if not similar_chunks:
                return {
                    "answer": "I couldn't find any relevant information in the documentation to answer your question.",
                    "context": "No relevant context found.",
                    "chunks_used": [],
                    "query": query
                }
            
            # Format context
            context = self.format_context_prompt(similar_chunks)
            
            # Create complete prompt
            prompt = self.create_ollama_prompt(query, context)
            
            # Get answer from Ollama
            answer = self.query_ollama(prompt)
            
            # Prepare response
            response = {
                "answer": answer,
                "context": context,
                "chunks_used": [
                    {
                        "id": chunk_data.get('id'),
                        "title": chunk_data.get('title'),
                        "type": chunk_data.get('type'),
                        "similarity_score": similarity,
                        "rank": chunk_data.get('rank')
                    }
                    for _, similarity, chunk_data in similar_chunks
                ],
                "query": query,
                "total_chunks_retrieved": len(similar_chunks)
            }
            
            logger.info("Query processing completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the loaded system."""
        return {
            "embedding_model": self.model_name,
            "ollama_model": self.ollama_model,
            "ollama_url": self.ollama_url,
            "total_chunks": len(self.chunks) if self.chunks else 0,
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "top_k": self.top_k
        }


# Cached engine for evaluation use to avoid repeated loads
_EVAL_ENGINE: Optional[RAGQueryEngine] = None

def run_query(query: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal helper for evaluation harnesses.

    Params may include:
      - top_k: int
      - temperature: float
      - max_tokens: int

    Returns:
      {"answer_text": str, "retrieved_titles": List[str]}
    """
    global _EVAL_ENGINE

    top_k = int(params.get("top_k", 5))
    temperature = float(params.get("temperature", 0.1))
    max_tokens = int(params.get("max_tokens", 1000))

    if _EVAL_ENGINE is None:
        engine = RAGQueryEngine(top_k=top_k)
        engine.load_models_and_data()
        _EVAL_ENGINE = engine
    else:
        engine = _EVAL_ENGINE
        # Allow changing retrieval count per call without reloading
        engine.top_k = top_k

    # Embed and retrieve
    query_embedding = engine.embed_query(query)
    similar = engine.find_similar_chunks(query_embedding)

    # Build context and prompt
    context = engine.format_context_prompt(similar)
    prompt = engine.create_ollama_prompt(query, context)

    # Generation with overrides tuned for CPU laptop
    answer_text = engine.query_ollama(prompt, options={
        "temperature": temperature,
        "max_tokens": max_tokens,
    })

    retrieved_titles: List[str] = [c[2].get("title", f"Chunk {c[0]}") for c in similar]

    return {
        "answer_text": answer_text,
        "retrieved_titles": retrieved_titles,
    }


def main():
    """Main function for testing the query engine."""
    try:
        # Create query engine instance
        engine = RAGQueryEngine()
        
        # Load models and data
        engine.load_models_and_data()
        
        # Display system info
        info = engine.get_system_info()
        print("\n" + "="*50)
        print("RAG QUERY ENGINE SYSTEM INFO")
        print("="*50)
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Interactive query loop
        print("\n" + "="*50)
        print("INTERACTIVE QUERY MODE")
        print("="*50)
        print("Enter your questions about aircraft repair and maintenance.")
        print("Type 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuestion: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Process query
                result = engine.process_query(query)
                
                # Display answer
                print("\n" + "-"*50)
                print("ANSWER:")
                print("-"*50)
                print(result['answer'])
                
                # Display context info
                print(f"\nRetrieved {result['total_chunks_retrieved']} relevant chunks:")
                for chunk in result['chunks_used']:
                    print(f"  - {chunk['title']} (similarity: {chunk['similarity_score']:.3f})")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 