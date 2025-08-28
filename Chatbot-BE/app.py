#!/usr/bin/env python3
"""
Aircraft Repair Manual Chatbot - CLI Interface

This is a simple command-line interface for testing the RAG chatbot system.
It loads the FAISS index and chunk metadata, then provides an interactive
query loop for asking questions about aircraft repair and maintenance.

Usage:
    python app.py
"""

import sys
import os
from pathlib import Path

# Add the rag directory to the Python path
sys.path.append(str(Path(__file__).parent / "rag"))

from query_engine import RAGQueryEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                Aircraft Repair Manual Chatbot                ║
║                                                              ║
║  A RAG-powered assistant for aircraft maintenance queries   ║
║  Using local LLMs via Ollama and semantic search via FAISS  ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """Print help information."""
    help_text = """
Available Commands:
  help, h          - Show this help message
  info, i          - Show system information
  quit, q, exit    - Exit the application
  clear, cls       - Clear the screen

Example Questions:
  - How do I check the engine oil level?
  - What are the safety procedures for engine maintenance?
  - How often should I inspect the landing gear?
  - What tools are needed for propeller maintenance?
  - How do I troubleshoot engine starting issues?

Note: Make sure Ollama is running with the Mistral model loaded.
    """
    print(help_text)


def print_system_info(engine):
    """Print system information."""
    info = engine.get_system_info()
    
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Embedding Model: {info['embedding_model']}")
    print(f"Ollama Model: {info['ollama_model']}")
    print(f"Ollama URL: {info['ollama_url']}")
    print(f"Total Chunks: {info['total_chunks']:,}")
    print(f"FAISS Index Size: {info['faiss_index_size']:,}")
    print(f"Top-K Retrieval: {info['top_k']}")
    print("="*50)


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_answer(result):
    """Format and display the answer with metadata."""
    print("\n" + "═"*60)
    print("🤖 ANSWER")
    print("═"*60)
    print(result['answer'])
    
    print("\n" + "─"*60)
    print("📚 SOURCES")
    print("─"*60)
    if result['chunks_used']:
        for i, chunk in enumerate(result['chunks_used'], 1):
            print(f"{i}. {chunk['title']}")
            print(f"   Type: {chunk['type']} | Similarity: {chunk['similarity_score']:.3f}")
    else:
        print("No relevant sources found.")
    
    print(f"\n📊 Retrieved {result['total_chunks_retrieved']} chunks for this query.")


def main():
    """Main CLI application."""
    print_banner()
    
    try:
        # Initialize the RAG query engine
        print("🔄 Initializing RAG system...")
        engine = RAGQueryEngine()
        
        # Load models and data
        print("📚 Loading FAISS index and metadata...")
        engine.load_models_and_data()
        
        print("✅ System ready! Type 'help' for available commands.\n")
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                query = input("\n❓ Question: ").strip()
                
                # Handle special commands
                if query.lower() in ['quit', 'q', 'exit']:
                    print("\n👋 Goodbye! Safe flying!")
                    break
                
                elif query.lower() in ['help', 'h']:
                    print_help()
                    continue
                
                elif query.lower() in ['info', 'i']:
                    print_system_info(engine)
                    continue
                
                elif query.lower() in ['clear', 'cls']:
                    clear_screen()
                    print_banner()
                    continue
                
                elif not query:
                    continue
                
                # Process the query
                print("🔍 Searching for relevant information...")
                result = engine.process_query(query)
                
                # Display the result
                format_answer(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Safe flying!")
                break
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'help' for assistance.")
                continue
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have run the preprocessing scripts first:")
        print("   1. cd scripts && python pdf_chunker.py")
        print("   2. python embed_store.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Make sure Ollama is running: ollama serve")
        print("   - Ensure Mistral model is loaded: ollama pull mistral")
        print("   - Check that all dependencies are installed")
        sys.exit(1)


if __name__ == "__main__":
    main()


