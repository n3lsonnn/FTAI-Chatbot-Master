# RAG Query Engine

This directory contains the Retrieval-Augmented Generation (RAG) system for the aircraft repair manual chatbot.

## Query Engine (`query_engine.py`)

The RAG query engine processes user queries and generates grounded answers using:
- FAISS similarity search for retrieving relevant chunks
- Sentence transformers for semantic understanding
- Ollama for local LLM inference

### Features

- **Semantic Search**: Uses FAISS to find the most relevant chunks based on semantic similarity
- **Context-Aware Generation**: Formats retrieved chunks into structured context for the LLM
- **Local LLM Integration**: Uses Ollama with the Mistral model for local inference
- **Safety-Focused**: Designed for aircraft maintenance with appropriate safety warnings
- **Interactive Mode**: Includes a command-line interface for testing

### Prerequisites

1. **Ollama Installation**: Make sure Ollama is installed and running
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the Mistral model
   ollama pull mistral
   
   # Start Ollama server
   ollama serve
   ```

2. **Python Dependencies**: Install required packages
   ```bash
   cd Chatbot-BE
   pip install -r requirements.txt
   ```

3. **Processed Data**: Ensure you have run the preprocessing scripts first
   ```bash
   cd scripts
   python pdf_chunker.py
   python embed_store.py
   ```

### Usage

#### Interactive Mode
```bash
cd Chatbot-BE/rag
python query_engine.py
```

This starts an interactive session where you can ask questions about aircraft repair and maintenance.

#### Programmatic Usage
```python
from query_engine import RAGQueryEngine

# Initialize the engine
engine = RAGQueryEngine()

# Load models and data
engine.load_models_and_data()

# Process a query
result = engine.process_query("How do I check the engine oil level?")

# Access the answer and metadata
print(result['answer'])
print(f"Retrieved {result['total_chunks_retrieved']} chunks")
```

### Configuration

You can customize the query engine by modifying these parameters:

```python
engine = RAGQueryEngine(
    model_name="all-MiniLM-L6-v2",  # Sentence transformer model
    ollama_url="http://localhost:11434",  # Ollama server URL
    ollama_model="mistral",  # Ollama model name
    top_k=5  # Number of chunks to retrieve
)
```

### Output Format

The query engine returns a dictionary with:

- `answer`: The generated response from Ollama
- `context`: The formatted context used for generation
- `chunks_used`: List of retrieved chunks with metadata
- `query`: The original user query
- `total_chunks_retrieved`: Number of chunks retrieved

### System Prompt

The engine uses a specialized system prompt for aircraft maintenance:

```
You are a helpful assistant for aircraft repair and maintenance. 
You have access to technical documentation and repair manuals. 
Always provide accurate, safety-focused answers based on the provided context.
If the context doesn't contain enough information to answer the question safely, say so.
Use clear, technical language appropriate for aircraft maintenance professionals.
```

### Troubleshooting

1. **Ollama Connection Error**: Make sure Ollama is running on `http://localhost:11434`
2. **Model Not Found**: Ensure the Mistral model is pulled: `ollama pull mistral`
3. **Missing Data**: Run the preprocessing scripts first to create the FAISS index
4. **Memory Issues**: Reduce `top_k` or use a smaller embedding model

### Performance Tips

- The first query may be slower as models are loaded into memory
- FAISS similarity search is very fast for large document collections
- Consider adjusting `top_k` based on your document structure and query complexity 