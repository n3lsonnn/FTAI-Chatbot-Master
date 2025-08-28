# Aircraft Repair Manual Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for aircraft repair and maintenance documentation. This system uses local LLMs via Ollama and semantic search via FAISS to provide grounded answers from technical manuals.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Ollama** installed and running
3. **Mistral model** pulled in Ollama

### Installation

1. **Install Python dependencies:**
   ```bash
   cd Chatbot-BE
   pip install -r requirements.txt
   ```

2. **Install and start Ollama:**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the Mistral model
   ollama pull mistral
   
   # Start Ollama server
   ollama serve
   ```

### Processing Pipeline

1. **Chunk the PDF manual:**
   ```bash
   cd scripts
   python pdf_chunker.py
   ```

2. **Create embeddings and FAISS index:**
   ```bash
   python embed_store.py
   ```

3. **Start the chatbot:**
   ```bash
   cd ..
   python app.py
   ```

## 📁 Project Structure

```
Chatbot-BE/
├── app.py                 # Main CLI interface
├── requirements.txt       # Python dependencies
├── data/
│   └── test_manual.pdf   # Input PDF manual
├── scripts/
│   ├── pdf_chunker.py    # PDF text extraction and chunking
│   ├── embed_store.py    # Embedding creation and FAISS storage
│   └── README.md         # Script documentation
├── rag/
│   ├── query_engine.py   # RAG query processing
│   └── README.md         # RAG system documentation
└── models/               # Generated files (created after processing)
    ├── chunks.json       # Extracted text chunks
    ├── index.faiss       # FAISS similarity index
    └── chunk_texts.pkl   # Chunk metadata
```

## 🎯 Usage

### Interactive CLI

Run the main application for an interactive experience:

```bash
python app.py
```

**Available Commands:**
- `help` - Show help information
- `info` - Display system information
- `clear` - Clear the screen
- `quit` - Exit the application

**Example Questions:**
- "How do I check the engine oil level?"
- "What are the safety procedures for engine maintenance?"
- "How often should I inspect the landing gear?"
- "What tools are needed for propeller maintenance?"

### Programmatic Usage

```python
from rag.query_engine import RAGQueryEngine

# Initialize and load the system
engine = RAGQueryEngine()
engine.load_models_and_data()

# Process a query
result = engine.process_query("How do I check the engine oil level?")
print(result['answer'])
```

## 🔧 Configuration

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimension**: 384
- **Performance**: Fast and efficient for semantic search

### LLM Configuration
- **Model**: `mistral` (via Ollama)
- **Temperature**: 0.1 (focused, consistent answers)
- **Max Tokens**: 1000
- **System Prompt**: Aircraft maintenance focused

### Retrieval Settings
- **Top-K**: 5 chunks retrieved per query
- **Similarity**: L2 distance in FAISS
- **Context**: Structured formatting with chunk titles

## 🛠️ Customization

### Change Models
```python
# Different embedding model
engine = RAGQueryEngine(model_name="all-mpnet-base-v2")

# Different LLM
engine = RAGQueryEngine(ollama_model="llama2")

# Adjust retrieval
engine = RAGQueryEngine(top_k=10)
```

### Add New PDFs
1. Place new PDF in `data/` directory
2. Update path in `scripts/pdf_chunker.py`
3. Re-run the processing pipeline

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Make sure Ollama is running
   ollama serve
   
   # Check if Mistral is available
   ollama list
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Missing Data Files**
   ```bash
   # Run the preprocessing pipeline
   cd scripts
   python pdf_chunker.py
   python embed_store.py
   ```

4. **Memory Issues**
   - Reduce `top_k` in query engine
   - Use smaller embedding model
   - Process smaller PDFs

### Performance Tips

- First query may be slower (model loading)
- FAISS search is very fast for large collections
- Consider batch processing for multiple queries
- Monitor memory usage with large documents

## 📊 System Information

The system provides detailed information about:
- Total chunks processed
- FAISS index size
- Embedding dimensions
- Model configurations
- Retrieval statistics

Use the `info` command in the CLI to view system details.

## 🔒 Safety Notes

- This system is designed for aircraft maintenance professionals
- Always verify critical procedures against official documentation
- The system includes safety warnings in its responses
- Use appropriate safety equipment and procedures

## 🤝 Contributing

To extend the system:
1. Add new structural patterns in `pdf_chunker.py`
2. Implement different embedding models
3. Create custom system prompts for specific domains
4. Add API endpoints for web integration
5. Implement conversation memory for multi-turn dialogues 