#!/usr/bin/env python3
"""
Embedding and Storage Script for RAG Chatbot

This script loads chunked text from JSON, creates embeddings using sentence-transformers,
and stores them in a FAISS index for efficient similarity search.

Features:
- Loads chunks from models/chunks.json
- Creates embeddings using all-MiniLM-L6-v2 model
- Stores embeddings in FAISS index
- Saves metadata for retrieval
"""

import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Handles embedding creation and FAISS storage for text chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding store.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        # File paths
        self.chunks_file = Path("../models/chunks.json")
        self.index_file = Path("../models/index.faiss")
        self.metadata_file = Path("../models/chunk_texts.pkl")
        
    def load_model(self):
        """Load the sentence-transformers model."""
        logger.info(f"Loading sentence-transformers model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """
        Load chunks from the JSON file.
        
        Returns:
            List of chunk dictionaries
        """
        if not self.chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")
        
        logger.info(f"Loading chunks from: {self.chunks_file}")
        
        try:
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.chunks = data.get('chunks', [])
            logger.info(f"Loaded {len(self.chunks)} chunks from {data.get('source_pdf', 'unknown')}")
            
            return self.chunks
            
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            raise
    
    def prepare_texts_for_embedding(self) -> List[str]:
        """
        Prepare chunk texts for embedding.
        
        Returns:
            List of text strings to embed
        """
        texts = []
        
        for chunk in self.chunks:
            # Combine title and content for better semantic representation
            title = chunk.get('title', '')
            content = chunk.get('content', '')
            
            # Create a structured text representation
            if title and content:
                text = f"{title}\n{content}"
            elif content:
                text = content
            else:
                text = title
            
            texts.append(text.strip())
        
        logger.info(f"Prepared {len(texts)} texts for embedding")
        return texts
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for the given texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        
        try:
            # Create embeddings in batches for better memory management
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)
                
                if (i + batch_size) % 100 == 0 or i + batch_size >= len(texts):
                    logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
            
            # Combine all embeddings
            self.embeddings = np.vstack(all_embeddings)
            
            logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
            return self.embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create a FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        logger.info(f"Creating FAISS index with dimension: {dimension}")
        
        try:
            # Create a simple L2 distance index
            # For larger datasets, you might want to use IndexIVFFlat or IndexHNSW
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add vectors to the index
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"FAISS index created with {self.index.ntotal} vectors")
            return self.index
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise
    
    def save_index(self):
        """Save the FAISS index to disk."""
        if not self.index:
            raise RuntimeError("No index to save. Call create_faiss_index() first.")
        
        # Ensure directory exists
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving FAISS index to: {self.index_file}")
        
        try:
            faiss.write_index(self.index, str(self.index_file))
            logger.info("FAISS index saved successfully")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise
    
    def save_metadata(self):
        """Save chunk metadata to pickle file."""
        if not self.chunks:
            raise RuntimeError("No chunks to save. Call load_chunks() first.")
        
        # Ensure directory exists
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving metadata to: {self.metadata_file}")
        
        try:
            # Prepare metadata for saving
            metadata = {
                'chunks': self.chunks,
                'total_chunks': len(self.chunks),
                'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None,
                'model_name': self.model_name
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise
    
    def process(self):
        """Main processing method."""
        logger.info("Starting embedding and storage process...")
        
        try:
            # Load the model
            self.load_model()
            
            # Load chunks
            chunks = self.load_chunks()
            
            if not chunks:
                logger.warning("No chunks found to process")
                return
            
            # Prepare texts for embedding
            texts = self.prepare_texts_for_embedding()
            
            # Create embeddings
            embeddings = self.create_embeddings(texts)
            
            # Create FAISS index
            index = self.create_faiss_index(embeddings)
            
            # Save index and metadata
            self.save_index()
            self.save_metadata()
            
            logger.info("Processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the processed data."""
        return {
            'total_chunks': len(self.chunks),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None,
            'index_size': self.index.ntotal if self.index else 0,
            'model_name': self.model_name,
            'chunks_file': str(self.chunks_file),
            'index_file': str(self.index_file),
            'metadata_file': str(self.metadata_file)
        }


def main():
    """Main function to run the embedding and storage process."""
    try:
        # Create embedding store instance
        store = EmbeddingStore()
        
        # Process the data
        store.process()
        
        # Get and display summary
        summary = store.get_summary()
        
        print("\n" + "="*50)
        print("EMBEDDING AND STORAGE SUMMARY")
        print("="*50)
        print(f"Model used: {summary['model_name']}")
        print(f"Total chunks processed: {summary['total_chunks']}")
        print(f"Embedding dimension: {summary['embedding_dimension']}")
        print(f"FAISS index size: {summary['index_size']}")
        print(f"Chunks file: {summary['chunks_file']}")
        print(f"Index file: {summary['index_file']}")
        print(f"Metadata file: {summary['metadata_file']}")
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 