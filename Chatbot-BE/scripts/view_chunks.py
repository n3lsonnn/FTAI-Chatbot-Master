#!/usr/bin/env python3
"""
View Chunks Script

This script loads and displays the chunks created by the PDF chunker
to help verify the quality of text extraction and chunking.
"""

import json
from pathlib import Path

def view_chunks():
    """Load and display chunks from the JSON file."""
    chunks_file = Path("../models/chunks.json")
    
    if not chunks_file.exists():
        print("❌ Chunks file not found. Run pdf_chunker.py first.")
        return
    
    print("📖 Loading chunks...")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunks', [])
    source_pdf = data.get('source_pdf', 'unknown')
    
    print(f"\n📊 CHUNK SUMMARY")
    print("="*50)
    print(f"Source PDF: {source_pdf}")
    print(f"Total chunks: {len(chunks)}")
    
    # Count chunk types
    type_counts = {}
    for chunk in chunks:
        chunk_type = chunk.get('type', 'unknown')
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
    
    print(f"\nChunk types:")
    for chunk_type, count in type_counts.items():
        print(f"  {chunk_type}: {count}")
    
    print(f"\n📋 CHUNK DETAILS")
    print("="*50)
    
    # Display first few chunks
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk.get('id')}")
        print(f"Type: {chunk.get('type')}")
        print(f"Title: {chunk.get('title', 'No title')}")
        print(f"Content length: {len(chunk.get('content', ''))} characters")
        
        # Show first 200 characters of content
        content = chunk.get('content', '')
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"Preview: {preview}")
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")
    
    # Show a random chunk from the middle
    if len(chunks) > 10:
        middle_chunk = chunks[len(chunks)//2]
        print(f"\n--- Middle Chunk (ID: {middle_chunk.get('id')}) ---")
        print(f"Type: {middle_chunk.get('type')}")
        print(f"Title: {middle_chunk.get('title', 'No title')}")
        content = middle_chunk.get('content', '')
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"Preview: {preview}")
    
    # Show last chunk
    if chunks:
        last_chunk = chunks[-1]
        print(f"\n--- Last Chunk (ID: {last_chunk.get('id')}) ---")
        print(f"Type: {last_chunk.get('type')}")
        print(f"Title: {last_chunk.get('title', 'No title')}")
        content = last_chunk.get('content', '')
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"Preview: {preview}")

if __name__ == "__main__":
    view_chunks() 