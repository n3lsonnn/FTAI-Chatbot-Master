#!/usr/bin/env python3
"""
PDF Chunker for Aircraft Repair Manuals

This script extracts structured text from PDF files and chunks it based on
structural patterns commonly found in technical manuals:
- Numbered sections (1., 2., 3., etc.)
- Lettered subsections (A., B., C., etc.)
- Parenthetical numbers ((1), (2), (3), etc.)
- Other hierarchical patterns

The script uses PyMuPDF for PDF text extraction and regex for pattern matching.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pypdf import PdfReader


class PDFChunker:
    """Extracts and chunks PDF text based on structural patterns."""
    
    def __init__(self, pdf_path: str):
        """
        Initialize the chunker with a PDF file path.
        
        Args:
            pdf_path: Path to the PDF file to process
        """
        self.pdf_path = Path(pdf_path)
        self.chunks = []
        
        # Regex patterns for different structural elements
        self.structural_patterns = [
            # Main numbered sections (1., 2., 3., etc.)
            r'^\s*(\d+)\.\s+',
            # Lettered subsections (A., B., C., etc.)
            r'^\s*([A-Z])\.\s+',
            # Parenthetical numbers ((1), (2), (3), etc.)
            r'^\s*\((\d+)\)\s+',
            # Roman numerals (I., II., III., etc.)
            r'^\s*([IVX]+)\.\s+',
            # Lowercase letters (a., b., c., etc.)
            r'^\s*([a-z])\.\s+',
            # Double parentheses ((a), (b), (c), etc.)
            r'^\s*\(([a-z])\)\s+',
        ]
        
        # Combined pattern for matching any structural element
        self.combined_pattern = '|'.join(self.structural_patterns)
    
    def extract_text_from_pdf(self) -> str:
        """
        Extract all text from the PDF file using pypdf.
        
        Returns:
            Extracted text as a string
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        print(f"Extracting text from: {self.pdf_path}")
        
        try:
            # Load the PDF using pypdf
            reader = PdfReader(self.pdf_path)
            
            # Check number of pages
            num_pages = len(reader.pages)
            print(f"Total pages in manual: {num_pages}")
            
            # Extract full text from all pages
            all_text = ""
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
                
                if (page_num + 1) % 10 == 0:
                    print(f"Processed page {page_num + 1}/{num_pages}")
            
            print(f"Successfully extracted text from {num_pages} pages")
            return all_text
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean the extracted text while preserving line structure.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace but preserve line breaks
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple empty lines to double line break
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def find_structural_boundaries(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Find structural boundaries in the text.
        
        Args:
            text: Cleaned text to analyze
            
        Returns:
            List of tuples: (position, pattern_type, content)
        """
        boundaries = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            match = re.match(self.combined_pattern, line)
            if match:
                # Determine pattern type
                pattern_type = "unknown"
                for i, pattern in enumerate(self.structural_patterns):
                    if re.match(pattern, line):
                        if i == 0:
                            pattern_type = "numbered_section"
                        elif i == 1:
                            pattern_type = "lettered_section"
                        elif i == 2:
                            pattern_type = "parenthetical_number"
                        elif i == 3:
                            pattern_type = "roman_numeral"
                        elif i == 4:
                            pattern_type = "lowercase_letter"
                        elif i == 5:
                            pattern_type = "parenthetical_letter"
                        break
                
                boundaries.append((line_num, pattern_type, line.strip()))
        
        return boundaries
    
    def refined_chunk_by_section(self, text: str) -> List[Dict]:
        """
        Chunk text by section headers using the refined approach.
        
        Args:
            text: Cleaned text to chunk
            
        Returns:
            List of chunk dictionaries
        """
        import re
        
        lines = text.split("\n")
        print(f"Total lines in text: {len(lines)}")
        
        # Debug: Show first 20 lines to see the structure
        print("\nFirst 20 lines of text:")
        for i, line in enumerate(lines[:20]):
            print(f"{i+1:2d}: '{line[:100]}'")
        
        # Debug: Search for numbered patterns in the entire text
        print(f"\nSearching for numbered patterns in the entire text:")
        numbered_lines = []
        for i, line in enumerate(lines):
            if re.match(r"^[0-9]+\.", line.strip()):
                numbered_lines.append((i+1, line.strip()[:100]))
                if len(numbered_lines) <= 20:  # Show first 20
                    print(f"Line {i+1}: '{line.strip()[:100]}'")
        
        print(f"Found {len(numbered_lines)} numbered lines in entire text")
        
        chunks = []
        current_chunk = ""
        chunk_id = 1
        header_count = 0
        
        # Pattern to match lines starting with a number followed by a period and space
        section_header_pattern = re.compile(r"^[0-9]+\.\s+.*")

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            header_match = section_header_pattern.match(line)

            # Debug: Show headers found
            if header_match:
                header_count += 1
                print(f"Found header {header_count}: '{line[:100]}'")

            # If the line matches a section header and we have content in the current chunk,
            # finalize the current chunk and start a new one
            if header_match and current_chunk:
                # Create chunk from current content
                chunk_title = self._extract_section_title(current_chunk)
                chunk = {
                    "id": chunk_id,
                    "type": "section_chunk",
                    "title": chunk_title,
                    "content": current_chunk.strip(),
                    "word_count": len(current_chunk.split())
                }
                chunks.append(chunk)
                chunk_id += 1
                current_chunk = line
            else:
                # Otherwise, add the line to the current chunk
                current_chunk += "\n" + line

        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_title = self._extract_section_title(current_chunk)
            chunk = {
                "id": chunk_id,
                "type": "section_chunk",
                "title": chunk_title,
                "content": current_chunk.strip(),
                "word_count": len(current_chunk.split())
            }
            chunks.append(chunk)

        print(f"\nTotal headers found: {header_count}")
        print(f"Total chunks created: {len(chunks)}")
        
        return chunks
    
    def _extract_section_title(self, chunk_text: str) -> str:
        """
        Extract the section title from chunk text.
        
        Args:
            chunk_text: Text content of the chunk
            
        Returns:
            Extracted section title
        """
        lines = chunk_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for section headers (numbered patterns)
            if re.match(r"^[0-9]+\.\s+.*", line):
                return line[:150]  # First 150 chars of the header
            
            # Look for other structural patterns
            if re.match(r'^[A-Z]\.\s+', line):
                return line[:150]
            
            if re.match(r'^\(\d+\)\s+', line):
                return line[:150]
            
            # If no structural pattern, use first meaningful line
            if len(line) > 10:
                return line[:150]
        
        # Fallback title
        return f"Section {len(chunk_text)} chars"
    
    def create_chunks(self, text: str, boundaries: List[Tuple[int, str, str]]) -> List[Dict]:
        """
        Create chunks using the refined section-based approach.
        
        Args:
            text: Cleaned text
            boundaries: List of structural boundaries (kept for compatibility)
            
        Returns:
            List of chunk dictionaries
        """
        print("Creating chunks using section-based approach...")
        
        # Use the refined section-based chunking method
        chunks = self.refined_chunk_by_section(text)
        
        # Limit context length for faster processing
        for chunk in chunks:
            if len(chunk['content']) > 2000:  # Limit to 2000 chars
                chunk['content'] = chunk['content'][:2000] + "..."
        
        print(f"Created {len(chunks)} chunks based on document sections")
        
        return chunks
    
    def process(self) -> List[Dict]:
        """
        Main processing method that extracts, cleans, and chunks the PDF.
        
        Returns:
            List of chunk dictionaries
        """
        print("Starting PDF processing...")
        
        # Extract text from PDF
        raw_text = self.extract_text_from_pdf()
        
        # Use raw text without cleaning
        print(f"Extracted {len(raw_text)} characters of text")
        
        # Debug: Show a sample of raw text
        print(f"\nSample of raw text (first 500 chars):")
        print("-" * 50)
        print(raw_text[:500])
        print("-" * 50)
        
        # Find structural boundaries (for reference)
        boundaries = self.find_structural_boundaries(raw_text)
        print(f"Found {len(boundaries)} structural boundaries (for reference)")
        
        # Create chunks using the raw text
        self.chunks = self.create_chunks(raw_text, boundaries)
        print(f"Created {len(self.chunks)} chunks")
        
        return self.chunks
    
    def save_chunks(self, output_path: str):
        """
        Save chunks to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        output_data = {
            "source_pdf": str(self.pdf_path),
            "total_chunks": len(self.chunks),
            "chunks": self.chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.chunks)} chunks to: {output_file}")


def main():
    """Main function to run the PDF chunker."""
    # Configuration
    pdf_path = "../data/test_manual.pdf"
    output_path = "../models/chunks.json"
    
    try:
        # Create chunker instance
        chunker = PDFChunker(pdf_path)
        
        # Process the PDF
        chunks = chunker.process()
        
        # Save results
        chunker.save_chunks(output_path)
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Source PDF: {pdf_path}")
        print(f"Total chunks created: {len(chunks)}")
        print(f"Output file: {output_path}")
        
        # Show chunk types distribution
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk['type']
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        print("\nChunk types distribution:")
        for chunk_type, count in type_counts.items():
            print(f"  {chunk_type}: {count}")
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 