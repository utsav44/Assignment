from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ManualChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        logger.info(f"Chunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict]:
        if not text.strip():
            return []
        
        chunks = self.splitter.split_text(text)
        
        return [
            {
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": i,
                    "chunk_total": len(chunks),
                    "chunk_size": len(chunk)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def chunk_pages(self, pages: List[Dict], metadata: Dict[str, Any]) -> List[Dict]:
        all_chunks = []
        chunk_id = 0
        
        for page_data in pages:
            page_text = page_data["text"]
            page_number = page_data["page_number"]
            
            if not page_text.strip():
                continue
            
            page_chunks = self.splitter.split_text(page_text)
            
            for chunk in page_chunks:
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "page": page_number,
                        "chunk_id": chunk_id,
                        "chunk_size": len(chunk)
                    }
                })
                chunk_id += 1
        
        # Update total chunk count
        for chunk in all_chunks:
            chunk["metadata"]["chunk_total"] = len(all_chunks)
        
        return all_chunks