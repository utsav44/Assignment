import PyPDF2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    def __init__(self):
        pass
    
    def extract_text(self, pdf_path: Path) -> str:
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
        
        return self._clean_text(text)
    
    def extract_text_by_page(self, pdf_path: Path) -> List[Dict]:
        pages = []
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        pages.append({
                            "text": self._clean_text(page_text),
                            "page_number": i
                        })
        except Exception as e:
            logger.error(f"Error extracting pages from {pdf_path}: {e}")
            raise
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Fix common OCR/extraction issues
        text = text.replace('\x00', '')
        
        # Remove very short lines (likely headers/footers)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 10 or not line.strip()]
        
        return '\n'.join(cleaned_lines).strip()