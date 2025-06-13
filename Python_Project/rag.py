import json
import os
import pandas as pd
from typing import List, Dict, Any, Union, Literal, Optional
import fitz  # PyMuPDF for PDF processing
from pathlib import Path
from sentence_transformers import SentenceTransformer

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import MockLLM
from pydantic import Field

class SentenceTransformersEmbedding(BaseEmbedding):
    model: SentenceTransformer = Field(alias='model')

    def __init__(self, model_name: str = '/app/bge-small-en-v1.5'):
        """
        Custom embedding class using Sentence Transformers

        Args:
            model_name: Name of the Sentence Transformers model
        """
        model = SentenceTransformer(model_name)
        super().__init__(**{'model': model})

    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for given texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        return self._embed_text([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        return self._embed_text(texts)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronous method to get text embedding"""
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronous method to get query embedding"""
        return self._get_text_embedding(query)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Synchronous method to get query embedding"""
        return self._get_text_embedding(query)

class OptimizedRAGManager:

    def __init__(self, model_name: str = '/app/bge-small-en-v1.5'):
        self.oem_list = ["mahindra", "mercedes"]
        self.model_name = model_name

        # Initialize embedding model once
        self.embedding_model = SentenceTransformersEmbedding(model_name)

        # Configure LlamaIndex settings
        Settings.llm = MockLLM()
        Settings.embed_model = self.embedding_model

        # Lazy-loaded components
        self._manual_indices = {}  # Manual RAG indices
        self._media_indices = {}  # Media recommendation indices
        self._media_data_cache = {}  # JSON data cache

        # Node parser for chunking
        self.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

    @lru_cache(maxsize=10)
    def _load_media_data(self, oem: str) -> Dict:
        """Cache media data loading"""
        if oem in self._media_data_cache:
            return self._media_data_cache[oem]

        filename = f"/app/{oem}-media-data-mapping.json"
        try:
            with open(filename, 'r') as f:
                video_data = json.load(f)
            self._media_data_cache[oem] = video_data
            return video_data
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return {}

    def _get_manual_index(self, oem: str) -> Optional[VectorStoreIndex]:
        """Lazy load manual RAG index"""
        if oem not in self._manual_indices:
            self._load_manual_index(oem)
        return self._manual_indices.get(oem)

    def _load_manual_index(self, oem: str):
        """Load or create manual index for OEM"""
        index_dir = f"/app/{oem}_manual_llama_index"

        try:
            # Try to load existing index
            if os.path.exists(index_dir):
                storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                index = load_index_from_storage(storage_context)
                self._manual_indices[oem] = index
                print(f"✅ Loaded existing manual index for {oem}")
            else:
                # Create new index
                self._create_manual_index(oem)
        except Exception as e:
            print(f"Error loading manual index for {oem}: {e}")

    def _create_manual_index(self, oem: str):
        """Create manual index from PDF"""
        manual_paths = {
            'mercedes': "/app/mercedes-owners-manual.pdf",
            'mahindra': "/app/mahindra-owners-manual.pdf"
        }

        manual_path = manual_paths.get(oem, "/app/mercedes-owners-manual.pdf")
        index_dir = f"/app/{oem}_manual_llamaindex"

        try:
            # Extract text from PDF
            text_content = self._extract_pdf_text(manual_path)
            if not text_content:
                return

            # Create text nodes
            nodes = self.node_parser.get_nodes_from_documents([
                TextNode(
                    text=text_content,
                    metadata={
                        "source_type": "manual",
                        "oem": oem,
                        "source": manual_path
                    }
                )
            ])

            # Create and persist index
            index = VectorStoreIndex(nodes, embed_model=self.embedding_model)
            os.makedirs(index_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=index_dir)

            self._manual_indices[oem] = index
            print(f"✅ Created manual index for {oem} with {len(nodes)} nodes")

        except Exception as e:
            print(f"Error creating manual index for {oem}: {e}")

    def _get_media_index(self, oem: str) -> Optional[VectorStoreIndex]:
        """Lazy load media recommendation index"""
        if oem not in self._media_indices:
            self._create_media_index(oem)
        return self._media_indices.get(oem)

    def _create_media_index(self, oem: str):
        """Create media recommendation index"""
        video_data = self._load_media_data(oem)
        if not video_data:
            return

        nodes = []
        for title, entries in video_data.items():
            for entry in entries:
                # Create multiple variants for better matching
                variants = [
                    title,
                    title.lower(),
                    title.replace('-', ' ').replace('_', ' ')
                ]

                for variant in variants:
                    nodes.append(TextNode(
                        text=variant,
                        metadata={
                            "source_type": "media",
                            "oem": oem,
                            "original_title": title,
                            "filename": entry['file'],
                            "filetype": entry['type']
                        }
                    ))

        if nodes:
            self._media_indices[oem] = VectorStoreIndex(nodes, embed_model=self.embedding_model)
            print(f"✅ Created media index for {oem} with {len(nodes)} nodes")

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text_content = []

            for page in doc:
                text = page.get_text("text")
                text_content.append(text)

            doc.close()
            return "\n".join(text_content)
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def query_manual(self, oem: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Query manual content for OEM"""
        manual_index = self._get_manual_index(oem)
        if not manual_index:
            return []

        try:
            query_engine = manual_index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query)

            results = []
            for node in response.source_nodes:
                results.append({
                    'text': node.text,
                    'source_type': 'manual',
                    'oem': node.metadata.get('oem', oem),
                    'score': getattr(node, 'score', 0.0),
                    'metadata': node.metadata
                })

            return results

        except Exception as e:
            print(f"Error querying manual for {oem}: {e}")
            return []

    def get_media_recommendations(self, oem: str, query: str, top_k: int = 4) -> List[str]:
        """Get media file recommendations"""
        media_index = self._get_media_index(oem)
        if not media_index:
            return ["No media index available"]

        try:
            query_engine = media_index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query)

            recommendations = []
            seen_files = set()

            for node in response.source_nodes:
                filename = node.metadata.get('filename', '')
                if filename and filename not in seen_files:
                    recommendations.append(filename)
                    seen_files.add(filename)

            return recommendations if recommendations else ["No Image/Video found"]

        except Exception as e:
            print(f"Error getting media recommendations for {oem}: {e}")
            return ["Error retrieving media"]

    def comprehensive_query(self, oem: str, user_query: str, include_manual: bool = True,
                            include_media: bool = True) -> Dict[str, Any]:
        """Comprehensive query combining manual and media results"""
        results = {
            'query': user_query,
            'oem': oem,
            'manual_results': [],
            'media_recommendations': [],
            'summary': {}
        }

        if include_manual:
            results['manual_results'] = self.query_manual(oem, user_query)

        if include_media:
            results['media_recommendations'] = self.get_media_recommendations(oem, user_query)

        # Create summary
        results['summary'] = {
            'manual_matches': len(results['manual_results']),
            'media_matches': len([r for r in results['media_recommendations']
                                  if r not in ["No Image/Video found", "Error retrieving media"]]),
            'best_manual_match': results['manual_results'][0] if results['manual_results'] else None,
            'top_media_file': results['media_recommendations'][0] if results['media_recommendations'] else None
        }

        return results

    def clear_cache(self, oem: str = None):
        """Clear cache for memory management"""
        if oem:
            self._manual_indices.pop(oem, None)
            self._media_indices.pop(oem, None)
            self._media_data_cache.pop(oem, None)
        else:
            self._manual_indices.clear()
            self._media_indices.clear()
            self._media_data_cache.clear()
        self._load_media_data.cache_clear()
