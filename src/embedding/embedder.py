from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", 
                 query_prefix: str = "Represent this sentence for searching relevant passages: "):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 32, 
                        show_progress: bool = False) -> np.ndarray:

        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:

        prefixed_query = self.query_prefix + query
        embedding = self.model.encode(
            [prefixed_query],
            normalize_embeddings=True
        )[0]
        return embedding
