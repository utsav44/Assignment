import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    def __init__(self, index_path: Path, dimension: int = 384):

        self.index_path = Path(index_path)
        self.dimension = dimension
        self.metadata_path = self.index_path.parent / "metadata.pkl"

        self.index: Optional[faiss.Index] = None
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []
        self.car_model_keys: Set[str] = set()

        self._load_or_create()
    
    def _load_or_create(self):

        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self._load()
                logger.info(f"Loaded index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Creating new.")
                self._create_new()
        else:
            self._create_new()
        
        self._validate_consistency()
    
    def _create_new(self):

        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []
        self.metadatas = []
        self.car_model_keys = set()
        
        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Created new FAISS index")
    
    def _load(self):

        self.index = faiss.read_index(str(self.index_path))
        
        with open(self.metadata_path, "rb") as f:
            saved_data = pickle.load(f)
            self.texts = saved_data.get("texts", [])
            self.metadatas = saved_data.get("metadatas", [])
            self.car_model_keys = set(saved_data.get("car_model_keys", []))
    
    def _save(self):

        faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadatas": self.metadatas,
                "car_model_keys": list(self.car_model_keys)
            }, f)
        
        logger.debug(f"Saved index with {self.index.ntotal} vectors")
    
    def _validate_consistency(self):
        if self.index is None:
            return
        
        ntotal = self.index.ntotal
        ntexts = len(self.texts)
        nmetadatas = len(self.metadatas)
        
        if ntotal != ntexts or ntotal != nmetadatas:
            logger.warning(
                f"Inconsistency detected: FAISS={ntotal}, "
                f"texts={ntexts}, metadatas={nmetadatas}"
            )
            # Reset to minimum consistent state
            min_size = min(ntotal, ntexts, nmetadatas)
            if min_size == 0:
                self._create_new()
            else:
                self.texts = self.texts[:min_size]
                self.metadatas = self.metadatas[:min_size]
                # Rebuild car_model_keys from metadatas
                self.car_model_keys = set(
                    m.get("car_model_key") for m in self.metadatas 
                    if m.get("car_model_key")
                )
                logger.info(f"Trimmed to {min_size} consistent entries")
    
    def add_documents(
        self, 
        embeddings: np.ndarray, 
        texts: List[str], 
        metadatas: List[Dict],
        car_model_key: str
    ):

        assert len(embeddings) == len(texts) == len(metadatas), \
            f"Length mismatch: embeddings={len(embeddings)}, texts={len(texts)}, metadatas={len(metadatas)}"
        
        if len(embeddings) == 0:
            return
        
        # Ensure car_model_key is in all metadatas
        for metadata in metadatas:
            metadata["car_model_key"] = car_model_key
        
        # Add to FAISS index
        embeddings_float32 = embeddings.astype('float32')
        self.index.add(embeddings_float32)
        
        # Add to metadata stores
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.car_model_keys.add(car_model_key)
        
        # Save
        self._save()
        
        logger.info(f"Added {len(texts)} documents for {car_model_key}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5
    ) -> Tuple[List[str], List[Dict], List[float]]:

        if self.index.ntotal == 0:
            return [], [], []
        
        query = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query, min(k, self.index.ntotal))
        
        texts = []
        metadatas = []
        result_scores = []
        
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.texts):
                texts.append(self.texts[idx])
                metadatas.append(self.metadatas[idx])
                result_scores.append(float(score))
        
        return texts, metadatas, result_scores
    
    def search_by_model(
        self, 
        query_embedding: np.ndarray, 
        car_model_key: str, 
        k: int = 5
    ) -> Tuple[List[str], List[Dict], List[float]]:

        if self.index.ntotal == 0:
            return [], [], []
        
        if car_model_key not in self.car_model_keys:
            logger.warning(f"Car model key not found: {car_model_key}")
            return [], [], []
        
        # Search more results to filter
        search_k = min(k * 20, self.index.ntotal)
        query = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query, search_k)
        
        texts = []
        metadatas = []
        result_scores = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            
            metadata = self.metadatas[idx]
            if metadata.get("car_model_key") == car_model_key:
                texts.append(self.texts[idx])
                metadatas.append(metadata)
                result_scores.append(float(score))
                
                if len(texts) >= k:
                    break
        
        return texts, metadatas, result_scores
    
    def clear(self):
        self._create_new()
        self._save()
        logger.info("Cleared all data from index")
    
    @property
    def total_documents(self) -> int:
        return self.index.ntotal if self.index else 0
