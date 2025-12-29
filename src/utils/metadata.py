import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class MetadataManager:
    def __init__(self, db_path: Path, embedder):

        self.db_path = Path(db_path)
        self.embedder = embedder
        self.data = self._load()
    
    def _load(self) -> Dict:
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                # Convert embeddings back to numpy arrays
                if "model_embeddings" in data:
                    data["model_embeddings"] = {
                        k: np.array(v) 
                        for k, v in data["model_embeddings"].items()
                    }
                logger.info(f"Loaded metadata with {len(data.get('manuals', {}))} manuals")
                return data
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}. Creating new.")
        
        return {
            "manuals": {},           # car_model_key -> manual info
            "model_embeddings": {},  # car_model_key -> embedding
            "version": "1.0"
        }
    
    def save(self):
        data_to_save = self.data.copy()
        
        # Convert numpy arrays to lists for JSON serialization
        data_to_save["model_embeddings"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in data_to_save["model_embeddings"].items()
        }
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps(data_to_save, indent=2))
        logger.debug("Saved metadata")
    
    def add_manual(
        self, 
        model: str, 
        filename: str, 
        brand: Optional[str] = None, 
        year: Optional[str] = None,
        chunk_count: int = 0
    ) -> str:

        # Generate car_model_key
        car_model_key = self._create_car_model_key(brand, model, year)
        display_name = self._create_display_name(brand, model, year)
        
        # Store manual info
        self.data["manuals"][car_model_key] = {
            "brand": brand,
            "model": model,
            "year": year,
            "filename": filename,
            "display_name": display_name,
            "chunk_count": chunk_count,
            "added_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        model_identifier = self._create_search_text(brand, model, year)
        model_embedding = self.embedder.embed_query(model_identifier)
        self.data["model_embeddings"][car_model_key] = model_embedding
        
        self.save()
        logger.info(f"Added manual: {display_name} ({car_model_key})")
        
        return car_model_key
    
    def find_matching_car_model(
        self, 
        query: str, 
        threshold: float = 0.65
    ) -> Optional[Dict]:

        if not self.data["model_embeddings"]:
            return None
        
        query_embedding = self.embedder.embed_query(query)
        
        best_match = None
        best_score = 0
        
        for car_model_key, stored_embedding in self.data["model_embeddings"].items():
            if isinstance(stored_embedding, list):
                stored_embedding = np.array(stored_embedding)
            
            # Cosine similarity (embeddings are normalized)
            similarity = float(np.dot(query_embedding, stored_embedding))
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = self.data["manuals"][car_model_key].copy()
                best_match["car_model_key"] = car_model_key
                best_match["similarity_score"] = similarity
        
        if best_match:
            logger.debug(
                f"Matched '{query}' to {best_match['display_name']} "
                f"(score: {best_score:.3f})"
            )
        else:
            logger.debug(f"No match found for '{query}' above threshold {threshold}")
        
        return best_match
    
    def list_available(self) -> List[Tuple[Optional[str], str, Optional[str]]]:

        return [
            (info["brand"], info["model"], info["year"])
            for info in self.data["manuals"].values()
        ]
    
    def list_all_manuals(self) -> List[Dict]:

        return [
            {**info, "car_model_key": key}
            for key, info in self.data["manuals"].items()
        ]
    
    def _create_car_model_key(
        self, 
        brand: Optional[str], 
        model: str, 
        year: Optional[str]
    ) -> str:

        brand_part = brand.lower().replace(" ", "_") if brand else "unknown"
        model_part = model.lower().replace(" ", "_")
        year_part = year if year else "unknown"
        return f"{brand_part}_{model_part}_{year_part}"
    
    def _create_display_name(
        self, 
        brand: Optional[str], 
        model: str, 
        year: Optional[str]
    ) -> str:

        parts = []
        if brand:
            parts.append(brand)
        parts.append(model)
        if year:
            parts.append(f"({year})")
        return " ".join(parts)
    
    def _create_search_text(
        self, 
        brand: Optional[str], 
        model: str, 
        year: Optional[str]
    ) -> str:

        parts = []
        if brand:
            parts.append(brand)
        parts.append(model)
        if year:
            parts.append(year)
        parts.append("car manual")
        return " ".join(parts)
    
    @property
    def manual_count(self) -> int:

        return len(self.data["manuals"])