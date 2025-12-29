import os
from pathlib import Path
from dataclasses import dataclass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
@dataclass
class Config:

    # Model settings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384  # BGE-small output dimension
    QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
    
    # Reranking settings
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    USE_HYBRID_RERANKING: bool = True
    KEYWORD_BOOST: float = 0.15
    
    # Storage paths
    DATA_DIR: Path = Path("data")
    MANUALS_DIR: Path = DATA_DIR / "manuals"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    VECTORSTORE_DIR: Path = DATA_DIR / "vectorstore"
    METADATA_DB: Path = DATA_DIR / "metadata.json"
    IMAGES_DIR: Path = DATA_DIR / "images"
    
    # Chunking settings
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 200

    # Specific settings for Tables
    TABLE_CHUNK_SIZE: int = 300
    TABLE_CHUNK_OVERLAP: int = 0

    # Table extraction settings
    TABLE_MIN_ACCURACY: float = 60.0
    SUBSTANTIAL_ROW_LENGTH: int = 100
    MIN_CHUNK_LENGTH: int = 50

    # Image extraction settings
    EXTRACT_IMAGES: bool = True
    MIN_IMAGE_SIZE: int = 5000
    MIN_WIDTH: int = 150
    MIN_HEIGHT: int = 150
    MAX_ASPECT_RATIO: float = 4.0
    MIN_VARIANCE_THRESHOLD: float = 250
    HEADER_FOOTER_CUTOFF: float = 0.10
    IMAGE_TAG_MAX_ATTEMPTS: int = 3
    
    # Retrieval settings
    TOP_K_RESULTS: int = 7
    RETRIEVAL_CANDIDATES: int = 60
    MODEL_MATCH_THRESHOLD: float = 0.65
    
    # LLM settings (OpenAI)
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    
    def __post_init__(self):
        self.MANUALS_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        self.IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
