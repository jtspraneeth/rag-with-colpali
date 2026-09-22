import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseModel):
    # System Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    chroma_persist_dir: Path = BASE_DIR / "data" / "chroma_db"
    documents_dir: Path = BASE_DIR / "data" / "documents"
    page_images_dir: Path = BASE_DIR / "data" / "page_images"

    # LLM Settings
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    mistral_api_key: str = Field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    default_llm_model: str = Field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-flash"))

    # Embedding & Vector DB
    embedding_model_name: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"))

    # Reranker Model
    reranker_model_name: str = Field(default_factory=lambda: os.getenv("RERANKER_MODEL_NAME", "ms-marco-MiniLM-L-6-v2"))

    # Retrieval Weights & Thresholds
    dense_weight: float = Field(default_factory=lambda: float(os.getenv("DENSE_WEIGHT", "0.4")))
    bm25_weight: float = Field(default_factory=lambda: float(os.getenv("BM25_WEIGHT", "0.3")))
    colpali_weight: float = Field(default_factory=lambda: float(os.getenv("COLPALI_WEIGHT", "0.3")))
    
    top_k_candidates: int = Field(default_factory=lambda: int(os.getenv("TOP_K_CANDIDATES", "20")))
    final_top_k: int = Field(default_factory=lambda: int(os.getenv("FINAL_TOP_K", "5")))

    # Confidence & Self-Correction
    confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")))
    max_retries: int = Field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "2")))

    # ColPali Visual Retrieval Settings
    enable_colpali: bool = Field(default_factory=lambda: os.getenv("ENABLE_COLPALI", "true").lower() == "true")
    colpali_model_name: str = Field(default_factory=lambda: os.getenv("COLPALI_MODEL_NAME", "vidore/colpali-v1.2"))
    force_visual_fallback: bool = Field(default_factory=lambda: os.getenv("FORCE_VISUAL_FALLBACK", "false").lower() == "true")

    # Application Modes
    debug_mode: bool = Field(default_factory=lambda: os.getenv("DEBUG_MODE", "true").lower() == "true")

    def ensure_directories(self):
        """Create necessary data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.page_images_dir.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()
settings.ensure_directories()
