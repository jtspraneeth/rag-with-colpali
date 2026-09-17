from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    """Represents a chunk extracted from a document with rich metadata."""
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int = 1
    section: Optional[str] = None
    text: str
    source_type: str = "pdf"  # pdf, docx, txt
    document_date: Optional[str] = None
    reporting_period: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    """Common normalized result structure for all retrieval methods (Dense, BM25, ColPali)."""
    document_id: str
    document_name: str
    page: int = 1
    chunk_id: str
    text: str
    score: float
    retrieval_method: str  # "dense", "bm25", "colpali", "fused", "reranked"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    image_path: Optional[str] = None
