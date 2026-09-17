import uuid
from pathlib import Path
from typing import List, Optional
from src.ingestion.metadata import DocumentChunk
from src.ingestion.chunker import StructureAwareChunker

class TXTParser:
    """Parser for plain text files."""
    def __init__(self, chunker: StructureAwareChunker = None):
        self.chunker = chunker or StructureAwareChunker()

    def parse_txt(self, file_path: str, document_id: Optional[str] = None) -> List[DocumentChunk]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"TXT file not found: {file_path}")

        document_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        document_name = path.name

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        chunks = self.chunker.split_text(
            text=content,
            document_id=document_id,
            document_name=document_name,
            page_number=1,
            source_type="txt"
        )
        return chunks
