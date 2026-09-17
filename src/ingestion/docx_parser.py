import uuid
from pathlib import Path
from typing import List, Tuple, Optional
import docx

from src.ingestion.metadata import DocumentChunk
from src.ingestion.chunker import StructureAwareChunker

class DOCXParser:
    """Parser for DOCX documents preserving headings, paragraphs, and document metadata."""
    def __init__(self, chunker: StructureAwareChunker = None):
        self.chunker = chunker or StructureAwareChunker()

    def parse_docx(self, file_path: str, document_id: Optional[str] = None) -> List[DocumentChunk]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        document_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        document_name = path.name

        doc = docx.Document(str(path))
        full_text = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        text_content = "\n\n".join(full_text)
        
        chunks = self.chunker.split_text(
            text=text_content,
            document_id=document_id,
            document_name=document_name,
            page_number=1,
            source_type="docx"
        )
        return chunks
