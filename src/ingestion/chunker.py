import re
import uuid
from typing import List, Dict, Any, Optional
from src.ingestion.metadata import DocumentChunk

class StructureAwareChunker:
    """Structure-aware chunker that splits document text into semantic chunks while preserving 

    page numbers, section headers, and temporal metadata.

    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(
        self,
        text: str,
        document_id: str,
        document_name: str,
        page_number: int = 1,
        section: Optional[str] = None,
        source_type: str = "pdf",
        document_date: Optional[str] = None,
        reporting_period: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Splits raw text into metadata-preserving chunks using semantic paragraph/sentence boundaries."""
        if not text or not text.strip():
            return []

        extra_metadata = extra_metadata or {}
        
        # Split text into paragraphs first
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        chunks: List[DocumentChunk] = []
        current_chunk_text = ""
        current_section = section
        chunk_idx = 0

        for para in paragraphs:
            # Check for header-like line (e.g., # Header, 1. Introduction, SECTION II)
            header_match = re.match(r'^(#+\s+.*|[0-9]+\.\s+[A-Z].*|SECTION\s+[IVXLCDM\d]+.*)', para, re.IGNORECASE)
            if header_match:
                current_section = header_match.group(0).strip("# ").strip()

            if len(current_chunk_text) + len(para) + 1 <= self.chunk_size:
                current_chunk_text += ("\n\n" if current_chunk_text else "") + para
            else:
                if current_chunk_text:
                    chunk_id = f"{document_id}_p{page_number}_c{chunk_idx:03d}"
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                        section=current_section,
                        text=current_chunk_text.strip(),
                        source_type=source_type,
                        document_date=document_date,
                        reporting_period=reporting_period,
                        metadata=extra_metadata
                    ))
                    chunk_idx += 1
                    
                    # Create overlap from end of current_chunk_text
                    overlap_text = current_chunk_text[-self.chunk_overlap:] if len(current_chunk_text) > self.chunk_overlap else ""
                    current_chunk_text = (overlap_text + "\n\n" + para).strip()
                else:
                    # Single paragraph exceeds chunk_size, split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sent in sentences:
                        if len(current_chunk_text) + len(sent) + 1 <= self.chunk_size:
                            current_chunk_text += (" " if current_chunk_text else "") + sent
                        else:
                            if current_chunk_text:
                                chunk_id = f"{document_id}_p{page_number}_c{chunk_idx:03d}"
                                chunks.append(DocumentChunk(
                                    chunk_id=chunk_id,
                                    document_id=document_id,
                                    document_name=document_name,
                                    page_number=page_number,
                                    section=current_section,
                                    text=current_chunk_text.strip(),
                                    source_type=source_type,
                                    document_date=document_date,
                                    reporting_period=reporting_period,
                                    metadata=extra_metadata
                                ))
                                chunk_idx += 1
                            current_chunk_text = sent

        if current_chunk_text and current_chunk_text.strip():
            chunk_id = f"{document_id}_p{page_number}_c{chunk_idx:03d}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                section=current_section,
                text=current_chunk_text.strip(),
                source_type=source_type,
                document_date=document_date,
                reporting_period=reporting_period,
                metadata=extra_metadata
            ))

        return chunks
