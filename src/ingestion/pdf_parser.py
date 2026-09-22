import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

try:
    import pymupdf as fitz  # Recommended PyMuPDF import
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from src.ingestion.metadata import DocumentChunk
from src.ingestion.chunker import StructureAwareChunker
from config import settings

class PDFParser:
    """PDF Document Parser supporting text extraction per page, visual page rendering,

    and temporal metadata extraction.

    """
    def __init__(self, chunker: StructureAwareChunker = None):
        self.chunker = chunker or StructureAwareChunker()

    def parse_pdf(self, file_path: str, document_id: Optional[str] = None) -> Tuple[List[DocumentChunk], List[str]]:
        """Parses PDF file, rendering page images and extracting metadata-preserved chunks.

        Returns (chunks, list_of_page_image_paths).

        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        document_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        document_name = path.name

        chunks: List[DocumentChunk] = []
        page_image_paths: List[str] = []

        # Extract document-level temporal metadata
        full_document_text = ""
        
        if fitz is not None:
            doc = fitz.open(str(path))
            try:
                total_pages = len(doc)

                for page_idx in range(total_pages):
                    page_num = page_idx + 1
                    page = doc.load_page(page_idx)
                    page_text = page.get_text("text") or ""
                    full_document_text += page_text + "\n"

                    # Render page image for visual evidence & ColPali
                    image_filename = f"{document_id}_p{page_num}.png"
                    image_path = settings.page_images_dir / image_filename
                    
                    try:
                        pix = page.get_pixmap(dpi=150)
                        pix.save(str(image_path))
                        page_image_paths.append(str(image_path))
                    except Exception as e:
                        print(f"Warning: Failed to render page {page_num} image: {e}")

                # Extract temporal dates
                doc_date, reporting_period = self._extract_temporal_metadata(full_document_text)

                # Chunk each page text preserving page_number & metadata
                for page_idx in range(total_pages):
                    page_num = page_idx + 1
                    page = doc.load_page(page_idx)
                    page_text = page.get_text("text") or ""

                    if not page_text.strip():
                        continue

                    page_chunks = self.chunker.split_text(
                        text=page_text,
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_num,
                        source_type="pdf",
                        document_date=doc_date,
                        reporting_period=reporting_period,
                        extra_metadata={
                            "page_image_path": str(settings.page_images_dir / f"{document_id}_p{page_num}.png"),
                            "total_pages": total_pages
                        }
                    )
                    chunks.extend(page_chunks)
            finally:
                doc.close()
        else:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF parsing and visual page rendering.")

        return chunks, page_image_paths

    def _extract_temporal_metadata(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts document date and reporting period (e.g. FY2024, Q3 2023, 2022-2025)."""
        doc_date = None
        reporting_period = None

        # Look for Fiscal Year / Reporting Period
        fy_match = re.search(r'\b(FY\s*\d{4}|Q[1-4]\s*\d{4}|Annual Report \d{4}|\d{4}-\d{4})\b', text, re.IGNORECASE)
        if fy_match:
            reporting_period = fy_match.group(0).strip()

        # Look for explicit dates (e.g., January 15, 2024 or 2024-05-12)
        date_match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b', text, re.IGNORECASE)
        if date_match:
            doc_date = date_match.group(0).strip()

        return doc_date, reporting_period
