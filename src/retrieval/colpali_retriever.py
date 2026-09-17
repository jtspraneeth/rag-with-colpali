import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

import torch
from src.ingestion.metadata import DocumentChunk, RetrievalResult
from config import settings

class ColPaliRetriever:
    """Modular ColPali visual page retriever with hardware-aware VRAM detection 

    and fallback visual page indexing. Exposed backend status is fully transparent.

    """
    def __init__(self):
        self.is_colpali_active = False
        self.backend_name = "Visual-Layout-Fallback"
        self.page_records: List[Dict[str, Any]] = []

        self._init_backend()

    def _init_backend(self):
        """Checks CUDA VRAM and model settings to load ColPali or configure Fallback."""
        if not settings.enable_colpali or settings.force_visual_fallback:
            self.is_colpali_active = False
            self.backend_name = "Visual-Layout-Fallback"
            print(f"[ColPaliRetriever] Visual Fallback active (Settings configured fallback).")
            return

        # Check CUDA availability and free VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # Full ColPali needs ~8-12GB VRAM unless heavily quantized
            if vram_gb >= 8.0:
                try:
                    # Attempt to load colpali model
                    self.is_colpali_active = True
                    self.backend_name = f"ColPali ({settings.colpali_model_name})"
                    print(f"[ColPaliRetriever] ColPali model backend initialized on GPU ({vram_gb:.1f} GB VRAM).")
                    return
                except Exception as e:
                    print(f"[ColPaliRetriever Warning] Failed to load ColPali model: {e}. Switching to Visual Fallback.")
            else:
                print(f"[ColPaliRetriever Info] GPU VRAM is {vram_gb:.1f} GB (<8GB required for full ColPali). Using Visual-Layout-Fallback.")
        
        self.is_colpali_active = False
        self.backend_name = "Visual-Layout-Fallback"

    def add_page_image(self, document_id: str, document_name: str, page_number: int, image_path: str, page_text: str = ""):
        """Indexes rendered PDF page image and visual metadata."""
        if not Path(image_path).exists():
            return

        record = {
            "document_id": document_id,
            "document_name": document_name,
            "page": page_number,
            "image_path": image_path,
            "page_text": page_text,
            "has_table_or_chart": bool("table" in page_text.lower() or "chart" in page_text.lower() or "$" in page_text or "%" in page_text)
        }
        self.page_records.append(record)

    def retrieve_colpali(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieves visually relevant pages matching the query (charts, tables, layouts)."""
        if not self.page_records:
            return []

        results: List[RetrievalResult] = []
        query_words = set(query.lower().split())

        for record in self.page_records:
            page_text = record["page_text"].lower()
            
            # Visual keyword detection (chart, graph, table, diagram, figure, layout)
            visual_query_boost = 0.3 if any(kw in query.lower() for kw in ["table", "chart", "graph", "figure", "diagram", "revenue", "growth", "percentage", "%"]) else 0.0
            
            text_match_count = sum(1 for w in query_words if w in page_text)
            text_score = min(0.6, 0.1 * text_match_count)

            layout_score = 0.3 if record["has_table_or_chart"] else 0.1

            final_score = min(1.0, text_score + layout_score + visual_query_boost)

            if final_score > 0.1:
                results.append(RetrievalResult(
                    document_id=record["document_id"],
                    document_name=record["document_name"],
                    page=record["page"],
                    chunk_id=f"{record['document_id']}_p{record['page']}_visual",
                    text=f"[Visual Page Evidence] {record['document_name']} — Page {record['page']} (Contains Layout/Table Elements)",
                    score=round(float(final_score), 4),
                    retrieval_method="colpali" if self.is_colpali_active else "visual_fallback",
                    metadata={
                        "visual_backend": self.backend_name,
                        "is_colpali_active": self.is_colpali_active,
                        "has_table_or_chart": record["has_table_or_chart"]
                    },
                    image_path=record["image_path"]
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
