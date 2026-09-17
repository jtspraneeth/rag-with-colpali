import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.ingestion.metadata import RetrievalResult

class ConflictReport(BaseModel):
    conflict_detected: bool
    description: str
    source_a: str
    source_b: str
    value_a: str
    value_b: str

class ConflictDetector:
    """Detects factual conflicts across retrieved source documents (e.g. differing numerical figures)."""

    def detect_conflicts(self, results: List[RetrievalResult]) -> ConflictReport:
        if len(results) < 2:
            return ConflictReport(
                conflict_detected=False,
                description="Insufficient sources to compare for conflicts.",
                source_a="", source_b="", value_a="", value_b=""
            )

        # Look for numerical patterns like $50M vs $55M or 12% vs 15% across different docs
        extracted_numbers: List[Dict[str, Any]] = []

        for r in results:
            matches = re.findall(r'(\$\s*\d+(?:\.\d+)?\s*(?:million|billion|M|B)?|\b\d+(?:\.\d+)?%)', r.text, re.IGNORECASE)
            for m in matches:
                extracted_numbers.append({
                    "doc": r.document_name,
                    "page": r.page,
                    "period": r.metadata.get("reporting_period", "N/A"),
                    "value": m
                })

        # Compare values from different docs
        for i in range(len(extracted_numbers)):
            for j in range(i + 1, len(extracted_numbers)):
                item_a = extracted_numbers[i]
                item_b = extracted_numbers[j]

                # Check if values differ between different documents for the same period
                if item_a["doc"] != item_b["doc"] and item_a["value"].lower() != item_b["value"].lower():
                    # If periods are identical but values differ -> Conflict!
                    if item_a["period"] == item_b["period"] and item_a["period"] != "N/A":
                        return ConflictReport(
                            conflict_detected=True,
                            description=f"Direct factual disagreement detected for period '{item_a['period']}': {item_a['doc']} reports {item_a['value']} whereas {item_b['doc']} reports {item_b['value']}.",
                            source_a=f"{item_a['doc']} (Page {item_a['page']})",
                            source_b=f"{item_b['doc']} (Page {item_b['page']})",
                            value_a=item_a["value"],
                            value_b=item_b["value"]
                        )

        return ConflictReport(
            conflict_detected=False,
            description="No direct factual conflicts detected across retrieved sources.",
            source_a="", source_b="", value_a="", value_b=""
        )
