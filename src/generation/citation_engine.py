from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from src.generation.claim_verifier import VerifiedClaim

class CitationRecord(BaseModel):
    citation_id: int
    claim_text: str
    document_name: str
    page_number: int
    section: str
    chunk_id: str
    retrieval_method: str
    evidence_snippet: str
    image_path: str = ""

class CitationEngine:
    """Constructs claim-level and sentence-level citation records linking answers to source documents and page images."""

    def build_citations(self, verified_claims: List[VerifiedClaim]) -> Tuple[List[CitationRecord], str]:
        """Generates structured citation records and formatted citation text block."""
        citation_records: List[CitationRecord] = []
        citation_lines = []

        for idx, claim in enumerate(verified_claims, 1):
            rec = CitationRecord(
                citation_id=idx,
                claim_text=claim.claim,
                document_name=claim.source_document,
                page_number=claim.page,
                section="N/A",
                chunk_id=f"{claim.source_document}_p{claim.page}",
                retrieval_method="verified_evidence",
                evidence_snippet=claim.evidence_text
            )
            citation_records.append(rec)
            citation_lines.append(f"[{idx}] {claim.source_document} — Page {claim.page} (Status: {claim.status})")

        formatted_citations = "\n".join(citation_lines)
        return citation_records, formatted_citations
