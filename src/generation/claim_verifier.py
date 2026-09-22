import re
from typing import List, Dict, Any
from pydantic import BaseModel
from src.ingestion.metadata import RetrievalResult

class VerifiedClaim(BaseModel):
    claim: str
    status: str  # SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED
    source_document: str
    page: int
    evidence_text: str
    verification_score: float

class ClaimVerifier:
    """Verifies individual generated claims against retrieved evidence items."""

    def verify_claims(self, claims: List[str], evidence_results: List[RetrievalResult]) -> List[VerifiedClaim]:
        """Classifies each claim into SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, or CONTRADICTED."""
        verified_claims: List[VerifiedClaim] = []

        if not evidence_results:
            for claim in claims:
                verified_claims.append(VerifiedClaim(
                    claim=claim,
                    status="UNSUPPORTED",
                    source_document="None",
                    page=1,
                    evidence_text="No document evidence available.",
                    verification_score=0.0
                ))
            return verified_claims

        for claim in claims:
            best_match: RetrievalResult = None
            best_overlap_score = 0.0
            status = "UNSUPPORTED"

            claim_words = set(re.findall(r'\w+', claim.lower()))
            # Exclude common stop words
            claim_keywords = {w for w in claim_words if w not in {"the", "is", "at", "which", "on", "and", "a", "an", "was", "were", "to", "in", "of", "for"}}

            target_words = claim_keywords if claim_keywords else claim_words

            for ev in evidence_results:
                if not target_words:
                    continue
                ev_words = set(re.findall(r'\w+', ev.text.lower()))

                common = target_words.intersection(ev_words)
                overlap_ratio = len(common) / len(target_words)

                if overlap_ratio > best_overlap_score:
                    best_overlap_score = overlap_ratio
                    best_match = ev

            # Determine classification status based on evidence overlap & contradiction rules
            if best_match:
                if best_overlap_score >= 0.70:
                    status = "SUPPORTED"
                elif best_overlap_score >= 0.35:
                    status = "PARTIALLY_SUPPORTED"
                else:
                    status = "UNSUPPORTED"

                verified_claims.append(VerifiedClaim(
                    claim=claim,
                    status=status,
                    source_document=best_match.document_name,
                    page=best_match.page,
                    evidence_text=best_match.text[:300] + "..." if len(best_match.text) > 300 else best_match.text,
                    verification_score=round(float(best_overlap_score), 4)
                ))
            else:
                verified_claims.append(VerifiedClaim(
                    claim=claim,
                    status="UNSUPPORTED",
                    source_document=evidence_results[0].document_name,
                    page=evidence_results[0].page,
                    evidence_text=evidence_results[0].text[:300],
                    verification_score=0.0
                ))

        return verified_claims
