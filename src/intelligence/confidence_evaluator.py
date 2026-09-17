from typing import List, Dict, Any
from pydantic import BaseModel
from src.ingestion.metadata import RetrievalResult
from config import settings

class ConfidenceResult(BaseModel):
    confidence_score: float
    confidence_level: str  # HIGH, MEDIUM, LOW
    reason: str

class ConfidenceEvaluator:
    """Evaluates candidate retrieval outputs across score magnitude, score distribution,

    and cross-retriever agreement to compute an overall Retrieval Confidence Score.

    """
    def __init__(self, high_threshold: float = 0.75, low_threshold: float = 0.50):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate_confidence(
        self,
        query: str,
        results: List[RetrievalResult],
        retrievers_used: List[str]
    ) -> ConfidenceResult:
        """Computes retrieval confidence score [0.0 - 1.0] and level (HIGH, MEDIUM, LOW)."""
        if not results:
            return ConfidenceResult(
                confidence_score=0.0,
                confidence_level="LOW",
                reason="No candidate documents retrieved."
            )

        top_score = results[0].score

        # Signal 1: Top Candidate Score
        score_signal = top_score

        # Signal 2: Score distribution gap (Top-1 vs Top-3)
        if len(results) >= 3:
            gap = top_score - results[2].score
            distribution_signal = min(1.0, gap * 2.0)
        else:
            distribution_signal = 0.5

        # Signal 3: Cross-retriever agreement (check if both dense and BM25/colpali produced matches)
        methods_present = set(r.retrieval_method for r in results[:5])
        agreement_signal = 0.9 if len(methods_present) > 1 else 0.7

        # Signal 4: Query term coverage in top result text
        query_terms = set(query.lower().split())
        matched_terms = sum(1 for term in query_terms if term in results[0].text.lower())
        coverage_signal = matched_terms / len(query_terms) if query_terms else 0.5

        # Combined Weighted Confidence Score
        composite_score = (
            0.50 * score_signal +
            0.20 * agreement_signal +
            0.15 * distribution_signal +
            0.15 * coverage_signal
        )

        composite_score = round(float(min(1.0, max(0.0, composite_score))), 4)

        if composite_score >= self.high_threshold:
            level = "HIGH"
            reason = f"Strong evidence match (Top score: {top_score:.2f}) with good retriever agreement."
        elif composite_score >= self.low_threshold:
            level = "MEDIUM"
            reason = f"Moderate retrieval confidence (Top score: {top_score:.2f}). Evidence is partially relevant."
        else:
            level = "LOW"
            reason = f"Low retrieval confidence (Top score: {top_score:.2f}). Evidence quality below threshold."

        return ConfidenceResult(
            confidence_score=composite_score,
            confidence_level=level,
            reason=reason
        )
