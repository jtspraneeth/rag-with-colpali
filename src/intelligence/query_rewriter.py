from typing import Dict, Any, List
from pydantic import BaseModel

class QueryRewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    expanded_queries: List[str]
    reason_for_rewrite: str

class QueryRewriter:
    """Normalizes and expands low-confidence or difficult queries for corrective retrieval."""
    
    def rewrite_query(self, query: str, attempt: int = 1, failure_reason: str = "") -> QueryRewriteResult:
        q_clean = query.strip()
        
        # Strategy 1: Expand acronyms & financial terms
        expansions = []
        rewritten = q_clean

        if "rev" in q_clean.lower() and "revenue" not in q_clean.lower():
            rewritten = rewritten.replace("rev", "revenue")
            expansions.append("financial revenue growth sales income")
            
        if "yoy" in q_clean.lower():
            rewritten = rewritten.replace("yoy", "year over year")
            expansions.append("year over year growth comparison")

        if attempt == 1:
            reason = f"Initial low confidence: {failure_reason}. Expanded query terms."
            expanded = [rewritten, f"{rewritten} overview summary report", f"{rewritten} data facts"]
        else:
            reason = f"Retry attempt {attempt}: Relaxing specific constraints to broaden search recall."
            # Remove filler words for broader matching
            words = [w for w in q_clean.split() if w.lower() not in ["what", "was", "the", "is", "of", "in", "for", "a", "an"]]
            rewritten = " ".join(words)
            expanded = [rewritten, f"{rewritten} details"]

        return QueryRewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            expanded_queries=expanded,
            reason_for_rewrite=reason
        )
