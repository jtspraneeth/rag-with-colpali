import re
from typing import Dict, Any, List
from pydantic import BaseModel

class QueryClassificationResult(BaseModel):
    query_type: str  # FACTUAL, SEMANTIC, VISUAL, TABLE, COMPARATIVE, MULTI_HOP, TEMPORAL
    confidence: float
    selected_retrievers: List[str]  # ["dense", "bm25", "colpali"]
    routing_reason: str

DEFAULT_ROUTING_POLICIES = {
    "FACTUAL": ["dense", "bm25"],
    "SEMANTIC": ["dense", "bm25"],
    "VISUAL": ["colpali"],
    "TABLE": ["colpali", "dense"],
    "COMPARATIVE": ["dense", "bm25"],
    "MULTI_HOP": ["dense", "bm25"],
    "TEMPORAL": ["dense", "bm25"]
}

class QueryClassifier:
    """Classifies user queries into intent categories and dynamically assigns retriever routing policies."""
    def __init__(self, routing_policies: Dict[str, List[str]] = None):
        self.routing_policies = routing_policies or DEFAULT_ROUTING_POLICIES

    def classify_query(self, query: str) -> QueryClassificationResult:
        q_lower = query.lower()

        # 1. Visual Intent
        if any(w in q_lower for w in ["picture", "chart", "diagram", "figure", "layout", "visual", "image", "graph", "plot"]):
            return QueryClassificationResult(
                query_type="VISUAL",
                confidence=0.92,
                selected_retrievers=self.routing_policies["VISUAL"],
                routing_reason="Detected explicit visual layout / diagram keywords."
            )

        # 2. Comparative Intent (evaluated before TABLE so 'compare revenue' is COMPARATIVE)
        if any(w in q_lower for w in ["compare", "versus", "vs", "difference between", "growth from", "compared to"]):
            return QueryClassificationResult(
                query_type="COMPARATIVE",
                confidence=0.88,
                selected_retrievers=self.routing_policies["COMPARATIVE"],
                routing_reason="Comparative question requires query decomposition across entities."
            )

        # 3. Table / Financial Table Intent
        if any(w in q_lower for w in ["table", "revenue", "balance sheet", "income statement", "row", "column", "percentage", "margin", "ebitda", "$", "%"]):
            return QueryClassificationResult(
                query_type="TABLE",
                confidence=0.89,
                selected_retrievers=self.routing_policies["TABLE"],
                routing_reason="Detected table/financial structure terms requiring visual & dense search."
            )

        # 4. Temporal Intent
        if re.search(r'\b(in \d{4}|fy\d{4}|latest|since|between \d{4} and \d{4}|quarter|q[1-4])\b', q_lower):
            return QueryClassificationResult(
                query_type="TEMPORAL",
                confidence=0.87,
                selected_retrievers=self.routing_policies["TEMPORAL"],
                routing_reason="Detected date or fiscal period specification requiring temporal filtering."
            )

        # 5. Multi-hop / Reasoning Intent
        if any(w in q_lower for w in ["why did", "how does", "what caused", "explain the relationship"]):
            return QueryClassificationResult(
                query_type="MULTI_HOP",
                confidence=0.82,
                selected_retrievers=self.routing_policies["MULTI_HOP"],
                routing_reason="Multi-step reasoning question requiring iterative evidence gathering."
            )

        # 6. Default Factual / Semantic
        if len(query.split()) < 6 or any(w in q_lower for w in ["who", "what", "where", "when", "which", "name", "id"]):
            return QueryClassificationResult(
                query_type="FACTUAL",
                confidence=0.85,
                selected_retrievers=self.routing_policies["FACTUAL"],
                routing_reason="Direct factual search routed to Dense + BM25 hybrid retrievers."
            )

        return QueryClassificationResult(
            query_type="SEMANTIC",
            confidence=0.80,
            selected_retrievers=self.routing_policies["SEMANTIC"],
            routing_reason="Semantic concept query routed to Dense + BM25 hybrid search."
        )
