import math
import time
from typing import List, Dict, Any

def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Recall@K: proportion of relevant documents present in top-K retrieved items."""
    if not relevant_ids:
        return 1.0
    top_k_retrieved = set(retrieved_ids[:k])
    matched = top_k_retrieved.intersection(set(relevant_ids))
    return len(matched) / len(relevant_ids)

def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Precision@K: proportion of top-K retrieved items that are relevant."""
    if not retrieved_ids or k == 0:
        return 0.0
    top_k_retrieved = retrieved_ids[:k]
    matched = [rid for rid in top_k_retrieved if rid in set(relevant_ids)]
    return len(matched) / len(top_k_retrieved)

def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank (MRR): reciprocal rank of the first relevant retrieved item."""
    rel_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in rel_set:
            return 1.0 / rank
    return 0.0

def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    rel_set = set(relevant_ids)
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], 1):
        rel = 1.0 if rid in rel_set else 0.0
        dcg += rel / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant_ids), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0

def compute_trustworthiness_metrics(verified_claims: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes Citation Accuracy, Unsupported Claim Rate, and Contradiction Rate."""
    if not verified_claims:
        return {"citation_accuracy": 1.0, "unsupported_claim_rate": 0.0, "contradiction_rate": 0.0}

    total = len(verified_claims)
    supported_count = sum(1 for c in verified_claims if c.get("status") in ["SUPPORTED", "PARTIALLY_SUPPORTED"])
    unsupported_count = sum(1 for c in verified_claims if c.get("status") == "UNSUPPORTED")
    contradicted_count = sum(1 for c in verified_claims if c.get("status") == "CONTRADICTED")

    return {
        "citation_accuracy": round(supported_count / total, 4),
        "unsupported_claim_rate": round(unsupported_count / total, 4),
        "contradiction_rate": round(contradicted_count / total, 4)
    }
