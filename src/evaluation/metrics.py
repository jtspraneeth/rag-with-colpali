import math
import time
from typing import List, Dict, Any, Union

def _is_match(retrieved_item: Any, relevant_ids: List[str]) -> bool:
    """Checks if a retrieved candidate item matches any target relevant doc or chunk ID."""
    if not relevant_ids:
        return False

    if isinstance(retrieved_item, dict):
        cid = str(retrieved_item.get("chunk_id", "")).lower()
        did = str(retrieved_item.get("document_id", "")).lower()
        dname = str(retrieved_item.get("document_name", "")).lower()
    else:
        cid = str(retrieved_item).lower()
        did = str(retrieved_item).lower()
        dname = str(retrieved_item).lower()

    for rel in relevant_ids:
        rel_norm = str(rel).lower().strip()
        if not rel_norm:
            continue
        if (rel_norm in cid or rel_norm in did or rel_norm in dname or 
            dname in rel_norm or did in rel_norm or cid in rel_norm):
            return True

    return False

def compute_recall_at_k(retrieved_items: List[Any], relevant_ids: List[str], k: int = 5) -> float:
    """Recall@K: proportion of relevant target documents present in top-K retrieved items."""
    if not relevant_ids:
        return 1.0
    top_k_items = retrieved_items[:k]
    
    matched_rel = 0
    for rel in relevant_ids:
        rel_norm = str(rel).lower().strip()
        if not rel_norm:
            continue
        if any(_is_match(item, [rel_norm]) for item in top_k_items):
            matched_rel += 1

    return round(matched_rel / len(relevant_ids), 4)

def compute_precision_at_k(retrieved_items: List[Any], relevant_ids: List[str], k: int = 5) -> float:
    """Precision@K: proportion of top-K retrieved items that are relevant."""
    top_k_items = retrieved_items[:k]
    if not top_k_items:
        return 0.0
    matched_count = sum(1 for item in top_k_items if _is_match(item, relevant_ids))
    return round(matched_count / len(top_k_items), 4)

def compute_mrr(retrieved_items: List[Any], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank (MRR): reciprocal rank of the first relevant retrieved item."""
    for rank, item in enumerate(retrieved_items, 1):
        if _is_match(item, relevant_ids):
            return round(1.0 / rank, 4)
    return 0.0

def compute_ndcg_at_k(retrieved_items: List[Any], relevant_ids: List[str], k: int = 5) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    dcg = 0.0
    for rank, item in enumerate(retrieved_items[:k], 1):
        rel = 1.0 if _is_match(item, relevant_ids) else 0.0
        dcg += rel / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant_ids), k) + 1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0

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

