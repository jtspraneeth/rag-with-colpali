import time
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from src.adaptive_rag_pipeline import AdaptiveMultimodalRAG
from src.evaluation.metrics import (
    compute_recall_at_k, compute_precision_at_k, compute_mrr, compute_ndcg_at_k, compute_trustworthiness_metrics
)

class BenchmarkRunner:
    """Runs reproducible evaluation comparing Basic RAG, Hybrid RAG, 
    and Adaptive Multimodal RAG across retrieval accuracy, trustworthiness, and latency.
    """
    def __init__(self, rag_system: Optional[AdaptiveMultimodalRAG] = None):
        self.rag = rag_system or AdaptiveMultimodalRAG()

    def run_benchmark(
        self, 
        test_queries: List[Dict[str, Any]], 
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Dict[str, Any]]:
        """Runs test query set against Mode 1, Mode 2, and Mode 3 and returns comparative metrics."""
        
        # Ensure at least one sample document is indexed if empty
        if not self.rag.indexed_documents:
            sample_txt = Path("data/documents/sample_report.txt")
            if not sample_txt.exists():
                sample_txt.parent.mkdir(parents=True, exist_ok=True)
                sample_txt.write_text(
                    "Annual Financial Report FY2024\n\n"
                    "Executive Summary: Revenue for FY2024 reached $150 million, representing a 20% increase year-over-year. "
                    "Total worldwide headcount expanded to 600 employees. The company opened offices in London and Tokyo.",
                    encoding="utf-8"
                )
            self.rag.ingest_document(str(sample_txt))

        modes = [
            ("basic_vector", "Mode 1: Basic Vector RAG"),
            ("hybrid", "Mode 2: Hybrid RAG (Dense + BM25)"),
            ("adaptive_multimodal", "Mode 3: Adaptive Multimodal RAG")
        ]

        results = {m_key: {"recall": [], "precision": [], "mrr": [], "citation_accuracy": [], "latency": []} for m_key, _ in modes}

        total_steps = max(1, len(test_queries) * len(modes))
        current_step = 0

        for q_idx, item in enumerate(test_queries):
            q_text = item["query"]
            rel_ids = item.get("relevant_doc_ids", [])
            if not rel_ids:
                rel_ids = [d.get("document_name", d.get("document_id", "*")) for d in self.rag.indexed_documents]

            for m_key, m_name in modes:
                current_step += 1
                if progress_callback:
                    pct = min(1.0, current_step / total_steps)
                    progress_callback(f"Evaluating {m_name} on Query {q_idx+1}/{len(test_queries)}...", pct)

                res = self.rag.query(q_text, mode=m_key)
                retrieved = res.get("retrieved_results", [])
                claims = res.get("verified_claims", [])
                latency = res.get("trace", {}).get("latencies", {}).get("total_sec", 0.0)

                results[m_key]["recall"].append(compute_recall_at_k(retrieved, rel_ids))
                results[m_key]["precision"].append(compute_precision_at_k(retrieved, rel_ids))
                results[m_key]["mrr"].append(compute_mrr(retrieved, rel_ids))
                results[m_key]["citation_accuracy"].append(compute_trustworthiness_metrics(claims)["citation_accuracy"])
                results[m_key]["latency"].append(latency)

        # Build clean formatted list of dicts for UI display
        summary_rows = []
        mode_labels = {
            "basic_vector": "Mode 1: Basic Vector RAG",
            "hybrid": "Mode 2: Hybrid RAG (Dense + BM25)",
            "adaptive_multimodal": "Mode 3: Adaptive Multimodal RAG (Proposed)"
        }

        for m_key, label in mode_labels.items():
            m_data = results[m_key]
            n = max(1, len(m_data["recall"]))
            summary_rows.append({
                "Pipeline Mode": label,
                "Recall@5": round(sum(m_data["recall"]) / n, 4),
                "Precision@5": round(sum(m_data["precision"]) / n, 4),
                "MRR": round(sum(m_data["mrr"]) / n, 4),
                "Citation Accuracy": f"{round((sum(m_data['citation_accuracy']) / n) * 100, 1)}%",
                "Avg Latency (sec)": round(sum(m_data["latency"]) / n, 3)
            })

        return summary_rows
