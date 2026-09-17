import time
from typing import List, Dict, Any
from src.adaptive_rag_pipeline import AdaptiveMultimodalRAG
from src.evaluation.metrics import (
    compute_recall_at_k, compute_precision_at_k, compute_mrr, compute_ndcg_at_k, compute_trustworthiness_metrics
)

class BenchmarkRunner:
    """Runs reproducible evaluation comparing Basic RAG, Hybrid RAG, 

    and Adaptive Multimodal RAG across retrieval accuracy, trustworthiness, and latency.

    """
    def __init__(self, rag_system: AdaptiveMultimodalRAG = None):
        self.rag = rag_system or AdaptiveMultimodalRAG()

    def run_benchmark(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs test query set against Mode 1, Mode 2, and Mode 3 and returns comparative metrics."""
        results = {
            "mode_1_basic": {"recall": [], "precision": [], "mrr": [], "citation_accuracy": [], "latency": []},
            "mode_2_hybrid": {"recall": [], "precision": [], "mrr": [], "citation_accuracy": [], "latency": []},
            "mode_3_adaptive": {"recall": [], "precision": [], "mrr": [], "citation_accuracy": [], "latency": []}
        }

        for item in test_queries:
            q_text = item["query"]
            rel_ids = item.get("relevant_doc_ids", [])

            # Mode 1: Basic Vector RAG
            res_m1 = self.rag.query(q_text, mode="basic_vector")
            r_ids_m1 = [r["chunk_id"] for r in res_m1["retrieved_results"]]
            results["mode_1_basic"]["recall"].append(compute_recall_at_k(r_ids_m1, rel_ids))
            results["mode_1_basic"]["precision"].append(compute_precision_at_k(r_ids_m1, rel_ids))
            results["mode_1_basic"]["mrr"].append(compute_mrr(r_ids_m1, rel_ids))
            results["mode_1_basic"]["citation_accuracy"].append(compute_trustworthiness_metrics(res_m1["verified_claims"])["citation_accuracy"])
            results["mode_1_basic"]["latency"].append(res_m1["trace"]["latencies"]["total_sec"])

            # Mode 2: Hybrid RAG
            res_m2 = self.rag.query(q_text, mode="hybrid")
            r_ids_m2 = [r["chunk_id"] for r in res_m2["retrieved_results"]]
            results["mode_2_hybrid"]["recall"].append(compute_recall_at_k(r_ids_m2, rel_ids))
            results["mode_2_hybrid"]["precision"].append(compute_precision_at_k(r_ids_m2, rel_ids))
            results["mode_2_hybrid"]["mrr"].append(compute_mrr(r_ids_m2, rel_ids))
            results["mode_2_hybrid"]["citation_accuracy"].append(compute_trustworthiness_metrics(res_m2["verified_claims"])["citation_accuracy"])
            results["mode_2_hybrid"]["latency"].append(res_m2["trace"]["latencies"]["total_sec"])

            # Mode 3: Adaptive Multimodal RAG
            res_m3 = self.rag.query(q_text, mode="adaptive_multimodal")
            r_ids_m3 = [r["chunk_id"] for r in res_m3["retrieved_results"]]
            results["mode_3_adaptive"]["recall"].append(compute_recall_at_k(r_ids_m3, rel_ids))
            results["mode_3_adaptive"]["precision"].append(compute_precision_at_k(r_ids_m3, rel_ids))
            results["mode_3_adaptive"]["mrr"].append(compute_mrr(r_ids_m3, rel_ids))
            results["mode_3_adaptive"]["citation_accuracy"].append(compute_trustworthiness_metrics(res_m3["verified_claims"])["citation_accuracy"])
            results["mode_3_adaptive"]["latency"].append(res_m3["trace"]["latencies"]["total_sec"])

        # Compute averages
        summary = {}
        for m_key, m_data in results.items():
            summary[m_key] = {
                "avg_recall_at_k": round(sum(m_data["recall"]) / max(1, len(m_data["recall"])), 4),
                "avg_precision_at_k": round(sum(m_data["precision"]) / max(1, len(m_data["precision"])), 4),
                "avg_mrr": round(sum(m_data["mrr"]) / max(1, len(m_data["mrr"])), 4),
                "avg_citation_accuracy": round(sum(m_data["citation_accuracy"]) / max(1, len(m_data["citation_accuracy"])), 4),
                "avg_latency_sec": round(sum(m_data["latency"]) / max(1, len(m_data["latency"])), 3)
            }

        return summary
