from typing import List, Dict, Any, Tuple
from pydantic import BaseModel

from src.ingestion.metadata import RetrievalResult
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.result_fusion import ResultFusion
from src.retrieval.reranker import CandidateReranker
from src.intelligence.confidence_evaluator import ConfidenceEvaluator, ConfidenceResult
from src.intelligence.query_rewriter import QueryRewriter
from config import settings

class CorrectionAttemptTrace(BaseModel):
    attempt: int
    query_used: str
    retrievers_used: List[str]
    candidates_retrieved_count: int
    confidence_score: float
    confidence_level: str
    action_taken: str

class AdaptiveRetrievalPipeline:
    """Orchestrates Adaptive Retrieval, Fusion, Reranking, Confidence Evaluation,

    and Self-Correcting Iterative Retries.

    """
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        colpali_retriever: ColPaliRetriever
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.colpali = colpali_retriever
        self.fusion = ResultFusion()
        self.reranker = CandidateReranker()
        self.confidence_evaluator = ConfidenceEvaluator(high_threshold=settings.confidence_threshold)
        self.rewriter = QueryRewriter()

    def execute_adaptive_retrieval(
        self,
        query: str,
        retriever_keys: List[str],
        top_k: int = 20,
        final_top_k: int = 5
    ) -> Dict[str, Any]:
        """Executes retrieval with automatic self-correcting loop if confidence is below threshold."""
        
        attempt_traces: List[CorrectionAttemptTrace] = []
        current_query = query
        current_retrievers = list(retriever_keys)
        max_retries = settings.max_retries
        
        final_results: List[RetrievalResult] = []
        final_confidence: ConfidenceResult = None

        for attempt in range(1, max_retries + 2):
            dense_res = []
            bm25_res = []
            colpali_res = []

            # Execute selected retrievers
            if "dense" in current_retrievers:
                dense_res = self.dense.retrieve_dense(current_query, top_k=top_k)
            if "bm25" in current_retrievers:
                bm25_res = self.bm25.retrieve_bm25(current_query, top_k=top_k)
            if "colpali" in current_retrievers:
                colpali_res = self.colpali.retrieve_colpali(current_query, top_k=top_k)

            # Fuse results
            fused_candidates = self.fusion.fuse_results(dense_res, bm25_res, colpali_res)

            # Rerank candidates
            reranked = self.reranker.rerank(current_query, fused_candidates, top_k=final_top_k)

            # Evaluate confidence
            conf = self.confidence_evaluator.evaluate_confidence(current_query, reranked, current_retrievers)

            action = "ACCEPTED" if conf.confidence_score >= settings.confidence_threshold or attempt > max_retries else "RETRY_TRIGGERED"

            attempt_traces.append(CorrectionAttemptTrace(
                attempt=attempt,
                query_used=current_query,
                retrievers_used=list(current_retrievers),
                candidates_retrieved_count=len(fused_candidates),
                confidence_score=conf.confidence_score,
                confidence_level=conf.confidence_level,
                action_taken=action
            ))

            final_results = reranked
            final_confidence = conf

            if conf.confidence_score >= settings.confidence_threshold:
                # High confidence achieved!
                break

            if attempt <= max_retries:
                # Trigger Self-Correction
                rewrite_res = self.rewriter.rewrite_query(query, attempt=attempt, failure_reason=conf.reason)
                current_query = rewrite_res.rewritten_query
                # Expand retriever set to all backends for retry
                current_retrievers = list(set(current_retrievers + ["dense", "bm25", "colpali"]))

        return {
            "query": query,
            "final_results": final_results,
            "confidence": final_confidence.model_dump(),
            "attempt_traces": [t.model_dump() for t in attempt_traces],
            "total_attempts": len(attempt_traces),
            "self_corrected": len(attempt_traces) > 1
        }
