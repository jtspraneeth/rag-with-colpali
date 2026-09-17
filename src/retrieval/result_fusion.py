from typing import List, Dict, Any
from src.ingestion.metadata import RetrievalResult
from config import settings

class ResultFusion:
    """Result fusion layer supporting Reciprocal Rank Fusion (RRF) 

    and Weighted Linear Score Combination across Dense, BM25, and ColPali retrievers.

    """
    def __init__(self, alpha: float = None, beta: float = None, gamma: float = None):
        self.alpha = alpha if alpha is not None else settings.dense_weight
        self.beta = beta if beta is not None else settings.bm25_weight
        self.gamma = gamma if gamma is not None else settings.colpali_weight

    def fuse_results(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        colpali_results: List[RetrievalResult],
        strategy: str = "weighted_linear",
        rrf_k: int = 60
    ) -> List[RetrievalResult]:
        """Combines multiple retriever outputs using chosen strategy."""
        
        if strategy == "rrf":
            return self._reciprocal_rank_fusion(dense_results, bm25_results, colpali_results, k=rrf_k)
        
        return self._weighted_linear_fusion(dense_results, bm25_results, colpali_results)

    def _weighted_linear_fusion(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        colpali_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Weighted linear score combination: alpha*dense + beta*bm25 + gamma*colpali."""
        chunk_map: Dict[str, RetrievalResult] = {}
        scores_map: Dict[str, Dict[str, float]] = {}

        def process_list(res_list: List[RetrievalResult], method_key: str):
            for item in res_list:
                cid = item.chunk_id
                if cid not in chunk_map:
                    chunk_map[cid] = item
                    scores_map[cid] = {"dense": 0.0, "bm25": 0.0, "colpali": 0.0}
                scores_map[cid][method_key] = item.score

        process_list(dense_results, "dense")
        process_list(bm25_results, "bm25")
        process_list(colpali_results, "colpali")

        fused_list: List[RetrievalResult] = []
        for cid, item in chunk_map.items():
            dense_s = scores_map[cid]["dense"]
            bm25_s = scores_map[cid]["bm25"]
            colpali_s = scores_map[cid]["colpali"]

            combined_score = (
                self.alpha * dense_s +
                self.beta * bm25_s +
                self.gamma * colpali_s
            )

            res_dict = item.model_dump()
            res_dict["score"] = round(float(combined_score), 4)
            res_dict["retrieval_method"] = "fused_weighted"
            res_dict["metadata"]["dense_score"] = dense_s
            res_dict["metadata"]["bm25_score"] = bm25_s
            res_dict["metadata"]["colpali_score"] = colpali_s

            fused_list.append(RetrievalResult(**res_dict))

        fused_list.sort(key=lambda x: x.score, reverse=True)
        return fused_list

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        colpali_results: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF) algorithm."""
        chunk_map: Dict[str, RetrievalResult] = {}
        rrf_scores: Dict[str, float] = {}

        def add_ranks(res_list: List[RetrievalResult]):
            for rank, item in enumerate(res_list, 1):
                cid = item.chunk_id
                if cid not in chunk_map:
                    chunk_map[cid] = item
                    rrf_scores[cid] = 0.0
                rrf_scores[cid] += 1.0 / (k + rank)

        add_ranks(dense_results)
        add_ranks(bm25_results)
        add_ranks(colpali_results)

        # Normalize RRF scores to [0, 1]
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

        fused_list: List[RetrievalResult] = []
        for cid, item in chunk_map.items():
            norm_score = rrf_scores[cid] / max_rrf if max_rrf > 0 else 0.0
            
            res_dict = item.model_dump()
            res_dict["score"] = round(float(norm_score), 4)
            res_dict["retrieval_method"] = "fused_rrf"
            res_dict["metadata"]["rrf_score"] = rrf_scores[cid]

            fused_list.append(RetrievalResult(**res_dict))

        fused_list.sort(key=lambda x: x.score, reverse=True)
        return fused_list
