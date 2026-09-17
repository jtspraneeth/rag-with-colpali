from typing import List, Tuple
from src.ingestion.metadata import RetrievalResult
from config import settings

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class CandidateReranker:
    """Candidate reranker model using CrossEncoder or FlashRank for semantic re-scoring."""
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model_name
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            if CrossEncoder is not None:
                try:
                    self._reranker = CrossEncoder(self.model_name)
                except Exception as e:
                    print(f"Warning: Failed to load CrossEncoder ({e}). Using heuristic reranker.")
                    self._reranker = "heuristic"
            else:
                self._reranker = "heuristic"
        return self._reranker

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """Reranks initial candidate results using Cross-Encoder semantic score."""
        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates[:top_k]

        reranked_results: List[RetrievalResult] = []

        if self.reranker != "heuristic" and hasattr(self.reranker, "predict"):
            pairs = [[query, c.text] for c in candidates]
            scores = self.reranker.predict(pairs)

            # Min-max normalize scores to [0, 1]
            max_s = max(scores)
            min_s = min(scores)
            
            for candidate, raw_score in zip(candidates, scores):
                norm_score = (raw_score - min_s) / (max_s - min_s) if max_s > min_s else 1.0
                
                # Copy candidate result with updated score and method
                c_dict = candidate.model_dump()
                c_dict["score"] = round(float(norm_score), 4)
                c_dict["retrieval_method"] = "reranked"
                c_dict["metadata"]["initial_score"] = candidate.score
                c_dict["metadata"]["initial_method"] = candidate.retrieval_method
                
                reranked_results.append(RetrievalResult(**c_dict))
        else:
            # Fallback heuristic reranking: boost items containing exact query term matches
            query_words = set(query.lower().split())
            for candidate in candidates:
                matched_count = sum(1 for w in query_words if w in candidate.text.lower())
                boost = 0.1 * matched_count
                new_score = min(1.0, candidate.score + boost)

                c_dict = candidate.model_dump()
                c_dict["score"] = round(float(new_score), 4)
                c_dict["retrieval_method"] = "reranked_heuristic"
                c_dict["metadata"]["initial_score"] = candidate.score
                c_dict["metadata"]["initial_method"] = candidate.retrieval_method
                
                reranked_results.append(RetrievalResult(**c_dict))

        # Sort descending by updated score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results[:top_k]
