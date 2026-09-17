import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.ingestion.metadata import DocumentChunk, RetrievalResult

def tokenize(text: str) -> List[str]:
    """Basic alphanumeric tokenization for lexical BM25 matching."""
    return re.findall(r'\w+', text.lower())

class BM25Retriever:
    """Lexical retriever using BM25Okapi with score normalization."""
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Adds document chunks to the BM25 lexical index."""
        if not chunks:
            return
        
        self.chunks.extend(chunks)
        self.corpus_tokens = [tokenize(c.text) for c in self.chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve_bm25(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        """Retrieves top_k relevant document chunks using lexical BM25 search."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        
        # Zip chunks with scores
        scored_chunks = []
        q_set = set(query_tokens)

        for chunk, r_score in zip(self.chunks, raw_scores):
            chunk_tokens = set(tokenize(chunk.text))
            overlap = len(q_set.intersection(chunk_tokens))
            
            # Combine BM25 raw score with keyword overlap signal
            final_score = r_score + (0.5 * overlap)
            if final_score > 0 or overlap > 0:
                scored_chunks.append((chunk, final_score))

        if not scored_chunks:
            # Fallback to returning top available chunk if corpus exists
            scored_chunks = [(c, 0.1) for c in self.chunks[:top_k]]

        # Sort descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_scored = scored_chunks[:top_k]

        max_score = top_scored[0][1] if top_scored else 1.0
        min_score = top_scored[-1][1] if len(top_scored) > 1 else 0.0

        results: List[RetrievalResult] = []
        for chunk, raw_score in top_scored:
            if max_score > min_score:
                norm_score = 0.1 + (0.9 * ((raw_score - min_score) / (max_score - min_score)))
            else:
                norm_score = 0.8

            results.append(RetrievalResult(
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                page=chunk.page_number,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=round(float(norm_score), 4),
                retrieval_method="bm25",
                metadata={
                    "section": chunk.section or "",
                    "source_type": chunk.source_type,
                    "document_date": chunk.document_date or "",
                    "reporting_period": chunk.reporting_period or "",
                    "raw_bm25_score": float(raw_score)
                },
                image_path=chunk.metadata.get("page_image_path")
            ))

        return results
