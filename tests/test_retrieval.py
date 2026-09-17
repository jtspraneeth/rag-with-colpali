import pytest
from src.ingestion.metadata import DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.result_fusion import ResultFusion
from src.retrieval.reranker import CandidateReranker

def test_bm25_retriever():
    bm25 = BM25Retriever()
    chunks = [
        DocumentChunk(chunk_id="c1", document_id="d1", document_name="doc1.txt", text="The annual revenue of the company increased by 15% in FY2024."),
        DocumentChunk(chunk_id="c2", document_id="d1", document_name="doc1.txt", text="The company opened three new offices in Europe and Asia.")
    ]
    bm25.add_chunks(chunks)

    res = bm25.retrieve_bm25("revenue increased FY2024", top_k=5)
    assert len(res) > 0
    assert res[0].chunk_id == "c1"
    assert res[0].retrieval_method == "bm25"

def test_result_fusion():
    fusion = ResultFusion(alpha=0.5, beta=0.5, gamma=0.0)
    bm25 = BM25Retriever()
    chunks = [DocumentChunk(chunk_id="c1", document_id="d1", document_name="doc1.txt", text="Revenue grew significantly.")]
    bm25.add_chunks(chunks)
    bm25_res = bm25.retrieve_bm25("revenue", top_k=1)

    fused = fusion.fuse_results([], bm25_res, [], strategy="weighted_linear")
    assert len(fused) == 1
    assert fused[0].retrieval_method == "fused_weighted"

def test_reranker():
    reranker = CandidateReranker()
    bm25 = BM25Retriever()
    chunks = [
        DocumentChunk(chunk_id="c1", document_id="d1", document_name="doc1.txt", text="Quarterly revenue reached $50 million."),
        DocumentChunk(chunk_id="c2", document_id="d1", document_name="doc1.txt", text="Weather report for Tuesday.")
    ]
    bm25.add_chunks(chunks)
    candidates = bm25.retrieve_bm25("revenue $50 million", top_k=2)

    reranked = reranker.rerank("revenue $50 million", candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].chunk_id == "c1"
