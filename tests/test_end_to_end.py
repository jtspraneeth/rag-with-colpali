import pytest
from src.adaptive_rag_pipeline import AdaptiveMultimodalRAG

def test_end_to_end_pipeline(tmp_path):
    # Create sample document
    file_p = tmp_path / "financial_report.txt"
    file_p.write_text(
        "Annual Report FY2024\n\n"
        "Executive Summary: Revenue for FY2024 reached $120 million, representing a 15% increase year-over-year. "
        "Total employee headcount grew to 450 staff members worldwide.",
        encoding="utf-8"
    )

    rag = AdaptiveMultimodalRAG()
    ingest_meta = rag.ingest_document(str(file_p))
    assert ingest_meta["chunks_count"] > 0

    # Test Factual Query
    res_factual = rag.query("What was the revenue for FY2024?", mode="adaptive_multimodal")
    assert "answer" in res_factual
    assert len(res_factual["retrieved_results"]) > 0
    assert res_factual["trace"]["retrieval_confidence"]["confidence_score"] is not None

    # Test Comparative Query
    res_comp = rag.query("Compare revenue and headcount growth between 2023 and 2024", mode="adaptive_multimodal")
    assert res_comp["trace"]["decomposition"]["is_decomposed"] is True
