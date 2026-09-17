import pytest
from src.intelligence.query_classifier import QueryClassifier
from src.intelligence.query_decomposer import QueryDecomposer
from src.intelligence.confidence_evaluator import ConfidenceEvaluator
from src.intelligence.query_rewriter import QueryRewriter

def test_query_classifier():
    classifier = QueryClassifier()
    
    res_visual = classifier.classify_query("Show me the chart on page 3")
    assert res_visual.query_type == "VISUAL"
    assert "colpali" in res_visual.selected_retrievers

    res_table = classifier.classify_query("What was the balance sheet revenue in 2024?")
    assert res_table.query_type == "TABLE"

    res_comp = classifier.classify_query("Compare revenue between 2023 and 2025")
    assert res_comp.query_type == "COMPARATIVE"

def test_query_decomposer():
    decomposer = QueryDecomposer()
    res = decomposer.decompose("Compare revenue and employee growth between 2023 and 2025.")
    
    assert res.is_decomposed is True
    assert len(res.sub_questions) >= 2

def test_confidence_evaluator():
    evaluator = ConfidenceEvaluator()
    from src.ingestion.metadata import RetrievalResult
    
    results = [
        RetrievalResult(document_id="d1", document_name="doc.pdf", page=1, chunk_id="c1", text="Revenue is $50M", score=0.90, retrieval_method="dense")
    ]
    conf = evaluator.evaluate_confidence("What is the revenue?", results, ["dense"])
    assert conf.confidence_level in ["HIGH", "MEDIUM", "LOW"]
    assert conf.confidence_score > 0.0
