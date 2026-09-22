import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.adaptive_rag_pipeline import AdaptiveMultimodalRAG

def test_pipeline():
    print("Initializing Adaptive Multimodal RAG Pipeline...")
    rag = AdaptiveMultimodalRAG()

    # Create temporary text document
    doc_path = Path("data/documents/sample_report.txt")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "Annual Financial Report FY2024\n\n"
        "Executive Summary: Revenue for FY2024 reached $150 million, representing a 20% increase year-over-year. "
        "Total worldwide headcount expanded to 600 employees. The company opened offices in London and Tokyo.",
        encoding="utf-8"
    )

    print(f"Ingesting {doc_path}...")
    info = rag.ingest_document(str(doc_path))
    print(f"Ingestion successful: {info}")

    # Query 1: Factual
    print("\n--- TEST 1: Factual Query ---")
    res1 = rag.query("What was the revenue for FY2024?", mode="adaptive_multimodal")
    print(f"Answer: {res1['answer']}")
    print(f"Citations:\n{res1['formatted_citations']}")
    print(f"Confidence: {res1['trace']['retrieval_confidence']}")

    # Query 2: Comparative / Multi-hop
    print("\n--- TEST 2: Comparative Query ---")
    res2 = rag.query("Compare revenue and headcount growth between 2023 and 2024.", mode="adaptive_multimodal")
    print(f"Decomposition: {res2['trace']['decomposition']['sub_questions']}")
    print(f"Answer: {res2['answer']}")

    print("\nAll pipeline sanity checks passed successfully!")

if __name__ == "__main__":
    test_pipeline()
