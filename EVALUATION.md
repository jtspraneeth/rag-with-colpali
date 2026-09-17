# Evaluation Methodology & Metrics Guide

## Overview

The evaluation framework provides quantitative, reproducible benchmarking comparing three distinct RAG configurations:
1. **MODE 1**: Basic Vector RAG (Dense Vector search + Direct LLM generation)
2. **MODE 2**: Hybrid RAG (Dense Vector + BM25 Lexical + Cross-Encoder Reranking)
3. **MODE 3**: Adaptive Multimodal RAG (Intent Classification + ColPali Visual Retrieval + Fusion + Reranking + Self-Correction + Claim Verification)

---

## Metric Definitions

### 1. Retrieval Quality Metrics
- **Recall@K**: The fraction of ground-truth relevant document chunks successfully retrieved within the top-K candidates.
  $$ \text{Recall@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|} $$
- **Precision@K**: The fraction of retrieved top-K candidates that are relevant.
  $$ \text{Precision@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K} $$
- **Mean Reciprocal Rank (MRR)**: Evaluates the rank position of the first relevant retrieved document.
  $$ \text{MRR} = \frac{1}{\text{rank}_1} $$
- **NDCG@K**: Normalized Discounted Cumulative Gain accounting for position-dependent relevance.

### 2. Trustworthiness & Faithfulness Metrics
- **Citation Accuracy**: Percentage of generated factual claims classified as `SUPPORTED` or `PARTIALLY_SUPPORTED` by retrieved source chunks.
- **Unsupported Claim Rate**: Proportion of generated claims lacking supporting source evidence (`UNSUPPORTED`).
- **Contradiction Rate**: Proportion of claims directly contradicted by source evidence (`CONTRADICTED`).

### 3. Performance Metrics
- **Retrieval Latency (sec)**: Time spent in vector, lexical, visual search, fusion, and reranking.
- **Generation Latency (sec)**: LLM text completion latency.
- **Total Latency (sec)**: End-to-end processing time per query.

---

## Running Benchmarks

### Via Streamlit UI
Navigate to the **EVALUATION** tab and click **Run Live Benchmark Evaluation Suite**.

### Via Python API
```python
from src.adaptive_rag_pipeline import AdaptiveMultimodalRAG
from src.evaluation.benchmark_runner import BenchmarkRunner

rag = AdaptiveMultimodalRAG()
runner = BenchmarkRunner(rag)

test_queries = [
    {"query": "What was the revenue growth in 2025?", "relevant_doc_ids": ["doc_001"]},
    {"query": "Compare revenue and employee growth between 2023 and 2025.", "relevant_doc_ids": ["doc_001"]}
]

summary = runner.run_benchmark(test_queries)
print(summary)
```
