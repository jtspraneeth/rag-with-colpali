# System Architecture Specification

## Overview

**Adaptive Multimodal Evidence-Grounded RAG** is engineered as a decoupled, modular system designed to select optimal retrieval strategies per query intent, evaluate evidence quality, perform self-correcting retries, and verify generated claims against source document chunks and visual page renderings.

---

## Module Breakdown

### 1. Ingestion Layer (`src/ingestion/`)
- `pdf_parser.py`: Parses PDF files per-page using PyMuPDF (`fitz`), extracts page text, and renders page images to `data/page_images/` for visual page evidence display and ColPali visual indexing.
- `docx_parser.py`: Parses Microsoft Word DOCX files preserving paragraph structures and heading sections.
- `txt_parser.py`: Ingests plain text files with metadata tagging.
- `chunker.py`: `StructureAwareChunker` implements semantic paragraph/sentence chunking while preserving `document_id`, `document_name`, `page_number`, `section`, `chunk_id`, `document_date`, and `reporting_period`.
- `metadata.py`: Data models (`DocumentChunk`, `RetrievalResult`).

### 2. Retrieval Layer (`src/retrieval/`)
- `dense_retriever.py`: Vector retriever backed by ChromaDB persistent vector database and SentenceTransformers embeddings (`all-MiniLM-L6-v2`).
- `bm25_retriever.py`: Lexical retriever using `BM25Okapi` with min-max score normalization into `[0.0, 1.0]`.
- `colpali_retriever.py`: ColPali visual page retriever with CUDA VRAM detection and visual page layout fallback engine. Exposes active backend transparently (`is_colpali_active`, `backend_name`).
- `result_fusion.py`: Fuses candidate results using Reciprocal Rank Fusion (RRF) or Weighted Linear Combination (`alpha*dense + beta*bm25 + gamma*colpali`).
- `reranker.py`: Candidate Reranker re-scoring initial top candidates (Top 20 → Top 5) using Cross-Encoder models (`ms-marco-MiniLM-L-6-v2`).
- `temporal_filter.py`: Filters and boosts document chunks based on fiscal year (e.g. FY2024) and date matches.

### 3. Intelligence Layer (`src/intelligence/`)
- `query_classifier.py`: Intent classifier tagging queries as `FACTUAL`, `SEMANTIC`, `VISUAL`, `TABLE`, `COMPARATIVE`, `MULTI_HOP`, `TEMPORAL` and assigning retriever routing policies.
- `query_decomposer.py`: Decomposes comparative or multi-entity questions into sub-questions (e.g. Q1, Q2, Q3, Q4) retrieved independently and synthesized.
- `query_rewriter.py`: Normalizes query terms and produces expanded search variations for corrective retrieval.
- `confidence_evaluator.py`: Computes a Retrieval Confidence Score considering top score magnitude, score distribution gap, retriever agreement, and query term coverage.
- `self_corrector.py`: `AdaptiveRetrievalPipeline` manages the self-correcting retry loop up to `max_retries`.
- `conflict_detector.py`: Scans retrieved sources for factual numerical disagreements across identical reporting periods.

### 4. Generation & Verification Layer (`src/generation/`)
- `llm_client.py`: Unified LLM client supporting Google Gemini API, OpenAI API, local Ollama, and offline mock generator fallback.
- `prompt_templates.py`: Non-overridable system prompts enforcing strict evidence grounding and prompt injection protection.
- `claim_extractor.py`: Parses LLM output into atomic sentence claims.
- `claim_verifier.py`: Verifies each claim against evidence chunks, assigning `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED`.
- `citation_engine.py`: Constructs claim-level and visual page citations.

### 5. Evaluation Layer (`src/evaluation/`)
- `metrics.py`: Computes Recall@K, Precision@K, MRR, NDCG@K, Faithfulness, Citation Accuracy, and Latencies.
- `benchmark_runner.py`: Compares Basic RAG (Mode 1), Hybrid RAG (Mode 2), and Adaptive Multimodal RAG (Mode 3).

### 6. User Interface (`app_ui.py`)
- Interactive 5-tab Streamlit dashboard providing Chat, Document Upload, Step-by-Step Research Trace, Evaluation Comparisons, and Configuration Settings.
