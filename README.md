# Adaptive Multimodal Evidence-Grounded RAG

> A production-quality research & capstone demonstration system combining Dense Vector, Lexical BM25, and ColPali Visual Page retrieval with dynamic query intent routing, reciprocal rank fusion, cross-encoder reranking, retrieval confidence estimation, self-correcting retries, grounded claim generation, evidence verification, and reproducible evaluation.

---

## 📌 Features Overview

- **Multimodal & Lexical Retrieval**:
  - **Dense Vector Search**: ChromaDB vector database with SentenceTransformers (`all-MiniLM-L6-v2`).
  - **Lexical Search**: BM25Okapi exact keyword, number, and code identifier matching.
  - **ColPali Visual Page Retrieval**: PDF page image rendering + visual page layout indexing (with transparent GPU/VRAM hardware detection & visual layout fallback).
- **Adaptive Query Intelligence**:
  - **Query Intent Classifier**: Categorizes queries into `FACTUAL`, `SEMANTIC`, `VISUAL`, `TABLE`, `COMPARATIVE`, `MULTI_HOP`, `TEMPORAL` and selects optimal retriever routing policy.
  - **Query Decomposer**: Splits complex multi-entity/comparative queries into independently retrievable sub-questions.
  - **Query Rewriter**: Normalizes acronyms and expands terms for corrective retrieval.
- **Fusion & Reranking**:
  - **Result Fusion**: Supports Reciprocal Rank Fusion (RRF) and Weighted Linear score combination.
  - **Candidate Reranker**: Cross-Encoder candidate re-scoring (Top 20 candidate pool → Top 5 refined context).
- **Self-Correction & Faithfulness**:
  - **Retrieval Confidence Scorer**: Evaluates score magnitude, score distribution gap, cross-retriever agreement, and query coverage.
  - **Self-Correcting Iterative Retry Loop**: Triggers query rewriting and retriever strategy expansion when confidence is below threshold.
  - **Grounded LLM Generation**: Non-overridable system prompts treating retrieved documents as untrusted data blocks.
  - **Claim Extraction & Evidence Verification**: Deconstructs generated answers into atomic claims and verifies each as `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED`.
  - **Citation System**: Claim-level and sentence-level citations referencing document, page, section, and rendered visual page images.
  - **Conflict & Temporal Engine**: Identifies source contradictions and filters by fiscal reporting periods.
- **Capstone Research UI**:
  - 5-Tab Interactive Streamlit Interface (`app_ui.py`): **CHAT & QUERY**, **DOCUMENTS**, **RETRIEVAL TRACE**, **EVALUATION**, **SETTINGS**.

---

## 🏗️ Architecture

```text
USER QUERY
    │
    ▼
QUERY CLASSIFICATION & DECOMPOSITION
    │
    ▼
ADAPTIVE RETRIEVER ROUTING
    ├── Dense Vector Retrieval (ChromaDB)
    ├── Lexical BM25 Retrieval
    └── ColPali Visual Page Search
    │
    ▼
RESULT FUSION (RRF / Weighted Linear)
    │
    ▼
CROSS-ENCODER RERANKING
    │
    ▼
RETRIEVAL CONFIDENCE SCORER
    │ ── (If Low Confidence) ──► SELF-CORRECTING RETRY LOOP (Query Rewrite + Strategy Expansion)
    ▼
GROUNDED CONTEXT GENERATION (LLM)
    │
    ▼
CLAIM EXTRACTION & EVIDENCE VERIFICATION (SUPPORTED / UNSUPPORTED / CONTRADICTED)
    │
    ▼
CITATIONS & FINAL ANSWER + RESEARCH DEBUG TRACE
```

---

## ⚙️ Installation & Setup

1. **Clone & Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables Configuration**:
   Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Set your preferred LLM Provider (`gemini`, `openai`, `ollama`, or default offline fallback):
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## 🚀 Running the Application

### 1. Interactive Capstone Web Interface (Streamlit)
```bash
streamlit run app_ui.py
```
Open your browser at `http://localhost:8501`.

### 2. Command Line Interface (CLI)
```bash
python main.py --query "Compare revenue and employee growth between FY2023 and FY2025."
```

### 3. REST API Server (FastAPI)
```bash
python main.py --serve --port 8000
```
Access interactive OpenAPI documentation at `http://localhost:8000/docs`.

---

## 🧪 Running Unit & Integration Tests

Run the full pytest suite:
```bash
python -m pytest tests/ -v
```

---

## 📊 Evaluation & Benchmark Methodology

The system provides built-in reproducible evaluation comparing:
- **MODE 1**: Basic Vector RAG
- **MODE 2**: Hybrid RAG (Dense + BM25 + Reranking)
- **MODE 3**: Adaptive Multimodal RAG (Proposed)

Evaluated metrics:
- **Retrieval**: Recall@K, Precision@K, MRR, NDCG
- **Trustworthiness**: Citation Accuracy, Unsupported Claim Rate, Contradiction Rate
- **Performance**: Retrieval Latency, Generation Latency, Total Latency

Run the evaluation suite via the **EVALUATION** tab in the Streamlit UI.

---

## 📄 License & Attribution

Capstone Research Demo — Adaptive Multimodal Evidence-Grounded RAG.
