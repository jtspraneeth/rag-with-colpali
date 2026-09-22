import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from PIL import Image

from src.adaptive_rag_pipeline import AdaptiveMultimodalRAG
from src.evaluation.benchmark_runner import BenchmarkRunner
from config import settings

st.set_page_config(
    page_title="Adaptive Multimodal Evidence-Grounded RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics & glassmorphism
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #4F46E5;
    }
    .badge-supported { background-color: #DEF7EC; color: #03543F; padding: 3px 8px; border-radius: 12px; font-weight: 600; }
    .badge-partial { background-color: #FEF3C7; color: #92400E; padding: 3px 8px; border-radius: 12px; font-weight: 600; }
    .badge-unsupported { background-color: #FDE8E8; color: #9B1C1C; padding: 3px 8px; border-radius: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State Singleton for Pipeline
if "rag_system" not in st.session_state:
    st.session_state.rag_system = AdaptiveMultimodalRAG()

rag = st.session_state.rag_system

# Sidebar - Document Explorer & Status
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.markdown("### System Architecture")
    st.info(f"**Visual Backend**: {rag.colpali_retriever.backend_name}")
    st.info(f"**LLM Engine**: {settings.llm_provider.upper()} ({settings.default_llm_model})")
    st.success(f"**Indexed Documents**: {len(rag.indexed_documents)}")

    st.markdown("---")
    st.markdown("### Select RAG Mode")
    rag_mode = st.radio(
        "Retrieval Mode:",
        ["Adaptive Multimodal RAG (Proposed)", "Hybrid RAG (Dense + BM25)", "Basic Vector RAG"],
        index=0
    )
    mode_key = "adaptive_multimodal" if "Adaptive" in rag_mode else ("hybrid" if "Hybrid" in rag_mode else "basic_vector")

# Header
st.markdown('<div class="main-title">Adaptive Multimodal Evidence-Grounded RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-retriever Routing • ColPali Visual Page Search • Self-Correction • Claim Verification</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_chat, tab_docs, tab_trace, tab_eval, tab_settings = st.tabs([
    "💬 CHAT & QUERY", "📄 DOCUMENTS", "🔍 RETRIEVAL TRACE", "📊 EVALUATION", "⚙️ SETTINGS"
])

# ---------------------------------------------------------
# TAB 1: CHAT & QUERY
# ---------------------------------------------------------
with tab_chat:
    col_q, col_ans = st.columns([1, 1.2])

    with col_q:
        st.markdown("### Ask a Question")
        user_query = st.text_area("Enter your query:", height=100, value="Compare the revenue growth and employee metrics between FY2023 and FY2025.")
        
        sample_queries = [
            "What was the revenue growth in 2025?",
            "Show me the visual table or chart on Page 2.",
            "Compare revenue and employee count between 2023 and 2025.",
            "Who was the CEO in FY2024?"
        ]
        selected_sample = st.selectbox("Or choose a sample test query:", ["Custom Query"] + sample_queries)
        if selected_sample != "Custom Query":
            user_query = selected_sample

        ask_btn = st.button("🚀 Execute Adaptive Query", type="primary", use_container_width=True)

    if ask_btn and user_query:
        with st.spinner("Executing Adaptive Retrieval & Evidence Grounded Generation..."):
            res = rag.query(user_query, mode=mode_key)
            st.session_state.last_result = res

    if "last_result" in st.session_state:
        res = st.session_state.last_result
        trace = res["trace"]

        with col_ans:
            st.markdown("### 🤖 Generated Answer")
            st.markdown(f"*{res['answer']}*")
            
            st.markdown("#### Citations")
            st.code(res["formatted_citations"], language="text")

            # Metrics pill bar
            m1, m2, m3 = st.columns(3)
            m1.metric("Retrieval Confidence", f"{trace['retrieval_confidence']['confidence_score']*100:.1f}%", trace['retrieval_confidence']['confidence_level'])
            m2.metric("Total Latency", f"{trace['latencies']['total_sec']} sec")
            m3.metric("Self-Corrected", "YES" if trace["self_corrected"] else "NO")

        st.markdown("---")
        st.markdown("### 📑 Verified Evidence & Source Panel")
        
        # Deduplicate evidence items strictly by normalized (document_name, page)
        unique_evidence = []
        seen_keys = set()
        
        for item in res["retrieved_results"]:
            doc_norm = str(item.get("document_name", "")).strip().lower()
            page_norm = int(item.get("page", 1))
            key = (doc_norm, page_norm)
            
            if key not in seen_keys:
                seen_keys.add(key)
                item_copy = dict(item)
                unique_evidence.append(item_copy)
            else:
                # Merge additional chunk text for the same page so context is preserved
                for existing in unique_evidence:
                    if (str(existing["document_name"]).strip().lower(), int(existing["page"])) == key:
                        if item["text"] not in existing["text"]:
                            existing["text"] += f"\n\n--- Chunk ({item['chunk_id']}) ---\n" + item["text"]
                        if not existing.get("image_path") and item.get("image_path"):
                            existing["image_path"] = item["image_path"]
                        break

        num_items = len(unique_evidence)
        
        if num_items > 0:
            st.markdown(f"#### Verified Context Pages ({num_items} Unique Page{'s' if num_items != 1 else ''})")
            
            # Display items in dynamic rows of up to 3 columns per row
            cols_per_row = 3
            for row_idx in range(0, num_items, cols_per_row):
                row_items = unique_evidence[row_idx : row_idx + cols_per_row]
                ev_cols = st.columns(len(row_items))
                
                for idx, item in enumerate(row_items):
                    item_num = row_idx + idx + 1
                    with ev_cols[idx]:
                        st.markdown(f"**[{item_num}] {item['document_name']} — Page {item['page']}**")
                        st.caption(f"Score: {item['score']} | Method: {item['retrieval_method']}")
                        st.text_area(
                            f"Context Text (Page {item['page']}):", 
                            item["text"], 
                            height=140, 
                            key=f"chunk_txt_{item['document_name']}_p{item['page']}_{item_num}"
                        )
                        
                        img_p = item.get("image_path")
                        if img_p and Path(img_p).exists():
                            st.image(
                                img_p, 
                                caption=f"{item['document_name']} — Page {item['page']} Rendered Visual Evidence", 
                                use_container_width=True
                            )
        else:
            st.info("No unique document evidence chunks retrieved for this query.")

# ---------------------------------------------------------
# TAB 2: DOCUMENTS INGESTION
# ---------------------------------------------------------
with tab_docs:
    st.markdown("### Document Ingestion & Metadata Management")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT documents:", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        for u_file in uploaded_files:
            save_path = settings.documents_dir / u_file.name
            with open(save_path, "wb") as f:
                f.write(u_file.read())
            
            with st.spinner(f"Ingesting & rendering visual pages for {u_file.name}..."):
                doc_info = rag.ingest_document(str(save_path))
                st.success(f"Indexed {u_file.name} successfully! Created {doc_info['chunks_count']} metadata-preserved chunks.")

    st.markdown("#### Currently Indexed Documents")
    if rag.indexed_documents:
        st.table(rag.indexed_documents)
    else:
        st.info("No documents uploaded yet. Upload a PDF/DOCX/TXT file above or use default workspace documents.")

# ---------------------------------------------------------
# TAB 3: RETRIEVAL TRACE (RESEARCH & DEBUG MODE)
# ---------------------------------------------------------
with tab_trace:
    st.markdown("### 🔬 Research & Debug Execution Trace")
    if "last_result" in st.session_state:
        trace = st.session_state.last_result["trace"]

        st.json({
            "Query": trace["query"],
            "1. Query Classification": trace["classification"],
            "2. Query Decomposition": trace["decomposition"],
            "3. Selected Retrievers": trace["selected_retrievers"],
            "4. Visual Backend Active": trace["visual_backend"],
            "5. Retrieval Confidence": trace["retrieval_confidence"],
            "6. Self-Correction Attempt Traces": trace["attempt_traces"],
            "7. Conflict Detection Report": trace["conflict_report"],
            "8. Execution Latencies": trace["latencies"]
        })
    else:
        st.info("Run a query in the CHAT tab to inspect the step-by-step research trace.")

# ---------------------------------------------------------
# TAB 4: EVALUATION DASHBOARD
# ---------------------------------------------------------
with tab_eval:
    st.markdown("### 📊 Baseline vs. Adaptive Multimodal RAG Evaluation")
    st.write("Compare empirical metrics across Basic Vector RAG, Hybrid RAG, and Adaptive Multimodal RAG.")

    if st.button("⚡ Run Live Benchmark Evaluation Suite"):
        doc_targets = [d.get("document_name", d.get("document_id", "")) for d in rag.indexed_documents] if rag.indexed_documents else []
        if not doc_targets:
            doc_targets = ["doc_001"]

        test_queries = [
            {"query": "What is the revenue growth in 2025?", "relevant_doc_ids": doc_targets},
            {"query": "Compare revenue and employee growth between 2023 and 2025.", "relevant_doc_ids": doc_targets},
            {"query": "Show table layout figures on Page 2.", "relevant_doc_ids": doc_targets}
        ]
        runner = BenchmarkRunner(rag)
        with st.spinner("Running evaluation benchmark suite across 3 modes..."):
            summary = runner.run_benchmark(test_queries)
            st.session_state.benchmark_summary = summary

    if "benchmark_summary" in st.session_state:
        st.table(st.session_state.benchmark_summary)

# ---------------------------------------------------------
# TAB 5: SETTINGS & CONFIGURATION
# ---------------------------------------------------------
with tab_settings:
    st.markdown("### ⚙️ Central System Configuration")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### Retrieval Weights")
        settings.dense_weight = st.slider("Dense Weight (alpha)", 0.0, 1.0, settings.dense_weight)
        settings.bm25_weight = st.slider("BM25 Weight (beta)", 0.0, 1.0, settings.bm25_weight)
        settings.colpali_weight = st.slider("ColPali Weight (gamma)", 0.0, 1.0, settings.colpali_weight)

    with col_s2:
        st.markdown("#### Self-Correction Thresholds")
        settings.confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, settings.confidence_threshold)
        settings.max_retries = st.number_input("Max Retries", 1, 5, settings.max_retries)

    st.markdown("---")
    st.markdown("#### 🤖 LLM Generation Engine Settings")
    provider_opts = ["gemini", "openai", "mistral", "ollama", "mock"]
    curr_prov = settings.llm_provider.lower()
    p_idx = provider_opts.index(curr_prov) if curr_prov in provider_opts else 0

    new_provider = st.selectbox("Active LLM Provider:", provider_opts, index=p_idx, help="Select LLM provider or 'mock' for smart offline synthesis.")
    if new_provider != settings.llm_provider:
        settings.llm_provider = new_provider
        rag.llm.provider = new_provider
        st.success(f"Switched LLM Provider to '{new_provider}'")

    if new_provider == "gemini":
        g_key = st.text_input("Gemini API Key:", value=settings.gemini_api_key, type="password", help="Enter your Google Gemini API Key")
        if g_key != settings.gemini_api_key:
            settings.gemini_api_key = g_key
            st.success("Updated Gemini API Key")
    elif new_provider == "openai":
        o_key = st.text_input("OpenAI API Key:", value=settings.openai_api_key, type="password", help="Enter your OpenAI API Key")
        if o_key != settings.openai_api_key:
            settings.openai_api_key = o_key
            st.success("Updated OpenAI API Key")
