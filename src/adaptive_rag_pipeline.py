import time
from typing import List, Dict, Any, Optional

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.docx_parser import DOCXParser
from src.ingestion.txt_parser import TXTParser
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.colpali_retriever import ColPaliRetriever
from src.retrieval.result_fusion import ResultFusion
from src.retrieval.reranker import CandidateReranker
from src.retrieval.temporal_filter import TemporalFilter
from src.intelligence.query_classifier import QueryClassifier
from src.intelligence.query_decomposer import QueryDecomposer
from src.intelligence.confidence_evaluator import ConfidenceEvaluator
from src.intelligence.self_corrector import AdaptiveRetrievalPipeline
from src.intelligence.conflict_detector import ConflictDetector
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import build_grounded_generation_prompt, SYSTEM_GROUNDED_INSTRUCTION
from src.generation.claim_extractor import ClaimExtractor
from src.generation.claim_verifier import ClaimVerifier
from src.generation.citation_engine import CitationEngine
from config import settings

class AdaptiveMultimodalRAG:
    """Production-quality Adaptive Multimodal Evidence-Grounded RAG Pipeline Coordinator."""

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.docx_parser = DOCXParser()
        self.txt_parser = TXTParser()
        
        self.dense_retriever = DenseRetriever()
        self.bm25_retriever = BM25Retriever()
        self.colpali_retriever = ColPaliRetriever()

        self.classifier = QueryClassifier()
        self.decomposer = QueryDecomposer()
        self.temporal_filter = TemporalFilter()
        self.conflict_detector = ConflictDetector()

        self.adaptive_pipeline = AdaptiveRetrievalPipeline(
            self.dense_retriever, self.bm25_retriever, self.colpali_retriever
        )

        self.llm = LLMClient()
        self.claim_extractor = ClaimExtractor()
        self.claim_verifier = ClaimVerifier()
        self.citation_engine = CitationEngine()

        self.indexed_documents: List[Dict[str, Any]] = []

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingests PDF, DOCX, or TXT document across text, lexical, and visual page backends."""
        start_t = time.time()
        
        if file_path.endswith(".pdf"):
            chunks, page_images = self.pdf_parser.parse_pdf(file_path)
            for chunk in chunks:
                img_path = chunk.metadata.get("page_image_path", "")
                if img_path:
                    self.colpali_retriever.add_page_image(
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        page_number=chunk.page_number,
                        image_path=img_path,
                        page_text=chunk.text
                    )
        elif file_path.endswith(".docx"):
            chunks = self.docx_parser.parse_docx(file_path)
        elif file_path.endswith(".txt"):
            chunks = self.txt_parser.parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported document format: {file_path}")

        # Index in Dense & BM25 retrievers
        self.dense_retriever.add_chunks(chunks)
        self.bm25_retriever.add_chunks(chunks)

        doc_meta = {
            "document_name": chunks[0].document_name if chunks else "unknown",
            "document_id": chunks[0].document_id if chunks else "unknown",
            "chunks_count": len(chunks),
            "source_type": chunks[0].source_type if chunks else "unknown",
            "file_path": file_path,
            "ingest_time_seconds": round(time.time() - start_t, 2)
        }
        self.indexed_documents.append(doc_meta)
        return doc_meta

    def query(self, query_text: str, mode: str = "adaptive_multimodal") -> Dict[str, Any]:
        """Executes full end-to-end RAG query flow with comprehensive research trace."""
        start_time = time.time()
        
        # Step 1: Query Understanding & Intent Classification
        classification = self.classifier.classify_query(query_text)
        
        # Step 2: Query Decomposition
        decomposition = self.decomposer.decompose(query_text)

        # Step 3: Adaptive Retrieval & Self-Correcting Loop
        retrieval_start = time.time()
        
        if mode == "basic_vector":
            retrieval_res = self.dense_retriever.retrieve_dense(query_text, top_k=settings.final_top_k)
            retrieval_output = {
                "final_results": retrieval_res,
                "confidence": {"confidence_score": 0.70, "confidence_level": "MEDIUM", "reason": "Basic Vector RAG mode"},
                "attempt_traces": [{"attempt": 1, "query_used": query_text, "confidence_score": 0.70}],
                "self_corrected": False
            }
        elif mode == "hybrid":
            dense_res = self.dense_retriever.retrieve_dense(query_text, top_k=settings.top_k_candidates)
            bm25_res = self.bm25_retriever.retrieve_bm25(query_text, top_k=settings.top_k_candidates)
            fused = self.adaptive_pipeline.fusion.fuse_results(dense_res, bm25_res, [])
            reranked = self.adaptive_pipeline.reranker.rerank(query_text, fused, top_k=settings.final_top_k)
            retrieval_output = {
                "final_results": reranked,
                "confidence": {"confidence_score": 0.80, "confidence_level": "HIGH", "reason": "Hybrid Dense + BM25 RAG"},
                "attempt_traces": [{"attempt": 1, "query_used": query_text, "confidence_score": 0.80}],
                "self_corrected": False
            }
        else: # Adaptive Multimodal RAG
            retrieval_output = self.adaptive_pipeline.execute_adaptive_retrieval(
                query=query_text,
                retriever_keys=classification.selected_retrievers,
                top_k=settings.top_k_candidates,
                final_top_k=settings.final_top_k
            )

        retrieved_candidates = retrieval_output["final_results"]
        retrieval_latency = time.time() - retrieval_start

        # Step 4: Temporal Filtering & Conflict Detection
        if classification.query_type == "TEMPORAL":
            retrieved_candidates = self.temporal_filter.filter_by_temporal_query(query_text, retrieved_candidates)
        
        conflict_report = self.conflict_detector.detect_conflicts(retrieved_candidates)

        # Step 5: LLM Generation
        gen_start = time.time()
        prompt = build_grounded_generation_prompt(query_text, retrieved_candidates)
        answer = self.llm.generate(prompt, system_instruction=SYSTEM_GROUNDED_INSTRUCTION)
        gen_latency = time.time() - gen_start

        # Step 6: Claim Extraction & Evidence Verification
        claims = self.claim_extractor.extract_claims(answer)
        verified_claims = self.claim_verifier.verify_claims(claims, retrieved_candidates)

        # Step 7: Citation Generation
        citation_records, formatted_citations = self.citation_engine.build_citations(verified_claims)

        total_latency = time.time() - start_time

        # Compile Research / Debug Trace
        trace = {
            "query": query_text,
            "mode": mode,
            "classification": classification.model_dump(),
            "decomposition": decomposition.model_dump(),
            "selected_retrievers": classification.selected_retrievers,
            "visual_backend": self.colpali_retriever.backend_name,
            "is_colpali_active": self.colpali_retriever.is_colpali_active,
            "retrieval_confidence": retrieval_output["confidence"],
            "attempt_traces": retrieval_output["attempt_traces"],
            "self_corrected": retrieval_output["self_corrected"],
            "conflict_report": conflict_report.model_dump(),
            "candidate_count": len(retrieved_candidates),
            "latencies": {
                "retrieval_sec": round(retrieval_latency, 3),
                "generation_sec": round(gen_latency, 3),
                "total_sec": round(total_latency, 3)
            }
        }

        return {
            "query": query_text,
            "answer": answer,
            "formatted_citations": formatted_citations,
            "citation_records": [c.model_dump() for c in citation_records],
            "verified_claims": [vc.model_dump() for vc in verified_claims],
            "retrieved_results": [r.model_dump() for r in retrieved_candidates],
            "trace": trace
        }
