from typing import List, Dict, Any
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.docx_parser import DOCXParser
from src.ingestion.txt_parser import TXTParser
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import build_grounded_generation_prompt, SYSTEM_GROUNDED_INSTRUCTION
from src.ingestion.metadata import RetrievalResult

class BaselineRAG:
    """Baseline RAG implementation (PDF/DOCX/TXT -> Chunks -> Embeddings -> Vector DB -> LLM Answer)."""
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.docx_parser = DOCXParser()
        self.txt_parser = TXTParser()
        self.retriever = DenseRetriever()
        self.llm = LLMClient()

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingests PDF, DOCX, or TXT document into the dense vector store."""
        if file_path.endswith(".pdf"):
            chunks, _ = self.pdf_parser.parse_pdf(file_path)
        elif file_path.endswith(".docx"):
            chunks = self.docx_parser.parse_docx(file_path)
        elif file_path.endswith(".txt"):
            chunks = self.txt_parser.parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        self.retriever.add_chunks(chunks)
        return {
            "status": "success",
            "chunks_indexed": len(chunks),
            "file_path": file_path
        }

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Executes Baseline RAG query flow."""
        results = self.retriever.retrieve_dense(query_text, top_k=top_k)
        prompt = build_grounded_generation_prompt(query_text, results)
        answer = self.llm.generate(prompt, system_instruction=SYSTEM_GROUNDED_INSTRUCTION)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_results": [r.model_dump() for r in results],
            "mode": "baseline_dense_rag"
        }
