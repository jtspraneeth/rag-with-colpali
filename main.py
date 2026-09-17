import argparse
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.adaptive_rag_pipeline import AdaptiveMultimodalRAG
from config import settings

app = FastAPI(
    title="Adaptive Multimodal Evidence-Grounded RAG API",
    version="1.0.0",
    description="Production-quality RAG system combining Dense, BM25, and ColPali visual retrieval with self-correction and claim verification."
)

rag_system = AdaptiveMultimodalRAG()

class QueryRequest(BaseModel):
    query: str
    mode: str = "adaptive_multimodal"

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Adaptive Multimodal Evidence-Grounded RAG",
        "visual_backend": rag_system.colpali_retriever.backend_name,
        "indexed_documents_count": len(rag_system.indexed_documents)
    }

@app.post("/query")
def query_endpoint(req: QueryRequest):
    return rag_system.query(req.query, mode=req.mode)

@app.post("/upload")
def upload_endpoint(file: UploadFile = File(...)):
    save_path = settings.documents_dir / file.filename
    with open(save_path, "wb") as f:
        f.write(file.file.read())
    info = rag_system.ingest_document(str(save_path))
    return info

def main():
    parser = argparse.ArgumentParser(description="Adaptive Multimodal RAG System")
    parser.add_argument("--query", type=str, help="Query text to execute")
    parser.add_argument("--mode", type=str, default="adaptive_multimodal", help="Retrieval mode")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI REST server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()

    if args.serve:
        print(f"Starting FastAPI server on http://localhost:{args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.query:
        res = rag_system.query(args.query, mode=args.mode)
        print("\n=== GENERATED ANSWER ===")
        print(res["answer"])
        print("\n=== CITATIONS ===")
        print(res["formatted_citations"])
        print("\n=== CONFIDENCE & TRACE ===")
        print(f"Confidence: {res['trace']['retrieval_confidence']}")
    else:
        print("Use --query 'your question' or --serve to launch API server. Launch UI with: streamlit run app_ui.py")

if __name__ == "__main__":
    main()
