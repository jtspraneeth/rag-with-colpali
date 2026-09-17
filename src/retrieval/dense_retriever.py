import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from src.ingestion.metadata import DocumentChunk, RetrievalResult
from config import settings

class DenseRetriever:
    """Dense vector retriever using ChromaDB and SentenceTransformers embeddings."""
    def __init__(self, collection_name: str = "dense_rag_collection"):
        self.collection_name = collection_name
        self.persist_dir = str(settings.chroma_persist_dir)
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Lazy load embedding model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not installed.")
            self._model = SentenceTransformer(settings.embedding_model_name)
        return self._model

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Embeds and indexes document chunks in ChromaDB."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

        metadatas = []
        for c in chunks:
            meta = {
                "document_id": c.document_id,
                "document_name": c.document_name,
                "page": c.page_number,
                "section": c.section or "",
                "source_type": c.source_type,
                "document_date": c.document_date or "",
                "reporting_period": c.reporting_period or "",
                "page_image_path": c.metadata.get("page_image_path", "")
            }
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def retrieve_dense(self, query: str, top_k: int = 20, filter_metadata: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Queries the vector database using dense embeddings and returns normalized RetrievalResults."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode([query]).tolist()

        where_clause = filter_metadata if filter_metadata else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            where=where_clause
        )

        retrieved_results: List[RetrievalResult] = []
        
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for chunk_id, distance, doc_text, meta in zip(ids, distances, documents, metadatas):
            # ChromaDB cosine distance range [0, 2]. Convert to similarity score [0, 1]
            similarity_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

            retrieved_results.append(RetrievalResult(
                document_id=meta.get("document_id", "unknown"),
                document_name=meta.get("document_name", "unknown"),
                page=int(meta.get("page", 1)),
                chunk_id=chunk_id,
                text=doc_text,
                score=round(float(similarity_score), 4),
                retrieval_method="dense",
                metadata=meta,
                image_path=meta.get("page_image_path")
            ))

        return sorted(retrieved_results, key=lambda x: x.score, reverse=True)
