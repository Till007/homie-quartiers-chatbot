"""
services/rag_retrieval_service/main.py
FastAPI-Microservice für Hybrid-Retrieval (ChromaDB + BM25 + RRF + Reranker + Quellen-Guard) auf Port 8003.
Startbefehl:
    uvicorn services.rag_retrieval_service.main:app --host 127.0.0.1 --port 8003 --reload
"""

import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException

from services.rag_retrieval_service.hybrid_engine import HybridRetrievalEngine
from shared.logger import get_service_logger
from shared.schemas import (
    DocumentChunk,
    IndexUpsertRequest,
    RetrievalRequest,
    RetrievalResponse,
)

logger = get_service_logger("rag-retrieval-api")

PROCESSED_CHUNKS_DIR = Path("data/processed_chunks")

app = FastAPI(
    title="HOMIE RAG-Retrieval Microservice",
    description="Hybrid-Search (ChromaDB + BM25 + RRF + BGE-Reranker) inkl. Quellen-Guard und TTL-Überwachung.",
    version="1.0.0",
)

engine = HybridRetrievalEngine()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "rag-retrieval-service",
        "port": 8003,
        "indexed_chunks": len(engine.chunk_store),
    }


@app.post("/api/v1/rag/upsert")
async def upsert_chunks_endpoint(req: IndexUpsertRequest):
    """Fügt eine Liste von DocumentChunks in ChromaDB und BM25 ein (inkl. § 10 A c Vektor-Ersetzung)."""
    try:
        count = engine.upsert_chunks(req.chunks, replace_existing_doc=req.replace_existing_doc)
        return {
            "success": True,
            "upserted_chunks": count,
            "total_indexed_chunks": len(engine.chunk_store),
        }
    except Exception as e:
        logger.error(f"Fehler beim Upsert in den Hybrid-Index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/sync-from-etl")
async def sync_from_etl_folder_endpoint():
    """
    Liest automatisch alle vom ETL-Service erzeugten Chunk-Dateien aus data/processed_chunks/
    und indexiert sie in ChromaDB und BM25.
    """
    if not PROCESSED_CHUNKS_DIR.exists():
        return {"success": True, "synced_files": 0, "total_indexed_chunks": len(engine.chunk_store)}

    synced_files = 0
    synced_chunks = 0
    try:
        for json_file in PROCESSED_CHUNKS_DIR.glob("*.json"):
            raw_list = json.loads(json_file.read_text(encoding="utf-8"))
            chunks: List[DocumentChunk] = [DocumentChunk.model_validate(item) for item in raw_list]
            if chunks:
                synced_chunks += engine.upsert_chunks(chunks, replace_existing_doc=True)
                synced_files += 1

        return {
            "success": True,
            "synced_files": synced_files,
            "synced_chunks": synced_chunks,
            "total_indexed_chunks": len(engine.chunk_store),
        }
    except Exception as e:
        logger.error(f"Fehler beim Synchronisieren aus data/processed_chunks/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sync-Fehler: {str(e)}")


@app.post("/api/v1/rag/retrieve", response_model=RetrievalResponse)
async def retrieve_endpoint(req: RetrievalRequest) -> RetrievalResponse:
    """
    Führt die Hybrid-Suche (BM25 + Dense + RRF + Cross-Encoder Reranker) durch
    und prüft den Quellen-Guard-Schwellenwert.
    """
    try:
        return engine.retrieve(req)
    except Exception as e:
        logger.error(f"Fehler beim Retrieval für Query '{req.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval-Fehler: {str(e)}")


@app.post("/api/v1/rag/check-ttl")
async def check_ttl_endpoint():
    """§ 10 B: Führt den Prüf-Lauf für abgelaufene Dokumente (valid_until) aus."""
    return engine.run_ttl_maintenance()