"""
services/rag_retrieval_service/main.py
FastAPI-Microservice für Hybrid-Retrieval (ChromaDB + BM25 + RRF + Reranker + Quellen-Guard) auf Port 8003.
Neu:
- Bereinigt veraltete JSON-Duplikate derselben Quelle in data/processed_chunks/
- Fehlertoleranter Delta-Sync mit automatischer JSON-Selbstheilung (kein 500-Absturz mehr)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

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
    description="Hybrid-Search (ChromaDB + BM25 + RRF + BGE-Reranker) inkl. Delta-Sync, Quellen-Guard und TTL-Überwachung.",
    version="1.3.0",
)

engine = HybridRetrievalEngine()


def _deduplicate_processed_chunk_files() -> int:
    """
    Falls dieselbe Datei oder URL mehrfach (mit unterschiedlichen Ständen/Hashes) in
    data/processed_chunks/ abgelegt wurde, wird nur die jeweils neueste Datei behalten.
    Verhindert, dass sich alte und neue Versionen beim Sync gegenseitig überschreiben.
    """
    if not PROCESSED_CHUNKS_DIR.exists():
        return 0

    latest_by_source: Dict[str, Tuple[float, Path]] = {}
    removed_old_files = 0

    for json_file in PROCESSED_CHUNKS_DIR.glob("*.json"):
        try:
            raw_list = json.loads(json_file.read_text(encoding="utf-8"))
            if not raw_list or not isinstance(raw_list, list):
                continue
            source_key = str(raw_list[0].get("metadata", {}).get("source_url_or_path", "")).strip().lower()
            if not source_key:
                continue

            mtime = json_file.stat().st_mtime
            if source_key in latest_by_source:
                prev_mtime, prev_path = latest_by_source[source_key]
                if mtime >= prev_mtime:
                    prev_path.unlink(missing_ok=True)
                    latest_by_source[source_key] = (mtime, json_file)
                else:
                    json_file.unlink(missing_ok=True)
                removed_old_files += 1
            else:
                latest_by_source[source_key] = (mtime, json_file)
        except Exception as e:
            logger.warning(f"Konnte {json_file.name} bei der Bereinigung nicht prüfen: {e}")

    return removed_old_files


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
async def sync_from_etl_folder_endpoint(force_rebuild: bool = False):
    """
    Fehlertoleranter Delta-Sync (§ 10 A a/c):
    1. Entfernt veraltete Vorversionen derselben Quelle aus data/processed_chunks/.
    2. Repariert automatisch Listen-/Objekt-Metadaten (z. B. metadata.phone) über den Schema-Validator.
    3. Überspringt bereits indexierte SHA-256 Hashes.
    """
    if not PROCESSED_CHUNKS_DIR.exists():
        return {
            "success": True,
            "synced_files": 0,
            "skipped_unchanged_files": 0,
            "total_indexed_chunks": len(engine.chunk_store),
        }

    cleaned_old = _deduplicate_processed_chunk_files()
    if cleaned_old > 0:
        logger.info(f"{cleaned_old} veraltete Chunk-Vorversion(en) aus data/processed_chunks/ bereinigt.")

    already_indexed_hashes = {
        ch.metadata.content_hash_sha256 for ch in engine.chunk_store.values()
    }

    synced_files = 0
    skipped_files = 0
    synced_chunks = 0
    failed_files: List[str] = []

    for json_file in PROCESSED_CHUNKS_DIR.glob("*.json"):
        try:
            raw_list = json.loads(json_file.read_text(encoding="utf-8"))
            if not raw_list:
                continue

            file_hash = raw_list[0].get("metadata", {}).get("content_hash_sha256")
            if not force_rebuild and file_hash and file_hash in already_indexed_hashes:
                skipped_files += 1
                continue

            chunks: List[DocumentChunk] = [DocumentChunk.model_validate(item) for item in raw_list]
            if chunks:
                # Speichere die durch den Validator reparierte Struktur direkt sauber zurück
                json_file.write_text(
                    json.dumps([c.model_dump(mode="json") for c in chunks], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                synced_chunks += engine.upsert_chunks(chunks, replace_existing_doc=True)
                synced_files += 1
                if file_hash:
                    already_indexed_hashes.add(file_hash)
        except Exception as e:
            logger.error(f"Fehler beim Synchronisieren von {json_file.name}: {e}", exc_info=True)
            failed_files.append(f"{json_file.name}: {str(e)[:120]}")

    logger.info(
        f"Delta-Sync abgeschlossen: {synced_files} neue Dokumente indexiert, "
        f"{skipped_files} unveränderte Dokumente übersprungen, {len(failed_files)} Fehler."
    )
    return {
        "success": len(failed_files) == 0,
        "synced_files": synced_files,
        "skipped_unchanged_files": skipped_files,
        "synced_chunks": synced_chunks,
        "total_indexed_chunks": len(engine.chunk_store),
        "failed_files": failed_files,
    }


@app.post("/api/v1/rag/retrieve", response_model=RetrievalResponse)
async def retrieve_endpoint(req: RetrievalRequest) -> RetrievalResponse:
    try:
        return engine.retrieve(req)
    except Exception as e:
        # Kein Anfragetext im Log (K1, 2.3.9: Echtbetrieb nur mit Zeiten, Scores und Kennungen)
        logger.error(f"Fehler beim Retrieval (Anfrage mit {len(req.query)} Zeichen): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval-Fehler: {str(e)}")


@app.post("/api/v1/rag/check-ttl")
async def check_ttl_endpoint():
    return engine.run_ttl_maintenance()