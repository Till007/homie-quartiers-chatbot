"""
services/etl_ingest_service/main.py
FastAPI-Microservice für ETL (Extract, Transform, Load-Vorbereitung) auf Port 8002.
Kommuniziert direkt mit dem Quality-Gate Microservice (Port 8001).
Startbefehl:
    uvicorn services.etl_ingest_service.main:app --host 127.0.0.1 --port 8002 --reload
"""

import json
import os
import shutil
from datetime import date
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from services.etl_ingest_service.extractor import DocumentExtractor
from services.etl_ingest_service.transformer import DocumentTransformer
from shared.logger import get_service_logger
from shared.schemas import (
    ETLProcessResponse,
    QualityAuditReport,
    QualityInspectionRequest,
    ScrapeUrlRequest,
)

logger = get_service_logger("etl-ingest-api")

QUALITY_GATE_URL = os.getenv("QUALITY_GATE_URL", "http://127.0.0.1:8001/api/v1/quality/inspect")

RAW_INBOX_DIR = Path("data/raw_inbox")
QUARANTINE_DIR = Path("data/quarantine")
PROCESSED_CHUNKS_DIR = Path("data/processed_chunks")

for d in (RAW_INBOX_DIR, QUARANTINE_DIR, PROCESSED_CHUNKS_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="HOMIE ETL-Ingest Microservice",
    description="Liest PDFs/TXT & Hildesheimer Webseiten, prüft sie über das Quality-Gate (Port 8001) und erzeugt semantische Chunks.",
    version="1.0.0",
)

extractor = DocumentExtractor()
transformer = DocumentTransformer()


def _get_existing_hashes() -> List[str]:
    """Liest bereits verarbeitete SHA-256 Hashes aus data/processed_chunks/, um Duplikate (§ 10 A) zu erkennen."""
    hashes = []
    for f in PROCESSED_CHUNKS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data and isinstance(data, list) and "metadata" in data[0]:
                h = data[0]["metadata"].get("content_hash_sha256")
                if h:
                    hashes.append(h)
        except Exception:
            continue
    return hashes


async def _run_quality_gate_and_transform(
    raw_text: str,
    title: str,
    source_path_or_url: str,
    publisher: Optional[str],
    doc_date: Optional[date],
    valid_until: Optional[date],
    use_llm_enrichment: bool,
    crawled_sub_urls: Optional[List[str]] = None,
) -> ETLProcessResponse:
    """
    Zentrale Pipeline:
    1. Ruft Quality-Gate Microservice (Port 8001) auf.
    2. Verschiebt abgelehnte Dokumente in data/quarantine/.
    3. Transformiert & chunkt freigegebene Dokumente.
    """
    inspection_req = QualityInspectionRequest(
        raw_text=raw_text,
        title=title,
        source_url_or_path=source_path_or_url,
        publisher=publisher,
        document_date=doc_date,
        valid_until=valid_until,
        existing_hashes=_get_existing_hashes(),
    )

    # 1. Quality-Gate Microservice via HTTP aufrufen
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                QUALITY_GATE_URL,
                content=inspection_req.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            audit_report = QualityAuditReport.model_validate(resp.json())
    except Exception as e:
        logger.error(f"Quality-Gate Service unter {QUALITY_GATE_URL} nicht erreichbar: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Quality-Gate Microservice (Port 8001) nicht erreichbar. Bitte sicherstellen, dass er läuft: {e}",
        )

    # 2. Wenn Quality-Gate blockiert -> Ab in die Quarantäne!
    if not audit_report.approved_for_ingestion:
        quarantine_file = QUARANTINE_DIR / f"{audit_report.audit_id}_REJECTED.json"
        quarantine_file.write_text(audit_report.model_dump_json(indent=2), encoding="utf-8")
        logger.warning(
            f"Dokument '{title}' vom Quality-Gate ABGELEHNT! Gespeichert unter {quarantine_file}"
        )
        return ETLProcessResponse(
            success=False,
            status_message=f"Dokument durch Quality-Gate blockiert: {'; '.join(audit_report.blocking_reasons)}",
            audit_report=audit_report,
            quarantine_path=str(quarantine_file),
            chunks=[],
            crawled_sub_urls=crawled_sub_urls or [],
        )

    # 3. Transformation: DIN-Abkürzungen (§ 3B), Glossar (§ 3Bc) & Mehrsprachige Keywords (§ 3D)
    cleaned_text, glossary_notes, glossary_keywords = transformer.clean_and_standardize_text(
        audit_report.sanitized_text
    )

    metadata = audit_report.extracted_metadata
    for kw in glossary_keywords:
        if kw not in metadata.multilingual_keywords:
            metadata.multilingual_keywords.append(kw)

    # 4. Kontaktdaten (FA-04) & Hildesheimer Stadtteil anreichern
    metadata = await transformer.enrich_metadata(
        cleaned_text, metadata, use_llm=use_llm_enrichment
    )

    # 5. Semantisches Chunking (300-500 Tokens, ~12% Overlap)
    chunks = transformer.create_semantic_chunks(cleaned_text, metadata, glossary_notes)

    # Chunks lokal als JSON-Zwischenstand sichern (für Review im Debug-UI & Ingestion in ChromaDB)
    chunk_export_path = PROCESSED_CHUNKS_DIR / f"{metadata.doc_id}.json"
    chunk_export_path.write_text(
        json.dumps([c.model_dump(mode="json") for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ETLProcessResponse(
        success=True,
        status_message=f"Qualitätsprüfung bestanden und {len(chunks)} semantische Chunks erzeugt.",
        audit_report=audit_report,
        chunks=chunks,
        crawled_sub_urls=crawled_sub_urls or [],
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "etl-ingest-service", "port": 8002}


@app.post("/api/v1/etl/ingest-file", response_model=ETLProcessResponse)
async def ingest_file_endpoint(
    file: UploadFile = File(..., description="PDF- oder TXT-Datei für die Quartiersdatenbank"),
    publisher: Optional[str] = Form(default="AWO Hildesheim"),
    valid_until: Optional[date] = Form(default=None, description="Optionales Verfallsdatum YYYY-MM-DD (§ 10 B)"),
    use_llm_enrichment: bool = Form(default=True),
) -> ETLProcessResponse:
    """
    Nimmt eine PDF- oder Textdatei entgegen, speichert sie unter data/raw_inbox/,
    extrahiert Text & Tabellen, prüft sie im Quality-Gate und erstellt semantische Chunks.
    """
    safe_filename = Path(file.filename or "upload.txt").name
    saved_path = RAW_INBOX_DIR / safe_filename

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text, doc_date, title = extractor.extract_from_file(saved_path)
        return await _run_quality_gate_and_transform(
            raw_text=raw_text,
            title=title,
            source_path_or_url=str(saved_path),
            publisher=publisher,
            doc_date=doc_date,
            valid_until=valid_until,
            use_llm_enrichment=use_llm_enrichment,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Datei-Ingest ({safe_filename}): {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Fehler bei der Dateiverarbeitung: {str(e)}")


@app.post("/api/v1/etl/ingest-url", response_model=ETLProcessResponse)
async def ingest_url_endpoint(req: ScrapeUrlRequest) -> ETLProcessResponse:
    """
    Crawlt eine Hildesheimer Whitelist-URL (inkl. strikter robots.txt-Prüfung nach § 9 A
    und optionalem Link-Crawling Tiefe = 1), prüft den Inhalt im Quality-Gate und erzeugt Chunks.
    """
    try:
        scraped_text, page_title, sub_urls = await extractor.scrape_url_with_depth(
            start_url=req.url, follow_depth_1=req.follow_links_depth_1
        )
        return await _run_quality_gate_and_transform(
            raw_text=scraped_text,
            title=page_title,
            source_path_or_url=req.url,
            publisher=req.publisher,
            doc_date=date.today(),
            valid_until=None,
            use_llm_enrichment=req.use_llm_enrichment,
            crawled_sub_urls=sub_urls,
        )
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Web-Scraping von {req.url}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Fehler beim Web-Scraping: {str(e)}")