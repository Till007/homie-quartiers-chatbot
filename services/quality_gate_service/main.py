"""
services/quality_gate_service/main.py
FastAPI-Microservice für die Dokumenten-Qualitätskontrolle (Port 8001).
Startbefehl aus dem Hauptverzeichnis:
    uvicorn services.quality_gate_service.main:app --host 127.0.0.1 --port 8001 --reload
"""

from fastapi import FastAPI, HTTPException
from shared.logger import get_service_logger
from shared.schemas import QualityAuditReport, QualityInspectionRequest
from services.quality_gate_service.pipeline import QualityGatePipeline

logger = get_service_logger("quality-gate-api")

app = FastAPI(
    title="HOMIE Quality-Gate Microservice",
    description="Prüft Dokumente schrittweise gegen das Regelwerk (§ 1 - § 11) vor der ETL-Aufbereitung.",
    version="1.0.0",
)

pipeline = QualityGatePipeline()


@app.get("/health")
async def health_check():
    """Health-Check für Orchestrierung und Debugging."""
    return {"status": "healthy", "service": "quality-gate-service", "port": 8001}


@app.post("/api/v1/quality/inspect", response_model=QualityAuditReport)
async def inspect_document_endpoint(request: QualityInspectionRequest) -> QualityAuditReport:
    """
    Prüft ein eingereichtes Dokument auf alle Regeln aus '2.0 Konzeption & Systementscheidung.txt'.
    Gibt einen detaillierten QualityAuditReport zurück.
    """
    try:
        report = pipeline.inspect_document(request)
        return report
    except Exception as e:
        logger.error(f"Unerwarteter Fehler bei der Dokumentenprüfung: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Interner Fehler im Quality-Gate Microservice: {str(e)}"
        )