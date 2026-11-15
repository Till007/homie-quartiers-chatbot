"""
services/orchestrator_service/main.py
FastAPI-Microservice für den Dialog-Controller / Orchestrator auf Port 8004.
Beinhaltet:
- Strategy-Pattern Umschaltung zwischen Pure RAG, Fine-Tuned LLM und Agentic AI (TA-04)
- Flüchtiges RAM-Sitzungsgedächtnis mit 60s Auto-Reset (TA-06)
- REST-Endpunkt (/api/v1/chat/query) & WebSocket-Endpunkt (/ws/robot) mit Satz-Streaming & RobotState (TA-05)

Startbefehl:
    uvicorn services.orchestrator_service.main:app --host 127.0.0.1 --port 8004 --reload
"""

import re
from typing import Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from services.orchestrator_service.memory import EphemeralSessionMemory
from services.orchestrator_service.strategies.agentic_ai import AgenticAIStrategy
from services.orchestrator_service.strategies.base import BaseInferenceStrategy
from services.orchestrator_service.strategies.fine_tuned import FineTunedStrategy
from services.orchestrator_service.strategies.pure_rag import PureRAGStrategy
from shared.logger import get_service_logger
from shared.schemas import ChatRequest, ChatResponse, ParadigmMode, RobotState

logger = get_service_logger("orchestrator-api")

app = FastAPI(
    title="HOMIE Orchestrator & Dialog-Controller Microservice",
    description="Steuert Dialoggedächtnis (60s RAM), Query-Rewriting und schaltet zwischen Pure RAG, Fine-Tuned und Agentic AI um.",
    version="1.0.0",
)

# 1. DSGVO-konformes Sitzungsgedächtnis (60 Sekunden Inaktivitäts-Timeout nach TA-06)
session_memory = EphemeralSessionMemory(ttl_seconds=60)

# 2. Austauschbare Architekturvarianten (Strategy Pattern nach TA-04)
STRATEGIES: Dict[ParadigmMode, BaseInferenceStrategy] = {
    ParadigmMode.PURE_RAG: PureRAGStrategy(),
    ParadigmMode.FINE_TUNED: FineTunedStrategy(),
    ParadigmMode.AGENTIC_AI: AgenticAIStrategy(),
}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "orchestrator-service",
        "port": 8004,
        "available_paradigms": [p.value for p in ParadigmMode],
    }


@app.post("/api/v1/chat/query", response_model=ChatResponse)
async def chat_query_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Haupt-Endpunkt für Text- und transkribierte Sprachanfragen.
    Lädt den Sitzungsverlauf aus dem RAM, führt die gewählte Strategie (Pure RAG, Fine-Tuned, Agentic AI)
    aus und speichert die Runde im flüchtigen RAM-Gedächtnis.
    """
    try:
        history = session_memory.get_history(req.session_id)
        strategy = STRATEGIES.get(req.paradigm, STRATEGIES[ParadigmMode.AGENTIC_AI])

        response = await strategy.execute(req, history)

        # Speichere Runde im RAM-Gedächtnis (wird nach 60s Inaktivität automatisch gelöscht)
        session_memory.append_turn(req.session_id, req.user_message, response.answer_text)
        return response

    except Exception as e:
        logger.error(f"Fehler im Orchestrator für Session '{req.session_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Orchestrator-Fehler: {str(e)}")


@app.post("/api/v1/session/reset")
async def reset_session_endpoint(session_id: str = "default-kiosk-session"):
    """TA-06: Sofortiges Löschen des Gesprächsverlaufs aus dem RAM (z. B. bei Nutzerwechsel am Terminal)."""
    deleted = session_memory.reset_session(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "was_active": deleted,
        "robot_state": RobotState.IDLE.value,
    }


@app.websocket("/ws/robot")
async def robot_websocket_endpoint(websocket: WebSocket):
    """
    TA-05 & UML-Sequenzdiagramm:
    WebSocket-Schnittstelle für den Navel-Roboter / Kiosk-Terminal.
    Sendet Zustandssignale ('listening', 'thinking', 'speaking', 'idle') für Roboter-LEDs/Gestik
    und streamt die Antwort satzweise für minimale TTS-Latenz (TA-03).
    """
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_json()
            req = ChatRequest.model_validate(raw_data)

            # Signalisiere dem Roboter: "Ich denke nach" (für LED-/Blick-Steuerung nach TA-05)
            await websocket.send_json({"event": "state_change", "robot_state": RobotState.THINKING.value})

            history = session_memory.get_history(req.session_id)
            strategy = STRATEGIES.get(req.paradigm, STRATEGIES[ParadigmMode.AGENTIC_AI])
            response = await strategy.execute(req, history)
            session_memory.append_turn(req.session_id, req.user_message, response.answer_text)

            # Signalisiere dem Roboter: "Ich spreche" und streame Satz für Satz (UML loop [pro fertigem Satz])
            await websocket.send_json({"event": "state_change", "robot_state": RobotState.SPEAKING.value})
            sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", response.answer_text) if s.strip()
            ]
            for idx, sentence in enumerate(sentences):
                await websocket.send_json(
                    {
                        "event": "sentence_chunk",
                        "sentence_index": idx,
                        "text": sentence,
                        "language": req.user_language,
                    }
                )

            # Sende vollständiges Metadaten-Paket (Quellen, QR-Code, Agent-Trace) und gehe auf 'idle'
            response.robot_state = RobotState.IDLE
            await websocket.send_json(
                {"event": "complete", "payload": response.model_dump(mode="json")}
            )

    except WebSocketDisconnect:
        logger.info("Roboter/Kiosk WebSocket-Verbindung getrennt.")
    except Exception as e:
        logger.error(f"WebSocket-Fehler: {e}", exc_info=True)
        await websocket.send_json(
            {"event": "error", "robot_state": RobotState.ERROR.value, "detail": str(e)}
        )