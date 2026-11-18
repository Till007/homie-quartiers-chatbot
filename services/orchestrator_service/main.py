"""
services/orchestrator_service/main.py
FastAPI-Microservice für den Dialog-Controller / Orchestrator auf Port 8004.
Beinhaltet:
- Deterministischen Safety- & Krisen-Interceptor (Suizid, Gewalt, Sucht, NSFW) gemäß FA-06 & SA-06
- Strategy-Pattern Umschaltung zwischen Pure RAG, Fine-Tuned LLM und Agentic AI (TA-04)
- Flüchtiges RAM-Sitzungsgedächtnis mit 60s Auto-Reset (TA-06)
"""

import re
from typing import Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from services.orchestrator_service.memory import EphemeralSessionMemory
from services.orchestrator_service.safety_guard import evaluate_safety_and_crisis
from services.orchestrator_service.strategies.agentic_ai import AgenticAIStrategy
from services.orchestrator_service.strategies.base import BaseInferenceStrategy
from services.orchestrator_service.strategies.fine_tuned import FineTunedStrategy
from services.orchestrator_service.strategies.pure_rag import PureRAGStrategy
from shared.logger import get_service_logger
from shared.schemas import ChatRequest, ChatResponse, ParadigmMode, RobotState

logger = get_service_logger("orchestrator-api")

app = FastAPI(
    title="HOMIE Orchestrator & Dialog-Controller Microservice",
    description="Steuert Safety-Guardrails, Dialoggedächtnis (60s RAM) und schaltet zwischen Pure RAG, Fine-Tuned und Agentic AI um.",
    version="1.3.0",
)

session_memory = EphemeralSessionMemory(ttl_seconds=60)

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
    1. Prüft über den Safety-Interceptor auf akute Krisen (Suizid, Gewalt, Sucht) oder Missbrauch (FA-06 / SA-06).
    2. Führt andernfalls die gewählte Strategie (Pure RAG, Fine-Tuned, Agentic AI) aus.
    """
    try:
        # 1. Deterministischer Safety- & Krisen-Check VOR dem LLM/RAG-Aufruf
        crisis_response, safety_net_appendix, safety_step = evaluate_safety_and_crisis(
            req, req.user_message
        )
        if crisis_response is not None:
            # Bei akuter Krise oder Missbrauch: Session-Kontext aus Datenschutz-/Sicherheitsgründen zurücksetzen
            session_memory.reset_session(req.session_id)
            return crisis_response

        # 2. Reguläre Ausführung über das gewählte Paradigma
        history = session_memory.get_history(req.session_id)
        strategy = STRATEGIES.get(req.paradigm, STRATEGIES[ParadigmMode.AGENTIC_AI])
        response = await strategy.execute(req, history)

        # Falls Stufe 2 (z. B. Sucht/Drogen) aktiv war: Verifiziertes Sicherheitsnetz anhängen & Diskretion erzwingen
        if safety_net_appendix:
            response.answer_text = f"{response.answer_text}{safety_net_appendix}"
            response.is_sensitive_topic = True
            if safety_step:
                response.agent_trace.insert(0, safety_step)

        if response.passed_source_guard and not response.answer_text.startswith("⚠️"):
            session_memory.append_turn(req.session_id, req.user_message, response.answer_text)

        return response

    except Exception as e:
        logger.error(f"Fehler im Orchestrator für Session '{req.session_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Orchestrator-Fehler: {str(e)}")


@app.post("/api/v1/session/reset")
async def reset_session_endpoint(session_id: str = "default-kiosk-session"):
    deleted = session_memory.reset_session(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "was_active": deleted,
        "robot_state": RobotState.IDLE.value,
    }


@app.websocket("/ws/robot")
async def robot_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_json()
            req = ChatRequest.model_validate(raw_data)

            await websocket.send_json({"event": "state_change", "robot_state": RobotState.THINKING.value})

            crisis_response, safety_net_appendix, safety_step = evaluate_safety_and_crisis(
                req, req.user_message
            )
            if crisis_response is not None:
                session_memory.reset_session(req.session_id)
                response = crisis_response
            else:
                history = session_memory.get_history(req.session_id)
                strategy = STRATEGIES.get(req.paradigm, STRATEGIES[ParadigmMode.AGENTIC_AI])
                response = await strategy.execute(req, history)
                if safety_net_appendix:
                    response.answer_text = f"{response.answer_text}{safety_net_appendix}"
                    response.is_sensitive_topic = True
                    if safety_step:
                        response.agent_trace.insert(0, safety_step)
                if response.passed_source_guard and not response.answer_text.startswith("⚠️"):
                    session_memory.append_turn(req.session_id, req.user_message, response.answer_text)

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