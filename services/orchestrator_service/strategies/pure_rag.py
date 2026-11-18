"""
services/orchestrator_service/strategies/pure_rag.py
Paradigma 1 (TA-04): Pure RAG (Single-Step Hybrid-Retrieval mit Quellen-Guard).
Unterstützt automatische Fremdsprachen-Erkennung (z. B. Arabisch, Türkisch, Ukrainisch)
und die Cross-Lingual Query-Brücke für hohe Trefferquote auf deutschen Quellen.
"""

import time
from typing import List

from services.orchestrator_service.prompts import (
    build_qr_code_payload,
    build_system_prompt,
    detect_sensitive_topic,
    format_chunks_for_prompt,
)
from services.orchestrator_service.strategies.base import BaseInferenceStrategy
from shared.schemas import (
    AgentTraceStep,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ParadigmMode,
    RobotState,
)


class PureRAGStrategy(BaseInferenceStrategy):
    """Klassische RAG-Pipeline ohne mehrstufige Agenten-Schleifen."""

    async def execute(self, req: ChatRequest, history: List[ChatMessage]) -> ChatResponse:
        t_start = time.perf_counter()
        trace: List[AgentTraceStep] = []

        # 0. Automatische Schrift- und Spracherkennung (z. B. Arabisch / Kyrillisch / Türkisch)
        effective_lang = self.detect_effective_language(req.user_message, req.user_language)
        req.user_language = effective_lang

        # 1. Query-Rewriter bzw. Cross-Lingual Übersetzung der Suchanfrage ins Deutsche
        search_query, rewrite_step = await self.rewrite_query_if_needed(
            req.user_message, history, effective_lang=effective_lang
        )
        if rewrite_step:
            trace.append(rewrite_step)

        is_sensitive = detect_sensitive_topic(search_query)

        # 2. Hybrid-Retrieval & Quellen-Guard aufrufen (auf der deutschen Suchfrage)
        ret_resp, ret_step = await self.query_retrieval_service(
            search_query=search_query,
            score_threshold=req.score_threshold,
            filters=req.manual_filters,
        )
        trace.append(ret_step)

        # 3. Falls Quellen-Guard anschlägt -> Kontrollierter Fallback an AWO-Beratung (FA-06)
        if not ret_resp.passed_source_guard:
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)
            return ChatResponse(
                session_id=req.session_id,
                paradigm_used=ParadigmMode.PURE_RAG,
                translation_variant_used=req.translation_variant,
                answer_text=ret_resp.fallback_message or "Bitte wenden Sie sich an die AWO-Beratung.",
                rewritten_query=search_query,
                passed_source_guard=False,
                is_sensitive_topic=is_sensitive,
                sources=ret_resp.results,
                robot_state=RobotState.SPEAKING,
                agent_trace=trace,
                total_latency_ms=total_ms,
            )

        # 4. Prompt-Builder & LLM-Generierung (inkl. V2a / V2b Unterstützung)
        context_block = format_chunks_for_prompt(ret_resp.results)
        user_prompt = (
            f"GEFUNDENE QUELLEN AUS DER QUARTIERSDATENBANK:\n{context_block}\n\n"
            f"FRAGE DER BEWOHNERIN / DES BEWOHNERS:\n{req.user_message}"
        )

        sys_de = build_system_prompt(req.language_level, "de", is_sensitive)
        sys_target = build_system_prompt(req.language_level, req.user_language, is_sensitive)

        final_answer, german_inter, gen_steps = await self.handle_translation_variant(
            req=req,
            system_prompt_de=sys_de,
            system_prompt_target=sys_target,
            user_prompt_with_context=user_prompt,
            german_question=search_query,
        )
        trace.extend(gen_steps)

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        return ChatResponse(
            session_id=req.session_id,
            paradigm_used=ParadigmMode.PURE_RAG,
            translation_variant_used=req.translation_variant,
            answer_text=final_answer,
            german_intermediate_text=german_inter,
            rewritten_query=search_query,
            passed_source_guard=True,
            is_sensitive_topic=is_sensitive,
            sources=ret_resp.results,
            qr_code_payload=build_qr_code_payload(ret_resp.results),
            robot_state=RobotState.SPEAKING,
            agent_trace=trace,
            total_latency_ms=total_ms,
        )