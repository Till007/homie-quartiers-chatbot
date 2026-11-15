"""
services/orchestrator_service/strategies/fine_tuned.py
Paradigma 2 (TA-04): Fine-Tuned LLM (QLoRA-Adapter für Quartiers-Tonalität & Leichte Sprache).
Verwendet einen schlanken System-Prompt, da Stil & Formatierung im Modell feinjustiert sind.
"""

import time
from typing import List

from services.orchestrator_service.prompts import (
    KNOWLEDGE_CUTOFF,
    LANGUAGE_NAMES,
    build_qr_code_payload,
    detect_sensitive_topic,
    format_chunks_for_prompt,
)
from services.orchestrator_service.strategies.base import (
    OLLAMA_FINETUNED_MODEL,
    BaseInferenceStrategy,
)
from shared.schemas import (
    AgentTraceStep,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ParadigmMode,
    RobotState,
)


class FineTunedStrategy(BaseInferenceStrategy):
    """
    Nutzt ein feinjustiertes Ollama-Modell (OLLAMA_FINETUNED_MODEL, z. B. 'homie-qwen-qlora')
    mit minimalem Prompt-Overhead für niedrige Latenz auf dem Jetson-Roboter.
    """

    async def execute(self, req: ChatRequest, history: List[ChatMessage]) -> ChatResponse:
        t_start = time.perf_counter()
        trace: List[AgentTraceStep] = []
        is_sensitive = detect_sensitive_topic(req.user_message)

        search_query, rewrite_step = await self.rewrite_query_if_needed(req.user_message, history)
        if rewrite_step:
            trace.append(rewrite_step)

        ret_resp, ret_step = await self.query_retrieval_service(
            search_query=search_query,
            score_threshold=req.score_threshold,
            filters=req.manual_filters,
            top_k_final=3,  # Kompakterer Kontext für Fine-Tuned-Variante
        )
        trace.append(ret_step)

        if not ret_resp.passed_source_guard:
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)
            return ChatResponse(
                session_id=req.session_id,
                paradigm_used=ParadigmMode.FINE_TUNED,
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

        # Kompakter System-Prompt für das Fine-Tuned Modell (Stil liegt im QLoRA-Adapter)
        lang_name = LANGUAGE_NAMES.get(req.user_language, "Deutsch")
        compact_sys_de = (
            f"[HOMIE-QLoRA | Modus: {req.language_level.value} | Stand: {KNOWLEDGE_CUTOFF}] "
            "Beantworte die Nutzerfrage ausschließlich anhand der folgenden Fakten inklusive Adresse, Telefon und Quelle auf Deutsch."
        )
        compact_sys_target = (
            f"[HOMIE-QLoRA | Modus: {req.language_level.value} | Stand: {KNOWLEDGE_CUTOFF}] "
            f"Beantworte die Nutzerfrage ausschließlich anhand der folgenden Fakten inklusive Adresse, Telefon und Quelle in der Sprache {lang_name}."
        )

        context_block = format_chunks_for_prompt(ret_resp.results)
        user_prompt = f"FAKTEN:\n{context_block}\n\nFRAGE: {req.user_message}"

        final_answer, german_inter, gen_steps = await self.handle_translation_variant(
            req=req,
            system_prompt_de=compact_sys_de,
            system_prompt_target=compact_sys_target,
            user_prompt_with_context=user_prompt,
            model_name=OLLAMA_FINETUNED_MODEL,
        )
        trace.extend(gen_steps)

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        return ChatResponse(
            session_id=req.session_id,
            paradigm_used=ParadigmMode.FINE_TUNED,
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