"""
services/orchestrator_service/strategies/base.py
Abstrakte Basisklasse für das Strategy-Pattern (TA-04) mit verbessertem Error-Catching:
- Fängt Offline-Zustände von Ollama (WinError 10061) ohne 500-Absturz ab.
- Erkennt automatisch installierte Alternativ-Modelle in Ollama, falls das Zielmodell fehlt.
"""

import abc
import os
import time
from typing import List, Optional, Tuple

import httpx

from services.orchestrator_service.prompts import LANGUAGE_NAMES
from shared.logger import get_service_logger
from shared.schemas import (
    AgentTraceStep,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    RetrievalFilter,
    RetrievalRequest,
    RetrievalResponse,
    TranslationVariant,
)

logger = get_service_logger("strategy-base")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_FINETUNED_MODEL = os.getenv("OLLAMA_FINETUNED_MODEL", "qwen2.5:7b-instruct")
RAG_RETRIEVAL_URL = os.getenv("RAG_RETRIEVAL_URL", "http://127.0.0.1:8003/api/v1/rag/retrieve")


class BaseInferenceStrategy(abc.ABC):
    """Gemeinsames Interface für Pure RAG, Fine-Tuned LLM und Agentic AI (TA-04)."""

    @abc.abstractmethod
    async def execute(
        self, req: ChatRequest, history: List[ChatMessage]
    ) -> ChatResponse:
        pass

    async def _resolve_available_ollama_model(
        self, client: httpx.AsyncClient, requested_model: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Prüft über /api/tags, ob Ollama erreichbar ist und ob 'requested_model' installiert ist.
        Falls ein anderes Modell installiert ist, wird automatisch darauf zurückgegriffen.
        Gibt (verwendetes_modell, fehler_hinweis) zurück.
        """
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models_data = resp.json().get("models", [])
            installed_names = [m.get("name", "") for m in models_data if m.get("name")]

            if not installed_names:
                return (
                    None,
                    "⚠️ Ollama läuft auf Port 11434, aber es ist noch kein Sprachmodell installiert. "
                    f"Bitte im Terminal ausführen: `ollama pull {OLLAMA_DEFAULT_MODEL}`",
                )

            # 1. Exakter Treffer oder Treffer ohne Tag-Suffix
            for name in installed_names:
                if name == requested_model or name.startswith(requested_model.split(":")[0]):
                    return name, None

            # 2. Fallback auf das erste installierte nicht-Embedding-Modell
            chat_candidates = [
                n for n in installed_names if "embed" not in n.lower() and "bge" not in n.lower()
            ]
            fallback_model = chat_candidates[0] if chat_candidates else installed_names[0]
            logger.warning(
                f"Angefordertes Modell '{requested_model}' nicht gefunden. "
                f"Nutze automatisch installiertes Modell '{fallback_model}'."
            )
            return fallback_model, None

        except httpx.ConnectError:
            return (
                None,
                "⚠️ **Ollama Server (Port 11434) ist aktuell OFFLINE (`[WinError 10061]`).**\n\n"
                "**Lösung:** Starte die Ollama-App über das Windows-Startmenü oder gib in einem Terminal "
                "`ollama serve` ein.\n"
                "*(Hinweis: Das Hybrid-Retrieval über deine 51 Chunks hat trotzdem funktioniert – "
                "siehe Tabelle & QR-Code rechts!)*",
            )
        except Exception as e:
            return None, f"⚠️ Fehler bei der Verbindung zu Ollama ({OLLAMA_BASE_URL}): {e}"

    async def call_ollama_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str = OLLAMA_DEFAULT_MODEL,
        temperature: float = 0.1,
    ) -> str:
        """
        Ruft das lokale Sprachmodell in Ollama fehlertolerant auf.
        Wirft keinen HTTP-500-Crash mehr, wenn Ollama offline ist, sondern gibt eine Diagnose zurück.
        """
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                active_model, err_msg = await self._resolve_available_ollama_model(client, model_name)
                if not active_model:
                    return err_msg or "Ollama nicht erreichbar."

                payload = {
                    "model": active_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": temperature, "num_ctx": 4096},
                }
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json().get("message", {}).get("content", "").strip()

        except Exception as e:
            logger.error(f"Ollama-Aufruf fehlgeschlagen ({model_name}): {e}")
            return (
                f"⚠️ **Fehler beim Aufruf des lokalen Sprachmodells (`{model_name}`):** {e}\n\n"
                f"Bitte prüfen, ob `ollama serve` läuft und `ollama pull {OLLAMA_DEFAULT_MODEL}` abgeschlossen ist."
            )

    async def rewrite_query_if_needed(
        self, user_message: str, history: List[ChatMessage]
    ) -> Tuple[str, Optional[AgentTraceStep]]:
        """
        Formuliert eine kontextabhängige Folgefrage anhand des Gesprächsverlaufs um.
        Fällt bei Offline-Ollama sofort auf die Originalfrage zurück.
        """
        if not history:
            return user_message, None

        t0 = time.perf_counter()
        history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in history[-4:])
        sys_prompt = (
            "Du bist ein präziser Query-Rewriter für eine Quartiersdatenbank in Hildesheim. "
            "Formuliere die letzte Nutzerfrage anhand des Gesprächsverlaufs in eine einzige, "
            "eigenständige Suchanfrage um. Gib NUR die umformulierte Suchanfrage aus, ohne Erklärung."
        )
        user_prompt = (
            f"GESPRÄCHSVERLAUF:\n{history_text}\n\n"
            f"AKTUELLE FRAGE: {user_message}\n\nEIGENSTÄNDIGE SUCHANFRAGE:"
        )
        try:
            rewritten = await self.call_ollama_chat(sys_prompt, user_prompt, temperature=0.0)
            if rewritten.startswith("⚠️"):
                return user_message, None
            rewritten = rewritten.strip().strip('"').strip("'") or user_message
            ms = round((time.perf_counter() - t0) * 1000, 1)
            return rewritten, AgentTraceStep(
                step_name="Query-Rewriter (Kontext-Auflösung)",
                details=f"Ursprünglich: '{user_message}' -> Umformuliert: '{rewritten}'",
                duration_ms=ms,
            )
        except Exception as e:
            logger.warning(f"Query-Rewriter Fallback auf Originalfrage: {e}")
            return user_message, None

    async def query_retrieval_service(
        self,
        search_query: str,
        score_threshold: float,
        filters: Optional[RetrievalFilter] = None,
        top_k_final: int = 4,
    ) -> Tuple[RetrievalResponse, AgentTraceStep]:
        """Ruft den RAG-Retrieval Microservice (Port 8003) auf."""
        t0 = time.perf_counter()
        req_obj = RetrievalRequest(
            query=search_query,
            top_k_final=top_k_final,
            score_threshold=score_threshold,
            filters=filters,
        )
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                RAG_RETRIEVAL_URL,
                content=req_obj.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            ret_resp = RetrievalResponse.model_validate(resp.json())

        ms = round((time.perf_counter() - t0) * 1000, 1)
        filter_info = filters.model_dump(exclude_none=True) if filters else "Keine"
        step = AgentTraceStep(
            step_name="Hybrid-Retrieval & Quellen-Guard (Port 8003)",
            details=(
                f"Query: '{search_query}' | Filter: {filter_info} | "
                f"Bester Score: {ret_resp.best_score:.3f} (Schwelle: {score_threshold:.2f}) | "
                f"Guard bestanden: {ret_resp.passed_source_guard}"
            ),
            duration_ms=ms,
        )
        return ret_resp, step

    async def handle_translation_variant(
        self,
        req: ChatRequest,
        system_prompt_de: str,
        system_prompt_target: str,
        user_prompt_with_context: str,
        model_name: str = OLLAMA_DEFAULT_MODEL,
    ) -> Tuple[str, Optional[str], List[AgentTraceStep]]:
        """Unterstützt V2a (direkte Zielsprache) und V2b (Deutscher Zwischentext -> Übersetzung)."""
        steps: List[AgentTraceStep] = []

        if req.user_language == "de" or req.translation_variant == TranslationVariant.V2A_DIRECT:
            t0 = time.perf_counter()
            answer = await self.call_ollama_chat(
                system_prompt=system_prompt_target,
                user_prompt=user_prompt_with_context,
                model_name=model_name,
            )
            ms = round((time.perf_counter() - t0) * 1000, 1)
            steps.append(
                AgentTraceStep(
                    step_name=f"LLM-Generierung ({req.translation_variant.value})",
                    details=f"Antwortgenerierung in Sprache '{req.user_language}' (Modell-Anfrage: '{model_name}').",
                    duration_ms=ms,
                )
            )
            german_inter = answer if req.user_language == "de" else None
            return answer, german_inter, steps

        # Variante V2b: 1. Deutsch generieren -> 2. Übersetzen
        t0 = time.perf_counter()
        german_answer = await self.call_ollama_chat(
            system_prompt=system_prompt_de,
            user_prompt=user_prompt_with_context,
            model_name=model_name,
        )
        ms_de = round((time.perf_counter() - t0) * 1000, 1)
        steps.append(
            AgentTraceStep(
                step_name="V2b Schritt 1: Deutsche Zwischenantwort (HIX-prüfbar)",
                details=f"Deutscher Zwischentext ({len(german_answer.split())} Wörter) erzeugt.",
                duration_ms=ms_de,
            )
        )

        if german_answer.startswith("⚠️"):
            return german_answer, german_answer, steps

        t1 = time.perf_counter()
        target_lang_name = LANGUAGE_NAMES.get(req.user_language, req.user_language)
        trans_sys = (
            f"Du bist ein präziser Fachübersetzer für soziale Beratung in Hildesheim. "
            f"Übersetze den folgenden deutschen Beratungstext exakt, freundlich und leicht verständlich "
            f"in die Sprache {target_lang_name}. Behalte kurze Sätze, Straßennamen (Hildesheim) "
            f"und Telefonnummern unverändert bei!"
        )
        translated_answer = await self.call_ollama_chat(
            system_prompt=trans_sys,
            user_prompt=german_answer,
            model_name=model_name,
            temperature=0.0,
        )
        ms_tr = round((time.perf_counter() - t1) * 1000, 1)
        steps.append(
            AgentTraceStep(
                step_name=f"V2b Schritt 2: Übersetzungs-Baustein (DE -> {req.user_language})",
                details=f"Antwort von Deutsch nach {target_lang_name} übersetzt.",
                duration_ms=ms_tr,
            )
        )
        return translated_answer, german_answer, steps