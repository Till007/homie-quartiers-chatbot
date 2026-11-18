"""
services/orchestrator_service/strategies/base.py
Abstrakte Basisklasse für das Strategy-Pattern (TA-04):
- Trennt strikt zwischen Eingabesprache (der Frage) und gewünschter Ausgabesprache (Dropdown)
- Verhindert unnötige Query-Übersetzungen bei deutschen Eingaben (spart ~5.5 Sekunden!)
- Schützt das Retrieval vor History-Pollution nach mehreren Gesprächsrunden
- CJK-Guard: Verhindert und repariert chinesische Zeichen (Qwen 2.5 Code-Switching) bei Türkisch/Arabisch/Ukrainisch
"""

import abc
import os
import re
import time
from typing import List, Optional, Tuple

import httpx
from langdetect import LangDetectException, detect

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

# Regex zur Erkennung von chinesischen/japanischen/koreanischen Zeichen (Qwen Code-Switching Guard)
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


class BaseInferenceStrategy(abc.ABC):
    """Gemeinsames Interface für Pure RAG, Fine-Tuned LLM und Agentic AI (TA-04)."""

    @abc.abstractmethod
    async def execute(
        self, req: ChatRequest, history: List[ChatMessage]
    ) -> ChatResponse:
        pass

    def detect_input_language(self, user_message: str) -> str:
        """
        Erkennt ausschließlich die Sprache der eingegebenen Nutzerfrage (unabhängig vom Zielsprachen-Dropdown!).
        Verhindert, dass eine deutsche Frage unnötig ins Deutsche übersetzt wird.
        """
        if re.search(r"[\u0600-\u06FF]", user_message):
            return "ar"
        if re.search(r"[\u0400-\u04FF]", user_message):
            if re.search(r"[іїєґІЇЄҐ]", user_message):
                return "uk"
            return "ru"

        # Typisch türkische Buchstaben oder Wörter
        if re.search(r"[ğışĞİŞ]", user_message) or any(
            w in user_message.lower().split()
            for w in ["nerede", "nasıl", "yardım", "bakım", "için", "var", "istiyorum"]
        ):
            return "tr"

        # Typisch deutsche Wörter sofort als Deutsch ('de') erkennen
        german_markers = [
            "ich", "wie", "wo", "was", "wann", "gibt", "brauche", "suche", "hilfe",
            "mit", "dem", "der", "die", "das", "nach", "für", "und", "oder", "bus",
            "bahnhof", "hildesheim", "pflege", "beratung", "geld", "antrag"
        ]
        words_lower = re.findall(r"\b\w+\b", user_message.lower())
        if any(w in german_markers for w in words_lower):
            return "de"

        if len(words_lower) >= 4:
            try:
                detected = str(detect(user_message))
                if detected in ("ar", "tr", "uk", "ru", "en", "de"):
                    return detected
            except LangDetectException:
                pass

        return "de"

    def detect_effective_language(self, user_message: str, selected_lang: str) -> str:
        """
        Bestimmt die gewünschte AUSGABESPRACHE:
        - Wenn der Nutzer im Dropdown explizit eine Fremdsprache (z.B. 'tr') gewählt hat -> 'tr'.
        - Wenn das Dropdown auf 'de' steht, der Nutzer aber auf Arabisch/Türkisch/Ukrainisch schreibt -> Sprache der Eingabe.
        """
        if selected_lang != "de":
            return selected_lang
        input_lang = self.detect_input_language(user_message)
        return input_lang if input_lang != "de" else "de"

    async def _resolve_available_ollama_model(
        self, client: httpx.AsyncClient, requested_model: str
    ) -> Tuple[Optional[str], Optional[str]]:
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

            for name in installed_names:
                if name == requested_model or name.startswith(requested_model.split(":")[0]):
                    return name, None

            chat_candidates = [
                n for n in installed_names if "embed" not in n.lower() and "bge" not in n.lower()
            ]
            fallback_model = chat_candidates[0] if chat_candidates else installed_names[0]
            return fallback_model, None

        except httpx.ConnectError:
            return (
                None,
                "⚠️ **Ollama Server (Port 11434) ist aktuell OFFLINE (`[WinError 10061]`).**\n\n"
                "**Lösung:** Starte die Ollama-App oder gib in einem Terminal `ollama serve` ein.",
            )
        except Exception as e:
            return None, f"⚠️ Fehler bei der Verbindung zu Ollama ({OLLAMA_BASE_URL}): {e}"

    async def call_ollama_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str = OLLAMA_DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 380,
    ) -> str:
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
                    "keep_alive": "30m",
                    "options": {
                        "temperature": temperature,
                        "num_ctx": 3072,
                        "num_predict": max_tokens,
                        "repeat_penalty": 1.12,
                    },
                }
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json().get("message", {}).get("content", "").strip()

        except Exception as e:
            logger.error(f"Ollama-Aufruf fehlgeschlagen ({model_name}): {e}")
            return f"⚠️ **Fehler beim Aufruf des lokalen Sprachmodells (`{model_name}`):** {e}"

    async def translate_query_to_german(
        self, user_message: str, input_lang: str
    ) -> Tuple[str, Optional[AgentTraceStep]]:
        """Übersetzt eine Anfrage NUR dann ins Deutsche, wenn der Eingabetext wirklich fremdsprachig ist."""
        if input_lang == "de":
            return user_message, None

        t0 = time.perf_counter()
        lang_name = LANGUAGE_NAMES.get(input_lang, input_lang)
        sys_prompt = (
            "Du bist ein präziser Übersetzer für soziale Beratungsfragen in Hildesheim. "
            "Übersetze die folgende Nutzerfrage ins Deutsche. "
            "Gib AUSSCHLIESSLICH die kurze deutsche Übersetzung der Frage aus, ohne Einleitung!"
        )
        de_query = await self.call_ollama_chat(
            sys_prompt, user_message, temperature=0.0, max_tokens=80
        )
        if de_query.startswith("⚠️") or not de_query:
            return user_message, None

        # Nimm nur die erste Zeile, falls das LLM Erklärungen anhängt
        de_query = de_query.splitlines()[0].strip().strip('"').strip("'")
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return de_query, AgentTraceStep(
            step_name=f"Cross-Lingual Query-Brücke ({lang_name} -> Deutsch)",
            details=f"Eingabe ({input_lang}): '{user_message}' -> Deutsche Suchfrage: '{de_query}'",
            duration_ms=ms,
        )

    def _is_contextual_followup(self, german_query: str) -> bool:
        """
        Prüft, ob die aktuelle Frage wirklich eine kontextabhängige Rückfrage zur vorherigen Runde ist
        (z. B. 'Und wann haben die offen?', 'Wie ist die Adresse dort?').
        Verhindert, dass eigenständige neue Fragen durch alte Chat-Historie verfälscht werden!
        """
        q_lower = german_query.lower().strip()
        words = re.findall(r"\b\w+\b", q_lower)

        # Sehr kurze Fragen (< 4 Wörter) wie "Und die Telefonnummer?" oder "Wo ist das?"
        if len(words) <= 3:
            return True

        referential_markers = [
            "dort", "da ", "diese", "dieser", "dieses", "diesem", "davon", "dafür",
            "die offen", "haben die", "sind die", "sprechen die", "kostet das",
            "wie komme ich dahin", "und wann", "und wo", "und wie"
        ]
        return any(marker in q_lower for marker in referential_markers)

    async def rewrite_query_if_needed(
        self, user_message: str, history: List[ChatMessage], effective_lang: str = "de"
    ) -> Tuple[str, Optional[AgentTraceStep]]:
        """
        1. Prüft die tatsächliche Eingabesprache der Frage: Ist sie fremdsprachig -> Übersetzung ins Deutsche.
        2. Ist es eine deutsche Frage: Nur dann mit der letzten Nutzerfrage kombinieren, wenn es eine echte Rückfrage ist!
        """
        input_lang = self.detect_input_language(user_message)
        working_query = user_message
        translation_step: Optional[AgentTraceStep] = None

        if input_lang != "de":
            working_query, translation_step = await self.translate_query_to_german(
                user_message, input_lang
            )

        # Wenn kein Verlauf existiert oder die Frage eigenständig ist -> Direkt suchen (kein History-Pollution!)
        if not history or not self._is_contextual_followup(working_query):
            return working_query, translation_step

        # Hole nur die letzten 2 NUTZER-Fragen aus dem Verlauf (keine langen Bot-Antworten!)
        prev_user_questions = [m.content for m in history if m.role == "user"][-2:]
        if not prev_user_questions:
            return working_query, translation_step

        t0 = time.perf_counter()
        context_str = " -> ".join(prev_user_questions)
        sys_prompt = (
            "Formuliere die aktuelle kurze Rückfrage mithilfe des vorherigen Gesprächsthemas in eine "
            "einzige, kurze deutsche Suchanfrage (maximal 12 Wörter) um. "
            "Gib NUR die Suchanfrage aus, niemals eine Antwort!"
        )
        user_prompt = f"Vorherige Frage(n): {context_str}\nAktuelle Rückfrage: {working_query}\nSuchanfrage:"
        try:
            rewritten = await self.call_ollama_chat(
                sys_prompt, user_prompt, temperature=0.0, max_tokens=45
            )
            rewritten = rewritten.splitlines()[0].strip().strip('"').strip("'")
            # Sicherheits-Check: Falls das LLM mehr als 18 Wörter generiert hat, verwerfen!
            if rewritten.startswith("⚠️") or not rewritten or len(rewritten.split()) > 18:
                return working_query, translation_step

            ms = round((time.perf_counter() - t0) * 1000, 1)
            step = AgentTraceStep(
                step_name="Query-Rewriter (Rückfrage-Auflösung)",
                details=f"Rückfrage: '{working_query}' -> Suchanfrage: '{rewritten}'",
                duration_ms=ms,
            )
            return rewritten, step
        except Exception as e:
            logger.warning(f"Query-Rewriter Fallback auf Originalfrage: {e}")
            return working_query, translation_step

    async def query_retrieval_service(
        self,
        search_query: str,
        score_threshold: float,
        filters: Optional[RetrievalFilter] = None,
        top_k_final: int = 4,
    ) -> Tuple[RetrievalResponse, AgentTraceStep]:
        t0 = time.perf_counter()
        req_obj = RetrievalRequest(
            query=search_query,
            top_k_candidates=8,
            top_k_final=top_k_final,
            score_threshold=score_threshold,
            filters=filters,
        )
        async with httpx.AsyncClient(timeout=25.0) as client:
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

    async def _sanitize_and_fix_cjk_leakage(
        self, text: str, target_lang_code: str, model_name: str
    ) -> Tuple[str, bool]:
        """
        Prüft, ob Qwen 2.5 versehentlich chinesische Zeichen (Hanzi) in eine türkische,
        arabische oder ukrainische Antwort gemischt hat, und übersetzt den Text bei Bedarf sauber neu.
        """
        if not CJK_PATTERN.search(text):
            return text, False

        logger.warning(
            f"CJK-Language-Drift (chinesische Zeichen) in '{target_lang_code}'-Antwort erkannt! Starte Auto-Bereinigung..."
        )
        # Entferne zuerst alle CJK-Zeichenblöcke und lasse das Modell den Text sauber in der Zielsprache glätten
        stripped = CJK_PATTERN.sub(" ", text)
        stripped = re.sub(r"\s+", " ", stripped).strip()

        target_lang_name = LANGUAGE_NAMES.get(target_lang_code, target_lang_code)
        fix_sys = (
            f"You are a native {target_lang_name} editor. Rewrite the following text cleanly and naturally "
            f"in {target_lang_name} ONLY. Remove any broken fragments or foreign words. "
            f"NEVER use Chinese characters!"
        )
        fixed = await self.call_ollama_chat(
            system_prompt=fix_sys,
            user_prompt=stripped,
            model_name=model_name,
            temperature=0.0,
            max_tokens=350,
        )
        # Falls immer noch CJK-Zeichen vorhanden sind, hart per Regex entfernen
        final_clean = CJK_PATTERN.sub("", fixed).strip()
        return (final_clean or stripped), True

    async def handle_translation_variant(
        self,
        req: ChatRequest,
        system_prompt_de: str,
        system_prompt_target: str,
        user_prompt_with_context: str,
        model_name: str = OLLAMA_DEFAULT_MODEL,
        german_question: Optional[str] = None,
    ) -> Tuple[str, Optional[str], List[AgentTraceStep]]:
        steps: List[AgentTraceStep] = []
        effective_lang = self.detect_effective_language(req.user_message, req.user_language)
        req.user_language = effective_lang
        target_lang_name = LANGUAGE_NAMES.get(effective_lang, effective_lang)

        de_q = german_question or req.user_message
        prompt_for_german_step = re.sub(
            r"FRAGE DER BEWOHNERIN / DES BEWOHNERS:\s*.*$",
            f"FRAGE DER BEWOHNERIN / DES BEWOHNERS (auf Deutsch):\n{de_q}\n"
            "(WICHTIG: Antworte ausschließlich anhand der obigen Quellen auf DEUTSCH. "
            "Wenn die Quellen keine Antwort auf genau diese Frage enthalten, erfinde NICHTS, sondern "
            "sage ehrlich, dass dazu keine Information in den vorliegenden Dokumenten steht!)",
            user_prompt_with_context,
            flags=re.DOTALL,
        )

        # -----------------------------------------------------------------
        # FALL 1: Zielsprache ist Deutsch
        # -----------------------------------------------------------------
        if effective_lang == "de":
            t0 = time.perf_counter()
            answer = await self.call_ollama_chat(
                system_prompt=system_prompt_de,
                user_prompt=prompt_for_german_step,
                model_name=model_name,
            )
            answer = CJK_PATTERN.sub("", answer).strip()
            ms = round((time.perf_counter() - t0) * 1000, 1)
            steps.append(
                AgentTraceStep(
                    step_name="LLM-Generierung (Deutsch)",
                    details=f"Antwort auf Deutsch mit Modell '{model_name}' erzeugt.",
                    duration_ms=ms,
                )
            )
            return answer, answer, steps

        # -----------------------------------------------------------------
        # FALL 2: Fremdsprache mit V2a (Direkte Antwort in Zielsprache)
        # -----------------------------------------------------------------
        if req.translation_variant == TranslationVariant.V2A_DIRECT:
            t0 = time.perf_counter()
            v2a_user_prompt = (
                f"{user_prompt_with_context}\n"
                f"[Deutsche Bedeutung der Frage: {de_q}]\n"
                f"[STRICT OUTPUT RULE: Write your entire response ONLY in {target_lang_name}! "
                f"NEVER use Chinese characters (中文/汉字)! Do NOT invent bus lines, apps, or phone numbers "
                f"that are not explicitly written in the sources above!]"
            )
            answer = await self.call_ollama_chat(
                system_prompt=system_prompt_target,
                user_prompt=v2a_user_prompt,
                model_name=model_name,
            )
            answer, was_cjk_fixed = await self._sanitize_and_fix_cjk_leakage(
                answer, effective_lang, model_name
            )
            ms = round((time.perf_counter() - t0) * 1000, 1)
            cjk_note = " (inkl. CJK-Sprachdrift-Bereinigung)" if was_cjk_fixed else ""
            steps.append(
                AgentTraceStep(
                    step_name=f"LLM-Generierung (V2a Direkt in {target_lang_name})",
                    details=f"Direkte Antwort in '{effective_lang}' erzeugt{cjk_note}.",
                    duration_ms=ms,
                )
            )
            return answer, None, steps

        # -----------------------------------------------------------------
        # FALL 3: Fremdsprache mit V2b (1. Deutscher Zwischentext -> 2. Übersetzung)
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        german_answer = await self.call_ollama_chat(
            system_prompt=system_prompt_de,
            user_prompt=prompt_for_german_step,
            model_name=model_name,
        )
        german_answer = CJK_PATTERN.sub("", german_answer).strip()
        ms_de = round((time.perf_counter() - t0) * 1000, 1)
        steps.append(
            AgentTraceStep(
                step_name="V2b Schritt 1: Deutsche Zwischenantwort (HIX-prüfbar)",
                details=f"Deutscher Zwischentext ({len(german_answer.split())} Wörter) aus deutschen Quellen erzeugt.",
                duration_ms=ms_de,
            )
        )

        if german_answer.startswith("⚠️"):
            return german_answer, german_answer, steps

        t1 = time.perf_counter()
        trans_sys = (
            f"You are a professional translator for social services in Hildesheim.\n"
            f"Translate the following German text accurately, politely, and in easy-to-understand language "
            f"into {target_lang_name}.\n"
            f"STRICT RULES:\n"
            f"1. Output ONLY the translated text in {target_lang_name}.\n"
            f"2. NEVER output Chinese characters (中文/汉字)!\n"
            f"3. Keep street names in Hildesheim and phone numbers unchanged."
        )
        translated_answer = await self.call_ollama_chat(
            system_prompt=trans_sys,
            user_prompt=german_answer,
            model_name=model_name,
            temperature=0.0,
        )
        translated_answer, _ = await self._sanitize_and_fix_cjk_leakage(
            translated_answer, effective_lang, model_name
        )
        ms_tr = round((time.perf_counter() - t1) * 1000, 1)
        steps.append(
            AgentTraceStep(
                step_name=f"V2b Schritt 2: Übersetzungs-Baustein (DE -> {effective_lang})",
                details=f"Deutscher Zwischentext sauber nach {target_lang_name} übersetzt.",
                duration_ms=ms_tr,
            )
        )
        return translated_answer, german_answer, steps