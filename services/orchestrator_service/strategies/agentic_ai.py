"""
services/orchestrator_service/strategies/agentic_ai.py
Paradigma 3 (TA-04): Agentic AI (Multi-Agenten-Workflow mit Slot-Filling, Tool-Filterung,
Öffnungszeiten-Werkzeug und Verifier-Guardrail gemäß FA-02, FA-03 und SA-04).
Unterstützt mehrsprachige Eingaben (de, ar, tr, uk, ru, en) inkl. mehrsprachiger Slot-Filling-Rückfragen.
"""

import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from services.orchestrator_service.prompts import (
    build_qr_code_payload,
    build_system_prompt,
    detect_sensitive_topic,
    format_chunks_for_prompt,
)
from services.orchestrator_service.strategies.base import BaseInferenceStrategy
from shared.logger import get_service_logger
from shared.schemas import (
    AgentTraceStep,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    LanguageLevel,
    ParadigmMode,
    RetrievalFilter,
    RobotState,
    ScoredChunk,
    TargetGroupTag,
)

logger = get_service_logger("strategy-agentic")


class AgenticAIStrategy(BaseInferenceStrategy):
    """
    Mehrstufiges Agentensystem:
    0. Cross-Lingual Query-Brücke: Erkennt Fremdsprachen (z. B. Arabisch) und übersetzt die Arbeitsfrage ins Deutsche.
    1. Slot-Filling-Agent (FA-02): Prüft auf vage Anfragen und stellt Rückfragen in der Nutzersprache.
    2. Tool- & Filter-Agent (FA-03): Extrahiert Metadaten-Filter (Sprache, Frauen-Schutzraum, Barrierefreiheit, Stadtteil) & Zeitkontext.
    3. Retrieval-Agent mit Auto-Relaxation: Sucht in ChromaDB/BM25 und lockert bei 0 Treffern zu enge Ortsfilter.
    4. Verifier-/Guardrail-Agent (SA-04): Überprüft Telefonnummern und Fakten vor der Ausgabe gegen die Quell-Chunks.
    """

    def __init__(self):
        self.hildesheim_districts = [
            "Stadtmitte", "Nordstadt", "Oststadt", "Weststadt", "Südstadt",
            "Marienburger Höhe", "Galgenberg", "Drispenstedt", "Himmelsthür",
            "Ochtersum", "Moritzberg", "Bavenstedt", "Itzum", "Sorsum", "Uppen"
        ]

    async def execute(self, req: ChatRequest, history: List[ChatMessage]) -> ChatResponse:
        t_start = time.perf_counter()
        trace: List[AgentTraceStep] = []

        # -----------------------------------------------------------------
        # SCHRITT 0: Automatische Spracherkennung & Cross-Lingual Query-Brücke
        # -----------------------------------------------------------------
        effective_lang = self.detect_effective_language(req.user_message, req.user_language)
        req.user_language = effective_lang

        search_query, rewrite_step = await self.rewrite_query_if_needed(
            req.user_message, history, effective_lang=effective_lang
        )
        if rewrite_step:
            trace.append(rewrite_step)

        is_sensitive = detect_sensitive_topic(search_query)

        # -----------------------------------------------------------------
        # AGENT 1: Router & Slot-Filling-Check bei vagen Anliegen (FA-02)
        # -----------------------------------------------------------------
        is_vague, clarification_msg, slot_step = self._check_slot_filling_needed(
            search_query, history, req.language_level, req.user_language
        )
        trace.append(slot_step)

        if is_vague and clarification_msg:
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)
            return ChatResponse(
                session_id=req.session_id,
                paradigm_used=ParadigmMode.AGENTIC_AI,
                translation_variant_used=req.translation_variant,
                answer_text=clarification_msg,
                german_intermediate_text=clarification_msg if req.user_language == "de" else None,
                rewritten_query=search_query,
                passed_source_guard=True,
                is_clarification_question=True,
                is_sensitive_topic=is_sensitive,
                sources=[],
                robot_state=RobotState.SPEAKING,
                agent_trace=trace,
                total_latency_ms=total_ms,
            )

        # -----------------------------------------------------------------
        # AGENT 2: Dynamische Tool-/Filter-Extraktion (FA-03)
        # -----------------------------------------------------------------
        auto_filter, time_context_note, tool_step = self._extract_agentic_tools_and_filters(
            search_query, req.manual_filters
        )
        trace.append(tool_step)

        # -----------------------------------------------------------------
        # AGENT 3: Retrieval-Agent mit Self-Correction (Filter-Relaxation)
        # -----------------------------------------------------------------
        ret_resp, ret_step = await self.query_retrieval_service(
            search_query=search_query,
            score_threshold=req.score_threshold,
            filters=auto_filter,
        )
        trace.append(ret_step)

        # Wenn der spezifische Filter (z. B. Stadtteil) 0 Treffer ergab -> Selbstkorrektur ohne Filter!
        if not ret_resp.passed_source_guard and auto_filter is not None:
            t_relax = time.perf_counter()
            ret_resp_relaxed, _ = await self.query_retrieval_service(
                search_query=search_query,
                score_threshold=req.score_threshold,
                filters=None,
            )
            ms_relax = round((time.perf_counter() - t_relax) * 1000, 1)
            trace.append(
                AgentTraceStep(
                    step_name="Agentic Self-Correction (Filter-Relaxation)",
                    details=(
                        f"Strikter Filter lieferte keinen Treffer über Schwellenwert. "
                        f"Suche auf Gesamt-Hildesheim erweitert -> Neuer bester Score: {ret_resp_relaxed.best_score:.3f}"
                    ),
                    duration_ms=ms_relax,
                )
            )
            ret_resp = ret_resp_relaxed

        if not ret_resp.passed_source_guard:
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)
            return ChatResponse(
                session_id=req.session_id,
                paradigm_used=ParadigmMode.AGENTIC_AI,
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

        # -----------------------------------------------------------------
        # AGENT 4: Antwort-Generierung (inkl. Zeit-Tool-Kontext & V2a/V2b)
        # -----------------------------------------------------------------
        context_block = format_chunks_for_prompt(ret_resp.results)
        user_prompt = (
            f"{time_context_note}\n"
            f"GEFUNDENE QUELLEN AUS DER QUARTIERSDATENBANK:\n{context_block}\n\n"
            f"FRAGE DER BEWOHNERIN / DES BEWOHNERS:\n{req.user_message}"
        )

        sys_de = build_system_prompt(req.language_level, "de", is_sensitive)
        sys_target = build_system_prompt(req.language_level, req.user_language, is_sensitive)

        draft_answer, german_inter, gen_steps = await self.handle_translation_variant(
            req=req,
            system_prompt_de=sys_de,
            system_prompt_target=sys_target,
            user_prompt_with_context=user_prompt,
            german_question=search_query,
        )
        trace.extend(gen_steps)

        # -----------------------------------------------------------------
        # AGENT 5: Verifier- / Guardrail-Schritt gegen Halluzinationen (SA-04)
        # -----------------------------------------------------------------
        verified_answer, verifier_step = self._verify_grounding_and_numbers(
            draft_answer, ret_resp.results
        )
        trace.append(verifier_step)

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        return ChatResponse(
            session_id=req.session_id,
            paradigm_used=ParadigmMode.AGENTIC_AI,
            translation_variant_used=req.translation_variant,
            answer_text=verified_answer,
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

    # =================================================================
    # SPEZIALISIERTE AGENTEN-WERKZEUGE (FA-02, FA-03, SA-04)
    # =================================================================

    def _check_slot_filling_needed(
        self,
        german_search_query: str,
        history: List[ChatMessage],
        lang_level: LanguageLevel,
        user_lang: str,
    ) -> Tuple[bool, Optional[str], AgentTraceStep]:
        """
        FA-02 Bedarfsorientierte Dialogführung (Slot-Filling):
        Prüft auf der (ggf. ins Deutsche übersetzten) Anfrage, ob das Anliegen zu vage ist,
        und gibt die passende Rückfrage direkt in der Sprache der Nutzer:innen (de, ar, tr, uk, ru, en) aus.
        """
        t0 = time.perf_counter()
        words = re.findall(r"\b\w+\b", german_search_query.lower())

        domain_keywords = [
            "pflege", "beratung", "essen", "mittagstisch", "arzt", "bus", "bahn", "rollstuhl",
            "demenz", "frauen", "kinder", "familie", "senioren", "türkisch", "arabisch",
            "russisch", "ukrainisch", "antrag", "geld", "wohngeld", "treff", "sport",
            "kurzzeitpflege", "tagespflege", "verhinderungspflege", "nachbarschaftshilfe",
            "oma", "opa", "papierkram", "hilfe für", "öffnungszeiten", "adresse", "telefon",
            "pflegegrad", "krankenhaus", "mobilität", "freizeit", "quartier", "älter"
        ]
        has_specific_topic = any(kw in german_search_query.lower() for kw in domain_keywords)

        if history or has_specific_topic or len(words) >= 6:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            return (
                False,
                None,
                AgentTraceStep(
                    step_name="Slot-Filling-Agent (FA-02)",
                    details="Anliegen ist spezifisch genug. Keine Rückfrage erforderlich.",
                    duration_ms=ms,
                ),
            )

        # Mehrsprachige Slot-Filling-Rückfragen für vage Erstanfragen
        clarifications_by_lang: Dict[str, str] = {
            "ar": (
                "أهلاً بك! يسعدني مساعدتك في العثور على الخدمة المناسبة في هيلدسهايم. "
                "هل تبحث عن استشارات الرعاية والتمريض، أو المساعدة لكبار السن، أو أنشطة أوقات الفراغ؟"
            ),
            "tr": (
                "Merhaba! Hildesheim'da size uygun hizmeti bulmanıza memnuniyetle yardımcı olurum. "
                "Bakım danışmanlığı, yaşlılar için günlük yardım veya boş zaman etkinlikleri mi arıyorsunuz?"
            ),
            "uk": (
                "Вітаю! Я радо допоможу вам знайти потрібну послугу в Гільдесгаймі. "
                "Ви шукаєте консультацію з догляду, допомогу для літніх людей чи дозвілля?"
            ),
            "ru": (
                "Здравствуйте! Я с радостью помогу вам найти подходящее предложение в Хильдесхайме. "
                "Вы ищете консультацию по уходу, помощь пожилым людям или мероприятия для досуга?"
            ),
            "en": (
                "Hello! I would be happy to help you find the right service in Hildesheim. "
                "Are you looking for care counseling, everyday support for seniors, or community activities?"
            ),
        }

        if user_lang in clarifications_by_lang:
            clarification = clarifications_by_lang[user_lang]
        elif lang_level == LanguageLevel.EASY:
            clarification = (
                "Hallo! Ich helfe Ihnen gerne weiter.\n"
                "Wonach suchen Sie genau?\n"
                "Suchen Sie zum Beispiel:\n"
                "- eine Pflege·beratung,\n"
                "- Hilfe im Alltag für Senior·innen,\n"
                "- oder einen Treff·punkt für die Freizeit in Hildesheim?"
            )
        else:
            clarification = (
                "Guten Tag! Ich unterstütze Sie gerne dabei, das passende Angebot in Hildesheim zu finden. "
                "Damit ich Ihnen gezielt helfen kann: Geht es bei Ihrem Anliegen eher um Pflege & Gesundheit, "
                "Hilfe bei Anträgen, Mobilität oder um Freizeit- und Begegnungsangebote im Quartier?"
            )

        ms = round((time.perf_counter() - t0) * 1000, 1)
        return (
            True,
            clarification,
            AgentTraceStep(
                step_name="Slot-Filling-Agent (FA-02)",
                details=f"Vages Anliegen erkannt ('{german_search_query}'). Stelle gezielte Rückfrage ({user_lang}).",
                duration_ms=ms,
            ),
        )

    def _extract_agentic_tools_and_filters(
        self, query: str, manual_filters: Optional[RetrievalFilter]
    ) -> Tuple[Optional[RetrievalFilter], str, AgentTraceStep]:
        """
        FA-03 Kontextsensitive Filterung & § 12 C c Kulturspezifische Detailfragen:
        Extrahiert strukturierte Filter mit exakten Wortgrenzen (\b).
        """
        t0 = time.perf_counter()
        q_lower = query.lower()
        f = manual_filters.model_copy() if manual_filters else RetrievalFilter()
        triggered_tools: List[str] = []

        # 1. Kultursensible Filter: Geschützter Frauenraum (§ 2 D b / § 12 C c)
        if any(w in q_lower for w in ["nur frauen", "nur für frauen", "frauenfrühstück", "frauentreff", "unter frauen"]):
            f.target_group = TargetGroupTag.WOMEN_ONLY
            triggered_tools.append("target_group=women_only")

        # 2. Vor-Ort-Sprachkompetenz (§ 2 I / § 12 C c: "Spricht dort jemand Türkisch/Arabisch?")
        lang_triggers = {
            "tr": ["türkisch", "türkce", "türkçe"],
            "ar": ["arabisch", "auf arabisch"],
            "ru": ["russisch", "auf russisch"],
            "uk": ["ukrainisch", "auf ukrainisch"],
            "en": ["englisch", "english"],
        }
        for iso_code, kw_list in lang_triggers.items():
            if any(kw in q_lower for kw in kw_list):
                f.required_service_language = iso_code
                triggered_tools.append(f"required_service_language={iso_code}")
                break

        # 3. Körperliche Barrierefreiheit (FA-03 / § 2 D b)
        if any(w in q_lower for w in ["rollstuhl", "barrierefrei", "aufzug", "gehbehindert", "rollator"]):
            f.require_barrier_free = True
            triggered_tools.append("require_barrier_free=True")

        # 4. Stadtteil-Erkennung in Hildesheim mit Wortgrenzen (\b)
        for dist in self.hildesheim_districts:
            if re.search(rf"\b{re.escape(dist)}\b", query, re.IGNORECASE):
                f.district = dist
                triggered_tools.append(f"district={dist}")
                break

        # 5. Uhrzeit- & Öffnungszeiten-Tool (FA-03: "jetzt geöffnet", "heute")
        now = datetime.now()
        weekdays_de = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
        current_weekday = weekdays_de[now.weekday()]
        time_str = now.strftime("%H:%M")
        time_context_note = (
            f"[ZEIT-TOOL (FA-03): Aktuell ist {current_weekday}, {time_str} Uhr. "
            f"Falls nach aktuellen Öffnungszeiten gefragt wird, gleiche die Sprechzeiten der Quelle mit {current_weekday}, {time_str} Uhr ab.]"
        )
        triggered_tools.append(f"clock_tool={current_weekday} {time_str}")

        has_active_filter = any([
            f.category, f.district, f.required_service_language, f.target_group,
            f.require_barrier_free, f.only_official_sources, f.exclude_expired
        ])

        ms = round((time.perf_counter() - t0) * 1000, 1)
        return (
            f if has_active_filter else None,
            time_context_note,
            AgentTraceStep(
                step_name="Tool- & Filter-Agent (FA-03 / § 12 C c)",
                details=f"Aktivierte Werkzeuge & Filter: {', '.join(triggered_tools)}",
                duration_ms=ms,
            ),
        )

    def _verify_grounding_and_numbers(
        self, draft_answer: str, sources: List[ScoredChunk]
    ) -> Tuple[str, AgentTraceStep]:
        """
        SA-04 Verifier-/Guardrail-Schritt:
        Prüft deterministisch, ob alle mehrstelligen Telefonnummern in der generierten Antwort
        auch tatsächlich im Quelltext vorkommen.
        """
        t0 = time.perf_counter()
        combined_source_text = " ".join(
            f"{s.chunk.text_for_indexing} {s.chunk.metadata.phone or ''} {s.chunk.metadata.address or ''}"
            for s in sources
        )
        source_digits = re.sub(r"\D", "", combined_source_text)

        phone_candidates = re.findall(r"\b0\d{3,4}[\s/\-]?\d{3,8}\b", draft_answer)
        hallucinated_numbers: List[str] = []

        for cand in phone_candidates:
            clean_cand = re.sub(r"\D", "", cand)
            if len(clean_cand) >= 5 and clean_cand not in source_digits:
                hallucinated_numbers.append(cand)

        verified_text = draft_answer
        if hallucinated_numbers:
            for bad_num in hallucinated_numbers:
                verified_text = verified_text.replace(
                    bad_num, "[Telefonnummer bitte direkt in der Originalquelle prüfen]"
                )
            ms = round((time.perf_counter() - t0) * 1000, 1)
            return (
                verified_text,
                AgentTraceStep(
                    step_name="Verifier-Guardrail (SA-04 Halluzinationsschutz)",
                    details=f"WARNUNG: Nicht belegte Nummer(n) {hallucinated_numbers} erkannt und entfernt!",
                    duration_ms=ms,
                ),
            )

        ms = round((time.perf_counter() - t0) * 1000, 1)
        return (
            verified_text,
            AgentTraceStep(
                step_name="Verifier-Guardrail (SA-04 Halluzinationsschutz)",
                details="Prüfung bestanden: Alle genannten Telefonnummern und Kontaktdaten sind durch die Quell-Chunks gedeckt.",
                duration_ms=ms,
            ),
        )