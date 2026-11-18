"""
shared/schemas.py
Zentrale Pydantic-Datenverträge (Data Contracts) für alle Microservices im HOMIE-Projekt.
Inklusive automatischer Selbstheilung (Validator) für LLM-Metadaten (Listen/Dicts -> String).
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


def _flatten_to_optional_str(val: Any) -> Optional[str]:
    """
    Konvertiert Listen oder Dictionaries (z. B. wenn Ollama mehrere Telefonnummern
    als [{'service': 'Schuldnerberatung', 'phone': '05121...'}] zurückgibt)
    automatisch in einen flachen, validen String.
    """
    if val is None:
        return None
    if isinstance(val, str):
        cleaned = val.strip()
        return cleaned if cleaned else None
    if isinstance(val, list):
        parts: List[str] = []
        for item in val:
            if isinstance(item, dict):
                kv_str = ": ".join(
                    str(v).strip() for v in item.values() if v is not None and str(v).strip()
                )
                if kv_str:
                    parts.append(kv_str)
            elif item is not None and str(item).strip():
                parts.append(str(item).strip())
        joined = " | ".join(parts)
        return joined if joined else None
    if isinstance(val, dict):
        kv_str = ", ".join(
            f"{k}: {v}" for k, v in val.items() if v is not None and str(v).strip()
        )
        return kv_str if kv_str else None
    return str(val).strip() or None


# =====================================================================
# 1. ENUMS FÜR METADATEN & ARCHITEKTUR-PARADIGMEN (TA-04, § 1, § 2, § 5)
# =====================================================================

class SourceStatus(str, Enum):
    OFFICIAL = "official"
    UNOFFICIAL = "unofficial"


class LanguageLevel(str, Enum):
    STANDARD = "standard"
    SIMPLE = "simple"
    EASY = "easy"


class TargetGroupTag(str, Enum):
    GENERAL = "general"
    SENIORS = "seniors"
    CAREGIVERS = "caregivers"
    DEMENTIA = "dementia"
    MIGRANTS = "migrants"
    FAMILIES = "families"
    WOMEN_ONLY = "women_only"
    YOUTH = "youth"


class DietaryOption(str, Enum):
    NONE = "none"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    KOSHER = "kosher"


class ParadigmMode(str, Enum):
    PURE_RAG = "pure_rag"
    FINE_TUNED = "fine_tuned"
    AGENTIC_AI = "agentic_ai"


class TranslationVariant(str, Enum):
    V2A_DIRECT = "v2a_direct"
    V2B_TRANSLATE_AFTER = "v2b_translated"


class RobotState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


# =====================================================================
# 2. METADATEN-SCHEMA FÜR DOKUMENTE & CHUNKS (§ 1 - § 11, FA-04, TA-01)
# =====================================================================

class DocumentMetadata(BaseModel):
    doc_id: str = Field(..., description="Eindeutige Dokument-ID")
    title: str = Field(..., description="Titel des Dokuments oder Angebots")
    source_url_or_path: str = Field(..., description="Ursprungspfad oder vollständige URL (§ 8 C)")
    publisher: Optional[str] = Field(default=None, description="Herausgeber / Fachabteilung für SSOT (§ 1 A)")
    status: SourceStatus = Field(default=SourceStatus.UNOFFICIAL, description="§ 1 B / § 5 A Status")

    created_or_updated_at: Optional[date] = Field(default=None, description="Zeitstempel der Quelle (§ 2 C)")
    valid_until: Optional[date] = Field(default=None, description="Ablaufdatum für TTL-Prüfung (§ 10 B a)")
    is_expired: bool = Field(default=False, description="§ 10 B c: Markierung als historisch/veraltet")
    content_hash_sha256: str = Field(..., description="§ 10 A a/b: SHA-256 Prüfsumme für Delta-Updates")
    knowledge_cutoff_date: date = Field(default_factory=date.today, description="§ 11 A: Stichtag")

    doc_lang: str = Field(default="de", description="§ 2 I: Primärsprache nach ISO 639-1")
    language_level: LanguageLevel = Field(default=LanguageLevel.STANDARD, description="§ 2 H a: Sprachniveau")
    service_languages: List[str] = Field(
        default_factory=lambda: ["de"],
        description="§ 2 I: Vor-Ort gesprochene Sprachen in der Beratungsstelle",
    )
    multilingual_keywords: List[str] = Field(
        default_factory=list,
        description="§ 3 D Variante B: Mehrsprachige Leit-Schlagworte für Cross-Lingual Retrieval",
    )

    category: str = Field(default="Allgemein", description="TA-01: Gesundheit, Freizeit, Beratung, Mobilität")
    district: str = Field(default="Hildesheim-Gesamt", description="Stadtteil / Quartier in Hildesheim")
    target_groups: List[TargetGroupTag] = Field(default_factory=lambda: [TargetGroupTag.GENERAL])
    dietary_options: List[DietaryOption] = Field(default_factory=list)
    is_religiously_neutral: bool = Field(default=True, description="§ 2 D b: Weltanschaulich offen vs. konfessionell")
    confessional_carrier: Optional[str] = Field(default=None, description="Falls gebunden: z.B. Diakonie, Caritas")
    barrier_free_wheelchair: bool = Field(default=False, description="§ 2 D b: Rollstuhlgerecht / Aufzug")

    facility_name: Optional[str] = Field(default=None, description="Name der Einrichtung")
    address: Optional[str] = Field(default=None, description="Straße, Hausnummer, PLZ, Ort")
    phone: Optional[str] = Field(default=None, description="Normalisierte Telefonnummer (TA-01)")
    opening_hours: Optional[str] = Field(default=None, description="Sprechzeiten für Tool-Filterung (FA-03)")
    public_transit_stop: Optional[str] = Field(default=None, description="Nächste ÖPNV-Haltestelle (FA-04)")

    @field_validator(
        "publisher",
        "confessional_carrier",
        "facility_name",
        "address",
        "phone",
        "opening_hours",
        "public_transit_stop",
        mode="before",
    )
    @classmethod
    def _normalize_string_fields(cls, v: Any) -> Optional[str]:
        return _flatten_to_optional_str(v)


# =====================================================================
# 3. QUALITY-GATE SCHEMAS (Für den Quality-Gate Microservice)
# =====================================================================

class RuleCheckStatus(str, Enum):
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"


class RuleEvaluationResult(BaseModel):
    rule_id: str = Field(..., description="z.B. '§ 1 A', '§ 2 A', '§ 2 F'")
    rule_name: str = Field(..., description="Bezeichnung der Regel")
    status: RuleCheckStatus
    score_or_metric: Optional[float] = Field(default=None, description="Messwert, z.B. Signal-to-Noise Ratio")
    details: str = Field(..., description="Nachvollziehbare Begründung für das Audit-Log")
    auto_fixed: bool = Field(default=False, description="Wurde das Problem automatisch bereinigt?")


class QualityInspectionRequest(BaseModel):
    raw_text: str = Field(..., description="Der aus PDF/TXT/Web extrahierte Rohtext")
    title: str = Field(..., description="Dokumenttitel oder Dateiname")
    source_url_or_path: str = Field(..., description="Dateipfad oder Quell-URL")
    publisher: Optional[str] = Field(default=None, description="Herausgeber (falls bekannt)")
    document_date: Optional[date] = Field(default=None, description="Datum des Dokuments (§ 2 C)")
    valid_until: Optional[date] = Field(default=None, description="Ablaufdatum (§ 10 B)")
    existing_hashes: List[str] = Field(default_factory=list, description="Bereits bekannte SHA-256 Hashes (§ 10 A)")


class QualityAuditReport(BaseModel):
    audit_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    approved_for_ingestion: bool = Field(..., description="True = Dokument darf in die DB; False = Quarantäne")
    sanitized_text: str = Field(..., description="Bereinigter, UTF-8 kodierter & pseudonymisierter Text")
    extracted_metadata: DocumentMetadata
    rule_results: List[RuleEvaluationResult] = Field(default_factory=list)
    blocking_reasons: List[str] = Field(default_factory=list)


# =====================================================================
# 4. ETL- & CHUNKING-SCHEMAS (Für etl_ingest_service & rag_retrieval_service)
# =====================================================================

class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Eindeutige Chunk-ID (z.B. DOC-xxx_CHUNK-01)")
    doc_id: str = Field(..., description="Referenz auf das Ursprungsdokument")
    chunk_index: int = Field(..., description="Laufende Nummer des Chunks im Dokument")
    text_raw: str = Field(..., description="Reiner, bereinigter Textabschnitt für die Anzeige")
    text_for_indexing: str = Field(
        ...,
        description="Mit Metadaten-Header, DIN-Langformen, Glossar und mehrsprachigen Keywords angereicherter Text",
    )
    token_estimate: int = Field(..., description="Geschätzte Token-Anzahl (Ziel: 300-500 Tokens)")
    metadata: DocumentMetadata


class ScrapeUrlRequest(BaseModel):
    url: str = Field(..., description="Start-URL")
    publisher: Optional[str] = Field(default=None, description="Optionaler Herausgeber")
    valid_until: Optional[date] = Field(default=None, description="Optionales Ablaufdatum")
    follow_links_depth_1: bool = Field(default=True, description="Kontextuelles Link-Crawling (Tiefe = 1)")
    use_llm_enrichment: bool = Field(
        default=True,
        description="Nutzt lokales Ollama-Modell für präzise Kontakt- & Keyword-Extraktion",
    )


class ETLProcessResponse(BaseModel):
    success: bool
    status_message: str
    audit_report: QualityAuditReport
    quarantine_path: Optional[str] = Field(default=None, description="Pfad, falls Dokument abgelehnt wurde")
    chunks: List[DocumentChunk] = Field(default_factory=list)
    crawled_sub_urls: List[str] = Field(default_factory=list)


# =====================================================================
# 5. RAG-RETRIEVAL & QUELLEN-GUARD SCHEMAS (Für rag_retrieval_service)
# =====================================================================

class RetrievalFilter(BaseModel):
    category: Optional[str] = Field(default=None, description="z.B. 'Gesundheit & Pflege', 'Mobilität'")
    district: Optional[str] = Field(default=None, description="Stadtteil in Hildesheim")
    required_service_language: Optional[str] = Field(
        default=None, description="ISO-Code der gewünschten Beratungssprache vor Ort (§ 2 I)"
    )
    target_group: Optional[TargetGroupTag] = Field(
        default=None, description="z.B. 'women_only' für geschützte Frauenangebote (§ 2 D b)"
    )
    require_barrier_free: bool = Field(default=False, description="Nur rollstuhlgerechte Angebote")
    only_official_sources: bool = Field(default=False, description="Nur Quellen mit status='official'")
    exclude_expired: bool = Field(default=False, description="Abgelaufene Veranstaltungen ausblenden")


class RetrievalRequest(BaseModel):
    query: str = Field(..., description="Suchanfrage")
    top_k_candidates: int = Field(default=15, description="Anzahl Kandidaten je Index vor dem Reranking")
    top_k_final: int = Field(default=4, description="Anzahl finaler Chunks nach Cross-Encoder Reranking")
    score_threshold: float = Field(default=0.25, description="Schwellenwert für den Quellen-Guard (0.0 bis 1.0)")
    filters: Optional[RetrievalFilter] = Field(default=None)


class ScoredChunk(BaseModel):
    chunk: DocumentChunk
    dense_rank: Optional[int] = Field(default=None, description="Rang in der ChromaDB Vektorsuche")
    bm25_rank: Optional[int] = Field(default=None, description="Rang in der lexikalischen BM25-Suche")
    rrf_score: float = Field(..., description="Kombinierter Reciprocal Rank Fusion Score")
    reranker_score: float = Field(..., description="Normalisierter Cross-Encoder Score (0.0 - 1.0)")
    final_score: float = Field(..., description="Endgültiger Score inkl. Official- & Leichte-Sprache-Boost")
    ttl_warning: Optional[str] = Field(default=None, description="§ 10 B c: Automatischer Warnhinweis")


class RetrievalResponse(BaseModel):
    query: str
    passed_source_guard: bool = Field(..., description="True = Score >= Schwellenwert; False = Fallback")
    best_score: float = Field(default=0.0)
    score_threshold: float
    fallback_message: Optional[str] = Field(default=None, description="FA-06: Verweis auf AWO-Beratung")
    results: List[ScoredChunk] = Field(default_factory=list)


class IndexUpsertRequest(BaseModel):
    chunks: List[DocumentChunk]
    replace_existing_doc: bool = Field(default=True, description="§ 10 A c: Alte Vektoren vorab löschen")


# =====================================================================
# 6. ORCHESTRATOR / DIALOG-CONTROLLER SCHEMAS (Für orchestrator_service)
# =====================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' oder 'assistant'")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentTraceStep(BaseModel):
    step_name: str
    details: str
    duration_ms: float = 0.0


class ChatRequest(BaseModel):
    session_id: str = Field(default="default-kiosk-session", description="Sitzungs-ID für das RAM-Gedächtnis")
    user_message: str = Field(..., description="Texteingabe oder transkribierte Spracheingabe")
    user_language: str = Field(default="de", description="Sprache nach ISO 639-1 (de, en, tr, ar, uk, ru)")
    language_level: LanguageLevel = Field(default=LanguageLevel.SIMPLE, description="SA-01: 'simple' oder 'easy'")
    paradigm: ParadigmMode = Field(default=ParadigmMode.AGENTIC_AI, description="TA-04 Paradigma")
    translation_variant: TranslationVariant = Field(default=TranslationVariant.V2A_DIRECT)
    score_threshold: float = Field(default=0.25, description="Schwellenwert für den Quellen-Guard")
    manual_filters: Optional[RetrievalFilter] = Field(default=None)


class ChatResponse(BaseModel):
    session_id: str
    paradigm_used: ParadigmMode
    translation_variant_used: TranslationVariant
    answer_text: str = Field(..., description="Finale Antwort in der Zielsprache")
    german_intermediate_text: Optional[str] = Field(default=None, description="Deutscher Zwischentext bei V2b")
    rewritten_query: str = Field(..., description="Eigenständige Suchanfrage")
    passed_source_guard: bool = Field(..., description="True = Gesicherte Quelle; False = Fallback")
    is_clarification_question: bool = Field(default=False, description="FA-02: Slot-Filling-Rückfrage")
    is_sensitive_topic: bool = Field(default=False, description="SA-06: Sensibles Thema erkannt")
    sources: List[ScoredChunk] = Field(default_factory=list)
    qr_code_payload: Optional[str] = Field(default=None, description="FA-07: vCard für QR-Code")
    robot_state: RobotState = Field(default=RobotState.SPEAKING)
    agent_trace: List[AgentTraceStep] = Field(default_factory=list)
    total_latency_ms: float = 0.0


# =====================================================================
# 7. SPEECH-SERVICE SCHEMAS (Für speech_service auf Port 8005)
# =====================================================================

class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transkribierter Text der Spracheingabe")
    detected_language: str = Field(..., description="Erkannte Sprache nach ISO 639-1")
    language_probability: float = Field(..., description="Konfidenz der Spracherkennung (0.0 - 1.0)")
    duration_ms: float = Field(..., description="Verarbeitungszeit in ms")


class SynthesisRequest(BaseModel):
    text: str = Field(..., description="Vorzulesender Antworttext")
    language: str = Field(default="de", description="Zielsprache nach ISO 639-1")
    speed_Factor: float = Field(default=0.85, description="SA-05: Sprechgeschwindigkeit")
    is_sensitive_topic: bool = Field(default=False, description="SA-06: Diskretions-Modus am Terminal")