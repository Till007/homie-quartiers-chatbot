"""
shared/schemas.py
Zentrale Pydantic-Datenverträge (Data Contracts) für alle Microservices im HOMIE-Projekt.
Deckt die Anforderungen aus '1.0 Anforderungen.pdf' sowie '2.0 Konzeption & Systementscheidung.txt' ab.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =====================================================================
# 1. ENUMS FÜR METADATEN & ARCHITEKTUR-PARADIGMEN (TA-04, § 1, § 2, § 5)
# =====================================================================

class SourceStatus(str, Enum):
    """§ 1 B & § 5 A: Kennzeichnung offizieller (SSOT) vs. inoffizieller Quellen."""
    OFFICIAL = "official"
    UNOFFICIAL = "unofficial"


class LanguageLevel(str, Enum):
    """§ 2 H a: Klassifizierung des Sprachniveaus."""
    STANDARD = "standard"          # Standardsprache / Fachsprache
    SIMPLE = "simple"              # Bürgernahe / Einfache Sprache (SA-01 Standard)
    EASY = "easy"                  # Leichte Sprache nach DIN 8581-1 (SA-01 Button)


class TargetGroupTag(str, Enum):
    """§ 2 D: Zielgruppen- und Diversitätsmerkmale."""
    GENERAL = "general"
    SENIORS = "seniors"
    CAREGIVERS = "caregivers"      # Pflegende Angehörige
    DEMENTIA = "dementia"          # Demenzerkrankte Personen
    MIGRANTS = "migrants"          # Menschen mit Migrationshintergrund
    FAMILIES = "families"
    WOMEN_ONLY = "women_only"      # § 2 D b: Geschützter Raum für Frauen
    YOUTH = "youth"


class DietaryOption(str, Enum):
    """§ 2 D b: Ernährungsspezifika bei Angeboten mit Verpflegung."""
    NONE = "none"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    KOSHER = "kosher"


class ParadigmMode(str, Enum):
    """TA-04: Umschaltbare Architekturvarianten für den wissenschaftlichen Vergleich (IT-MA5)."""
    PURE_RAG = "pure_rag"          # 1. Klassisches Single-Step Hybrid-RAG
    FINE_TUNED = "fine_tuned"      # 2. Fine-Tuned LLM (QLoRA) + Basis-Kontext
    AGENTIC_AI = "agentic_ai"      # 3. Multi-Step Agent (Slot-Filling, Tools, Verifier)


class TranslationVariant(str, Enum):
    """2.0 Konzeption PDF: Empirischer Vergleich der Übersetzungsvarianten."""
    V2A_DIRECT = "v2a_direct"              # LLM antwortet direkt in Zielsprache
    V2B_TRANSLATE_AFTER = "v2b_translated" # LLM antwortet auf Deutsch -> Übersetzer-Baustein


class RobotState(str, Enum):
    """TA-05: Zustandssignale für physische Endgeräte / Navel-Roboter (LEDs, Gestik)."""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


# =====================================================================
# 2. METADATEN-SCHEMA FÜR DOKUMENTE & CHUNKS (§ 1 - § 11, FA-04, TA-01)
# =====================================================================

class DocumentMetadata(BaseModel):
    """
    Vollständiger Metadaten-Satz, der jedem Dokument und Chunk in der ChromaDB
    beigefügt wird, um Traceability (§ 3) und Agentic Filtering (FA-03) zu garantieren.
    """
    doc_id: str = Field(..., description="Eindeutige Dokument-ID")
    title: str = Field(..., description="Titel des Dokuments oder Angebots")
    source_url_or_path: str = Field(..., description="Ursprungspfad oder vollständige URL (§ 8 C)")
    publisher: Optional[str] = Field(None, description="Herausgeber / Fachabteilung für SSOT (§ 1 A)")
    status: SourceStatus = Field(default=SourceStatus.UNOFFICIAL, description="§ 1 B / § 5 A Status")
    
    # § 2 C & § 10: Aktualität, Hash & Verfallsdatum (TTL)
    created_or_updated_at: Optional[date] = Field(None, description="Zeitstempel der Quelle (§ 2 C)")
    valid_until: Optional[date] = Field(None, description="Ablaufdatum für TTL-Prüfung (§ 10 B a)")
    is_expired: bool = Field(default=False, description="§ 10 B c: Markierung als historisch/veraltet")
    content_hash_sha256: str = Field(..., description="§ 10 A a/b: SHA-256 Prüfsumme für Delta-Updates")
    knowledge_cutoff_date: date = Field(default_factory=date.today, description="§ 11 A: Stichtag")

    # § 2 H & § 2 I: Sprach-Metadaten
    doc_lang: str = Field(default="de", description="§ 2 I: Primärsprache nach ISO 639-1 (de, tr, ar, uk, ru)")
    language_level: LanguageLevel = Field(default=LanguageLevel.STANDARD, description="§ 2 H a: Sprachniveau")
    service_languages: List[str] = Field(
        default_factory=lambda: ["de"],
        description="§ 2 I: Vor-Ort gesprochene Sprachen in der Beratungsstelle"
    )
    multilingual_keywords: List[str] = Field(
        default_factory=list,
        description="§ 3 D Variante B: Mehrsprachige Leit-Schlagworte für Cross-Lingual Retrieval"
    )

    # § 2 D & FA-03 / FA-04: Zielgruppen, Kultursensibilität & Ortsdaten (Hildesheim)
    category: str = Field(default="Allgemein", description="TA-01: Gesundheit, Freizeit, Beratung, Mobilität")
    district: str = Field(default="Hildesheim-Gesamt", description="Stadtteil / Quartier in Hildesheim")
    target_groups: List[TargetGroupTag] = Field(default_factory=lambda: [TargetGroupTag.GENERAL])
    dietary_options: List[DietaryOption] = Field(default_factory=list)
    is_religiously_neutral: bool = Field(default=True, description="§ 2 D b: Weltanschaulich offen vs. konfessionell")
    confessional_carrier: Optional[str] = Field(None, description="Falls gebunden: z.B. Diakonie, Caritas")
    barrier_free_wheelchair: bool = Field(default=False, description="§ 2 D b: Rollstuhlgerecht / Aufzug")
    
    # FA-04: Verifizierte Kontaktinformationen
    facility_name: Optional[str] = Field(None, description="Name der Einrichtung")
    address: Optional[str] = Field(None, description="Straße, Hausnummer, PLZ, Ort")
    phone: Optional[str] = Field(None, description="Normalisierte Telefonnummer (TA-01)")
    opening_hours: Optional[str] = Field(None, description="Sprechzeiten für Tool-Filterung (FA-03)")
    public_transit_stop: Optional[str] = Field(None, description="Nächste ÖPNV-Haltestelle (FA-04)")


# =====================================================================
# 3. QUALITY-GATE SCHEMAS (Für den Quality-Gate Microservice)
# =====================================================================

class RuleCheckStatus(str, Enum):
    PASSED = "PASSED"              # Regel vollständig erfüllt
    WARNING = "WARNING"            # Durchgelassen, aber mit automatischer Transformation/Hinweis
    FAILED = "FAILED"              # Kritischer Verstoß -> Dokument wird blockiert!


class RuleEvaluationResult(BaseModel):
    """Ergebnis einer einzelnen Regelprüfung aus dem Regelwerk (§ 1 - § 11)."""
    rule_id: str = Field(..., description="z.B. '§ 1 A', '§ 2 A', '§ 2 F'")
    rule_name: str = Field(..., description="Bezeichnung der Regel")
    status: RuleCheckStatus
    score_or_metric: Optional[float] = Field(None, description="Messwert, z.B. Signal-to-Noise Ratio")
    details: str = Field(..., description="Nachvollziehbare Begründung für das Audit-Log")
    auto_fixed: bool = Field(default=False, description="Wurde das Problem automatisch bereinigt (z.B. NER)?")


class QualityInspectionRequest(BaseModel):
    """Input für den Quality-Gate Microservice."""
    raw_text: str = Field(..., description="Der aus PDF/TXT/Web extrahierte Rohtext")
    title: str = Field(..., description="Dokumenttitel oder Dateiname")
    source_url_or_path: str = Field(..., description="Dateipfad oder Quell-URL")
    publisher: Optional[str] = Field(None, description="Herausgeber (falls bekannt)")
    document_date: Optional[date] = Field(None, description="Datum des Dokuments (§ 2 C)")
    valid_until: Optional[date] = Field(None, description="Ablaufdatum (§ 10 B)")
    existing_hashes: List[str] = Field(default_factory=list, description="Bereits bekannte SHA-256 Hashes (§ 10 A)")


class QualityAuditReport(BaseModel):
    """
    Output des Quality-Gate Microservice.
    Entscheidet verbindlich, ob ein Dokument in den ETL- & ChromaDB-Prozess darf.
    """
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
    """
    Repräsentiert einen semantisch kohärenten Textabschnitt (300-500 Tokens, 10-15% Overlap)
    inklusive angereichertem Header für ChromaDB und BM25 (TA-01, § 3 D).
    """
    chunk_id: str = Field(..., description="Eindeutige Chunk-ID (z.B. DOC-xxx_CHUNK-01)")
    doc_id: str = Field(..., description="Referenz auf das Ursprungsdokument")
    chunk_index: int = Field(..., description="Laufende Nummer des Chunks im Dokument")
    text_raw: str = Field(..., description="Reiner, bereinigter Textabschnitt für die Anzeige")
    text_for_indexing: str = Field(
        ...,
        description="Mit Metadaten-Header, DIN-Langformen, Glossar und mehrsprachigen Keywords angereicherter Text"
    )
    token_estimate: int = Field(..., description="Geschätzte Token-Anzahl (Ziel: 300-500 Tokens)")
    metadata: DocumentMetadata


class ScrapeUrlRequest(BaseModel):
    """Input für den Web-Scraper (§ 7 - § 9)."""
    url: str = Field(..., description="Start-URL aus dem Raum Hildesheim")
    publisher: Optional[str] = Field(default="Webquelle Hildesheim")
    follow_links_depth_1: bool = Field(default=True, description="Kontextuelles Link-Crawling (Tiefe = 1)")
    use_llm_enrichment: bool = Field(
        default=True,
        description="Nutzt lokales Ollama-Modell für präzise Kontakt- & Keyword-Extraktion"
    )


class ETLProcessResponse(BaseModel):
    """Rückgabe des ETL-Microservice nach Prüfung durch das Quality-Gate und Chunking."""
    success: bool
    status_message: str
    audit_report: QualityAuditReport
    quarantine_path: Optional[str] = Field(None, description="Pfad, falls Dokument abgelehnt wurde")
    chunks: List[DocumentChunk] = Field(default_factory=list)
    crawled_sub_urls: List[str] = Field(default_factory=list)

    # =====================================================================
# 5. RAG-RETRIEVAL & QUELLEN-GUARD SCHEMAS (Für rag_retrieval_service)
# =====================================================================

class RetrievalFilter(BaseModel):
    """
    Strukturierte Filterkriterien für Agentic Tool-Use (FA-03) und kultursensible Suche (§ 2 D / § 2 I).
    """
    category: Optional[str] = Field(None, description="z.B. 'Gesundheit & Pflege', 'Mobilität'")
    district: Optional[str] = Field(None, description="Stadtteil in Hildesheim, z.B. 'Oststadt', 'Nordstadt'")
    required_service_language: Optional[str] = Field(
        None, description="ISO-Code der gewünschten Beratungssprache vor Ort (§ 2 I), z.B. 'tr', 'ar', 'ru'"
    )
    target_group: Optional[TargetGroupTag] = Field(
        None, description="z.B. 'women_only' für geschützte Frauenangebote (§ 2 D b)"
    )
    require_barrier_free: bool = Field(
        default=False, description="Nur rollstuhlgerechte Angebote (§ 2 D b, FA-03)"
    )
    only_official_sources: bool = Field(
        default=False, description="Nur Quellen mit status='official' (§ 1 B)"
    )
    exclude_expired: bool = Field(
        default=False, description="Abgelaufene Veranstaltungen komplett ausblenden statt warnen (§ 10 B)"
    )


class RetrievalRequest(BaseModel):
    """Suchanfrage an den Hybrid-Retriever (BM25 + ChromaDB + RRF + Reranker)."""
    query: str = Field(..., description="Suchanfrage (ggf. durch Query-Rewriter präzisiert)")
    top_k_candidates: int = Field(default=15, description="Anzahl Kandidaten je Index vor dem Reranking")
    top_k_final: int = Field(default=4, description="Anzahl finaler Chunks nach Cross-Encoder Reranking")
    score_threshold: float = Field(
        default=0.25,
        description="Schwellenwert für den Quellen-Guard (0.0 bis 1.0). Darunter greift der AWO-Fallback."
    )
    filters: Optional[RetrievalFilter] = Field(default=None)


class ScoredChunk(BaseModel):
    """Ein abgerufener Chunk inklusive aller Einzel-Scores für das wissenschaftliche Debugging."""
    chunk: DocumentChunk
    dense_rank: Optional[int] = Field(None, description="Rang in der ChromaDB Vektorsuche")
    bm25_rank: Optional[int] = Field(None, description="Rang in der lexikalischen BM25-Suche")
    rrf_score: float = Field(..., description="Kombinierter Reciprocal Rank Fusion Score")
    reranker_score: float = Field(..., description="Normalisierter Cross-Encoder Score (0.0 - 1.0)")
    final_score: float = Field(..., description="Endgültiger Score inkl. Official- & Leichte-Sprache-Boost")
    ttl_warning: Optional[str] = Field(
        None, description="§ 10 B c: Automatischer Warnhinweis, falls das Dokument abgelaufen ist"
    )


class RetrievalResponse(BaseModel):
    """Antwort des RAG-Retrieval-Service inklusive Quellen-Guard-Entscheidung."""
    query: str
    passed_source_guard: bool = Field(
        ..., description="True = Score >= Schwellenwert; False = Keine gesicherte Quelle (Fallback!)"
    )
    best_score: float = Field(default=0.0)
    score_threshold: float
    fallback_message: Optional[str] = Field(
        None, description="FA-06: Verweis auf menschliche AWO-Anlaufstelle bei fehlender Quelle"
    )
    results: List[ScoredChunk] = Field(default_factory=list)


class IndexUpsertRequest(BaseModel):
    """Anfrage zum Einfügen oder Aktualisieren (§ 10 A c) von Chunks im Hybrid-Index."""
    chunks: List[DocumentChunk]
    replace_existing_doc: bool = Field(
        default=True, description="§ 10 A c: Alte Vektoren derselben doc_id/URL vorab löschen"
    )

    # =====================================================================
# 6. ORCHESTRATOR / DIALOG-CONTROLLER SCHEMAS (Für orchestrator_service)
# =====================================================================

class ChatMessage(BaseModel):
    """Einzelne Nachricht im flüchtigen RAM-Sitzungsgedächtnis (TA-06)."""
    role: str = Field(..., description="'user' oder 'assistant'")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentTraceStep(BaseModel):
    """Protokolliert jeden Zwischenschritt (Rewriting, Slot-Filling, Tool-Use, Verifier) für das Debug-UI."""
    step_name: str
    details: str
    duration_ms: float = 0.0


class ChatRequest(BaseModel):
    """Anfrage an den Dialog-Controller (Port 8004)."""
    session_id: str = Field(default="default-kiosk-session", description="Sitzungs-ID für das RAM-Gedächtnis")
    user_message: str = Field(..., description="Texteingabe oder transkribierte Spracheingabe")
    user_language: str = Field(
        default="de",
        description="Erkannte oder gewählte Sprache nach ISO 639-1 (de, en, tr, ar, uk, ru)"
    )
    language_level: LanguageLevel = Field(
        default=LanguageLevel.SIMPLE,
        description="SA-01: 'simple' (Bürgernahe Sprache) oder 'easy' (Leichte Sprache mit Mediopunkt)"
    )
    paradigm: ParadigmMode = Field(
        default=ParadigmMode.AGENTIC_AI,
        description="TA-04: Auswahl zwischen 'pure_rag', 'fine_tuned' und 'agentic_ai'"
    )
    translation_variant: TranslationVariant = Field(
        default=TranslationVariant.V2A_DIRECT,
        description="Empirischer Vergleich: 'v2a_direct' vs. 'v2b_translated'"
    )
    score_threshold: float = Field(default=0.25, description="Schwellenwert für den Quellen-Guard")
    manual_filters: Optional[RetrievalFilter] = Field(default=None, description="Optionale Filter aus dem UI")


class ChatResponse(BaseModel):
    """Vollständige Antwort des Orchestrators für Kiosk, Roboter und Debug-Oberfläche."""
    session_id: str
    paradigm_used: ParadigmMode
    translation_variant_used: TranslationVariant
    answer_text: str = Field(..., description="Finale Antwort in der Zielsprache (für Anzeige & TTS)")
    german_intermediate_text: Optional[str] = Field(
        None, description="Deutscher Zwischentext bei Variante V2b (für HIX- und Faithfulness-Messung)"
    )
    rewritten_query: str = Field(..., description="Vom Query-Rewriter erzeugte eigenständige Suchanfrage")
    passed_source_guard: bool = Field(..., description="True = Gesicherte Quelle; False = Fallback/Rückfrage")
    is_clarification_question: bool = Field(
        default=False, description="FA-02: True, wenn das System eine gezielte Slot-Filling-Rückfrage stellt"
    )
    is_sensitive_topic: bool = Field(
        default=False, description="SA-06: True bei Sucht/Schulden/Gewalt (unterdrückt laute TTS-Wiederholung)"
    )
    sources: List[ScoredChunk] = Field(default_factory=list, description="Verwendete Quell-Chunks inkl. Scores")
    qr_code_payload: Optional[str] = Field(
        None, description="FA-07: vCard/Text-Payload zur Generierung eines QR-Codes für das Smartphone"
    )
    robot_state: RobotState = Field(default=RobotState.SPEAKING, description="TA-05: Status für Roboter-Steuerung")
    agent_trace: List[AgentTraceStep] = Field(
        default_factory=list, description="Detaillierte Ausführungsschritte für die Evaluation"
    )
    total_latency_ms: float = 0.0

    # =====================================================================
# 7. SPEECH-SERVICE SCHEMAS (Für speech_service auf Port 8005)
# =====================================================================

class TranscriptionResponse(BaseModel):
    """Output von faster-whisper (STT) gemäß UML-Sequenzdiagramm & SA-07."""
    text: str = Field(..., description="Transkribierter Text der Spracheingabe")
    detected_language: str = Field(..., description="Erkannte Sprache nach ISO 639-1 (z.B. de, tr, ar, uk, ru)")
    language_probability: float = Field(..., description="Konfidenz der Spracherkennung (0.0 - 1.0)")
    duration_ms: float = Field(..., description="Verarbeitungszeit für Latenzmessung (TA-03)")


class SynthesisRequest(BaseModel):
    """Input für Piper TTS gemäß SA-05 & SA-06."""
    text: str = Field(..., description="Vorzulesender Antworttext")
    language: str = Field(default="de", description="Zielsprache nach ISO 639-1")
    speed_Factor: float = Field(
        default=0.85,
        description="SA-05: Sprechgeschwindigkeit (Standard 0.85x für Senior:innen, 1.0x Normal)"
    )
    is_sensitive_topic: bool = Field(
        default=False,
        description="SA-06: Bei True wird am öffentlichen Terminal nur ein diskreter Hinweis vorgelesen"
    )