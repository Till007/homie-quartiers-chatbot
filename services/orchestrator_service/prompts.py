"""
services/orchestrator_service/prompts.py
Zentraler Prompt-Builder für SA-01 (Einfache/Leichte Sprache mit Mediopunkt),
SA-04 (Striktes Grounding gegen Halluzinationen wie erfundene Buslinien),
SA-06 (Rogers-Tonalität & Diskretion), § 10 B c (TTL) und § 11 B (Knowledge Cutoff).
"""

import os
from typing import List, Optional
from shared.schemas import LanguageLevel, ScoredChunk

KNOWLEDGE_CUTOFF = os.getenv("KNOWLEDGE_CUTOFF_STR", "November 2026")

LANGUAGE_NAMES = {
    "de": "Deutsch (German)",
    "en": "Englisch (English)",
    "tr": "Türkisch (Turkish / Türkçe)",
    "ar": "Arabisch (Arabic / العربية)",
    "uk": "Ukrainisch (Ukrainian / Українська)",
    "ru": "Russisch (Russian / Русский)",
}

SENSITIVE_KEYWORDS = [
    "schulden", "insolvenz", "pfändung", "sucht", "alkohol", "drogen", "gewalt",
    "frauenhaus", "depression", "suizid", "psychiatrie", "notfall", "obdachlos"
]


def detect_sensitive_topic(text: str) -> bool:
    """SA-06: Erkennt sensible Themen (Sucht, Schulden, Gewalt), die am Terminal Diskretion erfordern."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SENSITIVE_KEYWORDS)


def format_chunks_for_prompt(scored_chunks: List[ScoredChunk]) -> str:
    """Bereitet die gefundenen Quell-Chunks übersichtlich (ohne verwirrenden Metadaten-Ballast) für das LLM auf."""
    blocks = []
    for idx, sc in enumerate(scored_chunks, start=1):
        m = sc.chunk.metadata
        ttl_line = f"\n[WICHTIGER HINWEIS: {sc.ttl_warning}]" if sc.ttl_warning else ""
        contact_info = (
            f"Einrichtung/Herausgeber: {m.facility_name or m.publisher or m.title} | "
            f"Stadtteil: {m.district} | "
            f"Adresse: {m.address or 'siehe Text'} | "
            f"Telefon: {m.phone or 'siehe Text'} | "
            f"Sprechzeiten: {m.opening_hours or 'siehe Text'}"
        )
        blocks.append(
            f"--- QUELLE {idx}: {m.title} ({m.source_url_or_path}) ---\n"
            f"KONTAKTDATEN: {contact_info}{ttl_line}\n"
            f"TEXTINHALT:\n{sc.chunk.text_raw}\n"
        )
    return "\n".join(blocks)


def build_system_prompt(
    language_level: LanguageLevel,
    target_language_iso: str,
    is_sensitive: bool,
) -> str:
    """Erstellt den System-Prompt gemäß SA-01, SA-04, SA-06, FA-04 und § 11 B."""
    lang_name = LANGUAGE_NAMES.get(target_language_iso, "Deutsch")

    if language_level == LanguageLevel.EASY:
        style_rules = (
            "REGELN FÜR LEICHTE SPRACHE (SA-01 / Netzwerk Leichte Sprache):\n"
            "1. Schreibe sehr kurze Sätze. Jeder Satz enthält nur EINEN Gedanken.\n"
            "2. Trenne lange zusammengesetzte Wörter auf Deutsch mit einem Mediopunkt (·), "
            "zum Beispiel: Pflege·beratung, Öffnungs·zeiten, Nachbarschafts·hilfe, Telefon·nummer.\n"
            "3. Verwende aktive Verben statt Passiv. Vermeide Fremdwörter.\n"
            "4. Beginne für jeden neuen Satz eine neue Zeile."
        )
    else:
        style_rules = (
            "REGELN FÜR BÜRGERNAHE / EINFACHE SPRACHE (SA-01):\n"
            "1. Formuliere klar, freundlich und alltagsnah in kurzen, übersichtlichen Sätzen.\n"
            "2. Vermeide Behördenjargon, Schachtelsätze und unerklärte Abkürzungen."
        )

    discretion_rule = ""
    if is_sensitive:
        discretion_rule = (
            "\nDISKRETIONS-REGEL (SA-06 SENSIBLES THEMA ERKANNT):\n"
            "Wiederhole sensible Begriffe (wie Schulden, Sucht, Gewalt oder Diagnosen) NICHT laut im Einleitungssatz. "
            "Formuliere neutral und würdevoll."
        )

    return (
        "Du bist HOMIE, der digitale Wegweiser für soziale Angebote und Dienstleistungen im Wohnquartier Hildesheim.\n"
        f"Dein eingefrorener Wissensstand (Knowledge Cutoff nach § 11 A/B) ist: {KNOWLEDGE_CUTOFF}.\n\n"
        "STRIKTE VERHALTENSREGELN:\n"
        "1. STRIKTES GROUNDING (SA-04 & FA-01): Antworte AUSSCHLIESSLICH auf Basis der bereitgestellten QUELLEN. "
        "Erfinde NIEMALS Buslinien (wie 'Linie 100'), Apps, Telefonnummern, Adressen oder Uhrzeiten, die nicht wörtlich in den Quellen stehen! "
        "Wenn die bereitgestellten Quellen die Frage nicht beantworten, sage ehrlich und freundlich, dass dazu keine Information "
        "in den vorliegenden Quartiersdokumenten steht, und verweise auf die AWO-Beratungsstelle Hildesheim.\n"
        "2. KONTAKTDATEN & QUELLE (FA-04): Nenne bei passenden Angeboten: Name der Einrichtung, Adresse, "
        "Telefonnummer, Sprechzeiten (sofern im Text vorhanden) sowie den Namen der Originalquelle.\n"
        "3. WERTSCHÄTZENDE TONALITÄT (SA-06): Begegne allen Menschen freundlich, geduldig und auf Augenhöhe.\n"
        f"4. AUSGABESPRACHE (SA-02): Verfasse deine gesamte Antwort AUSSCHLIESSLICH in der Sprache: {lang_name}. "
        "Verwende unter keinen Umständen chinesische Schriftzeichen (Hanzi / 中文)!\n\n"
        f"{style_rules}{discretion_rule}"
    )


def build_qr_code_payload(scored_chunks: List[ScoredChunk]) -> Optional[str]:
    """FA-07: Erzeugt einen vCard-Text aus dem besten Treffer für den Smartphone-QR-Code."""
    if not scored_chunks:
        return None
    top_meta = scored_chunks[0].chunk.metadata
    facility = top_meta.facility_name or top_meta.publisher or top_meta.title
    address = top_meta.address or "Hildesheim"
    phone = top_meta.phone or ""
    url = top_meta.source_url_or_path

    return (
        f"BEGIN:VCARD\nVERSION:3.0\n"
        f"FN:{facility[:60]}\n"
        f"ADR:;;{address};;;;\n"
        f"TEL:{phone}\n"
        f"NOTE:Sprechzeiten: {top_meta.opening_hours or 'siehe Quelle'} | Quelle: {url}\n"
        f"END:VCARD"
    )