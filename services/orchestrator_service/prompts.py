"""
services/orchestrator_service/prompts.py
Zentraler Prompt-Builder für SA-01 (Einfache/Leichte Sprache mit Mediopunkt),
SA-04 (Striktes Grounding), SA-06 (Rogers-Tonalität & Diskretion), § 10 B c (TTL) und § 11 B (Knowledge Cutoff).
"""

import os
from typing import List, Optional
from shared.schemas import LanguageLevel, ScoredChunk

KNOWLEDGE_CUTOFF = os.getenv("KNOWLEDGE_CUTOFF_STR", "November 2026")

LANGUAGE_NAMES = {
    "de": "Deutsch",
    "en": "Englisch (English)",
    "tr": "Türkisch (Türkçe)",
    "ar": "Arabisch (العربية)",
    "uk": "Ukrainisch (Українська)",
    "ru": "Russisch (Русский)",
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
    """Bereitet die gefundenen Quell-Chunks inkl. Kontaktdaten und TTL-Warnungen (§ 10 B c) für das LLM auf."""
    blocks = []
    for idx, sc in enumerate(scored_chunks, start=1):
        m = sc.chunk.metadata
        ttl_line = f"\n[WICHTIGER HINWEIS: {sc.ttl_warning}]" if sc.ttl_warning else ""
        contact_info = (
            f"Einrichtung: {m.facility_name or m.title} | "
            f"Adresse: {m.address or 'siehe Text'} | "
            f"Telefon: {m.phone or 'siehe Text'} | "
            f"Sprechzeiten: {m.opening_hours or 'siehe Text'} | "
            f"Haltestelle: {m.public_transit_stop or 'siehe Text'} | "
            f"Sprachen vor Ort: {', '.join(m.service_languages)}"
        )
        blocks.append(
            f"--- QUELLE {idx}: {m.title} ({m.source_url_or_path}) ---\n"
            f"VERIFIZIERTE METADATEN: {contact_info}{ttl_line}\n"
            f"INHALT:\n{sc.chunk.text_for_indexing}\n"
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
            "2. Trenne lange zusammengesetzte Wörter auf Deutsch immer mit einem Mediopunkt (·), "
            "zum Beispiel: Pflege·beratung, Öffnungs·zeiten, Nachbarschafts·hilfe, Telefon·nummer.\n"
            "3. Verwende aktive Verben statt Passiv. Vermeide Fremdwörter und Abkürzungen.\n"
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
            "Formuliere neutral und würdevoll, z. B.: 'Für Ihr persönliches Anliegen gibt es hier eine vertrauliche Beratungsstelle:'"
        )

    return (
        "Du bist HOMIE, der digitale Wegweiser für soziale Angebote und Dienstleistungen im Wohnquartier Hildesheim.\n"
        f"Dein eingefrorener Wissensstand (Knowledge Cutoff nach § 11 A/B) ist: {KNOWLEDGE_CUTOFF}.\n\n"
        "STRIKTE VERHALTENSREGELN:\n"
        "1. FAKTENTREUE (SA-04 & FA-01): Antworte AUSSCHLIESSLICH auf Basis der bereitgestellten QUELLEN. "
        "Erfinde niemals Telefonnummern, Adressen, Uhrzeiten oder Buslinien!\n"
        "2. KONTAKTDATEN & QUELLE (FA-04): Nenne bei jeder Empfehlung konkret: Name der Einrichtung, Adresse, "
        "Telefonnummer, Sprechzeiten (falls in der Quelle vorhanden) sowie am Ende den Namen der Originalquelle.\n"
        "3. WERTSCHÄTZENDE TONALITÄT (SA-06 nach Carl Rogers): Begegne allen Menschen auf Augenhöhe – freundlich, "
        "geduldig und ermutigend, aber OHNE bevormundende Verniedlichung (kein 'Elderspeak').\n"
        "4. VERALTETE QUELLEN (§ 10 B c): Wenn bei einer Quelle ein 'WICHTIGER HINWEIS' zur Überschreitung des "
        "Gültigkeitsdatums steht, weise in deiner Antwort transparent darauf hin (z. B. 'Laut einem Dokument aus dem Jahr...').\n"
        f"5. AUSGABESPRACHE (SA-02): Verfasse deine gesamte Antwort in der Sprache: {lang_name}.\n\n"
        f"{style_rules}{discretion_rule}"
    )


def build_qr_code_payload(scored_chunks: List[ScoredChunk]) -> Optional[str]:
    """
    FA-07: Erzeugt einen kompakten vCard-/Kontakt-Text aus dem besten Treffer,
    damit Bewohner:innen Adresse & Telefonnummer per QR-Code aufs Smartphone scannen können.
    """
    if not scored_chunks:
        return None
    top_meta = scored_chunks[0].chunk.metadata
    facility = top_meta.facility_name or top_meta.title
    address = top_meta.address or "Hildesheim"
    phone = top_meta.phone or ""
    url = top_meta.source_url_or_path

    return (
        f"BEGIN:VCARD\nVERSION:3.0\n"
        f"FN:{facility}\n"
        f"ADR:;;{address};;;;\n"
        f"TEL:{phone}\n"
        f"NOTE:Sprechzeiten: {top_meta.opening_hours or 'siehe Quelle'} | Quelle: {url}\n"
        f"END:VCARD"
    )