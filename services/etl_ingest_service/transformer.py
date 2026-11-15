"""
services/etl_ingest_service/transformer.py
Transformations- und Chunking-Schicht (Transform) gemäß TA-01, FA-04 und § 3 A-D.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import httpx

from shared.logger import get_service_logger
from shared.schemas import DocumentChunk, DocumentMetadata

logger = get_service_logger("etl-transformer")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")


class DocumentTransformer:
    """
    Führt Artefaktbereinigung, DIN-Abkürzungsauflösung, Glossar-Anreicherung,
    Metadaten-Extraktion und semantisches Chunking (300-500 Tokens, 10-15% Overlap) durch.
    """

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.abbreviations: Dict[str, str] = self._load_json("abbreviations_de.json")
        self.glossary: Dict[str, Dict] = self._load_json("cultural_glossary.json")

        # Bekannte Quartiere / Stadtteile in Hildesheim für automatische Zuordnung (FA-03 / TA-01)
        self.hildesheim_districts = [
            "Stadtmitte", "Zentrum", "Nordstadt", "Oststadt", "Weststadt", "Südstadt",
            "Marienburger Höhe", "Galgenberg", "Drispenstedt", "Himmelsthür", "Ochtersum",
            "Moritzberg", "Bavenstedt", "Einum", "Achtum", "Uppen", "Itzum", "Marienburg",
            "Neuhof", "Sorsum"
        ]

    def _load_json(self, filename: str) -> Dict:
        filepath = self.config_dir / filename
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden von {filepath}: {e}")
        return {}

    # =================================================================
    # 1. TEXT-NORMALISIERUNG, DIN-ABKÜRZUNGEN & GLOSSAR (§ 3 B & C)
    # =================================================================

    def clean_and_standardize_text(self, text: str) -> Tuple[str, List[str], List[str]]:
        """
        - Bereinigt PDF-/Web-Artefakte und vereinheitlicht Telefonnummern (TA-01, § 3 C)
        - Ersetzt Abkürzungen nach DIN 2340 (§ 3 B)
        - Erkennt Begriffe aus dem Kulturellen Glossar (§ 3 B c) und sammelt mehrsprachige Keywords (§ 3 D)
        """
        cleaned = text

        # 1. Typische Kopf-/Fußzeilen-Artefakte entfernen ("Seite 1 von 4", lange Trennlinien)
        cleaned = re.sub(r"(?im)^\s*seite\s+\d+\s*(von\s+\d+)?\s*$", "", cleaned)
        cleaned = re.sub(r"[-_=*]{5,}", "---", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        # 2. Telefonnummern auf einheitliches lesbares Format bringen (Vorwahl Hildesheim 05121 etc.)
        cleaned = re.sub(r"\(\s*0\s*51\s*21\s*\)", "05121 ", cleaned)

        # 3. § 3 B: Semantische Standardisierung von Abkürzungen (DIN 2340)
        for abbr, full_form in self.abbreviations.items():
            pattern = re.compile(rf"\b{re.escape(abbr)}\b")
            if pattern.search(cleaned):
                # Beim ersten Auftreten Langform + Kurzform einsetzen
                cleaned = pattern.sub(f"{full_form} ({abbr})", cleaned, count=1)

        # 4. § 3 B c & § 3 D: Kulturelles Glossar & mehrsprachige Keywords auslesen
        glossary_notes: List[str] = []
        multilingual_kw: List[str] = []

        for term, info in self.glossary.items():
            if re.search(rf"\b{re.escape(term)}\b", cleaned, re.IGNORECASE):
                explanation = info.get("explanation_simple", "")
                if explanation:
                    glossary_notes.append(f"- {term}: {explanation}")
                for kw in info.get("multilingual_keywords", []):
                    if kw not in multilingual_kw:
                        multilingual_kw.append(kw)

        return cleaned, glossary_notes, multilingual_kw

    # =================================================================
    # 2. METADATEN- & KONTAKTEXTRAKTION (FA-04, TA-01)
    # =================================================================

    async def enrich_metadata(
        self, text: str, metadata: DocumentMetadata, use_llm: bool = True
    ) -> DocumentMetadata:
        """
        Extrahiert strukturierte Kontaktinformationen (FA-04: Einrichtung, Adresse,
        Telefon, Sprechzeiten, ÖPNV-Haltestelle), Hildesheimer Stadtteil und Kategorie.
        """
        text_lower = text.lower()

        # 1. Regelbasierte Kategorie-Erkennung (TA-01)
        if any(w in text_lower for w in ["pflege", "arzt", "klinik", "gesundheit", "demenz", "krankenhaus"]):
            metadata.category = "Gesundheit & Pflege"
        elif any(w in text_lower for w in ["bus", "bahn", "öpnv", "svhi", "mobilität", "fahrdienst", "rollstuhl"]):
            metadata.category = "Mobilität"
        elif any(w in text_lower for w in ["treff", "frühstück", "sport", "chor", "verein", "freizeit", "café"]):
            metadata.category = "Freizeit & Begegnung"
        else:
            metadata.category = "Beratung & Unterstützung"

        # 2. Stadtteil in Hildesheim erkennen
        detected_districts = [d for d in self.hildesheim_districts if d.lower() in text_lower]
        if detected_districts:
            metadata.district = ", ".join(detected_districts)

        # 3. Regelbasierte Extraktion von Telefonnummer, Adresse, Öffnungszeiten & Haltestelle (FA-04)
        phone_match = re.search(
            r"(?:Tel(?:efon)?\.?|Fon|Rufnummer)?\s*[:.]?\s*((?:\+49|0)\s*\d{2,5}[\s/\-]?\d{2,10}[\s/\-]?\d{0,6})",
            text,
        )
        if phone_match:
            metadata.phone = re.sub(r"\s+", " ", phone_match.group(1)).strip()

        address_match = re.search(
            r"([A-ZÄÖÜ][a-zäöüß\-]+(?:straße|str\.|weg|platz|allee|ring|damm|wall|gasse)\s+\d+[a-zA-Z]?(?:\s*,\s*|\s+)311\d{2}\s+Hildesheim)",
            text,
            re.IGNORECASE,
        )
        if address_match:
            metadata.address = re.sub(r"\s+", " ", address_match.group(1)).strip()

        hours_match = re.search(
            r"((?:Mo(?:ntag)?|Di(?:enstag)?|Mi(?:ttwoch)?|Do(?:nnerstag)?|Fr(?:eitag)?|Sprechzeiten|Öffnungszeiten)[^\n]{5,120}\d{1,2}[:.]\d{2}\s*Uhr)",
            text,
            re.IGNORECASE,
        )
        if hours_match:
            metadata.opening_hours = re.sub(r"\s+", " ", hours_match.group(1)).strip()

        stop_match = re.search(
            r"(?:Haltestelle|Buslinie|Linie)\s*[:.]?\s*([^\n,.]{3,50})",
            text,
            re.IGNORECASE,
        )
        if stop_match:
            metadata.public_transit_stop = stop_match.group(1).strip()

        # 4. Optionale LLM-gestützte Anreicherung via lokalem Ollama (TA-01)
        if use_llm:
            metadata = await self._enrich_with_ollama(text, metadata)

        return metadata

    async def _enrich_with_ollama(self, text: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """
        Fragt das lokale Ollama-Modell nach fehlenden Kontaktdaten (FA-04) und
        5 mehrsprachigen Suchbegriffen (§ 3 D Variante B). Fällt bei Offline/Fehler sauber zurück.
        """
        prompt = (
            "Extrahiere aus dem folgenden Quartiers-Dokument (Hildesheim) ausschließlich belegte Fakten als JSON.\n"
            "Erfinde nichts! Wenn eine Angabe fehlt, setze null.\n"
            "Gewünschtes JSON-Format:\n"
            '{"facility_name": "...", "address": "...", "phone": "...", "opening_hours": "...", '
            '"public_transit_stop": "...", "multilingual_keywords": ["Begriff DE", "Term EN", "Terim TR", "مصطلح AR", "Термін UK"]}\n\n'
            f"DOKUMENTTEXT:\n{text[:2500]}"
        )
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "options": {"temperature": 0.0},
                    },
                )
                if resp.status_code == 200:
                    llm_json = json.loads(resp.json().get("response", "{}"))
                    metadata.facility_name = metadata.facility_name or llm_json.get("facility_name")
                    metadata.address = metadata.address or llm_json.get("address")
                    metadata.phone = metadata.phone or llm_json.get("phone")
                    metadata.opening_hours = metadata.opening_hours or llm_json.get("opening_hours")
                    metadata.public_transit_stop = metadata.public_transit_stop or llm_json.get("public_transit_stop")
                    for kw in llm_json.get("multilingual_keywords") or []:
                        if isinstance(kw, str) and kw not in metadata.multilingual_keywords:
                            metadata.multilingual_keywords.append(kw)
        except Exception as e:
            logger.info(
                f"Ollama-Anreicherung übersprungen (nutze regelbasierte Metadaten). Grund: {e}"
            )
        return metadata

    # =================================================================
    # 3. SEMANTISCHES CHUNKING (TA-01: 300-500 Tokens, 10-15% Overlap)
    # =================================================================

    def create_semantic_chunks(
        self,
        cleaned_text: str,
        metadata: DocumentMetadata,
        glossary_notes: List[str],
        target_tokens: int = 380,
        max_tokens: int = 480,
        overlap_ratio: float = 0.12,
    ) -> List[DocumentChunk]:
        """
        Zerlegt das Dokument entlang logischer Absätze in Sinnabschnitte von ca. 300-500 Tokens
        mit ~12 % Overlap. Injiziert in jeden Chunk den Kontakt- & Metadaten-Header (TA-01, § 3).
        """
        # Faustformel Deutsch: 1 Wort ≈ 1.35 Subword-Tokens (BGE-M3 / Qwen Tokenizer)
        token_factor = 1.35
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned_text) if len(p.strip()) > 0]

        raw_chunks: List[str] = []
        current_paras: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = int(len(para.split()) * token_factor)

            # Falls ein einzelner Absatz größer als max_tokens ist, an Satzgrenzen teilen
            if para_tokens > max_tokens:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    sent_tokens = int(len(sent.split()) * token_factor)
                    if current_tokens + sent_tokens > max_tokens and current_paras:
                        raw_chunks.append("\n\n".join(current_paras))
                        current_paras, current_tokens = self._compute_overlap(
                            current_paras, target_tokens, overlap_ratio, token_factor
                        )
                    current_paras.append(sent)
                    current_tokens += sent_tokens
                continue

            if current_tokens + para_tokens > target_tokens and current_paras:
                raw_chunks.append("\n\n".join(current_paras))
                current_paras, current_tokens = self._compute_overlap(
                    current_paras, target_tokens, overlap_ratio, token_factor
                )

            current_paras.append(para)
            current_tokens += para_tokens

        if current_paras:
            raw_chunks.append("\n\n".join(current_paras))

        # Baue den kontextbewahrenden Metadaten-Header für jeden Chunk (verhindert Trennung von Angebot & Adresse!)
        header_lines = [
            f"[DOKUMENT: {metadata.title} | KATEGORIE: {metadata.category} | STADTTEIL: {metadata.district}]",
            f"[QUELLE: {metadata.source_url_or_path} | STATUS: {metadata.status.value} | SPRACHEN VOR ORT: {', '.join(metadata.service_languages)}]",
        ]
        contact_parts = []
        if metadata.facility_name:
            contact_parts.append(f"Einrichtung: {metadata.facility_name}")
        if metadata.address:
            contact_parts.append(f"Adresse: {metadata.address}")
        if metadata.phone:
            contact_parts.append(f"Telefon: {metadata.phone}")
        if metadata.opening_hours:
            contact_parts.append(f"Sprechzeiten: {metadata.opening_hours}")
        if metadata.public_transit_stop:
            contact_parts.append(f"Haltestelle: {metadata.public_transit_stop}")
        if contact_parts:
            header_lines.append("[KONTAKT: " + " | ".join(contact_parts) + "]")

        if metadata.multilingual_keywords:
            header_lines.append(
                f"[MEHRSPRACHIGE SCHLAGWORTE (§ 3D): {', '.join(metadata.multilingual_keywords)}]"
            )
        if glossary_notes:
            header_lines.append("[BEGRIFFSERKLÄRUNG (§ 3Bc):\n" + "\n".join(glossary_notes) + "]")

        header_block = "\n".join(header_lines)

        final_chunks: List[DocumentChunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            est_tokens = int(len(chunk_text.split()) * token_factor)
            chunk_id = f"{metadata.doc_id}_CHUNK-{idx + 1:02d}"
            text_for_indexing = f"{header_block}\n\n{chunk_text}"

            final_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=metadata.doc_id,
                    chunk_index=idx + 1,
                    text_raw=chunk_text,
                    text_for_indexing=text_for_indexing,
                    token_estimate=est_tokens,
                    metadata=metadata,
                )
            )

        logger.info(
            f"Dokument '{metadata.title}' ({metadata.doc_id}) in {len(final_chunks)} semantische Chunks zerlegt."
        )
        return final_chunks

    def _compute_overlap(
        self, paras: List[str], target_tokens: int, overlap_ratio: float, token_factor: float
    ) -> Tuple[List[str], int]:
        """Behält die letzten Sätze/Absätze für 10-15% Overlap zum nächsten Chunk bei."""
        desired_overlap = int(target_tokens * overlap_ratio)
        overlap_paras: List[str] = []
        overlap_tokens = 0
        for p in reversed(paras):
            p_tok = int(len(p.split()) * token_factor)
            if overlap_tokens + p_tok <= max(desired_overlap, 60):
                overlap_paras.insert(0, p)
                overlap_tokens += p_tok
            else:
                break
        return overlap_paras, overlap_tokens