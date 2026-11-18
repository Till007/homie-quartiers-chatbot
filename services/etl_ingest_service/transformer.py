"""
services/etl_ingest_service/transformer.py
Transformations- und Chunking-Schicht (Transform) gemäß TA-01, FA-04 und § 3 A-D.
Normalisiert alle Ollama-JSON-Ausgaben typsicher zu Strings.
"""

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

from shared.logger import get_service_logger
from shared.schemas import DocumentChunk, DocumentMetadata, _flatten_to_optional_str

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
        self.glossary: Dict[str, Dict[str, Any]] = self._load_json("cultural_glossary.json")

        self.hildesheim_districts = [
            "Stadtmitte", "Zentrum", "Nordstadt", "Oststadt", "Weststadt", "Südstadt",
            "Marienburger Höhe", "Galgenberg", "Drispenstedt", "Himmelsthür", "Ochtersum",
            "Moritzberg", "Bavenstedt", "Einum", "Achtum", "Uppen", "Itzum", "Marienburg",
            "Neuhof", "Sorsum"
        ]

    def _load_json(self, filename: str) -> Dict[str, Any]:
        filepath = self.config_dir / filename
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden von {filepath}: {e}")
        return {}

    def clean_and_standardize_text(self, text: str) -> Tuple[str, List[str], List[str]]:
        cleaned = text
        cleaned = re.sub(r"(?im)^\s*seite\s+\d+\s*(von\s+\d+)?\s*$", "", cleaned)
        cleaned = re.sub(r"[-_=*]{5,}", "---", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        cleaned = re.sub(r"\(\s*0\s*51\s*21\s*\)", "05121 ", cleaned)

        for abbr, full_form in self.abbreviations.items():
            pattern = re.compile(rf"\b{re.escape(abbr)}\b")
            if pattern.search(cleaned):
                cleaned = pattern.sub(f"{full_form} ({abbr})", cleaned, count=1)

        glossary_notes: List[str] = []
        multilingual_kw: List[str] = []

        for term, info in self.glossary.items():
            if re.search(rf"\b{re.escape(term)}\b", cleaned, re.IGNORECASE):
                explanation = str(info.get("explanation_simple", ""))
                if explanation:
                    glossary_notes.append(f"- {term}: {explanation}")
                for kw in info.get("multilingual_keywords", []):
                    if isinstance(kw, str) and kw not in multilingual_kw:
                        multilingual_kw.append(kw)

        return cleaned, glossary_notes, multilingual_kw

    async def enrich_metadata(
        self, text: str, metadata: DocumentMetadata, use_llm: bool = True
    ) -> DocumentMetadata:
        text_lower = text.lower()

        if any(w in text_lower for w in ["pflege", "arzt", "klinik", "gesundheit", "demenz", "krankenhaus", "versorgung"]):
            metadata.category = "Gesundheit & Pflege"
        elif any(w in text_lower for w in ["bus", "bahn", "öpnv", "svhi", "mobilität", "fahrdienst", "rollstuhl"]):
            metadata.category = "Mobilität"
        elif any(w in text_lower for w in ["treff", "frühstück", "sport", "chor", "verein", "freizeit", "café"]):
            metadata.category = "Freizeit & Begegnung"
        else:
            metadata.category = "Beratung & Unterstützung"

        detected_districts: List[str] = []
        for d in self.hildesheim_districts:
            if re.search(rf"\b{re.escape(d)}\b", text, re.IGNORECASE):
                detected_districts.append(d)
        if detected_districts:
            metadata.district = ", ".join(detected_districts)
        else:
            metadata.district = "Hildesheim-Gesamt"

        phone_match = re.search(
            r"(?:Tel(?:efon)?\.?|Fon|Rufnummer)\s*[:.]?\s*((?:\+49|0)\s*\d{2,5}[\s/\-]+\d{3,10}(?:[\s\-]\d{1,5})?)",
            text,
            re.IGNORECASE,
        )
        if not phone_match:
            phone_match = re.search(
                r"\b(051\d{2}[\s/\-]+\d{3,8}(?:[\s\-]\d{1,4})?)\b",
                text,
            )
        if phone_match:
            candidate_phone = re.sub(r"\s+", " ", phone_match.group(1)).strip()
            if len(re.sub(r"\D", "", candidate_phone)) >= 7:
                metadata.phone = candidate_phone

        address_match = re.search(
            r"([A-ZÄÖÜ][a-zäöüß\-]+(?:straße|str\.|weg|platz|allee|ring|damm|wall|gasse)\s+\d+[a-zA-Z]?(?:\s*,\s*|\s+)31[01]\d{2}\s+[A-ZÄÖÜ][a-zäöüß]+)",
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
            r"\b(?:Haltestelle|Bushaltestelle|Buslinie)\s*[:\-]\s*([A-ZÄÖÜ0-9][^\n,.;]{2,45})",
            text,
        )
        if stop_match:
            metadata.public_transit_stop = stop_match.group(1).strip()

        if use_llm:
            metadata = await self._enrich_with_ollama(text, metadata)

        return metadata

    async def _enrich_with_ollama(self, text: str, metadata: DocumentMetadata) -> DocumentMetadata:
        prompt = (
            "Extrahiere aus dem folgenden Quartiers-Dokument (Hildesheim) ausschließlich belegte Fakten als JSON.\n"
            "WICHTIG: Gib für 'publisher', 'facility_name', 'address', 'phone', 'opening_hours' und 'public_transit_stop' "
            "immer nur einen einfachen String (Text) oder null zurück – NIEMALS Listen oder verschachtelte Objekte!\n"
            "Gewünschtes JSON-Format:\n"
            '{"publisher": "...", "valid_until": "YYYY-MM-DD", "facility_name": "...", "address": "...", '
            '"phone": "...", "opening_hours": "...", "public_transit_stop": "...", '
            '"multilingual_keywords": ["Begriff DE", "Term EN", "Terim TR", "مصطلح AR", "Термін UK"]}\n\n'
            f"DOKUMENTTEXT:\n{text[:2500]}"
        )
        try:
            async with httpx.AsyncClient(timeout=18.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "keep_alive": "30m",
                        "options": {"temperature": 0.0, "num_predict": 250},
                    },
                )
                if resp.status_code == 200:
                    llm_json = json.loads(resp.json().get("response", "{}"))
                    llm_pub = _flatten_to_optional_str(llm_json.get("publisher"))
                    if llm_pub and len(llm_pub) > 3:
                        if not any(sw in llm_pub.lower() for sw in ["adobe", "indesign", "word", "pdf"]):
                            if not metadata.publisher or "Quartiers-Dokument" in metadata.publisher or "Webquelle" in metadata.publisher:
                                metadata.publisher = llm_pub

                    raw_valid = _flatten_to_optional_str(llm_json.get("valid_until"))
                    if raw_valid and re.match(r"^20\d{2}-\d{2}-\d{2}$", raw_valid):
                        try:
                            parsed_v = date.fromisoformat(raw_valid)
                            if parsed_v >= date.today():
                                metadata.valid_until = parsed_v
                        except ValueError:
                            pass

                    metadata.facility_name = metadata.facility_name or _flatten_to_optional_str(llm_json.get("facility_name"))
                    metadata.address = metadata.address or _flatten_to_optional_str(llm_json.get("address"))
                    metadata.phone = metadata.phone or _flatten_to_optional_str(llm_json.get("phone"))
                    metadata.opening_hours = metadata.opening_hours or _flatten_to_optional_str(llm_json.get("opening_hours"))
                    metadata.public_transit_stop = metadata.public_transit_stop or _flatten_to_optional_str(llm_json.get("public_transit_stop"))

                    for kw in llm_json.get("multilingual_keywords") or []:
                        if isinstance(kw, str) and kw.strip() and kw.strip() not in metadata.multilingual_keywords:
                            metadata.multilingual_keywords.append(kw.strip())
        except Exception as e:
            logger.info(f"Ollama-Anreicherung übersprungen (nutze regelbasierte Metadaten): {e}")
        return metadata

    def create_semantic_chunks(
        self,
        cleaned_text: str,
        metadata: DocumentMetadata,
        glossary_notes: List[str],
        target_tokens: int = 380,
        max_tokens: int = 480,
        overlap_ratio: float = 0.12,
    ) -> List[DocumentChunk]:
        token_factor = 1.35
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned_text) if len(p.strip()) > 0]

        raw_chunks: List[str] = []
        current_paras: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = int(len(para.split()) * token_factor)

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

        header_lines = [
            f"[DOKUMENT: {metadata.title} | HERAUSGEBER: {metadata.publisher} | KATEGORIE: {metadata.category} | STADTTEIL: {metadata.district}]",
            f"[QUELLE: {metadata.source_url_or_path} | STATUS: {metadata.status.value} | GÜLTIG BIS: {metadata.valid_until} | SPRACHEN VOR ORT: {', '.join(metadata.service_languages)}]",
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

        return final_chunks

    def _compute_overlap(
        self, paras: List[str], target_tokens: int, overlap_ratio: float, token_factor: float
    ) -> Tuple[List[str], int]:
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