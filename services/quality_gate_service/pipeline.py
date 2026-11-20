"""
services/quality_gate_service/pipeline.py
Implementiert die schrittweise Qualitätskontrolle gemäß '2.0 Konzeption & Systementscheidung.txt'.
Vollständig Pylance-/Pyright-geprüft.
"""

import hashlib
import json
import re
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langdetect import LangDetectException, detect

from shared.logger import get_service_logger
from shared.schemas import (
    DietaryOption,
    DocumentMetadata,
    LanguageLevel,
    QualityAuditReport,
    QualityInspectionRequest,
    RuleCheckStatus,
    RuleEvaluationResult,
    SourceStatus,
    TargetGroupTag,
)

logger = get_service_logger("quality-gate-pipeline")


class QualityGatePipeline:
    """
    Sequenzielle Regel-Engine für die Dokumenten- und Quellensichtung.
    Lässt nur Dokumente durch, die keinen kritischen 'FAILED'-Status aufweisen.
    """

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.discriminatory_terms = self._load_json("discriminatory_terms.json")
        self.whitelist_cfg = self._load_json("whitelist_hildesheim.json")

    def _load_json(self, filename: str) -> Dict[str, Any]:
        filepath = self.config_dir / filename
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Fehler beim Laden von {filepath}: {e}", exc_info=True)
            return {}

    def inspect_document(self, req: QualityInspectionRequest) -> QualityAuditReport:
        audit_id = f"AUDIT-{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"[{audit_id}] Starte Qualitätsprüfung für '{req.title}' ({req.source_url_or_path})")

        self.whitelist_cfg = self._load_json("whitelist_hildesheim.json")
        rule_results: List[RuleEvaluationResult] = []
        blocking_reasons: List[str] = []

        # Schritt 0 (§ 3 A): Zeichensatzkodierung nach UTF-8
        working_text, res_utf8 = self._check_utf8_encoding(req.raw_text)
        rule_results.append(res_utf8)

        # Schritt 1 (§ 1 A & B, § 5 A): Offizieller Status (SSOT) vs. Inoffiziell
        source_status, res_ssot = self._check_ssot_status(req.publisher, req.source_url_or_path)
        rule_results.append(res_ssot)

        # Schritt 2 (§ 2 A): Signal-to-Noise Ratio (Rauschabstand prüfen)
        res_snr = self._check_signal_to_noise(working_text)
        rule_results.append(res_snr)

        # Schritt 3 (§ 2 B): Strukturelle Integrität
        res_struct = self._check_structural_integrity(working_text)
        rule_results.append(res_struct)

        # Schritt 4 (§ 2 C & § 10 B): Aktualität & automatisches Review-Datum (+365 Tage ab HEUTE nach § 10 C a)
        effective_doc_date = req.document_date or date.today()
        effective_valid_until = req.valid_until or (date.today() + timedelta(days=365))
        is_expired, res_recency = self._check_recency_and_ttl(
            effective_doc_date, effective_valid_until, was_auto_ttl=(req.valid_until is None)
        )
        rule_results.append(res_recency)

        # Schritt 5 (§ 2 G): Sicherheits-Audit (Blockieren echter Patientenakten!)
        res_sec = self._check_security_patient_records(working_text)
        rule_results.append(res_sec)

        # Schritt 6 (§ 2 F): Named Entity Recognition / PII-Pseudonymisierung
        working_text, res_pii = self._check_and_redact_pii(working_text)
        rule_results.append(res_pii)

        # Schritt 7 (§ 2 E): Kultursensible & diskriminierungsfreie Sprache prüfen/ersetzen
        working_text, res_disc = self._check_discriminatory_language(working_text)
        rule_results.append(res_disc)

        # Schritt 8 (§ 2 H): Sprachniveau (Standard/Einfach/Leicht)
        lang_level, res_level = self._classify_language_level(working_text)
        rule_results.append(res_level)

        # Schritt 9 (§ 2 I & § 2 D): Dokumentensprache, Vor-Ort-Service-Sprachen & Diversitäts-Tags
        doc_lang, service_langs, target_groups, diet_opts, barrier_free, res_meta = (
            self._extract_diversity_and_language_metadata(working_text)
        )
        rule_results.append(res_meta)

        # Schritt 10 (§ 10 A & § 11 D): SHA-256 Hash-Berechnung & intelligenter Duplikat-/Update-Check
        content_hash, res_hash = self._check_hash_and_duplicates(working_text, req.existing_hashes)
        rule_results.append(res_hash)

        for r in rule_results:
            if r.status == RuleCheckStatus.FAILED:
                blocking_reasons.append(f"[{r.rule_id}] {r.rule_name}: {r.details}")

        approved = len(blocking_reasons) == 0

        metadata = DocumentMetadata(
            doc_id=f"DOC-{content_hash[:10]}",
            title=req.title,
            source_url_or_path=req.source_url_or_path,
            publisher=req.publisher or "Automatisch ermittelt",
            status=source_status,
            created_or_updated_at=effective_doc_date,
            valid_until=effective_valid_until,
            is_expired=is_expired,
            content_hash_sha256=content_hash,
            doc_lang=doc_lang,
            language_level=lang_level,
            service_languages=service_langs,
            target_groups=target_groups,
            dietary_options=diet_opts,
            barrier_free_wheelchair=barrier_free,
            confessional_carrier=None,
            facility_name=None,
            address=None,
            phone=None,
            opening_hours=None,
            public_transit_stop=None,
        )

        report = QualityAuditReport(
            audit_id=audit_id,
            approved_for_ingestion=approved,
            sanitized_text=working_text,
            extracted_metadata=metadata,
            rule_results=rule_results,
            blocking_reasons=blocking_reasons,
        )

        self._persist_audit_log(report)
        return report

    def _check_utf8_encoding(self, text: str) -> Tuple[str, RuleEvaluationResult]:
        try:
            encoded = text.encode("utf-8", errors="replace").decode("utf-8")
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", encoded)
            return cleaned, RuleEvaluationResult(
                rule_id="§ 3 A",
                rule_name="UTF-8 Zeichensatzkodierung",
                status=RuleCheckStatus.PASSED,
                score_or_metric=None,
                details="Text erfolgreich nach UTF-8 normalisiert und Steuerzeichen bereinigt.",
                auto_fixed=(cleaned != text),
            )
        except Exception as e:
            return text, RuleEvaluationResult(
                rule_id="§ 3 A",
                rule_name="UTF-8 Zeichensatzkodierung",
                status=RuleCheckStatus.FAILED,
                score_or_metric=None,
                details=f"UTF-8 Kodierungsfehler: {e}",
                auto_fixed=False,
            )

    def _check_ssot_status(
        self, publisher: Optional[str], source_path: str
    ) -> Tuple[SourceStatus, RuleEvaluationResult]:
        trusted_domains = self.whitelist_cfg.get(
            "trusted_official_domains", self.whitelist_cfg.get("allowed_domains", [])
        )
        official_publishers = [
            "awo", "arbeiterwohlfahrt", "stadt hildesheim", "landkreis hildesheim",
            "ministerium", "caritas", "diakonie", "drk", "eutb", "paritätisch", "universität", "land niedersachsen"
        ]

        is_trusted_domain = any(str(domain) in source_path.lower() for domain in trusted_domains)
        is_official_pub = bool(
            publisher and any(op in publisher.lower() for op in official_publishers)
        )

        if is_trusted_domain or is_official_pub:
            return SourceStatus.OFFICIAL, RuleEvaluationResult(
                rule_id="§ 1 A/B",
                rule_name="SSOT & Offizielle Quelle",
                status=RuleCheckStatus.PASSED,
                score_or_metric=1.0,
                details=f"Als offizielle Quelle eingestuft (status: official | Herausgeber: '{publisher}').",
                auto_fixed=False,
            )
        return SourceStatus.UNOFFICIAL, RuleEvaluationResult(
            rule_id="§ 5 A/B",
            rule_name="Nicht-offizielle Quelle (Triangulation)",
            status=RuleCheckStatus.WARNING,
            score_or_metric=0.0,
            details=f"Herausgeber ('{publisher}') als 'status: unofficial' (§ 5 A) markiert – Dokument wird zugelassen.",
            auto_fixed=False,
        )

    def _check_signal_to_noise(self, text: str) -> RuleEvaluationResult:
        words = re.findall(r"\b\w+\b", text)
        if len(words) < 15:
            return RuleEvaluationResult(
                rule_id="§ 2 A",
                rule_name="Signal-to-Noise Ratio (Rauschabstand)",
                status=RuleCheckStatus.FAILED,
                score_or_metric=float(len(words)),
                details=f"Zu geringe Informationsdichte ({len(words)} Wörter < Minimum 15 Wörter).",
                auto_fixed=False,
            )

        alnum_chars = sum(c.isalnum() for c in text)
        total_chars = max(len(text.strip()), 1)
        snr_ratio = round(alnum_chars / total_chars, 2)

        if snr_ratio < 0.35:
            return RuleEvaluationResult(
                rule_id="§ 2 A",
                rule_name="Signal-to-Noise Ratio (Rauschabstand)",
                status=RuleCheckStatus.FAILED,
                score_or_metric=snr_ratio,
                details=f"Zu hohes Störrauschen (SNR: {snr_ratio} < 0.35).",
                auto_fixed=False,
            )
        return RuleEvaluationResult(
            rule_id="§ 2 A",
            rule_name="Signal-to-Noise Ratio (Rauschabstand)",
            status=RuleCheckStatus.PASSED,
            score_or_metric=snr_ratio,
            details=f"Gute Informationsdichte ({len(words)} Wörter, SNR: {snr_ratio}).",
            auto_fixed=False,
        )

    def _check_structural_integrity(self, text: str) -> RuleEvaluationResult:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 15]
        if len(paragraphs) == 0:
            return RuleEvaluationResult(
                rule_id="§ 2 B",
                rule_name="Strukturelle Integrität",
                status=RuleCheckStatus.FAILED,
                score_or_metric=0.0,
                details="Kein lesbarer Textabschnitt erkennbar.",
                auto_fixed=False,
            )
        return RuleEvaluationResult(
            rule_id="§ 2 B",
            rule_name="Strukturelle Integrität",
            status=RuleCheckStatus.PASSED,
            score_or_metric=float(len(paragraphs)),
            details=f"Struktur intakt ({len(paragraphs)} Sinnabschnitte erkannt).",
            auto_fixed=False,
        )

    def _check_recency_and_ttl(
        self, doc_date: date, valid_until: date, was_auto_ttl: bool
    ) -> Tuple[bool, RuleEvaluationResult]:
        today = date.today()
        is_expired = valid_until < today

        if is_expired:
            return True, RuleEvaluationResult(
                rule_id="§ 10 B b/c",
                rule_name="Time-to-Live (TTL) & Historisierung",
                status=RuleCheckStatus.WARNING,
                score_or_metric=None,
                details=(
                    f"Ermitteltes Ablaufdatum ({valid_until}) liegt in der Vergangenheit. "
                    f"Wird gemäß § 10 B b mit 'is_expired=True' für einen Prompt-Warnhinweis markiert."
                ),
                auto_fixed=False,
            )

        ttl_source_note = (
            "automatisch auf +1 Jahr Review-Zyklus (§ 10 C a) bzw. aus Text ermittelt"
            if was_auto_ttl
            else "manuell gesetzt"
        )
        return False, RuleEvaluationResult(
            rule_id="§ 2 C / § 10 B",
            rule_name="Aktualität & Gültigkeit (Auto-TTL)",
            status=RuleCheckStatus.PASSED,
            score_or_metric=None,
            details=f"Dokumentdatum: {doc_date} | Gültig bis: {valid_until} ({ttl_source_note}).",
            auto_fixed=was_auto_ttl,
        )

    def _check_security_patient_records(self, text: str) -> RuleEvaluationResult:
        forbidden_markers = [
            r"\bpatientenakte\b",
            r"\banamnesebogen\b",
            r"\bversichertennummer\b",
            r"\bkassen-nr\b",
            r"\bicd-10\s*:\s*[a-z]\d{2}",
            r"\bpflegebegutachtung\s+nach\s+§\s*18\b",
        ]
        found = [m for m in forbidden_markers if re.search(m, text, re.IGNORECASE)]
        if found:
            return RuleEvaluationResult(
                rule_id="§ 2 G",
                rule_name="Sicherheits-Audit (Schutz vor Patientenakten)",
                status=RuleCheckStatus.FAILED,
                score_or_metric=float(len(found)),
                details=f"KRITISCHER SICHERHEITSVERSTOSS: Verdacht auf echte Patientenakte ({found}).",
                auto_fixed=False,
            )
        return RuleEvaluationResult(
            rule_id="§ 2 G",
            rule_name="Sicherheits-Audit (Schutz vor Patientenakten)",
            status=RuleCheckStatus.PASSED,
            score_or_metric=0.0,
            details="Keine Merkmale persönlicher Patientenakten gefunden.",
            auto_fixed=False,
        )

    def _check_and_redact_pii(self, text: str) -> Tuple[str, RuleEvaluationResult]:
        redacted_text = text
        redactions_count = 0

        iban_pattern = r"\b[A-Z]{2}\d{2}[A-Z0-9\s]{12,30}\b"
        redacted_text, n_iban = re.subn(iban_pattern, "[KONTONUMMER GESCHWÄRZT]", redacted_text)
        redactions_count += n_iban

        dob_pattern = r"(geb(?:oren)?\.?\s*(?:am)?\s*\d{1,2}\.\d{1,2}\.\d{2,4})"
        redacted_text, n_dob = re.subn(
            dob_pattern, "[GEBURTSDATUM GESCHWÄRZT]", redacted_text, flags=re.IGNORECASE
        )
        redactions_count += n_dob

        if redactions_count > 0:
            return redacted_text, RuleEvaluationResult(
                rule_id="§ 2 F",
                rule_name="PII- & NER-Pseudonymisierung",
                status=RuleCheckStatus.WARNING,
                score_or_metric=float(redactions_count),
                details=f"{redactions_count} sensible Angaben (IBAN/Geburtsdaten) automatisch geschwärzt.",
                auto_fixed=True,
            )

        return redacted_text, RuleEvaluationResult(
            rule_id="§ 2 F",
            rule_name="PII- & NER-Pseudonymisierung",
            status=RuleCheckStatus.PASSED,
            score_or_metric=0.0,
            details="Keine sensiblen Privatdaten (IBAN/Geburtsdaten) im Text entdeckt.",
            auto_fixed=False,
        )

    def _check_discriminatory_language(self, text: str) -> Tuple[str, RuleEvaluationResult]:
        """
        § 2 E a: Ressourcenorientierung statt Defizitorientierung.
        Der Quelltext wird NICHT automatisch verändert. Die frühere Ersetzung als Teilzeichenkette
        (ohne Wortgrenzen, ohne Groß-/Kleinschreibung) hat Wörter zerstört, z. B.
        „Im Notfall rufen Sie die 112 an“ -> „Im NotBeratungsfall oder Unterstützungsanliegen ...“,
        „Haltestelle“ -> „Hältere Menschenstelle“, „Ausländerbehörde“ -> „Menschen ohne deutsche
        Staatsangehörigkeitbehörde“. Das widerspricht der Quellentreue (K2) und kann Notfallhinweise
        unlesbar machen. Funde werden als ganze Wörter erkannt und zur redaktionellen Prüfung gemeldet
        (§ 2 E a: „müssen redaktionell überarbeitet werden“). Eine Umformulierung gehört in die
        Antwort (Prompt-Glossar bzw. Post-Generation-Filter), nicht in die Quelle.
        """
        updated_text = text
        found_terms: List[str] = []

        for bad_term, neutral_term in self.discriminatory_terms.items():
            pattern = re.compile(r"(?<!\w)" + re.escape(bad_term) + r"(?!\w)")
            n_hits = len(pattern.findall(text))
            if n_hits:
                found_terms.append(f"'{bad_term}' ({n_hits}x, Vorschlag: '{neutral_term}')")

        if found_terms:
            return updated_text, RuleEvaluationResult(
                rule_id="§ 2 E a",
                rule_name="Kultursensible & diskriminierungsfreie Sprache",
                status=RuleCheckStatus.WARNING,
                score_or_metric=float(len(found_terms)),
                details=f"Defizitorientierte Begriffe gefunden, bitte redaktionell prüfen: {', '.join(found_terms)}",
                auto_fixed=False,
            )

        return updated_text, RuleEvaluationResult(
            rule_id="§ 2 E a",
            rule_name="Kultursensible & diskriminierungsfreie Sprache",
            status=RuleCheckStatus.PASSED,
            score_or_metric=0.0,
            details="Keine defizitorientierten Begriffe aus der Sperrliste gefunden.",
            auto_fixed=False,
        )

    def _classify_language_level(self, text: str) -> Tuple[LanguageLevel, RuleEvaluationResult]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0]
        words = re.findall(r"\b\w+\b", text)
        avg_sentence_len = len(words) / max(len(sentences), 1)
        long_words = [w for w in words if len(w) > 14]
        long_word_ratio = len(long_words) / max(len(words), 1)

        if "leichte sprache" in text.lower() or "·" in text or (avg_sentence_len <= 9 and long_word_ratio < 0.05):
            level = LanguageLevel.EASY
        elif avg_sentence_len <= 15 and long_word_ratio < 0.12:
            level = LanguageLevel.SIMPLE
        else:
            level = LanguageLevel.STANDARD

        status = RuleCheckStatus.PASSED
        details = (
            f"Sprachniveau klassifiziert als '{level.value}' "
            f"(Ø Satzlänge: {avg_sentence_len:.1f} Wörter, komplexe Wörter: {long_word_ratio:.1%})."
        )
        if avg_sentence_len > 24 or long_word_ratio > 0.20:
            status = RuleCheckStatus.WARNING
            details += " Hohe Dichte an Schachtelsätzen (§ 2 H c) – Vereinfachung im Prompt aktiv."

        return level, RuleEvaluationResult(
            rule_id="§ 2 H",
            rule_name="Sprachniveau & Verständlichkeit",
            status=status,
            score_or_metric=round(avg_sentence_len, 1),
            details=details,
            auto_fixed=False,
        )

    def _extract_diversity_and_language_metadata(
        self, text: str
    ) -> Tuple[str, List[str], List[TargetGroupTag], List[DietaryOption], bool, RuleEvaluationResult]:
        text_lower = text.lower()

        try:
            doc_lang = str(detect(text))
        except LangDetectException:
            doc_lang = "de"

        lang_map = {
            "de": ["deutsch"],
            "tr": ["türkisch", "türkce", "türkçe"],
            "ar": ["arabisch", "عربي"],
            "ru": ["russisch", "русский"],
            "uk": ["ukrainisch", "українська"],
            "en": ["englisch", "english"],
        }
        service_langs = ["de"]
        for iso_code, keywords in lang_map.items():
            if any(kw in text_lower for kw in keywords) and iso_code not in service_langs:
                service_langs.append(iso_code)

        target_groups: List[TargetGroupTag] = []
        if any(w in text_lower for w in ["nur für frauen", "frauenfrühstück", "frauentreff", "mädchen und frauen"]):
            target_groups.append(TargetGroupTag.WOMEN_ONLY)
        if any(w in text_lower for w in ["senior", "ältere menschen", "rentner", "pflegebedürftig"]):
            target_groups.append(TargetGroupTag.SENIORS)
        if any(w in text_lower for w in ["pflegende angehörige", "angehörigengruppe"]):
            target_groups.append(TargetGroupTag.CAREGIVERS)
        if "demenz" in text_lower or "alzheimer" in text_lower:
            target_groups.append(TargetGroupTag.DEMENTIA)
        if any(w in text_lower for w in ["migration", "zuwanderung", "geflüchtete", "integrations"]):
            target_groups.append(TargetGroupTag.MIGRANTS)
        if not target_groups:
            target_groups.append(TargetGroupTag.GENERAL)

        diet_opts: List[DietaryOption] = []
        if "halal" in text_lower:
            diet_opts.append(DietaryOption.HALAL)
        if "koscher" in text_lower or "kosher" in text_lower:
            diet_opts.append(DietaryOption.KOSHER)
        if "vegetarisch" in text_lower:
            diet_opts.append(DietaryOption.VEGETARIAN)
        if "vegan" in text_lower:
            diet_opts.append(DietaryOption.VEGAN)

        barrier_free = any(
            w in text_lower for w in ["barrierefrei", "rollstuhlgerecht", "aufzug vorhanden", "ebenerdig"]
        )

        return (
            doc_lang,
            service_langs,
            target_groups,
            diet_opts,
            barrier_free,
            RuleEvaluationResult(
                rule_id="§ 2 D / § 2 I",
                rule_name="Kultursensible Merkmale & Service-Sprachen",
                status=RuleCheckStatus.PASSED,
                score_or_metric=float(len(service_langs)),
                details=(
                    f"Erkannt: doc_lang='{doc_lang}', service_languages={service_langs}, "
                    f"Zielgruppen={[t.value for t in target_groups]}, Barrierefrei={barrier_free}."
                ),
                auto_fixed=False,
            ),
        )

    def _check_hash_and_duplicates(
        self, text: str, existing_hashes: List[str]
    ) -> Tuple[str, RuleEvaluationResult]:
        normalized = re.sub(r"\s+", " ", text.strip())
        sha256_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if sha256_hash in existing_hashes:
            return sha256_hash, RuleEvaluationResult(
                rule_id="§ 10 A / § 11 D",
                rule_name="SHA-256 Prüfsumme & Vektor-Update (§ 10 A c)",
                status=RuleCheckStatus.WARNING,
                score_or_metric=None,
                details=(
                    f"Inhalts-Hash ({sha256_hash[:12]}...) ist bereits im Vorzimmer/Index bekannt. "
                    f"Gemäß § 10 A c (Vektor-Ersetzung) wird das Dokument NICHT blockiert, sondern bestehende Chunks werden aktualisiert."
                ),
                auto_fixed=True,
            )

        return sha256_hash, RuleEvaluationResult(
            rule_id="§ 10 A / § 11 D",
            rule_name="SHA-256 Prüfsumme & Duplikat-Filter",
            status=RuleCheckStatus.PASSED,
            score_or_metric=None,
            details=f"Neuer, eindeutiger SHA-256 Fingerabdruck erzeugt ({sha256_hash[:12]}...).",
            auto_fixed=False,
        )

    def _persist_audit_log(self, report: QualityAuditReport) -> None:
        try:
            log_file = Path("data/audit_logs") / f"{report.audit_id}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Audit-Logs {report.audit_id}: {e}")