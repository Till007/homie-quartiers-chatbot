"""
services/quality_gate_service/pipeline.py
Implementiert die schrittweise Qualitätskontrolle gemäß '2.0 Konzeption & Systementscheidung.txt'.
Jeder Paragraph (§ 1 bis § 11) ist als eigene, nachvollziehbare Prüfmethode realisiert.
"""

import hashlib
import json
import re
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

from langdetect import detect, LangDetectException

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

    def _load_json(self, filename: str) -> Dict:
        """Lädt Konfigurationsdateien fehlertolerant mit sauberem Error-Catching."""
        filepath = self.config_dir / filename
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            logger.warning(f"Konfigurationsdatei {filepath} nicht gefunden. Nutze Fallback.")
            return {}
        except Exception as e:
            logger.error(f"Fehler beim Laden von {filepath}: {e}", exc_info=True)
            return {}

    def inspect_document(self, req: QualityInspectionRequest) -> QualityAuditReport:
        """
        Hauptmethode: Führt alle Prüfregeln nacheinander aus und erstellt den Audit-Report.
        """
        audit_id = f"AUDIT-{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"[{audit_id}] Starte Qualitätsprüfung für '{req.title}' ({req.source_url_or_path})")

        rule_results: List[RuleEvaluationResult] = []
        blocking_reasons: List[str] = []

        # Schritt 0 (§ 3 A): Zeichensatzkodierung nach UTF-8 sicherstellen
        working_text, res_utf8 = self._check_utf8_encoding(req.raw_text)
        rule_results.append(res_utf8)

        # Schritt 1 (§ 1 A & B, § 5 A): Offizieller Status (SSOT) vs. Inoffiziell
        source_status, res_ssot = self._check_ssot_status(req.publisher, req.source_url_or_path)
        rule_results.append(res_ssot)

        # Schritt 2 (§ 2 A): Signal-to-Noise Ratio (Rauschabstand prüfen)
        res_snr = self._check_signal_to_noise(working_text)
        rule_results.append(res_snr)

        # Schritt 3 (§ 2 B): Strukturelle Integrität (Absätze & Überschriften für Chunking)
        res_struct = self._check_structural_integrity(working_text)
        rule_results.append(res_struct)

        # Schritt 4 (§ 2 C & § 10 B): Aktualität (Recency) & Time-to-Live (TTL)
        is_expired, res_recency = self._check_recency_and_ttl(req.document_date, req.valid_until)
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

        # Schritt 8 (§ 2 H): Sprachniveau (Standard/Einfach/Leicht) & Passiv-/Nominalstil
        lang_level, res_level = self._classify_language_level(working_text)
        rule_results.append(res_level)

        # Schritt 9 (§ 2 I & § 2 D): Dokumentensprache, Vor-Ort-Service-Sprachen & Diversitäts-Tags
        doc_lang, service_langs, target_groups, diet_opts, barrier_free, res_meta = (
            self._extract_diversity_and_language_metadata(working_text)
        )
        rule_results.append(res_meta)

        # Schritt 10 (§ 10 A & § 11 D): SHA-256 Hash-Berechnung & Duplikat-Prüfung
        content_hash, res_hash = self._check_hash_and_duplicates(working_text, req.existing_hashes)
        rule_results.append(res_hash)

        # Auswertung: Gibt es einen kritischen Fehler (FAILED)?
        for r in rule_results:
            if r.status == RuleCheckStatus.FAILED:
                blocking_reasons.append(f"[{r.rule_id}] {r.rule_name}: {r.details}")

        approved = len(blocking_reasons) == 0

        # Metadaten-Objekt zusammenbauen
        metadata = DocumentMetadata(
            doc_id=f"DOC-{content_hash[:10]}",
            title=req.title,
            source_url_or_path=req.source_url_or_path,
            publisher=req.publisher,
            status=source_status,
            created_or_updated_at=req.document_date,
            valid_until=req.valid_until,
            is_expired=is_expired,
            content_hash_sha256=content_hash,
            doc_lang=doc_lang,
            language_level=lang_level,
            service_languages=service_langs,
            target_groups=target_groups,
            dietary_options=diet_opts,
            barrier_free_wheelchair=barrier_free,
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
        logger.info(
            f"[{audit_id}] Prüfung beendet. Freigegeben: {approved} "
            f"(Blockierende Gründe: {len(blocking_reasons)})"
        )
        return report

    # =================================================================
    # EINZELNE REGEL-METHODEN (Exakt nach Regelwerk § 1 bis § 11)
    # =================================================================

    def _check_utf8_encoding(self, text: str) -> Tuple[str, RuleEvaluationResult]:
        """§ 3 A: Zeichensatzkodierung zwingend nach UTF-8."""
        try:
            encoded = text.encode("utf-8", errors="replace").decode("utf-8")
            # Entferne unsichtbare Steuerzeichen (außer Zeilenumbruch/Tab)
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", encoded)
            return cleaned, RuleEvaluationResult(
                rule_id="§ 3 A",
                rule_name="UTF-8 Zeichensatzkodierung",
                status=RuleCheckStatus.PASSED,
                details="Text erfolgreich nach UTF-8 normalisiert und Steuerzeichen bereinigt.",
                auto_fixed=(cleaned != text),
            )
        except Exception as e:
            return text, RuleEvaluationResult(
                rule_id="§ 3 A",
                rule_name="UTF-8 Zeichensatzkodierung",
                status=RuleCheckStatus.FAILED,
                details=f"UTF-8 Kodierungsfehler: {e}",
            )

    def _check_ssot_status(
        self, publisher: str | None, source_path: str
    ) -> Tuple[SourceStatus, RuleEvaluationResult]:
        """§ 1 A/B & § 5 A: Prüfung auf offizielle Quelle (Single Source of Truth)."""
        allowed_domains = self.whitelist_cfg.get("allowed_domains", [])
        official_publishers = ["awo", "stadt hildesheim", "landkreis hildesheim", "ministerium", "caritas", "diakonie"]

        is_whitelisted_domain = any(domain in source_path.lower() for domain in allowed_domains)
        is_official_pub = bool(
            publisher and any(op in publisher.lower() for op in official_publishers)
        )

        if is_whitelisted_domain or is_official_pub:
            return SourceStatus.OFFICIAL, RuleEvaluationResult(
                rule_id="§ 1 A/B",
                rule_name="SSOT & Offizielle Quelle",
                status=RuleCheckStatus.PASSED,
                details=f"Als offizielle Quelle eingestuft (status: official). Herausgeber/Quelle verifiziert.",
            )
        return SourceStatus.UNOFFICIAL, RuleEvaluationResult(
            rule_id="§ 5 A/B",
            rule_name="Nicht-offizielle Quelle (Triangulation erforderlich)",
            status=RuleCheckStatus.WARNING,
            details="Kein offizieller Herausgeber erkannt. Markiert als 'status: unofficial' (§ 5 A) zur Triangulation.",
        )

    def _check_signal_to_noise(self, text: str) -> RuleEvaluationResult:
        """§ 2 A: Signal-to-Noise Ratio (Rauschabstand & Informationsdichte)."""
        words = re.findall(r"\b\w+\b", text)
        if len(words) < 25:
            return RuleEvaluationResult(
                rule_id="§ 2 A",
                rule_name="Signal-to-Noise Ratio (Rauschabstand)",
                status=RuleCheckStatus.FAILED,
                score_or_metric=float(len(words)),
                details=f"Zu geringe Informationsdichte ({len(words)} Wörter < Minimum 25 Wörter).",
            )

        # Verhältnis von Buchstaben/Zahlen zu Sonderzeichen/Whitespace-Müll berechnen
        alnum_chars = sum(c.isalnum() for c in text)
        total_chars = max(len(text.strip()), 1)
        snr_ratio = round(alnum_chars / total_chars, 2)

        if snr_ratio < 0.45:
            return RuleEvaluationResult(
                rule_id="§ 2 A",
                rule_name="Signal-to-Noise Ratio (Rauschabstand)",
                status=RuleCheckStatus.FAILED,
                score_or_metric=snr_ratio,
                details=f"Zu hohes Störrauschen (SNR: {snr_ratio} < 0.45). Vermutlich defektes OCR oder Tabellen-Artefakte.",
            )
        return RuleEvaluationResult(
            rule_id="§ 2 A",
            rule_name="Signal-to-Noise Ratio (Rauschabstand)",
            status=RuleCheckStatus.PASSED,
            score_or_metric=snr_ratio,
            details=f"Gute Informationsdichte ({len(words)} Wörter, SNR: {snr_ratio}).",
        )

    def _check_structural_integrity(self, text: str) -> RuleEvaluationResult:
        """§ 2 B: Strukturelle Integrität (Absätze & Struktur für kohärentes Chunking)."""
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
        has_headings = bool(re.search(r"(^#+\s|\n[A-ZÄÖÜ][^\n]{3,60}:\n)", text))

        if len(paragraphs) == 0:
            return RuleEvaluationResult(
                rule_id="§ 2 B",
                rule_name="Strukturelle Integrität",
                status=RuleCheckStatus.FAILED,
                details="Keine zusammenhängenden Sinnabschnitte erkennbar.",
            )
        if len(paragraphs) == 1 and len(text) > 1500 and not has_headings:
            return RuleEvaluationResult(
                rule_id="§ 2 B",
                rule_name="Strukturelle Integrität",
                status=RuleCheckStatus.WARNING,
                details="Langer Fließtext ohne klare Absatz-/Überschriftenstruktur. Semantisches Chunking im ETL nötig.",
            )
        return RuleEvaluationResult(
            rule_id="§ 2 B",
            rule_name="Strukturelle Integrität",
            status=RuleCheckStatus.PASSED,
            score_or_metric=float(len(paragraphs)),
            details=f"Struktur intakt ({len(paragraphs)} logische Absätze erkannt).",
        )

    def _check_recency_and_ttl(
        self, doc_date: date | None, valid_until: date | None
    ) -> Tuple[bool, RuleEvaluationResult]:
        """§ 2 C & § 10 B: Aktualität (Zeitstempel) und Verfallsdatum (TTL)."""
        today = date.today()
        is_expired = False

        if valid_until and valid_until < today:
            is_expired = True
            return is_expired, RuleEvaluationResult(
                rule_id="§ 10 B b/c",
                rule_name="Time-to-Live (TTL) & Historisierung",
                status=RuleCheckStatus.WARNING,
                details=(
                    f"Ablaufdatum ({valid_until}) überschritten. Dokument wird gemäß § 10 B b nicht gelöscht, "
                    f"sondern mit 'is_expired=True' für einen Prompt-Warnhinweis markiert."
                ),
            )

        if not doc_date and not valid_until:
            return is_expired, RuleEvaluationResult(
                rule_id="§ 2 C",
                rule_name="Aktualität (Recency)",
                status=RuleCheckStatus.WARNING,
                details="Kein explizites Dokumentdatum übergeben. Ingest-Datum wird als Fallback-Zeitstempel gesetzt.",
            )

        return is_expired, RuleEvaluationResult(
            rule_id="§ 2 C / § 10 B",
            rule_name="Aktualität & Gültigkeit",
            status=RuleCheckStatus.PASSED,
            details=f"Zeitstempel geprüft (Datum: {doc_date}, Gültig bis: {valid_until or 'unbefristet'}).",
        )

    def _check_security_patient_records(self, text: str) -> RuleEvaluationResult:
        """§ 2 G: Sicherheits-Audit – Blockieren echter Patientenakten oder Diagnosedokumente."""
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
                details=f"KRITISCHER SICHERHEITSVERSTOSS: Verdacht auf echte Patientenakte (Muster: {found}). Ingest blockiert!",
            )
        return RuleEvaluationResult(
            rule_id="§ 2 G",
            rule_name="Sicherheits-Audit (Schutz vor Patientenakten)",
            status=RuleCheckStatus.PASSED,
            details="Keine Merkmale persönlicher Patientenakten gefunden.",
        )

    def _check_and_redact_pii(self, text: str) -> Tuple[str, RuleEvaluationResult]:
        """
        § 2 F: Named Entity Recognition / PII-Schutz.
        Schwärzt IBAN-Kontodaten und persönliche Fallbeispiele vor der Vektorisierung.
        (Hinweis: Offizielle Kontakttelefonnummern von Beratungsstellen bleiben für FA-04 erhalten!)
        """
        redacted_text = text
        redactions_count = 0

        # 1. Deutsche & Internationale IBANs schwärzen
        iban_pattern = r"\b[A-Z]{2}\d{2}[A-Z0-9\s]{12,30}\b"
        redacted_text, n_iban = re.subn(iban_pattern, "[KONTONUMMER GESCHWÄRZT]", redacted_text)
        redactions_count += n_iban

        # 2. Geburtsdaten in Fallbeispielen schwärzen (z.B. "geb. am 12.04.1948")
        dob_pattern = r"(geb(?:oren)?\.?\s*(?:am)?\s*\d{1,2}\.\d{1,2}\.\d{2,4})"
        redacted_text, n_dob = re.subn(dob_pattern, "[GEBURTSDATUM GESCHWÄRZT]", redacted_text, flags=re.IGNORECASE)
        redactions_count += n_dob

        if redactions_count > 0:
            return redacted_text, RuleEvaluationResult(
                rule_id="§ 2 F",
                rule_name="PII- & NER-Pseudonymisierung",
                status=RuleCheckStatus.WARNING,
                score_or_metric=float(redactions_count),
                details=f"{redactions_count} sensible personenbezogene Angaben (IBAN/Geburtsdaten) automatisch geschwärzt.",
                auto_fixed=True,
            )

        return redacted_text, RuleEvaluationResult(
            rule_id="§ 2 F",
            rule_name="PII- & NER-Pseudonymisierung",
            status=RuleCheckStatus.PASSED,
            score_or_metric=0.0,
            details="Keine sensiblen Privatdaten (IBAN/Geburtsdaten) im Text entdeckt.",
        )

    def _check_discriminatory_language(self, text: str) -> Tuple[str, RuleEvaluationResult]:
        """§ 2 E a: Ressourcenorientierung statt Defizitorientierung (automatische Ersetzung)."""
        updated_text = text
        replaced_terms = []

        for bad_term, neutral_term in self.discriminatory_terms.items():
            pattern = re.compile(re.escape(bad_term), re.IGNORECASE)
            if pattern.search(updated_text):
                updated_text = pattern.sub(neutral_term, updated_text)
                replaced_terms.append(f"'{bad_term}' -> '{neutral_term}'")

        if replaced_terms:
            return updated_text, RuleEvaluationResult(
                rule_id="§ 2 E a",
                rule_name="Kultursensible & diskriminierungsfreie Sprache",
                status=RuleCheckStatus.WARNING,
                details=f"Defizitorientierte Begriffe automatisch ersetzt: {', '.join(replaced_terms)}",
                auto_fixed=True,
            )

        return updated_text, RuleEvaluationResult(
            rule_id="§ 2 E a",
            rule_name="Kultursensible & diskriminierungsfreie Sprache",
            status=RuleCheckStatus.PASSED,
            details="Sprachduktus geprüft: Keine defizitorientierten Begriffe aus der Sperrliste gefunden.",
        )

    def _classify_language_level(self, text: str) -> Tuple[LanguageLevel, RuleEvaluationResult]:
        """§ 2 H a/c: Klassifizierung des Sprachniveaus & Prüfung auf Behördenjargon/Nominalstil."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0]
        words = re.findall(r"\b\w+\b", text)
        avg_sentence_len = len(words) / max(len(sentences), 1)
        long_words = [w for w in words if len(w) > 14]
        long_word_ratio = len(long_words) / max(len(words), 1)

        # Erkennung von Mediopunkten (z.B. Pflege·beratung) oder explizitem Hinweis auf Leichte Sprache
        if "leichte sprache" in text.lower() or "·" in text or (avg_sentence_len <= 9 and long_word_ratio < 0.05):
            level = LanguageLevel.EASY
        elif avg_sentence_len <= 15 and long_word_ratio < 0.12:
            level = LanguageLevel.SIMPLE
        else:
            level = LanguageLevel.STANDARD

        status = RuleCheckStatus.PASSED
        details = (
            f"Sprachniveau klassifiziert als '{level.value}' "
            f"(Ø Satzlänge: {avg_sentence_len:.1f} Wörter, Anteil komplexer Wörter: {long_word_ratio:.1%})."
        )
        if avg_sentence_len > 24 or long_word_ratio > 0.20:
            status = RuleCheckStatus.WARNING
            details += " Hohe Dichte an Behördenjargon/Schachtelsätzen (§ 2 H c) – Vereinfachung im Prompt/ETL empfohlen."

        return level, RuleEvaluationResult(
            rule_id="§ 2 H",
            rule_name="Sprachniveau & Verständlichkeit",
            status=status,
            score_or_metric=round(avg_sentence_len, 1),
            details=details,
        )

    def _extract_diversity_and_language_metadata(
        self, text: str
    ) -> Tuple[str, List[str], List[TargetGroupTag], List[DietaryOption], bool, RuleEvaluationResult]:
        """§ 2 D b & § 2 I: Extrahiert Dokumentensprache, Vor-Ort-Sprachen und Diversitätsmerkmale."""
        text_lower = text.lower()

        # 1. Primärsprache nach ISO 639-1 erkennen (§ 2 I)
        try:
            doc_lang = detect(text)
        except LangDetectException:
            doc_lang = "de"

        # 2. Vor-Ort gesprochene Service-Sprachen erkennen (§ 2 I)
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

        # 3. Zielgruppen erkennen (§ 2 D)
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

        # 4. Ernährungsspezifika (§ 2 D b)
        diet_opts: List[DietaryOption] = []
        if "halal" in text_lower:
            diet_opts.append(DietaryOption.HALAL)
        if "koscher" in text_lower or "kosher" in text_lower:
            diet_opts.append(DietaryOption.KOSHER)
        if "vegetarisch" in text_lower:
            diet_opts.append(DietaryOption.VEGETARIAN)
        if "vegan" in text_lower:
            diet_opts.append(DietaryOption.VEGAN)

        # 5. Barrierefreiheit (§ 2 D b)
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
                details=(
                    f"Erkannt: doc_lang='{doc_lang}', service_languages={service_langs}, "
                    f"Zielgruppen={[t.value for t in target_groups]}, Barrierefrei={barrier_free}."
                ),
            ),
        )

    def _check_hash_and_duplicates(
        self, text: str, existing_hashes: List[str]
    ) -> Tuple[str, RuleEvaluationResult]:
        """§ 10 A a & § 11 D: Berechnet SHA-256 Prüfsumme und verhindert Duplikat-Ingestion."""
        normalized = re.sub(r"\s+", " ", text.strip())
        sha256_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if sha256_hash in existing_hashes:
            return sha256_hash, RuleEvaluationResult(
                rule_id="§ 10 A / § 11 D",
                rule_name="SHA-256 Prüfsumme & Duplikat-Filter",
                status=RuleCheckStatus.FAILED,
                details=f"Exaktes Duplikat erkannt (SHA-256: {sha256_hash[:12]}...). Dokument ist bereits im Index.",
            )

        return sha256_hash, RuleEvaluationResult(
            rule_id="§ 10 A / § 11 D",
            rule_name="SHA-256 Prüfsumme & Duplikat-Filter",
            status=RuleCheckStatus.PASSED,
            details=f"Eindeutiger SHA-256 Fingerabdruck erzeugt ({sha256_hash[:12]}...).",
        )

    def _persist_audit_log(self, report: QualityAuditReport) -> None:
        """Speichert das Prüfprotokoll revisionssicher als JSON unter data/audit_logs/."""
        try:
            log_file = Path("data/audit_logs") / f"{report.audit_id}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Audit-Logs {report.audit_id}: {e}")