"""
services/etl_ingest_service/extractor.py
Extraktions-Schicht (Extract) für PDF-Dateien, Textdateien und Webseiten.
Vollständig Pylance-/Pyright-geprüft.
"""

import asyncio
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import fitz  # PyMuPDF
import httpx
import pdfplumber
import trafilatura
from bs4 import BeautifulSoup, Tag

from shared.logger import get_service_logger

logger = get_service_logger("etl-extractor")


class DocumentExtractor:
    """Liest lokale PDF-/TXT-Dateien sowie Webseiten (mit Blacklist & automatischer Relevanzprüfung) aus."""

    def __init__(self, config_dir: Path = Path("config")):
        self.config_path = config_dir / "whitelist_hildesheim.json"
        self.filter_cfg = self._load_filter_config()
        self.user_agent = "HOMIE-QuartiersBot-HULLS/1.2 (+https://uni-hildesheim.de)"
        self.politeness_delay_sec = 1.0  # § 9 C: Politeness Policy

    def _load_filter_config(self) -> Dict[str, Any]:
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der URL-Filter-Konfiguration: {e}")
        return {
            "blocked_domains": [],
            "blocked_url_patterns": [],
            "trusted_official_domains": [],
            "relevance_keywords": ["hildesheim", "beratung", "pflege", "hilfe", "angebot"],
            "min_relevance_hits": 2,
            "max_crawl_depth": 1,
        }

    # =================================================================
    # AUTOMATISCHE ERMITTLUNG VON HERAUSGEBER & ABLAUFDATUM (§ 1 A, § 10 B)
    # =================================================================

    def detect_publisher_automatically(
        self, text: str, source_path_or_url: str, meta_author: Optional[str] = None
    ) -> str:
        """
        Ermittelt den Herausgeber automatisch.
        Priorität 1: Bekannte soziale/kommunale Träger in Titel, Dateiname, URL oder Text.
        Priorität 2: Explizite 'Herausgeber:'-Nennung im Text.
        Priorität 3: PDF-/HTML-Author-Tag (gefiltert gegen Layout-Programme wie Adobe InDesign).
        """
        src_lower = source_path_or_url.lower()
        text_lower = text[:5000].lower()

        known_orgs = [
            (["landkreis hildesheim", "landkreishildesheim.de", "im landkreis hildeshei"], "Landkreis Hildesheim"),
            (["stadt hildesheim", "hildesheim.de"], "Stadt Hildesheim"),
            (["awo-hildesheim", "awo hildesheim", "arbeiterwohlfahrt kreisverband hildesheim"], "AWO Hildesheim"),
            (["awo-bs", "awo bezirksverband braunschweig", "arbeiterwohlfahrt"], "AWO (Arbeiterwohlfahrt)"),
            (["caritas-hildesheim", "caritasverband", "caritas"], "Caritas Hildesheim"),
            (["diakonie-hildesheim", "diakonisches werk", "diakonie"], "Diakonie Hildesheim"),
            (["drk-hildesheim", "deutsches rotes kreuz"], "DRK Hildesheim"),
            (["eutb", "ergänzende unabhängige teilhabeberatung"], "EUTB Teilhabeberatung Hildesheim"),
            (["paritätische", "paritaetischer"], "Der Paritätische Hildesheim"),
            (["uni-hildesheim", "universität hildesheim"], "Universität Hildesheim"),
            (["niedersächsisches ministerium", "land niedersachsen"], "Land Niedersachsen"),
        ]
        for triggers, org_name in known_orgs:
            if any(t in src_lower or t in text_lower for t in triggers):
                return org_name

        pub_match = re.search(
            r"(?:Herausgeber(?:in)?|Träger|Verantwortlich\s+für\s+den\s+Inhalt|Anbieter)\s*[:\-]\s*([^\n|]{4,70})",
            text[:4000],
            re.IGNORECASE,
        )
        if pub_match:
            candidate = re.sub(r"\s+", " ", pub_match.group(1)).strip()
            if len(candidate) > 3:
                return candidate

        software_blacklist = [
            "adobe", "indesign", "illustrator", "photoshop", "acrobat", "distiller",
            "microsoft", "word", "powerpoint", "excel", "publisher", "libreoffice",
            "openoffice", "quark", "corel", "latex", "tex", "ghostscript", "pdf",
            "scanner", "canon", "hp", "brother", "epson", "admin", "user", "unbekannt"
        ]
        if meta_author and len(meta_author.strip()) > 2:
            cleaned_author = meta_author.strip()
            if not any(sw in cleaned_author.lower() for sw in software_blacklist):
                return cleaned_author

        if source_path_or_url.startswith(("http://", "https://")):
            domain = urlparse(source_path_or_url).netloc.lower().replace("www.", "")
            if domain:
                return f"Webquelle ({domain})"

        return "Quartiers-Dokument Hildesheim"

    def detect_valid_until_automatically(self, text: str) -> Optional[date]:
        """Sucht im Text automatisch nach einem Gültigkeits- oder Veranstaltungsdatum (§ 10 B a)."""
        patterns = [
            r"(?:gültig\s+bis(?:\s+zum)?|befristet\s+bis|endet\s+am|ablaufdatum\s*:?)\s*(\d{1,2})\.(\d{1,2})\.(20\d{2})",
            r"(?:veranstaltung\s+am|termin\s*:?\s*(?:am)?)\s*(\d{1,2})\.(\d{1,2})\.(20\d{2})",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                except ValueError:
                    continue

        year_match = re.search(
            r"(?:gültig\s+(?:für|in)|programm|jahresprogramm)\s+(202[5-9])", text, re.IGNORECASE
        )
        if year_match:
            try:
                return date(int(year_match.group(1)), 12, 31)
            except ValueError:
                pass

        return None

    # =================================================================
    # 1. PDF- & TEXT-EXTRAKTION (TA-01, § 4 A, § 10 A b)
    # =================================================================

    def extract_from_file(
        self, file_path: Path
    ) -> Tuple[str, Optional[date], str, str, Optional[date]]:
        """
        Extrahiert Text, Tabellen und automatische Metadaten aus PDF- oder TXT-Dateien.
        Gibt (extrahierter_text, dokument_datum, titel, auto_publisher, auto_valid_until) zurück.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        suffix = file_path.suffix.lower()
        title = file_path.stem.replace("_", " ").strip()

        if suffix in (".txt", ".md"):
            logger.info(f"Lese Textdatei: {file_path.name}")
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            mtime = date.fromtimestamp(file_path.stat().st_mtime)
            auto_pub = self.detect_publisher_automatically(raw_text, str(file_path))
            auto_valid = self.detect_valid_until_automatically(raw_text)
            return raw_text, mtime, title, auto_pub, auto_valid

        if suffix == ".pdf":
            return self._extract_pdf_layout_aware(file_path, title)

        raise ValueError(f"Nicht unterstütztes Dateiformat: {suffix}. Erlaubt sind .pdf, .txt, .md")

    def _extract_pdf_layout_aware(
        self, pdf_path: Path, default_title: str
    ) -> Tuple[str, Optional[date], str, str, Optional[date]]:
        """
        Schnelles, mehrstufiges PDF-Parsing (begrenzt pdfplumber auf max. 15 Seiten gegen Timeouts).
        """
        logger.info(f"Starte Layout-sensitives PDF-Parsing für: {pdf_path.name}")
        extracted_pages: List[str] = []
        doc_date: Optional[date] = None
        doc_title = default_title
        meta_author: Optional[str] = None

        try:
            with fitz.open(pdf_path) as doc:
                meta = doc.metadata or {}
                raw_title = meta.get("title")
                if isinstance(raw_title, str) and len(raw_title.strip()) > 3:
                    clean_t = raw_title.strip()
                    if not clean_t.lower().endswith((".indd", ".pdf", ".docx")):
                        doc_title = clean_t

                raw_author = meta.get("author")
                if isinstance(raw_author, str):
                    meta_author = raw_author

                raw_pdf_date = str(meta.get("modDate") or meta.get("creationDate") or "")
                date_match = re.search(r"D:(\d{4})(\d{2})(\d{2})", raw_pdf_date)
                if date_match:
                    try:
                        doc_date = date(
                            int(date_match.group(1)),
                            int(date_match.group(2)),
                            int(date_match.group(3)),
                        )
                    except ValueError:
                        doc_date = None

                for page in doc:
                    raw_blocks = page.get_text("blocks")
                    blocks: List[Any] = list(raw_blocks) if isinstance(raw_blocks, list) else []
                    blocks.sort(key=lambda b: (round(float(b[1]), 1), round(float(b[0]), 1)))
                    page_text = "\n\n".join(
                        str(b[4]).strip() for b in blocks if len(b) > 6 and b[6] == 0 and len(str(b[4]).strip()) > 0
                    )
                    if len(page_text.strip()) < 20:
                        raw_fallback = page.get_text("text")
                        page_text = raw_fallback.strip() if isinstance(raw_fallback, str) else ""
                    extracted_pages.append(page_text)

            with pdfplumber.open(pdf_path) as plumber_doc:
                max_table_pages = min(len(plumber_doc.pages), 15)
                for idx in range(max_table_pages):
                    p_page = plumber_doc.pages[idx]
                    if idx < len(extracted_pages) and len(extracted_pages[idx].strip()) < 20:
                        p_text = (p_page.extract_text() or "").strip()
                        if p_text:
                            extracted_pages[idx] = p_text

                    tables = p_page.extract_tables()
                    if tables:
                        md_tables = []
                        for table in tables:
                            cleaned_rows = [
                                [re.sub(r"\s+", " ", (cell or "").strip()) for cell in row]
                                for row in table
                                if any(cell and cell.strip() for cell in row)
                            ]
                            if len(cleaned_rows) >= 2:
                                header = " | ".join(cleaned_rows[0])
                                sep = " | ".join(["---"] * len(cleaned_rows[0]))
                                body = "\n".join(" | ".join(r) for r in cleaned_rows[1:])
                                md_tables.append(f"\n[TABELLE SEITE {idx + 1}]\n{header}\n{sep}\n{body}\n")
                        if md_tables and idx < len(extracted_pages):
                            extracted_pages[idx] += "\n\n" + "\n".join(md_tables)

            if not doc_date:
                doc_date = date.fromtimestamp(pdf_path.stat().st_mtime)

            full_text = "\n\n".join(p for p in extracted_pages if p.strip())
            auto_pub = self.detect_publisher_automatically(full_text, str(pdf_path), meta_author)
            auto_valid = self.detect_valid_until_automatically(full_text)
            return full_text, doc_date, doc_title, auto_pub, auto_valid

        except Exception as e:
            logger.error(f"Fehler beim PDF-Parsing von {pdf_path}: {e}", exc_info=True)
            raise RuntimeError(f"PDF konnte nicht gelesen werden: {e}")

    # =================================================================
    # 2. WEB-SCRAPER MIT MANUELLER BLACKLIST & AUTO-RELEVANZ (§ 7 - § 9)
    # =================================================================

    def is_url_allowed(self, url: str) -> Tuple[bool, str]:
        self.filter_cfg = self._load_filter_config()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False, f"Ungültiges URL-Format: '{url}'. Bitte mit http:// oder https:// angeben."

        domain = parsed.netloc.lower().replace("www.", "")
        path_lower = parsed.path.lower()

        blocked_domains = [str(d).lower().replace("www.", "") for d in self.filter_cfg.get("blocked_domains", [])]
        for b_dom in blocked_domains:
            if domain == b_dom or domain.endswith("." + b_dom):
                return False, f"Domain '{domain}' steht auf der manuellen Blacklist (blocked_domains)."

        blocked_patterns = [str(p) for p in self.filter_cfg.get("blocked_url_patterns", [])]
        for pattern in blocked_patterns:
            if pattern.lower() in path_lower:
                return False, f"URL-Pfad enthält gesperrtes Blacklist-Muster '{pattern}' (§ 7 B a)."

        return True, "URL ist nicht auf der Blacklist – zugelassen zur automatischen Inhaltsprüfung."

    def evaluate_content_relevance(self, text: str, url: str) -> Tuple[bool, str]:
        keywords = [str(k) for k in self.filter_cfg.get("relevance_keywords", [])]
        min_hits = int(self.filter_cfg.get("min_relevance_hits", 2))
        combined_lower = (text + " " + url).lower()

        matched_kw = [kw for kw in keywords if kw.lower() in combined_lower]
        if len(matched_kw) < min_hits:
            return (
                False,
                f"Automatische Relevanzprüfung (§ 7 B) nicht bestanden: Die Webseite enthält zu wenige "
                f"Quartiers-/Beratungs-Schlüsselwörter (Gefunden: {matched_kw}, benötigt: mind. {min_hits}).",
            )
        return (
            True,
            f"Automatische Relevanzprüfung bestanden ({len(matched_kw)} relevante Themenbegriffe erkannt).",
        )

    async def check_robots_txt(self, url: str, client: httpx.AsyncClient) -> Tuple[bool, str]:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            resp = await client.get(robots_url, timeout=6.0)
            if resp.status_code == 200 and resp.text.strip():
                rp = RobotFileParser()
                rp.parse(resp.text.splitlines())
                if not rp.can_fetch(self.user_agent, url) and not rp.can_fetch("*", url):
                    return False, f"§ 9 A: robots.txt von {parsed.netloc} verbietet das Auslesen dieser URL explizit."
                return True, "robots.txt geprüft und Zugriff erlaubt (§ 9 A)."
            return True, f"Keine einschränkende robots.txt unter {robots_url} – Zugriff erlaubt."
        except Exception as e:
            logger.info(f"robots.txt Hinweis für {robots_url}: {e} – fahre mit Crawling fort.")
            return True, "robots.txt nicht erreichbar – Standard-Crawling erlaubt."

    async def scrape_url_with_depth(
        self, start_url: str, follow_depth_1: bool = True
    ) -> Tuple[str, str, List[str], str, Optional[date]]:
        allowed, reason = self.is_url_allowed(start_url)
        if not allowed:
            raise ValueError(reason)

        headers = {"User-Agent": self.user_agent}
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            robots_ok, robots_msg = await self.check_robots_txt(start_url, client)
            if not robots_ok:
                raise PermissionError(robots_msg)

            main_text, main_title, context_links, meta_author = await self._fetch_and_clean_page(
                start_url, client
            )

            is_relevant, rel_msg = self.evaluate_content_relevance(main_text, start_url)
            if not is_relevant:
                raise ValueError(rel_msg)

            combined_sections = [f"# {main_title}\nQuelle: {start_url}\n\n{main_text}"]
            crawled_sub_urls: List[str] = []

            if follow_depth_1 and context_links:
                for sub_url in context_links[:3]:
                    await asyncio.sleep(self.politeness_delay_sec)
                    sub_allowed, _ = self.is_url_allowed(sub_url)
                    if not sub_allowed:
                        continue
                    try:
                        sub_text, sub_title, _, _ = await self._fetch_and_clean_page(sub_url, client)
                        sub_rel, _ = self.evaluate_content_relevance(sub_text, sub_url)
                        if len(sub_text.split()) >= 25 and sub_rel:
                            combined_sections.append(
                                f"## Unterseite: {sub_title}\nQuelle: {sub_url}\n\n{sub_text}"
                            )
                            crawled_sub_urls.append(sub_url)
                    except Exception as e:
                        logger.warning(f"Konnte Unterseite {sub_url} nicht crawlen: {e}")

            full_scraped_text = "\n\n---\n\n".join(combined_sections)
            auto_pub = self.detect_publisher_automatically(full_scraped_text, start_url, meta_author)
            auto_valid = self.detect_valid_until_automatically(full_scraped_text)

            return full_scraped_text, main_title, crawled_sub_urls, auto_pub, auto_valid

    async def _fetch_and_clean_page(
        self, url: str, client: httpx.AsyncClient
    ) -> Tuple[str, str, List[str], Optional[str]]:
        resp = await client.get(url, timeout=15.0)
        resp.raise_for_status()
        html_content = resp.text

        soup = BeautifulSoup(html_content, "html.parser")
        title = soup.title.string.strip() if (soup.title and soup.title.string) else url

        # Pylance-sicheres Auslesen der HTML-Meta-Tags (og:site_name, author, publisher)
        meta_author: Optional[str] = None
        target_meta_keys = {"og:site_name", "author", "publisher"}
        for node in soup.find_all("meta"):
            if not isinstance(node, Tag):
                continue
            prop_val = node.get("property") or node.get("name")
            prop_str = (prop_val[0] if isinstance(prop_val, list) else str(prop_val or "")).lower().strip()
            if prop_str in target_meta_keys:
                content_val = node.get("content")
                content_str = (content_val[0] if isinstance(content_val, list) else str(content_val or "")).strip()
                if len(content_str) > 2:
                    meta_author = content_str
                    break

        context_links: List[str] = []
        context_keywords = [
            "beratung", "pflege", "angebot", "kontakt", "sprechstunde",
            "hilfe", "senioren", "familie", "hildesheim", "quartier"
        ]

        # Pylance-sicheres Auflösen aller <a href="..."> Links
        for node in soup.find_all("a"):
            if not isinstance(node, Tag):
                continue
            href_val = node.get("href")
            if not href_val:
                continue
            raw_href = (href_val[0] if isinstance(href_val, list) else str(href_val)).strip()
            if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            abs_url = urljoin(url, raw_href)
            node["href"] = abs_url
            link_text = node.get_text(" ", strip=True)
            if link_text:
                node.string = f"{link_text} ({abs_url})"

            if any(kw in abs_url.lower() or kw in link_text.lower() for kw in context_keywords):
                if abs_url != url and abs_url not in context_links:
                    context_links.append(abs_url)

        modified_html = str(soup)
        extracted_md = trafilatura.extract(
            modified_html,
            include_links=True,
            include_tables=True,
            output_format="txt",
        )

        if not extracted_md or len(extracted_md.strip()) < 50:
            for noisy in soup(["script", "style", "nav", "footer", "header", "aside"]):
                noisy.decompose()
            extracted_md = soup.get_text(separator="\n\n", strip=True)

        return extracted_md.strip(), title, context_links, meta_author