"""
services/etl_ingest_service/extractor.py
Extraktions-Schicht (Extract) für PDF-Dateien, Textdateien und Webseiten (Raum Hildesheim).
Implementiert TA-01 sowie § 4, § 7, § 8, § 9 und § 10 A b des Regelwerks.
"""

import asyncio
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import fitz  # PyMuPDF
import httpx
import pdfplumber
import trafilatura
from bs4 import BeautifulSoup

from shared.logger import get_service_logger

logger = get_service_logger("etl-extractor")


class DocumentExtractor:
    """Liest lokale PDF-/TXT-Dateien sowie freigegebene Webseiten aus Hildesheim aus."""

    def __init__(self, config_dir: Path = Path("config")):
        self.whitelist_cfg = self._load_whitelist(config_dir / "whitelist_hildesheim.json")
        self.user_agent = "HOMIE-QuartiersBot-HULLS/1.0 (+https://uni-hildesheim.de)"
        self.politeness_delay_sec = 1.2  # § 9 C: Politeness Policy (Rate Limit)

    def _load_whitelist(self, path: Path) -> Dict:
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Whitelist {path}: {e}")
        return {"allowed_domains": [], "blocked_url_patterns": [], "max_crawl_depth": 1}

    # =================================================================
    # 1. PDF- & TEXT-EXTRAKTION (TA-01, § 4 A, § 10 A b)
    # =================================================================

    def extract_from_file(self, file_path: Path) -> Tuple[str, Optional[date], str]:
        """
        Extrahiert Text und Tabellen aus PDF- oder TXT-Dateien.
        Gibt (extrahierter_text, dokument_datum, titel) zurück.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

        suffix = file_path.suffix.lower()
        title = file_path.stem.replace("_", " ").strip()

        if suffix == ".txt" or suffix == ".md":
            logger.info(f"Lese Textdatei: {file_path.name}")
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            mtime = date.fromtimestamp(file_path.stat().st_mtime)
            return raw_text, mtime, title

        if suffix == ".pdf":
            return self._extract_pdf_layout_aware(file_path, title)

        raise ValueError(f"Nicht unterstütztes Dateiformat: {suffix}. Erlaubt sind .pdf, .txt, .md")

    def _extract_pdf_layout_aware(self, pdf_path: Path, default_title: str) -> Tuple[str, Optional[date], str]:
        """
        Kombiniert PyMuPDF (Fließtext & PDF-Metadaten nach § 10 A b)
        und pdfplumber (Tabellenextraktion für Öffnungszeiten nach TA-01 & § 4 A).
        """
        logger.info(f"Starte Layout-sensitives PDF-Parsing für: {pdf_path.name}")
        extracted_pages: List[str] = []
        doc_date: Optional[date] = None
        doc_title = default_title

        try:
            # 1. PyMuPDF: Metadaten (§ 10 A b) und Basis-Textstruktur lesen
            with fitz.open(pdf_path) as doc:
                meta = doc.metadata or {}
                if meta.get("title") and len(meta["title"].strip()) > 3:
                    doc_title = meta["title"].strip()

                # PDF ModDate / CreationDate auslesen (Format: D:YYYYMMDD...)
                raw_pdf_date = meta.get("modDate") or meta.get("creationDate") or ""
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

                for page_num, page in enumerate(doc):
                    blocks = page.get_text("blocks")
                    # Sortiere Blöcke von oben nach unten, links nach rechts
                    blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
                    page_text = "\n\n".join(
                        b[4].strip() for b in blocks if b[6] == 0 and len(b[4].strip()) > 0
                    )
                    extracted_pages.append(page_text)

            # 2. pdfplumber: Tabellen (z.B. Sprechzeiten) als Markdown-Tabellen ergänzen
            with pdfplumber.open(pdf_path) as plumber_doc:
                for idx, p_page in enumerate(plumber_doc.pages):
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

            full_text = "\n\n".join(extracted_pages)
            return full_text, doc_date, doc_title

        except Exception as e:
            logger.error(f"Fehler beim PDF-Parsing von {pdf_path}: {e}", exc_info=True)
            raise RuntimeError(f"PDF konnte nicht gelesen werden: {e}")

    # =================================================================
    # 2. WEB-SCRAPER FÜR HILDESHEIM (§ 7, § 8, § 9)
    # =================================================================

    def is_url_allowed(self, url: str) -> Tuple[bool, str]:
        """Prüft § 7 A (Whitelist-Domains) und § 7 B a (URL-Filterung gegen Web-Rauschen)."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path_lower = parsed.path.lower()

        allowed_domains = self.whitelist_cfg.get("allowed_domains", [])
        if not any(domain == d or domain.endswith("." + d) for d in allowed_domains):
            return False, f"Domain '{domain}' steht nicht auf der Hildesheim-Whitelist (§ 7 A)."

        blocked_patterns = self.whitelist_cfg.get("blocked_url_patterns", [])
        for pattern in blocked_patterns:
            if pattern.lower() in path_lower:
                return False, f"URL-Pfad enthält blockiertes Muster '{pattern}' (§ 7 B a)."

        return True, "URL durch Whitelist- und Pfadfilter freigegeben."

    async def check_robots_txt(self, url: str, client: httpx.AsyncClient) -> Tuple[bool, str]:
        """
        § 9 A Strikte Robots.txt-Prüfung:
        - Ist eine robots.txt vorhanden? -> Inhalt strikt befolgen.
        - Ist KEINE robots.txt vorhanden? -> URL ignorieren und als solche markieren!
        """
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            resp = await client.get(robots_url, timeout=8.0)
            if resp.status_code != 200 or not resp.text.strip():
                return (
                    False,
                    f"§ 9 A Verstoß: Keine gültige robots.txt unter {robots_url} gefunden (Status {resp.status_code}). URL wird ignoriert.",
                )

            rp = RobotFileParser()
            rp.parse(resp.text.splitlines())
            if not rp.can_fetch(self.user_agent, url) and not rp.can_fetch("*", url):
                return False, f"§ 9 A Verstoß: robots.txt verbietet das Crawlen von {url}."

            return True, "robots.txt vorhanden und Zugriff erlaubt (§ 9 A)."
        except Exception as e:
            return False, f"§ 9 A Verstoß: robots.txt konnte nicht geprüft werden ({e}). URL wird ignoriert."

    async def scrape_url_with_depth(
        self, start_url: str, follow_depth_1: bool = True
    ) -> Tuple[str, str, List[str]]:
        """
        Crawlt eine Whitelist-URL (und optional relevante Kontext-Links mit Tiefe = 1).
        Gibt (gesammelter_markdown_text, seitentitel, gecrawlte_sub_urls) zurück.
        """
        allowed, reason = self.is_url_allowed(start_url)
        if not allowed:
            raise ValueError(reason)

        headers = {"User-Agent": self.user_agent}
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            robots_ok, robots_msg = await self.check_robots_txt(start_url, client)
            if not robots_ok:
                raise PermissionError(robots_msg)

            main_text, main_title, context_links = await self._fetch_and_clean_page(start_url, client)
            combined_sections = [f"# {main_title}\nQuelle: {start_url}\n\n{main_text}"]
            crawled_sub_urls: List[str] = []

            # § 8 C a: Kontextuelles Link-Crawling (Maximale Tiefe = 1, max. 3 relevante Unterseiten)
            if follow_depth_1 and context_links:
                for sub_url in context_links[:3]:
                    await asyncio.sleep(self.politeness_delay_sec)  # § 9 C Politeness Policy
                    sub_allowed, _ = self.is_url_allowed(sub_url)
                    if not sub_allowed:
                        continue
                    try:
                        sub_text, sub_title, _ = await self._fetch_and_clean_page(sub_url, client)
                        if len(sub_text.split()) >= 30:
                            combined_sections.append(
                                f"## Unterseite: {sub_title}\nQuelle: {sub_url}\n\n{sub_text}"
                            )
                            crawled_sub_urls.append(sub_url)
                    except Exception as e:
                        logger.warning(f"Konnte Unterseite {sub_url} nicht crawlen: {e}")

            return "\n\n---\n\n".join(combined_sections), main_title, crawled_sub_urls

    async def _fetch_and_clean_page(
        self, url: str, client: httpx.AsyncClient
    ) -> Tuple[str, str, List[str]]:
        """
        § 8 A (Boilerplate-Removal), § 8 B (HTML -> Markdown) und § 8 C (Relative Links -> Absolute URLs).
        """
        resp = await client.get(url, timeout=12.0)
        resp.raise_for_status()
        html_content = resp.text

        soup = BeautifulSoup(html_content, "html.parser")
        title = soup.title.string.strip() if (soup.title and soup.title.string) else url

        # § 8 C: Alle relativen Links (<a href="/...">) im DOM in absolute URLs auflösen
        context_links: List[str] = []
        context_keywords = ["beratung", "pflege", "angebot", "kontakt", "sprechstunde", "hilfe", "senioren", "familie"]

        for a_tag in soup.find_all("a", href=True):
            raw_href = a_tag["href"].strip()
            if raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            abs_url = urljoin(url, raw_href)
            a_tag["href"] = abs_url
            link_text = a_tag.get_text(" ", strip=True)
            if link_text:
                a_tag.string = f"{link_text} ({abs_url})"

            # Prüfe für § 8 C a.1, ob der Link im relevanten Quartiers-Kontext steht
            if any(kw in abs_url.lower() or kw in link_text.lower() for kw in context_keywords):
                if abs_url != url and abs_url not in context_links:
                    context_links.append(abs_url)

        modified_html = str(soup)

        # § 8 A & B: Boilerplate-Removal (Navbars, Footer, Cookie-Banner) via Trafilatura
        extracted_md = trafilatura.extract(
            modified_html,
            include_links=True,
            include_tables=True,
            output_format="txt",
        )

        if not extracted_md or len(extracted_md.strip()) < 50:
            # Fallback auf bereinigten BeautifulSoup-Text
            for noisy in soup(["script", "style", "nav", "footer", "header", "aside"]):
                noisy.decompose()
            extracted_md = soup.get_text(separator="\n\n", strip=True)

        return extracted_md.strip(), title, context_links