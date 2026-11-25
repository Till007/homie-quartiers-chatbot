"""
server_runtime/app.py
Schlanker Production-Voice-Server für Laptop (im Hotspot) oder günstigen vServer.
Entkoppelt von Quality-Gate, ETL und Gradio-UI.

Funktionen:
- Groq Whisper STT (whisper-large-v3-turbo)
- Deterministischer 3-Stufen Safety- & Krisen-Interceptor (FA-06 / SA-06)
- Offline-Hybrid-Suche (Wort-BM25 + Zeichen-4-Gramm-BM25) auf server_runtime/knowledge_bundle.json
- Gestuftes Antwort-Routing:
    * Starker Quellenbezug (striktes Grounding) für alle Pflege-, Beratungs- & Hildesheim-Angebote
    * Natürliche Antworten auf Smalltalk & harmloses Allgemeinwissen (z. B. "Wie geht es dir?", Bundesländer)
    * Kontrollierter AWO-Fallback, wenn eine lokale Beratungsfrage nicht in den Dokumenten steht
- Hybrid-TTS: 'acapela' (Navel Lip-Sync für DE/EN) & 'audio' (edge-tts MP3 für TR/AR/UK/RU)
"""

import asyncio
import base64
import html
import json
import math
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import edge_tts
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from langdetect import LangDetectException, detect
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

load_dotenv()

# =====================================================================
# 1. KONFIGURATION (aus .env)
# =====================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo").strip()
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile").strip()
GROQ_FALLBACK_MODELS = [
    GROQ_CHAT_MODEL,
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

ROBOT_API_KEY = os.getenv("ROBOT_API_KEY", "homie-lokal-2026").strip()
BUNDLE_PATH = Path(os.getenv("KNOWLEDGE_BUNDLE_PATH", "server_runtime/knowledge_bundle.json"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.32"))

ENABLE_QR_URL = os.getenv("ENABLE_QR_URL", "false").strip().lower() == "true"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8080").rstrip("/")

NAVEL_ACAPELA_LANGS = {
    lang.strip().lower()
    for lang in os.getenv("NAVEL_ACAPELA_LANGS", "de,en").split(",")
    if lang.strip()
}
ACAPELA_LANG_TAGS = {
    "de": "de1", "en": "en1", "es": "es1", "it": "it1", "nl": "nl1", "pt": "pt1", "sw": "sw1"
}

EDGE_TTS_VOICES = {
    "de": "de-DE-KatjaNeural",
    "en": "en-US-AriaNeural",
    "tr": "tr-TR-EmelNeural",
    "ar": "ar-SA-ZariyahNeural",
    "uk": "uk-UA-PolinaNeural",
    "ru": "ru-RU-SvetlanaNeural",
}

LANGUAGE_NAMES = {
    "de": "Deutsch",
    "en": "Englisch (English)",
    "tr": "Türkisch (Türkçe)",
    "ar": "Arabisch (العربية)",
    "uk": "Ukrainisch (Українська)",
    "ru": "Russisch (Русский)",
}

GERMAN_STOPWORDS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "ein", "eine", "einen", "einem", "einer",
    "der", "die", "das", "den", "dem", "des", "und", "oder", "aber", "mit", "von", "zu",
    "zum", "zur", "in", "im", "am", "an", "auf", "für", "bei", "nach", "aus", "über",
    "ist", "sind", "war", "kann", "können", "habe", "haben", "will", "möchte", "gibt",
    "wo", "wie", "was", "wann", "wer", "hier", "dort", "bitte", "mir", "mich", "uns",
}

ANTI_DISCRIMINATION_MAP = {
    r"\bAltenheim(?:e|en|s)?\b": "Seniorenzentrum",
    r"\bAltersheim(?:e|en|s)?\b": "Seniorenzentrum",
    r"\bGreis(?:e|en|in)?\b": "älterer Mensch",
    r"\bBehinderte\b": "Menschen mit Behinderung",
    r"\bAusländer\b": "Menschen mit Migrationserfahrung",
}

# =====================================================================
# 2. DETERMINISTISCHER 3-STUFEN SAFETY- & KRISEN-INTERCEPTOR (FA-06/SA-06)
# =====================================================================

SUICIDE_TRIGGERS = [
    r"\bsuizid", r"\bselbstmord", r"\bbringen?\s+mich\s+um\b", r"\bwill\s+nicht\s+mehr\s+leben\b",
    r"\bleben\s+beenden\b", r"\bmich\s+umbringen\b", r"\bselbstverletz",
    r"\bsuicide\b", r"\bkill\s+myself\b", r"\bintihar\b", r"انتحار", r"самогубств", r"суицид",
]
VIOLENCE_TRIGGERS = [
    r"\bvergewaltig", r"\bsexuell(?:e|er|en)?\s+(?:missbrauch|übergriff|gewalt)",
    r"\bhäusliche\s+gewalt\b", r"\bschlägt\s+mich\b", r"\bfrauenhaus\b",
    r"\bdomestic\s+violence\b", r"\btecavüz\b", r"عنف منزلي", r"насильств",
]
DRUG_TRIGGERS = [
    r"\büberdosis\b", r"\bdrogennotfall\b", r"\bdrogensucht\b", r"\bsuchtberatung\b",
    r"\bdrogenberatung\b", r"\balkoholsucht\b",
]
NSFW_TRIGGERS = [
    r"\bporno\b", r"\bpornografie\b", r"\bsexchat\b", r"\bfick(?:en|dich)\b",
]
SENSITIVE_TOPICS = [
    "schulden", "insolvenz", "sucht", "alkohol", "drogen", "gewalt",
    "frauenhaus", "depression", "suizid", "notfall", "obdachlos",
]


def check_safety_guard(text: str, lang: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    low = text.lower()

    if any(re.search(p, low, re.IGNORECASE) for p in SUICIDE_TRIGGERS):
        spoken = {
            "de": "Es tut mir sehr leid, dass es Ihnen gerade so schlecht geht. Sie sind damit nicht allein. Bitte rufen Sie sofort kostenfrei und anonym die TelefonSeelsorge unter 0800 111 0 111 oder den Notruf 112 an.",
            "en": "I am very sorry you are going through such a difficult time. Please reach out for free, confidential help right now by calling 0800 111 0 111 or emergency number 112.",
            "tr": "Böyle zor bir durumda olmanıza çok üzüldüm. Yalnız değilsiniz. Lütfen hemen ücretsiz ve gizli yardım hattı 0800 111 0 111 veya acil 112 numarasını arayın.",
            "ar": "يؤسفني جداً أنك تمر بهذا الظرف الصعب. أنت لست وحدك. يرجى الاتصال فوراً بخط المساعدة المجاني 0800 111 0 111 أو الطوارئ 112.",
            "uk": "Мені дуже шкода, що вам зараз так важко. Будь ласка, зверніться по безкоштовну допомогу за номером 0800 111 0 111 або 112.",
            "ru": "Мне очень жаль, что вам сейчас так тяжело. Пожалуйста, обратитесь за бесплатной помощью по номеру 0800 111 0 111 или 112.",
        }.get(lang, "Es tut mir sehr leid, dass es Ihnen so schlecht geht. Bitte rufen Sie sofort die TelefonSeelsorge unter 0800 111 0 111 oder 112 an.")
        return (
            {
                "answer_text": spoken,
                "emotion": "sad",
                "is_sensitive_topic": True,
                "card": {
                    "title": "Akute Krisenhilfe Hildesheim (24/7)",
                    "phone": "0800 111 0 111 (TelefonSeelsorge) | 05121 309-7371 (Sozialpsych. Dienst) | 112",
                    "address": "Bischof-Janssen-Str. 31, 31134 Hildesheim",
                    "note": "Kostenlos, anonym und rund um die Uhr erreichbar.",
                },
            },
            None,
        )

    if any(re.search(p, low, re.IGNORECASE) for p in VIOLENCE_TRIGGERS):
        spoken = {
            "de": "Wenn Sie Gewalt oder Bedrohung erleben, gibt es sofort vertraulichen Schutz. Das Hilfetelefon erreichen Sie rund um die Uhr unter 116 016, das Frauenhaus Hildesheim unter 05121 15454 und bei akuter Gefahr die Polizei unter 110.",
            "en": "If you are experiencing violence or threats, confidential help is available 24/7 at 116 016, the Hildesheim women's shelter at 05121 15454, or police at 110.",
            "tr": "Şiddet veya tehdit yaşıyorsanız 7/24 gizli yardım hattı 116 016, Hildesheim kadın sığınma evi 05121 15454 veya polis 110 üzerinden hemen destek alabilirsiniz.",
            "ar": "إذا كنت تتعرض للعنف أو التهديد، يمكنك الحصول على مساعدة سرية على مدار الساعة عبر الرقم 116 016 أو الشرطة 110.",
        }.get(lang, "Vertrauliche Hilfe bei Gewalt erhalten Sie rund um die Uhr unter 116 016 oder bei der Polizei unter 110.")
        return (
            {
                "answer_text": spoken,
                "emotion": "sad",
                "is_sensitive_topic": True,
                "card": {
                    "title": "Schutz & Hilfe bei Gewalt (Hildesheim)",
                    "phone": "116 016 (Hilfetelefon 24/7) | 05121 15454 (Frauenhaus) | 110",
                    "address": "BISS Hildesheim: 05121 6971418",
                    "note": "Vertraulich, kostenfrei und in 18 Sprachen.",
                },
            },
            None,
        )

    if any(re.search(p, low, re.IGNORECASE) for p in NSFW_TRIGGERS):
        return (
            {
                "answer_text": (
                    "Ich bin ein Wegweiser für Pflege- und Beratungsangebote in Hildesheim "
                    "und beantworte keine unangemessenen Anfragen. Bei Fragen zu Hilfsangeboten bin ich gerne für Sie da."
                ),
                "emotion": "neutral",
                "is_sensitive_topic": True,
                "card": None,
            },
            None,
        )

    if any(re.search(p, low, re.IGNORECASE) for p in DRUG_TRIGGERS):
        return (
            None,
            "Bei Fragen zu Sucht oder Drogen hilft Ihnen die DROBS Hildesheim vertraulich unter 05121 696050 weiter.",
        )

    return None, None


# =====================================================================
# 3. OFFLINE-HYBRID-SUCHE (Wort-BM25 + Zeichen-4-Gramm-BM25)
# =====================================================================

class OfflineHybridStore:
    def __init__(self, bundle_path: Path):
        self.chunks: List[Dict[str, Any]] = []
        self.word_bm25: Optional[BM25Okapi] = None
        self.ngram_bm25: Optional[BM25Okapi] = None
        self.load(bundle_path)

    def _word_tokens(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\säöüßÄÖÜçğıöşüÇĞİÖŞÜ]", " ", text.lower())
        return [t for t in cleaned.split() if len(t) > 1 and t not in GERMAN_STOPWORDS]

    def _char_ngrams(self, text: str, n: int = 4) -> List[str]:
        words = self._word_tokens(text)
        grams: List[str] = []
        for w in words:
            if len(w) <= n:
                grams.append(w)
            else:
                for i in range(len(w) - n + 1):
                    grams.append(w[i : i + n])
        return grams

    def load(self, bundle_path: Path) -> None:
        if not bundle_path.exists():
            print(f"[WARNUNG] Kein Wissenspaket unter {bundle_path} gefunden! Starte leer.", flush=True)
            return
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        self.chunks = data.get("chunks", [])
        if not self.chunks:
            return

        word_corpus = [self._word_tokens(c.get("text_for_indexing", "")) for c in self.chunks]
        ngram_corpus = [self._char_ngrams(c.get("text_for_indexing", "")) for c in self.chunks]
        self.word_bm25 = BM25Okapi(word_corpus)
        self.ngram_bm25 = BM25Okapi(ngram_corpus)
        print(f"[OK] Offline-Wissenspaket geladen: {len(self.chunks)} Chunks (Wort + N-Gramm BM25).", flush=True)

    def search(self, german_query: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], float]:
        if not self.chunks or self.word_bm25 is None or self.ngram_bm25 is None:
            return [], 0.0

        q_words = self._word_tokens(german_query)
        q_ngrams = self._char_ngrams(german_query)
        if not q_words and not q_ngrams:
            return [], 0.0

        w_scores = self.word_bm25.get_scores(q_words) if q_words else [0.0] * len(self.chunks)
        n_scores = self.ngram_bm25.get_scores(q_ngrams) if q_ngrams else [0.0] * len(self.chunks)

        top_w = sorted(range(len(self.chunks)), key=lambda i: float(w_scores[i]), reverse=True)[:12]
        top_n = sorted(range(len(self.chunks)), key=lambda i: float(n_scores[i]), reverse=True)[:12]

        rrf: Dict[int, float] = {}
        for rank, idx in enumerate(top_w, start=1):
            if float(w_scores[idx]) > 0:
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(top_n, start=1):
            if float(n_scores[idx]) > 0:
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)

        if not rrf:
            return [], 0.0

        candidates = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[: max(top_k * 2, 6)]
        scored: List[Dict[str, Any]] = []

        for idx in candidates:
            ch = self.chunks[idx]
            meta = ch["metadata"]
            w_raw = max(0.0, float(w_scores[idx]))
            n_raw = max(0.0, float(n_scores[idx]))

            w_norm = 1.0 - math.exp(-w_raw / 4.5)
            n_norm = 1.0 - math.exp(-n_raw / 22.0)
            base_score = (0.65 * w_norm) + (0.35 * n_norm)

            boost = 0.03 if meta.get("status") == "official" else 0.0
            if meta.get("language_level") in ("easy", "simple"):
                boost += 0.02

            final_score = min(round(base_score + boost, 4), 1.0)
            scored.append({"chunk": ch, "score": final_score})

        scored.sort(key=lambda r: r["score"], reverse=True)
        top_hits = scored[:top_k]
        best_score = top_hits[0]["score"] if top_hits else 0.0
        return top_hits, best_score


# =====================================================================
# 4. GROQ API HELPER (STT & CHAT MIT AUTOMATISCHEM MODEL-FALLBACK)
# =====================================================================

async def _groq_post(
    client: httpx.AsyncClient, path: str, **kwargs: Any
) -> httpx.Response:
    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_hier_deinen_key_einfuegen":
        raise HTTPException(status_code=500, detail="GROQ_API_KEY ist in .env nicht korrekt gesetzt.")

    url = f"https://api.groq.com/openai/v1/{path.lstrip('/')}"
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {GROQ_API_KEY}"

    for attempt in range(3):
        resp = await client.post(url, headers=headers, **kwargs)
        if resp.status_code == 429:
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        return resp
    raise HTTPException(status_code=429, detail="Groq API Rate-Limit erreicht. Bitte kurz warten.")


async def groq_transcribe(audio_bytes: bytes, filename: str) -> Tuple[str, str]:
    async with httpx.AsyncClient(timeout=35.0) as client:
        resp = await _groq_post(
            client,
            "audio/transcriptions",
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={"model": GROQ_STT_MODEL, "response_format": "verbose_json"},
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Groq STT Fehler ({resp.status_code}): {resp.text[:250]}")
        data = resp.json()
        text = (data.get("text") or "").strip()
        raw_lang = (data.get("language") or "").strip().lower()
        lang = _normalize_lang(raw_lang, text)
        return text, lang


async def groq_chat(
    system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 550, temperature: float = 0.15
) -> str:
    payload_msgs = [{"role": "system", "content": system_prompt}] + messages
    tried = set()
    last_error = ""

    async with httpx.AsyncClient(timeout=40.0) as client:
        for model_name in GROQ_FALLBACK_MODELS:
            if not model_name or model_name in tried:
                continue
            tried.add(model_name)
            resp = await _groq_post(
                client,
                "chat/completions",
                json={
                    "model": model_name,
                    "messages": payload_msgs,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"] or ""
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content
            last_error = f"{model_name} -> HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code in (400, 404):
                continue
            break

    raise HTTPException(status_code=502, detail=f"Groq Chat Fehler: {last_error}")


async def translate_query_to_german(text: str, lang: str) -> str:
    if lang == "de":
        return text
    sys_p = (
        "Übersetze die folgende gesprochene Frage in eine kurze, präzise deutsche Suchanfrage. "
        "Gib AUSSCHLIESSLICH die deutsche Übersetzung aus!"
    )
    try:
        translated = await groq_chat(
            sys_p, [{"role": "user", "content": text}], max_tokens=80, temperature=0.0
        )
        return translated.splitlines()[0].strip().strip('"').strip("'") or text
    except Exception:
        return text


def _normalize_lang(raw_lang: str, text: str) -> str:
    mapping = {
        "de": "de", "german": "de", "deu": "de",
        "en": "en", "english": "en", "eng": "en",
        "tr": "tr", "turkish": "tr", "tur": "tr",
        "ar": "ar", "arabic": "ar", "ara": "ar",
        "uk": "uk", "ukrainian": "uk", "ukr": "uk",
        "ru": "ru", "russian": "ru", "rus": "ru",
    }
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(r"[\u0400-\u04FF]", text):
        return "uk" if re.search(r"[іїєґІЇЄҐ]", text) else "ru"
    if raw_lang in mapping:
        return mapping[raw_lang]
    if len(text.split()) >= 3:
        try:
            d = str(detect(text))
            if d in mapping:
                return mapping[d]
        except LangDetectException:
            pass
    return "de"


def _apply_post_filters(text: str, lang: str, allowed_sources_text: str = "") -> str:
    """
    1. Bereinigt Markdown/Sternchen/URLs für natürliche Sprachausgabe.
    2. Wendet Antidiskriminierungs-Ersetzungen im Deutschen an.
    3. Halluzinationsschutz (SA-04): Entfernt lokale Telefonnummern (05121...), die NICHT in den Quellen vorkommen.
    """
    cleaned = text.replace("·", "")
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"[*_#`~\[\]()<>|]", " ", cleaned)

    if lang == "de":
        for pattern, replacement in ANTI_DISCRIMINATION_MAP.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Deterministische Telefonnummern-Prüfung gegen die übergebenen Quellen
    source_digits = re.sub(r"\D", "", allowed_sources_text)
    for phone_cand in re.findall(r"\b0\d{3,4}[\s/\-]?\d{3,8}\b", cleaned):
        digits = re.sub(r"\D", "", phone_cand)
        # Bekannte Notruf-/Hotline-Nummern aus dem Safety-Guard ausnehmen
        if digits in ("08001110111", "08001110222", "0512115454", "05121696050", "051213097371", "051216971418"):
            continue
        if len(digits) >= 6 and digits not in source_digits:
            cleaned = cleaned.replace(phone_cand, "die Nummer der AWO-Beratungsstelle")

    return re.sub(r"\s+", " ", cleaned).strip()


async def generate_edge_tts_base64(text: str, lang: str, speed_pct: int = -12) -> Optional[str]:
    if not text:
        return None
    voice = EDGE_TTS_VOICES.get(lang, "de-DE-KatjaNeural")
    try:
        comm = edge_tts.Communicate(text=text, voice=voice, rate=f"{speed_pct:+d}%")
        audio_chunks: List[bytes] = []
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                audio_data = chunk.get("data")
                if isinstance(audio_data, bytes):
                    audio_chunks.append(audio_data)
        if audio_chunks:
            return base64.b64encode(b"".join(audio_chunks)).decode("ascii")
    except Exception as e:
        print(f"[WARNUNG] edge-tts fehlgeschlagen: {e}", flush=True)
    return None


# =====================================================================
# 5. RAM-SESSION-GEDÄCHTNIS (60s TTL) & OPTIONALER QR-KARTEN-SPEICHER
# =====================================================================

_SESSIONS: Dict[str, Tuple[float, List[Dict[str, str]]]] = {}
_CONTACT_CARDS: Dict[str, Tuple[float, Dict[str, str]]] = {}


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    now = time.time()
    entry = _SESSIONS.get(session_id)
    if not entry or (now - entry[0]) > 60.0:
        _SESSIONS.pop(session_id, None)
        return []
    return list(entry[1])


def append_session_turn(session_id: str, user_text: str, bot_text: str) -> None:
    hist = get_session_history(session_id)
    hist.append({"role": "user", "content": user_text})
    hist.append({"role": "assistant", "content": bot_text})
    _SESSIONS[session_id] = (time.time(), hist[-4:])


def register_short_qr_card(card: Optional[Dict[str, str]]) -> Optional[str]:
    if not ENABLE_QR_URL or not card:
        return None
    code = secrets.token_hex(2).upper()
    _CONTACT_CARDS[code] = (time.time(), card)
    now = time.time()
    for k in [k for k, (ts, _) in _CONTACT_CARDS.items() if now - ts > 900]:
        _CONTACT_CARDS.pop(k, None)
    return f"{PUBLIC_BASE_URL.upper()}/C/{code}"


# =====================================================================
# 6. FASTAPI APP & ENDPUNKTE
# =====================================================================

app = FastAPI(
    title="HOMIE Robot Runtime (Groq + Offline Hybrid-RAG)",
    version="2.1.0",
)
store = OfflineHybridStore(BUNDLE_PATH)


def verify_api_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != ROBOT_API_KEY:
        raise HTTPException(status_code=401, detail="Ungültiger oder fehlender X-API-Key.")


class TextTestRequest(BaseModel):
    text: str
    session_id: str = "test-laptop-01"
    language_override: Optional[str] = None


class RobotConverseResponse(BaseModel):
    transcript: str
    search_query_de: str = ""
    language: str
    response_mode: str = "rag_sources"  # "rag_sources", "general_or_fallback", "safety_guard", "empty"
    best_score: float = 0.0
    answer_text: str
    tts_mode: str  # "acapela" (Navel Lip-Sync) oder "audio" (edge-tts MP3 Base64)
    acapela_text: Optional[str] = None
    audio_base64: Optional[str] = None
    emotion: str = "happy"
    is_sensitive_topic: bool = False
    short_qr_url: Optional[str] = None
    latency_ms: float = 0.0


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "chunks_loaded": len(store.chunks),
        "groq_key_set": bool(GROQ_API_KEY and GROQ_API_KEY != "gsk_hier_deinen_key_einfuegen"),
        "stt_model": GROQ_STT_MODEL,
        "chat_model": GROQ_CHAT_MODEL,
        "acapela_langs": sorted(NAVEL_ACAPELA_LANGS),
    }


@app.get("/c/{code}", response_class=HTMLResponse)
@app.get("/C/{code}", response_class=HTMLResponse)
async def show_contact_card(code: str):
    entry = _CONTACT_CARDS.get(code.upper())
    if not entry:
        return HTMLResponse("<h2>Kontaktkarte abgelaufen.</h2>", status_code=404)
    c = entry[1]
    return HTMLResponse(
        f"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>{html.escape(c.get('title', 'HOMIE Kontakt'))}</title>
        <style>body{{font-family:system-ui,sans-serif;max-width:480px;margin:2rem auto;padding:1.2rem;line-height:1.6}}
        .card{{border:2px solid #2563eb;border-radius:12px;padding:1.2rem;background:#f8fafc}}</style></head>
        <body><div class="card"><h2>{html.escape(c.get('title', 'Beratungsstelle Hildesheim'))}</h2>
        <p><strong>Telefon:</strong> {html.escape(c.get('phone') or 'siehe Einrichtung')}</p>
        <p><strong>Adresse:</strong> {html.escape(c.get('address') or 'Hildesheim')}</p>
        <p>{html.escape(c.get('note') or '')}</p></div></body></html>"""
    )


async def _process_dialogue_turn(
    transcript: str, lang: str, session_id: str, t0: float
) -> RobotConverseResponse:
    print(f"\n🎙️ [SERVER-EINGANG | {lang.upper()}] Gesagt: \"{transcript}\"", flush=True)

    if not transcript.strip():
        return RobotConverseResponse(
            transcript="",
            language=lang,
            response_mode="empty",
            answer_text="",
            tts_mode="acapela",
            emotion="neutral",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # 1. Deterministischer Safety- & Krisen-Check VOR der Suche
    crisis_hit, addiction_note = check_safety_guard(transcript, lang)
    if crisis_hit is not None:
        _SESSIONS.pop(session_id, None)
        qr_url = register_short_qr_card(crisis_hit.get("card"))
        print("🛡️ [SAFETY-GUARD] Krisen-/Schutz-Intervention ausgelöst.", flush=True)
        return await _build_tts_response(
            transcript=transcript,
            search_query_de="[SAFETY-GUARD]",
            lang=lang,
            response_mode="safety_guard",
            best_score=1.0,
            answer_text=crisis_hit["answer_text"],
            emotion=crisis_hit["emotion"],
            is_sensitive=crisis_hit["is_sensitive_topic"],
            qr_url=qr_url,
            t0=t0,
        )

    # 2. Cross-Lingual Query-Brücke (nur bei Nicht-Deutsch)
    search_query_de = await translate_query_to_german(transcript, lang)
    is_sensitive = any(kw in search_query_de.lower() for kw in SENSITIVE_TOPICS)

    # 3. Offline-Hybrid-Retrieval (Wort-BM25 + Zeichen-4-Gramm-BM25)
    hits, best_score = store.search(search_query_de, top_k=3)
    has_strong_sources = bool(hits and best_score >= SCORE_THRESHOLD)
    response_mode = "rag_sources" if has_strong_sources else "general_or_fallback"

    lang_name = LANGUAGE_NAMES.get(lang, "Deutsch")
    history = get_session_history(session_id)
    allowed_sources_text = ""

    if has_strong_sources:
        context_blocks = []
        for i, h in enumerate(hits, start=1):
            ch = h["chunk"]
            m = ch["metadata"]
            block = (
                f"[QUELLE {i}: {m.get('title')} | Einrichtung: {m.get('facility_name') or m.get('publisher')} | "
                f"Adresse: {m.get('address') or 'siehe Text'} | Tel: {m.get('phone') or 'siehe Text'} | "
                f"Zeiten: {m.get('opening_hours') or 'siehe Text'}]\n{ch.get('text_raw')}"
            )
            context_blocks.append(block)
        allowed_sources_text = "\n\n".join(context_blocks)

        system_prompt = (
            "Du bist HOMIE, ein herzlicher, wertschätzender Sprach-Roboter für Pflege- und Versorgungsberatung in Hildesheim (AWO).\n"
            "ANTWORT-HIERARCHIE:\n"
            "1. STARKER QUELLENBEZUG (HÖCHSTE PRIORITÄT): Für alle Fragen zu Pflege, Versorgung, Beratungsstellen, "
            "Angeboten, Adressen, Telefonnummern, Öffnungszeiten oder Terminen in Hildesheim nutze AUSSCHLIESSLICH "
            "die untenstehenden QUELLEN. Erfinde NIEMALS Telefonnummern, Adressen, Uhrzeiten oder lokale Angebote! "
            "Wenn eine konkrete lokale Beratungs- oder Kontaktfrage nicht in den Quellen steht, sage ehrlich, dass du "
            "dazu keine gesicherte Information in deinen Unterlagen hast, und verweise freundlich an die AWO-Beratungsstelle in Hildesheim.\n"
            "2. SMALLTALK & ALLGEMEINWISSEN: Wenn die Person dich nur begrüßt, fragt wie es dir geht, wer du bist oder "
            "eine harmlose allgemeine Wissensfrage stellt (z. B. Geografie, wo Hildesheim in Deutschland liegt, wie viele "
            "Bundesländer Deutschland hat), darfst du freundlich, kurz und sachlich richtig antworten.\n"
            "3. GESPROCHENE SPRACHE & TONALITÄT: Maximal 3 bis 4 kurze, leicht verständliche Sätze. Keine Aufzählungen, "
            "Sternchen oder URLs. Kommuniziere empathisch und auf Augenhöhe (Carl Rogers).\n"
            f"4. SPRACHE: Antworte vollständig und natürlich in der Sprache: {lang_name}."
        )
        user_turn = (
            f"GEFUNDENE QUELLEN AUS DER HILDESHEIM-DATENBANK:\n{allowed_sources_text}\n\n"
            f"GESPROCHENE FRAGE: {transcript}\n(Deutsche Bedeutung: {search_query_de})"
        )
    else:
        # Keine passenden Hildesheim-Dokumente gefunden -> Smalltalk/Allgemeinwissen erlauben, lokale Fakten aber strikt blocken!
        system_prompt = (
            "Du bist HOMIE, ein herzlicher, wertschätzender Sprach-Roboter der AWO für Pflege- und Versorgungsberatung in Hildesheim.\n"
            "Zu dieser Frage wurden KEINE passenden Dokumente in deiner Hildesheimer Datenbank gefunden. Wende streng folgende Regel an:\n"
            "1. SMALLTALK & HARMLOSES ALLGEMEINWISSEN: Wenn es sich um eine Begrüßung, Smalltalk (z. B. 'Wie geht es dir?', 'Wer bist du?') "
            "oder eine harmlose allgemeine Wissensfrage handelt (z. B. 'Wo liegt Hildesheim in Deutschland?', 'Wie viele Bundesländer hat Deutschland?'), "
            "antworte freundlich, kurz (maximal 2 bis 3 einfache Sätze) und korrekt.\n"
            "2. LOKALE BERATUNG / PFLEGE / KONTAKTE (STRIKTER SCHUTZ): Wenn nach konkreten Pflegeangeboten, Beratungsstellen, "
            "Telefonnummern, Adressen, Öffnungszeiten, Veranstaltungen, Buslinien oder medizinischer/rechtlicher Hilfe gefragt wird, "
            "ERFINDE NICHTS! Sage freundlich, dass du dazu leider keine gesicherte Information in deinen Unterlagen hast, "
            "und verweise direkt an die persönliche AWO-Beratungsstelle in Hildesheim.\n"
            "3. FORMAT: Gut vorlesbare, leicht verständliche Sprache ohne Aufzählungen, Sternchen oder Links.\n"
            f"4. SPRACHE: Antworte vollständig und natürlich in der Sprache: {lang_name}."
        )
        user_turn = f"GESPROCHENE FRAGE: {transcript}\n(Deutsche Bedeutung: {search_query_de})"

    raw_answer = await groq_chat(system_prompt, history + [{"role": "user", "content": user_turn}])
    answer = _apply_post_filters(raw_answer, lang, allowed_sources_text=allowed_sources_text)
    if addiction_note and lang == "de":
        answer = f"{answer} {addiction_note}"

    append_session_turn(session_id, transcript, answer)

    qr_url = None
    if has_strong_sources:
        top_meta = hits[0]["chunk"]["metadata"]
        if top_meta.get("phone") or top_meta.get("address"):
            qr_url = register_short_qr_card(
                {
                    "title": str(top_meta.get("facility_name") or top_meta.get("title") or "Beratungsstelle"),
                    "phone": str(top_meta.get("phone") or ""),
                    "address": str(top_meta.get("address") or ""),
                    "note": f"Sprechzeiten: {top_meta.get('opening_hours') or 'siehe Quelle'}",
                }
            )

    print(
        f"🤖 [SERVER-AUSGANG | Modus={response_mode} | Score={best_score:.2f}] Antwort: \"{answer}\"",
        flush=True,
    )

    return await _build_tts_response(
        transcript=transcript,
        search_query_de=search_query_de,
        lang=lang,
        response_mode=response_mode,
        best_score=best_score,
        answer_text=answer,
        emotion="neutral" if is_sensitive else "happy",
        is_sensitive=is_sensitive,
        qr_url=qr_url,
        t0=t0,
    )


async def _build_tts_response(
    transcript: str,
    search_query_de: str,
    lang: str,
    response_mode: str,
    best_score: float,
    answer_text: str,
    emotion: str,
    is_sensitive: bool,
    qr_url: Optional[str],
    t0: float,
) -> RobotConverseResponse:
    if lang in NAVEL_ACAPELA_LANGS and lang in ACAPELA_LANG_TAGS:
        tag = ACAPELA_LANG_TAGS[lang]
        return RobotConverseResponse(
            transcript=transcript,
            search_query_de=search_query_de,
            language=lang,
            response_mode=response_mode,
            best_score=best_score,
            answer_text=answer_text,
            tts_mode="acapela",
            acapela_text=f"<lang,{tag}><rspd,85>{answer_text}",
            audio_base64=None,
            emotion=emotion,
            is_sensitive_topic=is_sensitive,
            short_qr_url=qr_url,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    audio_b64 = await generate_edge_tts_base64(answer_text, lang)
    return RobotConverseResponse(
        transcript=transcript,
        search_query_de=search_query_de,
        language=lang,
        response_mode=response_mode,
        best_score=best_score,
        answer_text=answer_text,
        tts_mode="audio",
        acapela_text=None,
        audio_base64=audio_b64,
        emotion=emotion,
        is_sensitive_topic=is_sensitive,
        short_qr_url=qr_url,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post(
    "/api/v1/robot/converse",
    response_model=RobotConverseResponse,
    dependencies=[Depends(verify_api_key)],
)
async def robot_converse(
    audio_file: UploadFile = File(..., description="Aufgenommene WAV-Datei vom Navel-Mikrofon (odas_1)"),
    session_id: str = Form(default="navel-hildesheim-01"),
    language_override: Optional[str] = Form(default=None),
) -> RobotConverseResponse:
    t0 = time.perf_counter()
    audio_bytes = await audio_file.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audiodatei zu kurz oder leer.")

    transcript, detected_lang = await groq_transcribe(
        audio_bytes, audio_file.filename or "speech.wav"
    )
    lang = language_override.strip().lower() if language_override else detected_lang
    return await _process_dialogue_turn(transcript, lang, session_id, t0)


@app.post(
    "/api/v1/robot/converse-text",
    response_model=RobotConverseResponse,
    dependencies=[Depends(verify_api_key)],
)
async def robot_converse_text(req: TextTestRequest) -> RobotConverseResponse:
    t0 = time.perf_counter()
    lang = req.language_override.strip().lower() if req.language_override else _normalize_lang("", req.text)
    return await _process_dialogue_turn(req.text, lang, req.session_id, t0)