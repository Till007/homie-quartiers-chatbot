"""
services/ui_debug_client/app.py
Interaktive Gradio-Benutzeroberfläche für Debugging, Governance (Quality-Gate/ETL)
und Kiosk-/Roboter-Simulation (Port 7860).
Neu:
- Aufgeräumte Oberfläche: Keine langen Erklärungstexte mehr unter den Bedienelementen!
- Klickbares 'ℹ️'-Symbol bei jeder Funktion, das ein schwebendes Popup-Fenster darüber öffnet.
- Unterstützt mehrsprachige Audio-Ausgabe (.wav & .mp3) für Deutsch, Arabisch, Türkisch, Ukrainisch, Russisch & Englisch.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx
import qrcode
from PIL import Image
from qrcode.image.pil import PilImage

from shared.logger import get_service_logger

logger = get_service_logger("ui-debug-client")

QUALITY_URL = "http://127.0.0.1:8001"
ETL_URL = "http://127.0.0.1:8002"
RAG_URL = "http://127.0.0.1:8003"
ORCHESTRATOR_URL = "http://127.0.0.1:8004"
SPEECH_URL = "http://127.0.0.1:8005"

# CSS für SA-05 Großschrift + Klickbares ℹ️-Popup-Fenster (schwebt beim Klick über dem Element)
CUSTOM_CSS = """
.kiosk-chat p, .kiosk-chat li {
    font-size: 20px !important;
    line-height: 1.55 !important;
}
.status-badge {
    font-weight: bold;
    padding: 8px 12px;
    border-radius: 6px;
    background-color: #1e293b;
    color: #f8fafc;
}
.info-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: -6px;
    padding: 2px 4px;
}
.info-bar-title {
    font-weight: 600;
    font-size: 14px;
}
details.info-pop {
    position: relative;
    display: inline-block;
}
details.info-pop summary {
    list-style: none;
    cursor: pointer;
    user-select: none;
    font-size: 15px;
    background: #2563eb;
    color: #ffffff;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,0.25);
}
details.info-pop summary::-webkit-details-marker {
    display: none;
}
details.info-pop .info-pop-box {
    position: absolute;
    bottom: 30px;
    right: 0;
    width: 320px;
    background: #0f172a;
    color: #f8fafc;
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 13px;
    line-height: 1.45;
    z-index: 99999;
    box-shadow: 0 12px 28px rgba(0,0,0,0.55);
}
"""


def info_label(title: str, popup_html: str) -> gr.HTML:
    """
    Erzeugt eine kompakte Titelzeile mit einem anklickbaren 'i'-Symbol.
    Beim Klick auf 'i' öffnet sich ein kleines schwebendes Fenster darüber.
    """
    return gr.HTML(
        f"""
        <div class="info-bar">
            <span class="info-bar-title">{title}</span>
            <details class="info-pop">
                <summary title="Klicken für Erklärung">i</summary>
                <div class="info-pop-box">
                    <strong>ℹ️ {title}</strong><br><br>
                    {popup_html}
                </div>
            </details>
        </div>
        """
    )


def generate_qr_image(vcard_payload: Optional[str]) -> Optional[Image.Image]:
    if not vcard_payload:
        return None
    qr = qrcode.QRCode(version=1, box_size=6, border=2, image_factory=PilImage)
    qr.add_data(vcard_payload)
    qr.make(fit=True)
    pil_wrapper = qr.make_image(fill_color="black", back_color="white")
    return pil_wrapper.get_image().convert("RGB")


def _append_to_chat_history(
    chat_history: Optional[List[Any]], user_msg: str, bot_msg: str
) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for item in (chat_history or []):
        if isinstance(item, dict) and "role" in item and "content" in item:
            history.append({"role": str(item["role"]), "content": str(item["content"])})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            history.append({"role": "user", "content": str(item[0])})
            history.append({"role": "assistant", "content": str(item[1])})

    history.append({"role": "user", "content": str(user_msg)})
    history.append({"role": "assistant", "content": str(bot_msg)})
    return history


def _create_kiosk_chatbot() -> gr.Chatbot:
    base_kwargs: Dict[str, Any] = {
        "label": "Quartiers-Dialog (Großschrift 20pt nach SA-05 Zwei-Sinne-Prinzip)",
        "height": 430,
        "elem_classes": ["kiosk-chat"],
    }
    try:
        with_type: Dict[str, Any] = {**base_kwargs, "type": "messages"}
        return gr.Chatbot(**with_type)
    except TypeError:
        return gr.Chatbot(**base_kwargs)


# =====================================================================
# TAB 1 LOGIK: CHAT, AUDIO (STT/TTS) & PARADIGMEN-VERGLEICH
# =====================================================================

def transcribe_mic_input(audio_filepath: Optional[str]) -> Tuple[str, str, str]:
    if not audio_filepath:
        return "", "de", "🟢 ROBO-STATE: IDLE (Kein Audio aufgenommen)"
    try:
        with open(audio_filepath, "rb") as f:
            files = {"audio_file": (Path(audio_filepath).name, f, "audio/wav")}
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{SPEECH_URL}/api/v1/stt/transcribe", files=files)
                resp.raise_for_status()
                data = resp.json()

        detected_lang = data.get("detected_language", "de")
        supported = ["de", "en", "tr", "ar", "uk", "ru"]
        lang_out = detected_lang if detected_lang in supported else "de"
        status = (
            f"🎤 STT erfolgreich ({data.get('duration_ms')} ms) | "
            f"Erkannte Sprache: {detected_lang.upper()} ({data.get('language_probability', 0):.0%})"
        )
        return data.get("text", ""), lang_out, status
    except Exception as e:
        return "", "de", f"🔴 STT-Fehler (Port 8005 prüfen): {e}"


def handle_chat_turn(
    user_message: str,
    chat_history: List[Any],
    session_id: str,
    paradigm: str,
    language_level: str,
    user_language: str,
    translation_variant: str,
    score_threshold: float,
    enable_tts: bool,
    tts_speed: float,
) -> Tuple[List[Any], str, str, str, List[List[Any]], Optional[Image.Image], Optional[str], str]:
    if not user_message or not user_message.strip():
        return (
            chat_history,
            "",
            "Bitte eine Frage eingeben oder einsprechen.",
            "",
            [],
            None,
            None,
            "🟢 ROBO-STATE: IDLE",
        )

    payload = {
        "session_id": session_id,
        "user_message": user_message.strip(),
        "user_language": user_language,
        "language_level": language_level,
        "paradigm": paradigm,
        "translation_variant": translation_variant,
        "score_threshold": float(score_threshold),
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{ORCHESTRATOR_URL}/api/v1/chat/query", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        err_msg = f"Fehler beim Aufruf des Orchestrators (Port 8004): {e}"
        chat_history = _append_to_chat_history(chat_history, user_message, err_msg)
        return chat_history, "", err_msg, "", [], None, None, "🔴 ROBO-STATE: ERROR"

    answer_text = data.get("answer_text", "")
    chat_history = _append_to_chat_history(chat_history, user_message, answer_text)

    trace_lines = [
        f"⏱️ Gesamt-Latenz: {data.get('total_latency_ms')} ms | "
        f"Paradigma: {data.get('paradigm_used')} | "
        f"Quellen-Guard bestanden: {data.get('passed_source_guard')} | "
        f"Sensibles Thema (SA-06): {data.get('is_sensitive_topic')}",
        f"🔍 Deutsche Suchanfrage (Query-Rewriter/Brücke): '{data.get('rewritten_query')}'",
        "-" * 70,
    ]
    for idx, step in enumerate(data.get("agent_trace", []), start=1):
        trace_lines.append(
            f"[{idx}] {step.get('step_name')} ({step.get('duration_ms')} ms):\n    {step.get('details')}"
        )

    tts_audio_path = None
    if enable_tts and answer_text:
        try:
            tts_req = {
                "text": answer_text,
                "language": user_language,
                "speed_Factor": float(tts_speed),
                "is_sensitive_topic": bool(data.get("is_sensitive_topic", False)),
            }
            with httpx.Client(timeout=45.0) as client:
                tts_resp = client.post(f"{SPEECH_URL}/api/v1/tts/synthesize", json=tts_req)
                if tts_resp.status_code == 200 and len(tts_resp.content) > 200:
                    audio_dir = Path("data/audio_temp")
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    ctype = tts_resp.headers.get("content-type", "")
                    ext = ".mp3" if "mpeg" in ctype else ".wav"
                    out_audio = audio_dir / f"ui_response_{int(time.time() * 1000)}{ext}"
                    out_audio.write_bytes(tts_resp.content)
                    tts_audio_path = str(out_audio.resolve())
                    trace_lines.append(f"🔊 Sprachausgabe (TTS) erfolgreich generiert: {out_audio.name}")
                else:
                    trace_lines.append(f"⚠️ TTS-Warnung: HTTP {tts_resp.status_code}")
        except Exception as e:
            logger.warning(f"TTS-Wiedergabe im UI fehlgeschlagen: {e}")
            trace_lines.append(f"⚠️ TTS-Fehler (Port 8005 prüfen): {e}")

    trace_output = "\n".join(trace_lines)
    german_inter = data.get("german_intermediate_text") or (
        "(Kein separater Zwischentext, da V2a direkt gewählt wurde)"
    )

    source_rows = []
    for sc in data.get("sources", []):
        ch = sc.get("chunk", {})
        meta = ch.get("metadata", {})
        source_rows.append(
            [
                ch.get("chunk_id"),
                meta.get("title"),
                meta.get("district"),
                sc.get("dense_rank"),
                sc.get("bm25_rank"),
                sc.get("rrf_score"),
                sc.get("reranker_score"),
                sc.get("final_score"),
                ",".join(meta.get("service_languages", [])),
            ]
        )

    qr_img = generate_qr_image(data.get("qr_code_payload"))
    robot_badge = f"🗣️ ROBO-STATE: {data.get('robot_state', 'speaking').upper()} (Danach Auto-Idle)"
    return (
        chat_history,
        "",
        trace_output,
        german_inter,
        source_rows,
        qr_img,
        tts_audio_path,
        robot_badge,
    )


def reset_kiosk_session(session_id: str) -> Tuple[List, str, str]:
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(f"{ORCHESTRATOR_URL}/api/v1/session/reset", params={"session_id": session_id})
        return [], "Sitzungsgedächtnis im RAM vollständig gelöscht (DSGVO TA-06).", "🟢 ROBO-STATE: IDLE"
    except Exception as e:
        return [], f"Fehler beim Session-Reset: {e}", "🔴 ROBO-STATE: ERROR"


# =====================================================================
# TAB 2 LOGIK: ETL-INGEST & QUALITY-GATE GOVERNANCE (§ 1 - § 11)
# =====================================================================

def sync_etl_to_chromadb() -> str:
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{RAG_URL}/api/v1/rag/sync-from-etl")
            resp.raise_for_status()
            data = resp.json()
        return (
            f"✅ Delta-Synchronisation erfolgreich! "
            f"Neu indexierte Dokumente: {data.get('synced_files')} | "
            f"Übersprungene (bereits vorhandene) Dokumente: {data.get('skipped_unchanged_files', 0)} | "
            f"Gesamtanzahl Chunks in ChromaDB + BM25: {data.get('total_indexed_chunks')}"
        )
    except Exception as e:
        return f"❌ Fehler bei der Index-Synchronisation (Port 8003 prüfen): {e}"


def run_file_ingest_and_audit(
    file_obj: Any, publisher: str, valid_until_str: str, use_llm: bool, auto_sync: bool
) -> Tuple[str, List[List[Any]], str, str, str]:
    if file_obj is None:
        return "Bitte zuerst eine PDF- oder TXT-Datei auswählen.", [], "{}", "", ""

    filepath = Path(file_obj.name)
    data_form: Dict[str, Any] = {
        "use_llm_enrichment": str(use_llm).lower(),
    }
    if publisher and publisher.strip():
        data_form["publisher"] = publisher.strip()
    if valid_until_str and valid_until_str.strip():
        data_form["valid_until"] = valid_until_str.strip()

    try:
        with open(filepath, "rb") as f:
            files = {"file": (filepath.name, f)}
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(f"{ETL_URL}/api/v1/etl/ingest-file", data=data_form, files=files)
                resp.raise_for_status()
                res = resp.json()
        summary, rule_rows, meta_json, chunks_preview = _format_etl_response_for_ui(res)
        sync_msg = ""
        if auto_sync and res.get("success"):
            sync_msg = sync_etl_to_chromadb()
        return summary, rule_rows, meta_json, chunks_preview, sync_msg
    except Exception as e:
        return f"❌ Fehler beim Datei-Ingest: {e}", [], "{}", "", ""


def run_url_scrape_and_audit(
    url: str, publisher: str, valid_until_str: str, follow_depth_1: bool, use_llm: bool, auto_sync: bool
) -> Tuple[str, List[List[Any]], str, str, str]:
    if not url or not url.strip():
        return "Bitte eine gültige URL eingeben.", [], "{}", "", ""

    payload: Dict[str, Any] = {
        "url": url.strip(),
        "publisher": publisher.strip() if (publisher and publisher.strip()) else None,
        "valid_until": valid_until_str.strip() if (valid_until_str and valid_until_str.strip()) else None,
        "follow_links_depth_1": follow_depth_1,
        "use_llm_enrichment": use_llm,
    }
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{ETL_URL}/api/v1/etl/ingest-url", json=payload)
            if resp.status_code != 200:
                return f"❌ Scraper/Regelwerk-Ablehnung ({resp.status_code}): {resp.text}", [], "{}", "", ""
            res = resp.json()
        summary, rule_rows, meta_json, chunks_preview = _format_etl_response_for_ui(res)
        sync_msg = ""
        if auto_sync and res.get("success"):
            sync_msg = sync_etl_to_chromadb()
        return summary, rule_rows, meta_json, chunks_preview, sync_msg
    except Exception as e:
        return f"❌ Fehler beim Web-Scraping: {e}", [], "{}", "", ""


def _format_etl_response_for_ui(res: Dict[str, Any]) -> Tuple[str, List[List[Any]], str, str]:
    audit = res.get("audit_report", {})
    meta = audit.get("extracted_metadata", {})
    approved = audit.get("approved_for_ingestion", False)
    badge = "✅ FREIGEGEBEN DURCH QUALITY-GATE" if approved else "🚫 BLOCKIERT (IN QUARANTÄNE VERSCHOBEN)"
    summary = (
        f"{badge} | Audit-ID: {audit.get('audit_id')}\n"
        f"Erkannter Herausgeber: {meta.get('publisher')} | Gültig bis (TTL): {meta.get('valid_until')}\n"
        f"Status: {res.get('status_message')}"
    )

    rule_rows = []
    for r in audit.get("rule_results", []):
        rule_rows.append(
            [
                r.get("rule_id"),
                r.get("rule_name"),
                r.get("status"),
                r.get("score_or_metric"),
                r.get("auto_fixed"),
                r.get("details"),
            ]
        )

    meta_json = json.dumps(meta, indent=2, ensure_ascii=False)
    chunks_preview = "\n\n====================\n\n".join(
        f"CHUNK {c.get('chunk_id')} (~{c.get('token_estimate')} Tokens):\n{c.get('text_for_indexing')}"
        for c in res.get("chunks", [])
    )
    return summary, rule_rows, meta_json, chunks_preview


def check_all_microservices_health() -> str:
    services = [
        ("Quality-Gate Service (Port 8001)", f"{QUALITY_URL}/health"),
        ("ETL-Ingest Service (Port 8002)", f"{ETL_URL}/health"),
        ("RAG-Retrieval Service (Port 8003)", f"{RAG_URL}/health"),
        ("Orchestrator Service (Port 8004)", f"{ORCHESTRATOR_URL}/health"),
        ("Speech Service STT/TTS (Port 8005)", f"{SPEECH_URL}/health"),
        ("Ollama Local LLM Server (Port 11434)", "http://127.0.0.1:11434/api/tags"),
    ]
    lines = []
    with httpx.Client(timeout=4.0) as client:
        for name, url in services:
            try:
                r = client.get(url)
                if r.status_code == 200:
                    lines.append(f"🟢 {name}: ONLINE ({r.text[:100]})")
                else:
                    lines.append(f"🟡 {name}: WARNUNG (HTTP {r.status_code})")
            except Exception as e:
                lines.append(f"🔴 {name}: OFFLINE ({e})")
    return "\n".join(lines)


# =====================================================================
# GRADIO UI AUFBAU MIT KOMPAKTEN KLICKBAREN 'i'-POPUPS
# =====================================================================

with gr.Blocks(title="HOMIE - Quartiers-Chatbot & Debug-Steuerzentrale", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# 🤖 HOMIE – Quartiers-Chatbot Hildesheim (Development- & Evaluations-Zentrale)\n"
        "**IT-Studienprojekt (IT-MA5) | HULLS Use-Case 1:** Klicke bei jeder Einstellung auf das blaue **`i`**, um eine Erklärung in einem kleinen Popup-Fenster zu öffnen."
    )

    with gr.Tabs():
        # -------------------------------------------------------------
        # TAB 1: KIOSK-SIMULATION & PARADIGMEN-EVALUATION
        # -------------------------------------------------------------
        with gr.TabItem("💬 1. Kiosk-Terminal & Paradigmen-Vergleich (TA-04)"):
            with gr.Row():
                with gr.Column(scale=6):
                    robot_status_box = gr.Markdown("🟢 **ROBO-STATE: IDLE**", elem_classes=["status-badge"])
                    chatbot = _create_kiosk_chatbot()

                    info_label(
                        "Frage eingeben (Alle Quartierssprachen)",
                        "Gib deine Frage auf Deutsch, Arabisch, Türkisch, Ukrainisch, Russisch oder Englisch ein. Das System erkennt fremde Schriftsysteme (z. B. Arabisch) automatisch.",
                    )
                    with gr.Row():
                        user_input = gr.Textbox(
                            show_label=False,
                            placeholder="z.B. 'Wo gibt es Hilfe für Senior:innen?' oder auf Arabisch/Türkisch...",
                            scale=4,
                        )
                        send_btn = gr.Button("Senden 🚀", variant="primary", scale=1)

                    with gr.Accordion("🎙️ Sprach-Ein-/Ausgabe (faster-whisper STT & Mehrsprachiges TTS)", open=True):
                        with gr.Row():
                            mic_input = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Mikrofon-Aufnahme (STT Port 8005)",
                            )
                            tts_output = gr.Audio(
                                label="Sprachausgabe der Antwort (TTS Port 8005)",
                                type="filepath",
                                autoplay=True,
                            )
                        with gr.Row():
                            with gr.Column():
                                info_label(
                                    "Automatische Sprachausgabe (SA-05)",
                                    "Liest die Antwort in der gewählten Zielsprache (Deutsch, Arabisch, Türkisch, Ukrainisch, Russisch, Englisch) laut vor.",
                                )
                                enable_tts_chk = gr.Checkbox(value=True, label="Antwort automatisch vorlesen")
                            with gr.Column():
                                info_label(
                                    "Sprechgeschwindigkeit (SA-05)",
                                    "0.85x spricht etwas langsamer und deutlicher (ideal für Senior:innen); 1.0x ist Normaltempo.",
                                )
                                tts_speed_slider = gr.Slider(
                                    minimum=0.65, maximum=1.25, value=0.85, step=0.05, show_label=False
                                )

                    with gr.Row():
                        reset_btn = gr.Button("🗑️ Gespräch beenden & RAM löschen (DSGVO TA-06)", variant="stop")
                        session_id_box = gr.Textbox(value="kiosk-hildesheim-01", label="Session-ID (60s RAM-TTL)")

                with gr.Column(scale=5):
                    gr.Markdown("### ⚙️ Architektur- & Sprach-Konfiguration")
                    with gr.Row():
                        with gr.Column():
                            info_label(
                                "KI-Paradigma (TA-04)",
                                "• <b>agentic_ai:</b> Fragt bei unklaren Fragen nach (Slot-Filling), nutzt Filter-Tools und prüft Telefonnummern.<br>"
                                "• <b>pure_rag:</b> Klassische Suche + direkte Antwort.<br>"
                                "• <b>fine_tuned:</b> Nutzt ein auf den Quartiers-Tonfall trainiertes Modell.",
                            )
                            paradigm_radio = gr.Radio(
                                choices=["agentic_ai", "pure_rag", "fine_tuned"],
                                value="agentic_ai",
                                show_label=False,
                            )
                        with gr.Column():
                            info_label(
                                "Sprachniveau (SA-01)",
                                "• <b>simple:</b> Bürgernahe, freundliche Alltagssprache.<br>"
                                "• <b>easy:</b> Leichte Sprache mit sehr kurzen Sätzen und Mediopunkt (z. B. Pflege·beratung).<br>"
                                "• <b>standard:</b> Normale Standardsprache.",
                            )
                            lang_level_radio = gr.Radio(
                                choices=["simple", "easy", "standard"],
                                value="simple",
                                show_label=False,
                            )

                    with gr.Row():
                        with gr.Column():
                            info_label(
                                "Zielsprache (SA-02)",
                                "Bestimmt die Sprache für Textantwort und Audio-Ausgabe: de=Deutsch, tr=Türkisch, ar=Arabisch, uk=Ukrainisch, ru=Russisch, en=Englisch.",
                            )
                            user_lang_drop = gr.Dropdown(
                                choices=["de", "tr", "ar", "uk", "ru", "en"],
                                value="de",
                                show_label=False,
                            )
                        with gr.Column():
                            info_label(
                                "Übersetzungsverfahren",
                                "• <b>v2a_direct:</b> Das LLM antwortet direkt in der Zielsprache (schneller).<br>"
                                "• <b>v2b_translated:</b> Das LLM erzeugt erst einen deutschen Zwischentext (für HIX-Messung) und übersetzt ihn danach.",
                            )
                            trans_var_radio = gr.Radio(
                                choices=["v2a_direct", "v2b_translated"],
                                value="v2a_direct",
                                show_label=False,
                            )

                    info_label(
                        "Quellen-Guard Schwellenwert (FA-06)",
                        "Mindest-Score (0.0 bis 1.0) einer gefundenen Textstelle. Liegt der beste Treffer darunter, rät der Bot nicht, sondern verweist an die AWO-Beratung.",
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.05, maximum=0.85, value=0.25, step=0.05, show_label=False
                    )

                    info_label(
                        "Live Agent-Trace & Latenz-Aufschlüsselung",
                        "Zeigt jeden ausgeführten Teilschritt (Query-Übersetzung, Retrieval, LLM, TTS) samt Dauer in Millisekunden.",
                    )
                    trace_box = gr.Textbox(show_label=False, lines=6, interactive=False)

                    info_label(
                        "Deutscher Zwischentext (für HIX-Prüfung bei V2b)",
                        "Zeigt bei aktivem 'v2b_translated' den deutschen Ursprungstext vor der Übersetzung in die Fremdsprache.",
                    )
                    german_inter_box = gr.Textbox(show_label=False, lines=3, interactive=False)

                    sources_table = gr.Dataframe(
                        headers=[
                            "Chunk-ID", "Titel", "Stadtteil", "Dense-Rang",
                            "BM25-Rang", "RRF", "Reranker", "Final-Score", "Sprachen"
                        ],
                        label="📚 Abgerufene Quell-Chunks (ChromaDB + BM25 + Cross-Encoder)",
                        interactive=False,
                    )
                    qr_image_box = gr.Image(
                        label="📱 QR-Code zum Mitnehmen aufs Smartphone (FA-07)",
                        height=170,
                    )

        # -------------------------------------------------------------
        # TAB 2: QUALITY-GATE & ETL-PIPELINE (SA-03, § 1 - § 11)
        # -------------------------------------------------------------
        with gr.TabItem("🛡️ 2. Quality-Gate & ETL-Ingest (Regelwerk § 1–§ 11)"):
            info_label(
                "Automatische Übernahme in ChromaDB & BM25 (Delta-Sync)",
                "Wenn aktiv, werden neu geprüfte Dokumente sofort in die Vektor- und Stichwortdatenbank des Chatbots übernommen (unveränderte Dokumente werden übersprungen).",
            )
            auto_sync_chk = gr.Checkbox(
                value=True,
                label="Nach Prüfung automatisch in ChromaDB & BM25 synchronisieren",
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📄 A. PDF- oder Textdatei hochladen")
                    upload_file = gr.File(
                        label="PDF- oder TXT-Dokument auswählen",
                        file_types=[".pdf", ".txt", ".md"],
                    )
                    info_label(
                        "Herausgeber & Ablaufdatum (Optional)",
                        "Lasse beide Felder leer, damit das System den Herausgeber (z. B. Landkreis Hildesheim, AWO) und die Gültigkeit automatisch ermittelt.",
                    )
                    pub_input = gr.Textbox(
                        value="",
                        placeholder="Herausgeber (Optional – Leer lassen für Auto-Erkennung)",
                        show_label=False,
                    )
                    valid_until_input = gr.Textbox(
                        value="",
                        placeholder="Ablaufdatum YYYY-MM-DD (Optional – Leer lassen für Auto-TTL)",
                        show_label=False,
                    )
                    use_llm_chk = gr.Checkbox(
                        value=False,
                        label="Ollama-LLM zusätzlich für Metadaten nutzen (Dauert ~8s länger; Aus = Schneller Regex-Modus)",
                    )
                    ingest_file_btn = gr.Button("1. Datei prüfen (Quality-Gate) & Chunks erzeugen", variant="primary")

                with gr.Column():
                    gr.Markdown("#### 🌐 B. Webseite scrapen (Blacklist & Auto-Relevanz)")
                    info_label(
                        "Ziel-URL & Automatische Relevanzprüfung (§ 7 – § 9)",
                        "Alle URLs außer den Einträgen auf der manuellen Blacklist sind erlaubt. Der Scraper prüft automatisch, ob der Seiteninhalt zum Quartier passt.",
                    )
                    scrape_url_input = gr.Textbox(
                        placeholder="https://...",
                        show_label=False,
                    )
                    scrape_pub_input = gr.Textbox(
                        value="",
                        placeholder="Herausgeber der Webseite (Optional – Leer lassen für Auto-Erkennung)",
                        show_label=False,
                    )
                    scrape_valid_input = gr.Textbox(
                        value="",
                        placeholder="Ablaufdatum YYYY-MM-DD (Optional – Leer lassen für Auto-TTL)",
                        show_label=False,
                    )
                    depth1_chk = gr.Checkbox(
                        value=False,
                        label="Kontextuelles Link-Crawling Tiefe = 1 (Aus = Nur Hauptseite scrapen, 3x schneller!)",
                    )
                    scrape_btn = gr.Button("1. Webseite scrapen & im Quality-Gate prüfen", variant="secondary")

            etl_status_box = gr.Textbox(label="Status & Quality-Gate Entscheidung", lines=3, interactive=False)
            sync_db_btn = gr.Button(
                "2. 🔄 Alle freigegebenen Chunks manuell in ChromaDB & BM25 synchronisieren (Delta-Sync)",
                variant="primary",
            )
            sync_status_box = gr.Textbox(label="Datenbank-Synchronisations-Status", interactive=False)

            rules_df = gr.Dataframe(
                headers=["Paragraph", "Regelname", "Status", "Messwert", "Auto-Fix?", "Begründung / Audit-Log"],
                label="📋 Schrittweises Prüfprotokoll des Quality-Gate Microservice (§ 1 bis § 11)",
                interactive=False,
            )
            with gr.Row():
                meta_preview_box = gr.Code(label="Extrahierte & geprüfte Metadaten (JSON)", language="json")
                chunks_preview_box = gr.Textbox(
                    label="Erzeugte Semantische Chunks inkl. Header & DIN-Abkürzungen",
                    lines=14,
                )

        # -------------------------------------------------------------
        # TAB 3: SYSTEM-HEALTH & MICROSERVICE-MONITOR
        # -------------------------------------------------------------
        with gr.TabItem("🩺 3. Microservice Health-Monitor"):
            info_label(
                "System-Status aller Microservices (Port 8001–8005 & Ollama 11434)",
                "Prüft per HTTP-Health-Check, ob alle Hintergrunddienste aktiv sind und wie viele Chunks bzw. TTS-Stimmen geladen sind.",
            )
            health_btn = gr.Button("🔄 Status aller Services prüfen", variant="primary")
            health_output = gr.Textbox(show_label=False, lines=8, interactive=False)

    # Event-Bindings Tab 1
    mic_input.change(
        fn=transcribe_mic_input,
        inputs=[mic_input],
        outputs=[user_input, user_lang_drop, robot_status_box],
    )
    send_btn.click(
        fn=handle_chat_turn,
        inputs=[
            user_input, chatbot, session_id_box, paradigm_radio, lang_level_radio,
            user_lang_drop, trans_var_radio, threshold_slider, enable_tts_chk, tts_speed_slider,
        ],
        outputs=[
            chatbot, user_input, trace_box, german_inter_box,
            sources_table, qr_image_box, tts_output, robot_status_box,
        ],
    )
    user_input.submit(
        fn=handle_chat_turn,
        inputs=[
            user_input, chatbot, session_id_box, paradigm_radio, lang_level_radio,
            user_lang_drop, trans_var_radio, threshold_slider, enable_tts_chk, tts_speed_slider,
        ],
        outputs=[
            chatbot, user_input, trace_box, german_inter_box,
            sources_table, qr_image_box, tts_output, robot_status_box,
        ],
    )
    reset_btn.click(
        fn=reset_kiosk_session,
        inputs=[session_id_box],
        outputs=[chatbot, trace_box, robot_status_box],
    )

    # Event-Bindings Tab 2
    ingest_file_btn.click(
        fn=run_file_ingest_and_audit,
        inputs=[upload_file, pub_input, valid_until_input, use_llm_chk, auto_sync_chk],
        outputs=[etl_status_box, rules_df, meta_preview_box, chunks_preview_box, sync_status_box],
    )
    scrape_btn.click(
        fn=run_url_scrape_and_audit,
        inputs=[scrape_url_input, scrape_pub_input, scrape_valid_input, depth1_chk, use_llm_chk, auto_sync_chk],
        outputs=[etl_status_box, rules_df, meta_preview_box, chunks_preview_box, sync_status_box],
    )
    sync_db_btn.click(fn=sync_etl_to_chromadb, inputs=[], outputs=[sync_status_box])

    # Event-Bindings Tab 3
    health_btn.click(fn=check_all_microservices_health, inputs=[], outputs=[health_output])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)