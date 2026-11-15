"""
services/ui_debug_client/app.py
Interaktive Gradio-Benutzeroberfläche für Debugging, Governance (Quality-Gate/ETL)
und Kiosk-/Roboter-Simulation (Port 7860).

Startbefehl:
    python -m services.ui_debug_client.app
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx
import qrcode
from PIL import Image

from shared.logger import get_service_logger

logger = get_service_logger("ui-debug-client")

QUALITY_URL = "http://127.0.0.1:8001"
ETL_URL = "http://127.0.0.1:8002"
RAG_URL = "http://127.0.0.1:8003"
ORCHESTRATOR_URL = "http://127.0.0.1:8004"
SPEECH_URL = "http://127.0.0.1:8005"

# CSS für SA-05 (Zwei-Sinne-Prinzip: Kontraststarker Text & große Schrift 18-22pt im Chat)
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
"""


def generate_qr_image(vcard_payload: Optional[str]) -> Optional[Image.Image]:
    """FA-07: Erzeugt ein scannbares QR-Code-Bild aus den Kontaktdaten der Empfehlung."""
    if not vcard_payload:
        return None
    qr = qrcode.QRCode(version=1, box_size=6, border=2)
    qr.add_data(vcard_payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img.convert("RGB")


# =====================================================================
# TAB 1 LOGIK: CHAT, AUDIO (STT/TTS) & PARADIGMEN-VERGLEICH
# =====================================================================

def transcribe_mic_input(audio_filepath: Optional[str]) -> Tuple[str, str, str]:
    """Sendet die Mikrofonaufnahme an den Speech-Service (Port 8005) und gibt Text + erkannte Sprache zurück."""
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
    chat_history: List[Dict[str, str]],
    session_id: str,
    paradigm: str,
    language_level: str,
    user_language: str,
    translation_variant: str,
    score_threshold: float,
    enable_tts: bool,
    tts_speed: float,
) -> Tuple[List[Dict[str, str]], str, str, str, List[List[Any]], Optional[Image.Image], Optional[str], str]:
    """
    Sendet die Anfrage an den Orchestrator (Port 8004), aktualisiert das Chat-Fenster,
    bereitet die Debug-Tabellen auf und erzeugt optional die Sprachausgabe (Port 8005) + QR-Code (FA-07).
    """
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
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{ORCHESTRATOR_URL}/api/v1/chat/query", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        err_msg = f"Fehler beim Aufruf des Orchestrators (Port 8004): {e}"
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": err_msg})
        return chat_history, "", err_msg, "", [], None, None, "🔴 ROBO-STATE: ERROR"

    answer_text = data.get("answer_text", "")
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": answer_text})

    # 1. Agent-Trace & Metriken formatieren
    trace_lines = [
        f"⏱️ Gesamt-Latenz: {data.get('total_latency_ms')} ms | "
        f"Paradigma: {data.get('paradigm_used')} | "
        f"Quellen-Guard bestanden: {data.get('passed_source_guard')} | "
        f"Sensibles Thema (SA-06): {data.get('is_sensitive_topic')}",
        f"🔍 Eigenständige Suchanfrage (Query-Rewriter): '{data.get('rewritten_query')}'",
        "-" * 70,
    ]
    for idx, step in enumerate(data.get("agent_trace", []), start=1):
        trace_lines.append(
            f"[{idx}] {step.get('step_name')} ({step.get('duration_ms')} ms):\n    {step.get('details')}"
        )
    trace_output = "\n".join(trace_lines)

    # 2. Deutscher Zwischentext (für V2b HIX-/Leichte-Sprache-Evaluation)
    german_inter = data.get("german_intermediate_text") or (
        "(Kein separater Zwischentext, da V2a direkt oder Sprache = Deutsch)"
    )

    # 3. Quell-Chunks & Scores als Tabelle aufbereiten
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

    # 4. QR-Code (FA-07) generieren
    qr_img = generate_qr_image(data.get("qr_code_payload"))

    # 5. Optionale TTS-Sprachausgabe (Port 8005)
    tts_audio_path = None
    if enable_tts and answer_text:
        try:
            tts_req = {
                "text": answer_text,
                "language": user_language,
                "speed_Factor": float(tts_speed),
                "is_sensitive_topic": bool(data.get("is_sensitive_topic", False)),
            }
            with httpx.Client(timeout=30.0) as client:
                tts_resp = client.post(f"{SPEECH_URL}/api/v1/tts/synthesize", json=tts_req)
                if tts_resp.status_code == 200:
                    out_wav = Path("data/audio_temp") / "ui_last_response.wav"
                    out_wav.parent.mkdir(parents=True, exist_ok=True)
                    out_wav.write_bytes(tts_resp.content)
                    tts_audio_path = str(out_wav)
        except Exception as e:
            logger.warning(f"TTS-Wiedergabe im UI fehlgeschlagen: {e}")

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
    """TA-06: Löscht den Sitzungsverlauf sofort aus dem RAM des Orchestrators."""
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(f"{ORCHESTRATOR_URL}/api/v1/session/reset", params={"session_id": session_id})
        return [], "Sitzungsgedächtnis im RAM vollständig gelöscht (DSGVO TA-06).", "🟢 ROBO-STATE: IDLE"
    except Exception as e:
        return [], f"Fehler beim Session-Reset: {e}", "🔴 ROBO-STATE: ERROR"


# =====================================================================
# TAB 2 LOGIK: ETL-INGEST & QUALITY-GATE GOVERNANCE (§ 1 - § 11)
# =====================================================================

def run_file_ingest_and_audit(
    file_obj: Any, publisher: str, valid_until_str: str, use_llm: bool
) -> Tuple[str, List[List[Any]], str, str]:
    """Sendet eine hochgeladene PDF/TXT an den ETL-Service (Port 8002) und zeigt den Quality-Gate-Report."""
    if file_obj is None:
        return "Bitte zuerst eine PDF- oder TXT-Datei auswählen.", [], "{}", ""

    filepath = Path(file_obj.name)
    data_form: Dict[str, Any] = {
        "publisher": publisher or "AWO Hildesheim",
        "use_llm_enrichment": str(use_llm).lower(),
    }
    if valid_until_str and valid_until_str.strip():
        data_form["valid_until"] = valid_until_str.strip()

    try:
        with open(filepath, "rb") as f:
            files = {"file": (filepath.name, f)}
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(f"{ETL_URL}/api/v1/etl/ingest-file", data=data_form, files=files)
                resp.raise_for_status()
                res = resp.json()
        return _format_etl_response_for_ui(res)
    except Exception as e:
        return f"❌ Fehler beim Datei-Ingest: {e}", [], "{}", ""


def run_url_scrape_and_audit(
    url: str, publisher: str, follow_depth_1: bool, use_llm: bool
) -> Tuple[str, List[List[Any]], str, str]:
    """Crawlt eine Hildesheimer Whitelist-URL über Port 8002 und zeigt den Quality-Gate-Report."""
    if not url or not url.strip():
        return "Bitte eine gültige Hildesheimer URL eingeben.", [], "{}", ""

    payload = {
        "url": url.strip(),
        "publisher": publisher or "Webquelle Hildesheim",
        "follow_links_depth_1": follow_depth_1,
        "use_llm_enrichment": use_llm,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{ETL_URL}/api/v1/etl/ingest-url", json=payload)
            if resp.status_code != 200:
                return f"❌ Scraper/Regelwerk-Ablehnung ({resp.status_code}): {resp.text}", [], "{}", ""
            res = resp.json()
        return _format_etl_response_for_ui(res)
    except Exception as e:
        return f"❌ Fehler beim Web-Scraping: {e}", [], "{}", ""


def _format_etl_response_for_ui(res: Dict[str, Any]) -> Tuple[str, List[List[Any]], str, str]:
    audit = res.get("audit_report", {})
    approved = audit.get("approved_for_ingestion", False)
    badge = "✅ FREIGEGEBEN DURCH QUALITY-GATE" if approved else "🚫 BLOCKIERT (IN QUARANTÄNE VERSCHOBEN)"
    summary = (
        f"{badge} | Audit-ID: {audit.get('audit_id')}\n"
        f"Status: {res.get('status_message')}\n"
        f"Erzeugte Chunks: {len(res.get('chunks', []))}"
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

    meta_json = json.dumps(audit.get("extracted_metadata", {}), indent=2, ensure_ascii=False)
    chunks_preview = "\n\n====================\n\n".join(
        f"CHUNK {c.get('chunk_id')} (~{c.get('token_estimate')} Tokens):\n{c.get('text_for_indexing')}"
        for c in res.get("chunks", [])
    )
    return summary, rule_rows, meta_json, chunks_preview


def sync_etl_to_chromadb() -> str:
    """Synchronisiert alle geprüften Chunks aus data/processed_chunks/ in ChromaDB & BM25 (Port 8003)."""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{RAG_URL}/api/v1/rag/sync-from-etl")
            resp.raise_for_status()
            data = resp.json()
        return (
            f"✅ Synchronisation erfolgreich! "
            f"Verarbeitete Dokumente: {data.get('synced_files')} | "
            f"Gesamtanzahl Chunks im Hybrid-Index (ChromaDB + BM25): {data.get('total_indexed_chunks')}"
        )
    except Exception as e:
        return f"❌ Fehler bei der Index-Synchronisation (Port 8003 prüfen): {e}"


# =====================================================================
# TAB 3 LOGIK: SYSTEM-HEALTH & ISOLIERTER RAG-INSPECTOR
# =====================================================================

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
# GRADIO UI AUFBAU
# =====================================================================

with gr.Blocks(title="HOMIE - Quartiers-Chatbot & Debug-Steuerzentrale", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# 🤖 HOMIE – Quartiers-Chatbot Hildesheim (Development- & Evaluations-Zentrale)\n"
        "**IT-Studienprojekt (IT-MA5) | HULLS Use-Case 1:** Modularer Vergleich von *Pure RAG*, *Fine-Tuned LLM* und *Agentic AI* inkl. Quality-Gate (§ 1–§ 11)"
    )

    with gr.Tabs():
        # -------------------------------------------------------------
        # TAB 1: KIOSK-SIMULATION & PARADIGMEN-EVALUATION
        # -------------------------------------------------------------
        with gr.TabItem("💬 1. Kiosk-Terminal & Paradigmen-Vergleich (TA-04)"):
            with gr.Row():
                # Linke Spalte: Barrierefreier Kiosk-Chat & Audio
                with gr.Column(scale=6):
                    robot_status_box = gr.Markdown("🟢 **ROBO-STATE: IDLE**", elem_classes=["status-badge"])
                    chatbot = gr.Chatbot(
                        #type="messages",
                        label="Quartiers-Dialog (Großschrift 20pt nach SA-05 Zwei-Sinne-Prinzip)",
                        height=430,
                        elem_classes=["kiosk-chat"],
                    )
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="Frage eingeben (Deutsch, Türkisch, Arabisch, Ukrainisch, Russisch, Englisch)",
                            placeholder="z.B. 'Wo gibt es eine Pflegeberatung in Hildesheim, wo jemand Türkisch spricht?'",
                            scale=4,
                        )
                        send_btn = gr.Button("Senden 🚀", variant="primary", scale=1)

                    with gr.Accordion("🎙️ Sprach-Ein-/Ausgabe (faster-whisper STT & Piper TTS)", open=True):
                        with gr.Row():
                            mic_input = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Frage einsprechen (STT Port 8005)",
                            )
                            tts_output = gr.Audio(
                                label="Sprachausgabe (TTS Port 8005)",
                                type="filepath",
                                autoplay=True,
                            )
                        with gr.Row():
                            enable_tts_chk = gr.Checkbox(value=True, label="Antwort automatisch vorlesen (SA-05)")
                            tts_speed_slider = gr.Slider(
                                minimum=0.65,
                                maximum=1.25,
                                value=0.85,
                                step=0.05,
                                label="Sprechgeschwindigkeit (SA-05: 0.85x empfohlen für Senior:innen)",
                            )

                    with gr.Row():
                        reset_btn = gr.Button("🗑️ Gespräch beenden & RAM löschen (DSGVO TA-06)", variant="stop")
                        session_id_box = gr.Textbox(value="kiosk-hildesheim-01", label="Session-ID (60s RAM-TTL)")

                # Rechte Spalte: Konfigurations-Umschalter, Agent-Trace & QR-Code
                with gr.Column(scale=5):
                    gr.Markdown("### ⚙️ Architektur- & Sprach-Konfiguration (Live umschaltbar)")
                    with gr.Row():
                        paradigm_radio = gr.Radio(
                            choices=["agentic_ai", "pure_rag", "fine_tuned"],
                            value="agentic_ai",
                            label="KI-Paradigma (TA-04 Vergleichsarchitektur)",
                        )
                        lang_level_radio = gr.Radio(
                            choices=["simple", "easy", "standard"],
                            value="simple",
                            label="Sprachniveau (SA-01: 'easy' = Leichte Sprache mit Mediopunkt ·)",
                        )
                    with gr.Row():
                        user_lang_drop = gr.Dropdown(
                            choices=["de", "tr", "ar", "uk", "ru", "en"],
                            value="de",
                            label="Zielsprache (SA-02)",
                        )
                        trans_var_radio = gr.Radio(
                            choices=["v2a_direct", "v2b_translated"],
                            value="v2a_direct",
                            label="Übersetzungsvariante (V2a Direkt vs. V2b DE-Zwischentext)",
                        )
                    threshold_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.85,
                        value=0.25,
                        step=0.05,
                        label="Quellen-Guard Schwellenwert (Darunter greift AWO-Fallback nach FA-06)",
                    )

                    trace_box = gr.Textbox(
                        label="🔬 Live Agent-Trace & Latenz-Aufschlüsselung",
                        lines=7,
                        interactive=False,
                    )
                    german_inter_box = gr.Textbox(
                        label="🇩🇪 Deutscher Zwischentext (für HIX- & Leichte-Sprache-Prüfung bei V2b)",
                        lines=3,
                        interactive=False,
                    )
                    sources_table = gr.Dataframe(
                        headers=[
                            "Chunk-ID", "Titel", "Stadtteil", "Dense-Rang",
                            "BM25-Rang", "RRF", "Reranker", "Final-Score", "Sprachen"
                        ],
                        label="📚 Abgerufene Quell-Chunks (ChromaDB + BM25 + Cross-Encoder)",
                        interactive=False,
                    )
                    qr_image_box = gr.Image(
                        label="📱 QR-Code zum Mitnehmen der Kontaktdaten aufs Smartphone (FA-07)",
                        height=180,
                    )

        # -------------------------------------------------------------
        # TAB 2: QUALITY-GATE & ETL-PIPELINE (SA-03, § 1 - § 11)
        # -------------------------------------------------------------
        with gr.TabItem("🛡️ 2. Quality-Gate & ETL-Ingest (Regelwerk § 1–§ 11)"):
            gr.Markdown(
                "### Dokumenten- & Webquellen-Aufnahme mit automatischer Qualitätsprüfung\n"
                "Jede Quelle durchläuft zuerst den **Quality-Gate Microservice (Port 8001)**. "
                "Nur freigegebene Quellen werden nach DIN 2340 standardisiert, mit dem Kulturellen Glossar angereichert und gechunkt."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📄 A. PDF- oder Textdatei hochladen")
                    upload_file = gr.File(label="PDF- oder TXT-Dokument auswählen", file_types=[".pdf", ".txt", ".md"])
                    pub_input = gr.Textbox(value="AWO Hildesheim", label="Herausgeber / Fachabteilung (§ 1 A SSOT)")
                    valid_until_input = gr.Textbox(
                        placeholder="YYYY-MM-DD (optional, z.B. 2026-12-31)",
                        label="Ablaufdatum / TTL (§ 10 B a)",
                    )
                    use_llm_chk = gr.Checkbox(
                        value=True, label="Lokales Ollama-Modell für Kontaktdaten- & Keyword-Extraktion nutzen"
                    )
                    ingest_file_btn = gr.Button("1. Datei prüfen (Quality-Gate) & Chunks erzeugen", variant="primary")

                with gr.Column():
                    gr.Markdown("#### 🌐 B. Hildesheimer Webseite scrapen (§ 7 – § 9)")
                    scrape_url_input = gr.Textbox(
                        placeholder="https://www.hildesheim.de/...",
                        label="Ziel-URL (Muss auf Whitelist stehen & gültige robots.txt besitzen)",
                    )
                    scrape_pub_input = gr.Textbox(value="Stadt Hildesheim", label="Herausgeber")
                    depth1_chk = gr.Checkbox(value=True, label="Kontextuelles Link-Crawling (Tiefe = 1) aktivieren (§ 8 C a)")
                    scrape_btn = gr.Button("1. Webseite scrapen & im Quality-Gate prüfen", variant="secondary")

            etl_status_box = gr.Textbox(label="Status & Quality-Gate Entscheidung", lines=3, interactive=False)
            sync_db_btn = gr.Button(
                "2. 🔄 Alle freigegebenen Chunks jetzt in ChromaDB & BM25 (production_bundle) synchronisieren!",
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
                    label="Erzeugte Semantische Chunks inkl. Header & DIN-Abkürzungen", lines=14
                )

        # -------------------------------------------------------------
        # TAB 3: SYSTEM-HEALTH & MICROSERVICE-MONITOR
        # -------------------------------------------------------------
        with gr.TabItem("🩺 3. Microservice Health-Monitor"):
            gr.Markdown("### Live-Status aller lokalen Microservices & Ollama")
            health_btn = gr.Button("🔄 Status aller Services prüfen", variant="primary")
            health_output = gr.Textbox(label="System-Status", lines=8, interactive=False)

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
        inputs=[upload_file, pub_input, valid_until_input, use_llm_chk],
        outputs=[etl_status_box, rules_df, meta_preview_box, chunks_preview_box],
    )
    scrape_btn.click(
        fn=run_url_scrape_and_audit,
        inputs=[scrape_url_input, scrape_pub_input, depth1_chk, use_llm_chk],
        outputs=[etl_status_box, rules_df, meta_preview_box, chunks_preview_box],
    )
    sync_db_btn.click(fn=sync_etl_to_chromadb, inputs=[], outputs=[sync_status_box])

    # Event-Bindings Tab 3
    health_btn.click(fn=check_all_microservices_health, inputs=[], outputs=[health_output])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)