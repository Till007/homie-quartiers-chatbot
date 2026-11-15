"""
services/speech_service/main.py
FastAPI-Microservice für lokale Spracherkennung (faster-whisper STT) und
lokale Sprachausgabe (Piper TTS mit Offline-Fallback) auf Port 8005.
Implementiert FA-05, TA-03, SA-05, SA-06 und SA-07.

Startbefehl:
    uvicorn services.speech_service.main:app --host 127.0.0.1 --port 8005 --reload
"""

import os
import re
import shutil
import tempfile
import time
import wave
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

from shared.logger import get_service_logger
from shared.schemas import SynthesisRequest, TranscriptionResponse

logger = get_service_logger("speech-service")

MODELS_CACHE_DIR = Path(os.getenv("MODELS_CACHE_DIR", "data/models_cache"))
WHISPER_CACHE_DIR = MODELS_CACHE_DIR / "whisper"
PIPER_CACHE_DIR = MODELS_CACHE_DIR / "piper"
AUDIO_OUT_DIR = Path("data/audio_temp")

for d in (WHISPER_CACHE_DIR, PIPER_CACHE_DIR, AUDIO_OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Laut '2.0 Konzeption & Systementscheidung.pdf': small (int8) auf CPU benötigt 0 GB VRAM
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

app = FastAPI(
    title="HOMIE Speech Microservice (STT & TTS)",
    description="Lokales Speech-to-Text (faster-whisper) und Text-to-Speech (Piper TTS) für Kiosk & Roboter.",
    version="1.0.0",
)

logger.info(f"Lade faster-whisper Modell '{WHISPER_MODEL_SIZE}' (int8) auf {WHISPER_DEVICE}...")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type="int8",
    download_root=str(WHISPER_CACHE_DIR),
    cpu_threads=8,
)
logger.info("faster-whisper erfolgreich geladen.")


# =====================================================================
# HILFSFUNKTIONEN FÜR PIPER TTS & OFFLINE-SYNTHESE (SA-05, SA-06)
# =====================================================================

PIPER_DE_ONNX_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
)
PIPER_DE_JSON_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json"
)


def _ensure_piper_german_voice() -> Path | None:
    """Lädt im Development einmalig das deutsche Piper-ONNX-Sprachmodell in den lokalen Cache."""
    onnx_path = PIPER_CACHE_DIR / "de_DE-thorsten-medium.onnx"
    json_path = PIPER_CACHE_DIR / "de_DE-thorsten-medium.onnx.json"
    if onnx_path.exists() and json_path.exists():
        return onnx_path
    try:
        logger.info("Lade deutsche Piper-TTS-Stimme (de_DE-thorsten-medium) für Offline-Betrieb herunter...")
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            r_onnx = client.get(PIPER_DE_ONNX_URL)
            r_onnx.raise_for_status()
            onnx_path.write_bytes(r_onnx.content)

            r_json = client.get(PIPER_DE_JSON_URL)
            r_json.raise_for_status()
            json_path.write_bytes(r_json.content)
        logger.info(f"Piper-Stimme erfolgreich gespeichert unter {onnx_path}")
        return onnx_path
    except Exception as e:
        logger.warning(f"Piper-Stimme konnte nicht heruntergeladen werden (nutze lokalen SAPI5-Fallback): {e}")
        return None


def _clean_text_for_tts(text: str) -> str:
    """
    Entfernt Mediopunkte ('Pflege·beratung' -> 'Pflegeberatung'), Markdown-Symbole
    und URLs, damit die Sprachausgabe natürlich klingt (SA-01 / SA-05).
    """
    cleaned = text.replace("·", "")
    cleaned = re.sub(r"https?://\S+", "siehe Internetlink auf dem Bildschirm", cleaned)
    cleaned = re.sub(r"[*_#`~\[\]()]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _synthesize_wav(text: str, language: str, speed_factor: float, output_wav: Path) -> None:
    """
    Versucht primär die Synthese über Piper TTS (mit anpassbarer length_scale für SA-05).
    Fällt bei fehlender Stimmdatei unter Windows sauber auf pyttsx3 (Offline SAPI5) zurück.
    """
    # 1. Versuch: Piper Python-API (falls installiert und Modell vorhanden)
    if language == "de":
        onnx_model = _ensure_piper_german_voice()
        if onnx_model:
            try:
                from piper.voice import PiperVoice

                voice = PiperVoice.load(str(onnx_model))
                # length_scale > 1.0 bedeutet langsameres Sprechen (z.B. 1 / 0.85 = 1.18 für Senior:innen)
                length_scale = round(1.0 / max(min(speed_factor, 1.5), 0.5), 2)
                with wave.open(str(output_wav), "wb") as wav_file:
                    voice.synthesize(text, wav_file, length_scale=length_scale)
                return
            except Exception as e:
                logger.warning(f"Piper-Synthese Hinweis (nutze Fallback): {e}")

    # 2. Zuverlässiger Offline-Fallback über pyttsx3 (Windows SAPI5 / Linux espeak)
    try:
        import pyttsx3

        engine = pyttsx3.init()
        # Standard-Sprechrate ~185 WPM; mit speed_factor (z.B. 0.85x) skalieren
        base_rate = 185
        engine.setProperty("rate", int(base_rate * speed_factor))
        engine.save_to_file(text, str(output_wav))
        engine.runAndWait()
        if output_wav.exists() and output_wav.stat().st_size > 100:
            return
    except Exception as e:
        logger.warning(f"pyttsx3 Fallback-Fehler: {e}")

    # 3. Notfall-Fallback: Erzeuge kurze stille WAV-Datei, damit das Frontend niemals abstürzt
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 8000)


# =====================================================================
# FASTAPI ENDPUNKTE (Port 8005)
# =====================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "speech-service",
        "port": 8005,
        "whisper_model": WHISPER_MODEL_SIZE,
    }


@app.post("/api/v1/stt/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(
    audio_file: UploadFile = File(..., description="WAV/MP3/WebM Mikrofon-Aufnahme")
) -> TranscriptionResponse:
    """
    SA-07 & FA-05: Transkribiert Spracheingaben lokal über faster-whisper inkl.
    VAD-Störgeräuschfilterung (für belebte Quartiersräume) und automatischer Spracherkennung.
    """
    t0 = time.perf_counter()
    suffix = Path(audio_file.filename or "recording.wav").suffix or ".wav"
    temp_path = AUDIO_OUT_DIR / f"stt_in_{int(time.time() * 1000)}{suffix}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        segments, info = whisper_model.transcribe(
            str(temp_path),
            beam_size=5,
            vad_filter=True,  # SA-07: Filtert Hintergrundgeräusche im Stadtteilcafé heraus
            vad_parameters=dict(min_silence_duration_ms=400),
        )
        full_text = " ".join(seg.text.strip() for seg in segments).strip()
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            f"STT abgeschlossen in {duration_ms}ms | Sprache: {info.language} "
            f"({info.language_probability:.2f}) | Text: '{full_text}'"
        )
        return TranscriptionResponse(
            text=full_text,
            detected_language=info.language or "de",
            language_probability=round(float(info.language_probability), 3),
            duration_ms=duration_ms,
        )
    except Exception as e:
        logger.error(f"Fehler bei der STT-Transkription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"STT-Fehler: {str(e)}")
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/v1/tts/synthesize")
async def synthesize_speech_endpoint(req: SynthesisRequest):
    """
    SA-05 & SA-06: Erzeugt eine WAV-Audiodatei mit anpassbarer Sprechgeschwindigkeit (z.B. 0.85x).
    Wahrt bei sensiblen Themen (SA-06) Diskretion am öffentlichen Terminal.
    """
    try:
        if req.is_sensitive_topic:
            # SA-06: Keine laute Wiederholung sensibler Beratungsdetails im öffentlichen Raum
            spoken_text = (
                "Ich habe Ihnen die passenden vertraulichen Kontaktstellen auf dem Bildschirm "
                "und als QR-Code zum Mitnehmen angezeigt."
            )
        else:
            spoken_text = _clean_text_for_tts(req.text)

        out_file = AUDIO_OUT_DIR / f"tts_out_{int(time.time() * 1000)}.wav"
        _synthesize_wav(spoken_text, req.language, req.speed_Factor, out_file)

        return FileResponse(
            path=str(out_file),
            media_type="audio/wav",
            filename="homie_response.wav",
        )
    except Exception as e:
        logger.error(f"Fehler bei der TTS-Synthese: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS-Fehler: {str(e)}")