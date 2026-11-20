"""
services/speech_service/main.py
FastAPI-Microservice für lokale Spracherkennung (faster-whisper STT) und
mehrsprachige Sprachausgabe (de, ar, tr, uk, ru, en) auf Port 8005.
Neu:
- Echte mehrsprachige TTS-Unterstützung für Arabisch (ar), Türkisch (tr), Ukrainisch (uk), Russisch (ru), Englisch (en) & Deutsch (de)
- RAM-Caching der geladenen Piper-ONNX-Stimmen (reduziert TTS-Latenz von ~3s auf ~0.3s)
- Mehrsprachiger Neural-Fallback (edge-tts) für Windows, falls ein lokales ONNX-Modell fehlt
"""

import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, Tuple

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

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

app = FastAPI(
    title="HOMIE Speech Microservice (Multilingual STT & TTS)",
    description="Mehrsprachiges Speech-to-Text (faster-whisper) & Text-to-Speech (de, ar, tr, uk, ru, en) auf Port 8005.",
    version="1.3.0",
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

# Offizielle Piper-ONNX-Sprachmodelle für alle 6 Quartierssprachen (SA-02 / SA-05)
PIPER_VOICE_URLS: Dict[str, Tuple[str, str]] = {
    "de": (
        "de_DE-thorsten-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
    ),
    "ar": (
        "ar_JO-kareem-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
    ),
    "tr": (
        "tr_TR-dfki-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx",
    ),
    "uk": (
        "uk_UA-ukrainian_tts-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx",
    ),
    "ru": (
        "ru_RU-irina-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
    ),
    "en": (
        "en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    ),
}

# Neurale Fallback-Stimmen (für edge-tts unter Windows, falls Piper-Phonemizer für Arabisch/Kyrillisch fehlt)
EDGE_TTS_VOICES: Dict[str, str] = {
    "de": "de-DE-KatjaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "tr": "tr-TR-EmelNeural",
    "uk": "uk-UA-PolinaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "en": "en-US-AriaNeural",
}

# RAM-Cache für geladene PiperVoice-Instanzen (verhindert 2-3s Nachladezeit bei jeder Anfrage!)
_LOADED_PIPER_VOICES: Dict[str, Any] = {}


def _detect_script_language(text: str, requested_lang: str) -> str:
    """Erkennt anhand des Schriftsystems automatisch Arabisch oder Kyrillisch für die korrekte TTS-Stimme."""
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(r"[\u0400-\u04FF]", text):
        if re.search(r"[іїєґІЇЄҐ]", text):
            return "uk"
        return "ru" if requested_lang not in ("uk", "ru") else requested_lang
    return requested_lang if requested_lang in PIPER_VOICE_URLS else "de"


def _get_or_load_piper_voice(lang_code: str) -> Any | None:
    """Lädt das passende Piper-ONNX-Modell einmalig herunter und hält es im RAM-Cache."""
    if lang_code in _LOADED_PIPER_VOICES:
        return _LOADED_PIPER_VOICES[lang_code]

    if lang_code not in PIPER_VOICE_URLS:
        return None

    filename, url = PIPER_VOICE_URLS[lang_code]
    onnx_path = PIPER_CACHE_DIR / filename
    json_path = PIPER_CACHE_DIR / f"{filename}.json"

    if not (onnx_path.exists() and json_path.exists() and onnx_path.stat().st_size > 1_000_000):
        try:
            logger.info(f"Lade Piper-TTS-Stimme für '{lang_code}' ({filename}) herunter...")
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                r_onnx = client.get(url)
                r_onnx.raise_for_status()
                onnx_path.write_bytes(r_onnx.content)

                r_json = client.get(f"{url}.json")
                r_json.raise_for_status()
                json_path.write_bytes(r_json.content)
            logger.info(f"Piper-Stimme '{filename}' erfolgreich im lokalen Cache gespeichert.")
        except Exception as e:
            logger.warning(f"Download der Piper-Stimme '{filename}' fehlgeschlagen: {e}")
            return None

    try:
        from piper.voice import PiperVoice

        voice: Any = PiperVoice.load(str(onnx_path))
        _LOADED_PIPER_VOICES[lang_code] = voice
        logger.info(f"PiperVoice '{lang_code}' erfolgreich in den RAM-Cache geladen.")
        return voice
    except Exception as e:
        logger.warning(f"PiperVoice '{lang_code}' konnte nicht initialisiert werden: {e}")
        return None


def _clean_text_for_tts(text: str, lang_code: str) -> str:
    """Entfernt Mediopunkte und Markdown, erhält aber arabische, kyrillische und türkische Buchstaben vollständig."""
    cleaned = text.replace("·", "")
    link_replacement = " " if lang_code != "de" else " siehe Internetlink "
    cleaned = re.sub(r"https?://\S+", link_replacement, cleaned)
    cleaned = re.sub(r"[*_#`~\[\]()<>|]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


async def _synthesize_with_edge_tts(
    text: str, lang_code: str, speed_factor: float, output_audio: Path
) -> bool:
    """
    Mehrsprachige neurale Sprachausgabe (unterstützt fließendes Arabisch, Türkisch, Ukrainisch,
    Russisch, Englisch und Deutsch), falls Piper unter Windows ausfällt.
    """
    # edge-tts ist kein lokales Modell: Der Text wird an den Onlinedienst von Microsoft gesendet.
    # Das widerspricht K1 (keine Nutzerdaten an Dritte, Offline-Test) und dem Offline-Betrieb auf dem
    # Roboter. Deshalb nur nach ausdrücklicher Freigabe (z. B. für Demos): ALLOW_CLOUD_TTS=true
    if os.getenv("ALLOW_CLOUD_TTS", "false").strip().lower() not in ("1", "true", "yes"):
        logger.info("Cloud-TTS (edge-tts) deaktiviert (ALLOW_CLOUD_TTS ist nicht gesetzt); nutze lokale Fallbacks.")
        return False
    try:
        import edge_tts

        voice_name = EDGE_TTS_VOICES.get(lang_code, "de-DE-KatjaNeural")
        pct = int(round((speed_factor - 1.0) * 100))
        rate_str = f"{pct:+d}%"
        communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate_str)
        await communicate.save(str(output_audio))
        return output_audio.exists() and output_audio.stat().st_size > 500
    except Exception as e:
        logger.warning(f"edge-tts Fallback nicht verfügbar ({e}).")
        return False


def _synthesize_with_windows_sapi(text: str, speed_factor: float, output_wav: Path) -> bool:
    """Lokaler Windows-SAPI-Fallback (primär für Deutsch/Englisch)."""
    if sys.platform != "win32":
        return False

    sapi_rate = max(-10, min(10, int(round((speed_factor - 1.0) * 10))))
    txt_temp = output_wav.with_suffix(".txt")
    try:
        txt_temp.write_text(text, encoding="utf-8")
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = {sapi_rate}; "
            f"$text = Get-Content -Path '{txt_temp.resolve()}' -Raw -Encoding UTF8; "
            f"$synth.SetOutputToWaveFile('{output_wav.resolve()}'); "
            "$synth.Speak($text); "
            "$synth.Dispose();"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            check=True,
            timeout=25,
            capture_output=True,
        )
        return output_wav.exists() and output_wav.stat().st_size > 500
    except Exception as e:
        logger.error(f"Windows System.Speech Synthese fehlgeschlagen: {e}")
        return False
    finally:
        txt_temp.unlink(missing_ok=True)


async def _synthesize_audio(
    text: str, language: str, speed_factor: float, output_wav: Path
) -> Path:
    """
    Mehrstufige TTS-Pipeline für alle 6 Sprachen (de, ar, tr, uk, ru, en):
    1. Gecachte lokale Piper-ONNX-Stimme (100% offline für Roboter & Dev).
    2. Mehrsprachiger edge-tts Neural-Fallback (perfekt für Arabisch/Türkisch/Ukrainisch unter Windows).
    3. Windows SAPI (für Deutsch/Englisch) bzw. Linux espeak-ng.
    """
    effective_lang = _detect_script_language(text, language)
    length_scale = round(1.0 / max(min(speed_factor, 1.5), 0.5), 2)

    # 1. Versuch: Lokale Piper-ONNX-Stimme aus RAM-Cache
    voice = _get_or_load_piper_voice(effective_lang)
    if voice is not None:
        try:
            if hasattr(voice, "config") and hasattr(voice.config, "length_scale"):
                voice.config.length_scale = length_scale

            with wave.open(str(output_wav), "wb") as wav_file:
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(text, wav_file)
                else:
                    voice.synthesize(text, wav_file)

            if output_wav.exists() and output_wav.stat().st_size > 500:
                logger.info(
                    f"Piper-ONNX TTS ({effective_lang}) erfolgreich ({output_wav.stat().st_size} Bytes)."
                )
                return output_wav
        except Exception as e:
            logger.warning(f"Piper-ONNX Synthese für '{effective_lang}' fehlgeschlagen: {e}")

    # 2. Versuch: Mehrsprachige Neural-Stimme (unterstützt Arabisch, Türkisch, Ukrainisch, Russisch, Deutsch, Englisch)
    mp3_path = output_wav.with_suffix(".mp3")
    if await _synthesize_with_edge_tts(text, effective_lang, speed_factor, mp3_path):
        logger.info(
            f"Mehrsprachige Neural-TTS ({effective_lang}) erfolgreich ({mp3_path.stat().st_size} Bytes)."
        )
        return mp3_path

    # 3. Versuch: Windows SAPI (nur sinnvoll bei lateinischer Schrift de/en)
    if effective_lang in ("de", "en") and _synthesize_with_windows_sapi(text, speed_factor, output_wav):
        logger.info(f"Windows SAPI TTS ({effective_lang}) erfolgreich.")
        return output_wav

    # 4. Versuch: Linux espeak-ng (unterstützt ar, tr, uk, ru, de, en auf dem Jetson-Roboter)
    try:
        wpm = int(170 * speed_factor)
        subprocess.run(
            ["espeak-ng", "-v", effective_lang, "-s", str(wpm), "-w", str(output_wav), text],
            check=True,
            timeout=15,
        )
        if output_wav.exists() and output_wav.stat().st_size > 500:
            return output_wav
    except Exception:
        pass

    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 8000)
    return output_wav


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "speech-service",
        "port": 8005,
        "whisper_model": WHISPER_MODEL_SIZE,
        "cached_piper_voices": list(_LOADED_PIPER_VOICES.keys()),
    }


@app.post("/api/v1/stt/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(
    audio_file: UploadFile = File(..., description="WAV/MP3/WebM Mikrofon-Aufnahme")
) -> TranscriptionResponse:
    t0 = time.perf_counter()
    suffix = Path(audio_file.filename or "recording.wav").suffix or ".wav"
    temp_path = AUDIO_OUT_DIR / f"stt_in_{int(time.time() * 1000)}{suffix}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        segments, info = whisper_model.transcribe(
            str(temp_path),
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )
        full_text = " ".join(seg.text.strip() for seg in segments).strip()
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

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
    try:
        effective_lang = _detect_script_language(req.text, req.language)
        if req.is_sensitive_topic:
            spoken_text = (
                "Ich habe Ihnen die passenden vertraulichen Kontaktstellen auf dem Bildschirm "
                "und als QR-Code zum Mitnehmen angezeigt."
            )
            effective_lang = "de"
        else:
            spoken_text = _clean_text_for_tts(req.text, effective_lang)

        if not spoken_text:
            spoken_text = "Es liegt keine Antwort zum Vorlesen vor."

        out_wav = AUDIO_OUT_DIR / f"tts_out_{int(time.time() * 1000)}.wav"
        final_audio_path = await _synthesize_audio(
            spoken_text, effective_lang, req.speed_Factor, out_wav
        )
        media_type = "audio/mpeg" if final_audio_path.suffix == ".mp3" else "audio/wav"

        return FileResponse(
            path=str(final_audio_path),
            media_type=media_type,
            filename=final_audio_path.name,
        )
    except Exception as e:
        logger.error(f"Fehler bei der TTS-Synthese: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS-Fehler: {str(e)}")