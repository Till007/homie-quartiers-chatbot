#!/usr/bin/env python3
"""
robot_client/homie_navel_app.py
Lauffähiger Voice-Client für den Navel-Roboter (und lokalen Windows-Test via --simulate).
Zeigt im Konsolenfenster live an, was verstanden wurde, welcher Modus greift und was HOMIE antwortet.
"""

import argparse
import asyncio
import base64
import inspect
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, List, Optional, Tuple

import httpx
import numpy as np
import pyaudio

# Audio-Konstanten laut Navel SDK (odas_1 liefert 48 kHz int16 Mono)
NAVEL_DEVICE_NAME = "odas_1"
CAPTURE_RATE = 48000
UPLOAD_RATE = 16000  # 48000 / 3 = exakt 16 kHz -> 3x kleinerer & schnellerer Upload
CHUNK_MS = 100
CHUNK_FRAMES = int(CAPTURE_RATE * CHUNK_MS / 1000)

# VAD-Schwellenwerte (Spracherkennung & Stille-Erkennung)
SILENCE_DURATION_S = 1.2
MIN_SPEECH_S = 0.5
MAX_RECORD_S = 14.0
ENERGY_MULTIPLIER = 2.6
MIN_RMS_THRESHOLD = 350.0


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _find_input_device_index(p: pyaudio.PyAudio, simulate: bool) -> Optional[int]:
    if simulate:
        return None
    try:
        alsa_info = p.get_host_api_info_by_type(pyaudio.paALSA)
        dev_count = int(str(alsa_info.get("deviceCount", 0)))
        for i in range(dev_count):
            dev = p.get_device_info_by_host_api_device_index(0, i)
            if str(dev.get("name", "")) == NAVEL_DEVICE_NAME:
                return i
    except Exception:
        pass

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if NAVEL_DEVICE_NAME in str(dev.get("name", "")):
            return i

    print("[Hinweis] Gerät 'odas_1' nicht explizit gefunden – verwende System-Standard-Mikrofon.", flush=True)
    return None


def calibrate_room_noise(p: pyaudio.PyAudio, device_index: Optional[int]) -> float:
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=CAPTURE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_FRAMES,
    )
    rms_values: List[float] = []
    try:
        for _ in range(10):
            raw = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16)
            rms_values.append(float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))))
    finally:
        stream.stop_stream()
        stream.close()

    baseline = float(np.median(rms_values)) if rms_values else 150.0
    print(f"🔧 Raumgeräusch kalibriert (Basis-RMS={baseline:.0f}). Bereit – sprich einfach los!", flush=True)
    return baseline


def record_single_utterance_blocking(
    p: pyaudio.PyAudio, device_index: Optional[int], base_noise_rms: float
) -> Tuple[bytes, float]:
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=CAPTURE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_FRAMES,
    )
    threshold = max(MIN_RMS_THRESHOLD, base_noise_rms * ENERGY_MULTIPLIER)
    pre_buffer: List[bytes] = []
    recorded_frames: List[bytes] = []
    speaking = False
    silence_chunks = 0
    speech_chunks = 0
    max_chunks = int(MAX_RECORD_S * 1000 / CHUNK_MS)
    silence_limit = int(SILENCE_DURATION_S * 1000 / CHUNK_MS)
    min_speech_chunks = int(MIN_SPEECH_S * 1000 / CHUNK_MS)

    try:
        while True:
            raw = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

            if not speaking:
                pre_buffer.append(raw)
                if len(pre_buffer) > 3:
                    pre_buffer.pop(0)

                if rms > threshold:
                    speaking = True
                    recorded_frames.extend(pre_buffer)
                    speech_chunks = 1
                    silence_chunks = 0
                    print(f"\n🎤 Sprache erkannt (RMS={rms:.0f}) – nehme auf...", flush=True)
                else:
                    base_noise_rms = 0.96 * base_noise_rms + 0.04 * rms
                    threshold = max(MIN_RMS_THRESHOLD, base_noise_rms * ENERGY_MULTIPLIER)
            else:
                recorded_frames.append(raw)
                if rms > threshold * 0.75:
                    speech_chunks += 1
                    silence_chunks = 0
                else:
                    silence_chunks += 1

                if silence_chunks >= silence_limit or len(recorded_frames) >= max_chunks:
                    if speech_chunks >= min_speech_chunks:
                        break
                    speaking = False
                    recorded_frames.clear()
                    speech_chunks = 0
                    silence_chunks = 0
    finally:
        stream.stop_stream()
        stream.close()

    pcm_48k = np.frombuffer(b"".join(recorded_frames), dtype=np.int16)
    pcm_16k = pcm_48k[::3].tobytes()

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(UPLOAD_RATE)
        wf.writeframes(pcm_16k)
    return wav_io.getvalue(), base_noise_rms


def play_mp3_blocking(mp3_bytes: bytes) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        tf.write(mp3_bytes)
        tmp_path = Path(tf.name)

    try:
        if sys.platform == "win32":
            if shutil.which("ffplay"):
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(tmp_path)],
                    check=False,
                )
            else:
                ps_cmd = (
                    "Add-Type -AssemblyName presentationCore; "
                    "$p = New-Object System.Windows.Media.MediaPlayer; "
                    f"$p.Open([Uri]'{tmp_path.resolve()}'); $p.Play(); "
                    "Start-Sleep -Milliseconds 600; "
                    "while (-not $p.NaturalDuration.HasTimeSpan) { Start-Sleep -Milliseconds 100 }; "
                    "$sec = [math]::Ceiling($p.NaturalDuration.TimeSpan.TotalSeconds); "
                    "Start-Sleep -Seconds $sec; $p.Close();"
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=False)
        else:
            if shutil.which("ffplay"):
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(tmp_path)],
                    check=False,
                )
            elif shutil.which("mpg123"):
                subprocess.run(["mpg123", "-q", str(tmp_path)], check=False)
            else:
                print("[FEHLER] Weder ffplay noch mpg123 installiert! Auf Navel: sudo apt install ffmpeg", flush=True)
    finally:
        tmp_path.unlink(missing_ok=True)


async def set_robot_expression(robot: Any, emotion: str, simulate: bool) -> None:
    if simulate or robot is None:
        return
    try:
        if emotion == "happy":
            await _maybe_await(robot.head_facial_expression(smile=0.55, happy=0.3))
        elif emotion == "sad":
            await _maybe_await(robot.head_facial_expression(sad=0.45, neutral=0.2))
        elif emotion == "thinking":
            await _maybe_await(robot.head_facial_expression(neutral=0.6, surprise=0.15))
        else:
            await _maybe_await(robot.head_facial_expression(neutral=0.7, smile=0.15))
    except Exception as e:
        print(f"[Warnung Mimik]: {e}", flush=True)


async def speak_response(robot: Any, data: dict, simulate: bool) -> None:
    tts_mode = data.get("tts_mode", "acapela")
    emotion = data.get("emotion", "happy")
    answer_text = data.get("answer_text", "")
    qr_url = data.get("short_qr_url")

    await set_robot_expression(robot, emotion, simulate)

    if qr_url:
        if simulate or robot is None:
            print(f"📱 [Augen-QR-Code]: {qr_url}", flush=True)
        else:
            try:
                await _maybe_await(robot.head_qr_code(qr_url, duration_s=20))
            except Exception as e:
                print(f"[Warnung QR-Code]: {e}", flush=True)

    if tts_mode == "acapela" and data.get("acapela_text"):
        if simulate or robot is None:
            print(f"🗣️ [Acapela + Lip-Sync ({data.get('language')}) | Mimik={emotion}]", flush=True)
        else:
            await _maybe_await(robot.say(data["acapela_text"]))
    elif data.get("audio_base64"):
        print(f"🔊 [Server-Audio ({data.get('language')}) | Mimik={emotion}] Spiele ab...", flush=True)
        mp3_bytes = base64.b64decode(data["audio_base64"])
        await asyncio.to_thread(play_mp3_blocking, mp3_bytes)
    else:
        print(f"💬 [Text-Fallback]: {answer_text}", flush=True)

    await set_robot_expression(robot, "neutral", simulate)


async def conversation_loop(
    robot: Any, server_url: str, api_key: str, session_id: str, simulate: bool
) -> None:
    p = pyaudio.PyAudio()
    device_index = _find_input_device_index(p, simulate=simulate)
    noise_rms = await asyncio.to_thread(calibrate_room_noise, p, device_index)
    endpoint = f"{server_url.rstrip('/')}/api/v1/robot/converse"

    print(f"🚀 HOMIE Voice-Loop aktiv (Server: {endpoint}). Strg+C zum Beenden.\n", flush=True)
    await set_robot_expression(robot, "neutral", simulate)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                wav_bytes, noise_rms = await asyncio.to_thread(
                    record_single_utterance_blocking, p, device_index, noise_rms
                )

                print("⏳ Sende Aufnahme zur Verarbeitung an den Server...", flush=True)
                await set_robot_expression(robot, "thinking", simulate)
                t0 = time.perf_counter()
                try:
                    resp = await client.post(
                        endpoint,
                        headers={"X-API-Key": api_key},
                        files={"audio_file": ("navel_odas1.wav", wav_bytes, "audio/wav")},
                        data={"session_id": session_id},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    print(f"❌ Server-Fehler: {e}", flush=True)
                    await set_robot_expression(robot, "neutral", simulate)
                    continue

                roundtrip_ms = round((time.perf_counter() - t0) * 1000)
                transcript = (data.get("transcript") or "").strip()
                if not transcript:
                    print("ℹ️ Keine verständliche Sprache erkannt – lausche weiter.", flush=True)
                    await set_robot_expression(robot, "neutral", simulate)
                    continue

                lang = str(data.get("language", "?")).upper()
                mode = str(data.get("response_mode", "?"))
                search_de = str(data.get("search_query_de", ""))
                score = float(data.get("best_score", 0.0))
                answer = str(data.get("answer_text", ""))

                print("=" * 68, flush=True)
                print(f"🗣️ DU HAST GESAGT ({lang}):  \"{transcript}\"", flush=True)
                if search_de and search_de != transcript:
                    print(f"🔍 DEUTSCHE SUCHE:          \"{search_de}\"", flush=True)
                print(f"🧠 VERARBEITUNG:            Modus={mode} | Quellen-Score={score:.2f} | {roundtrip_ms} ms", flush=True)
                print(f"🤖 HOMIE ANTWORTET:         \"{answer}\"", flush=True)
                print("=" * 68, flush=True)

                await speak_response(robot, data, simulate=simulate)
                await asyncio.sleep(0.4)
    finally:
        p.terminate()


async def main() -> None:
    parser = argparse.ArgumentParser(description="HOMIE Voice-Client für den Navel-Roboter")
    parser.add_argument("--server", default=os.getenv("HOMIE_SERVER_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--api-key", default=os.getenv("ROBOT_API_KEY", "homie-lokal-2026"))
    parser.add_argument("--session-id", default="navel-hildesheim-01")
    parser.add_argument("--simulate", action="store_true", help="Testmodus auf Windows-Laptop ohne Navel-Hardware")
    args = parser.parse_args()

    if args.simulate:
        await conversation_loop(None, args.server, args.api_key, args.session_id, simulate=True)
    else:
        import navel  # type: ignore[import-not-found]

        async with navel.Robot() as robot:
            await conversation_loop(robot, args.server, args.api_key, args.session_id, simulate=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeendet.")