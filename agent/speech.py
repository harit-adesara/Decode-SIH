import io
import logging
import os
import re
import wave
import asyncio
import time
from typing import Generator, Optional
from dotenv import load_dotenv
from google import genai
import miniaudio
import edge_tts

load_dotenv()

logger = logging.getLogger("bharatswasthya.speech")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
sarvam_client = None
_sarvam_available = bool(SARVAM_API_KEY)
_sarvam_last_error_ts = 0.0
SARVAM_COOLDOWN_SECONDS = 300.0  # 5 minute cooldown on failure

if SARVAM_API_KEY:
    try:
        from sarvamai import SarvamAI
        sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    except Exception as e:
        logger.warning("Failed to initialize SarvamAI client: %s", e)
        _sarvam_available = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning("Failed to initialize Gemini client for speech fallback: %s", e)

EDGE_VOICE_MAP = {
    "hi-IN": "hi-IN-MadhurNeural",
    "gu-IN": "gu-IN-NiranjanNeural",
    "en-IN": "en-IN-PrabhatNeural",
    "mr-IN": "mr-IN-ManoharNeural",
    "bn-IN": "bn-IN-BashkarNeural",
    "ta-IN": "ta-IN-ValluvarNeural",
    "te-IN": "te-IN-MohanNeural",
    "kn-IN": "kn-IN-GaganNeural",
    "ml-IN": "ml-IN-MidhunNeural",
}

AUDIO_CACHE: dict[str, bytes] = {}


def _is_sarvam_usable() -> bool:
    """Check if Sarvam is enabled and not in cooldown due to quota/network failures."""
    global _sarvam_available, _sarvam_last_error_ts
    if not sarvam_client or not _sarvam_available:
        if _sarvam_last_error_ts and (time.monotonic() - _sarvam_last_error_ts > SARVAM_COOLDOWN_SECONDS):
            # Attempt recovery after cooldown
            _sarvam_available = True
            return True
        return False
    return True


def _disable_sarvam_temporarily(reason: str):
    """Trip Sarvam circuit breaker on error to avoid repeated multi-second timeouts."""
    global _sarvam_available, _sarvam_last_error_ts
    logger.warning("Disabling Sarvam temporarily due to error: %s", reason)
    _sarvam_available = False
    _sarvam_last_error_ts = time.monotonic()


def clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting, symbols, brackets, and links so TTS produces natural speech."""
    if not text:
        return ""
    t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    t = re.sub(r"[*_#`~>|]", " ", t)
    t = re.sub(r"(\d+)\.\s+", r"\1 ", t)  # turn "1. " into "1 "
    t = re.sub(r"[-–—]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def wav_to_pcm(wav_bytes: bytes) -> bytes:
    """Extract raw 16-bit 8kHz PCM from WAV bytes."""
    if not wav_bytes:
        return b""
    if wav_bytes.startswith(b"RIFF"):
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                return wf.readframes(wf.getnframes())
        except Exception as e:
            logger.warning("Error reading WAV header: %s", e)
            return wav_bytes[44:] if len(wav_bytes) > 44 else wav_bytes
    return wav_bytes


async def _generate_edge_tts_pcm(text: str, language_code: str) -> bytes:
    """Generate 8kHz 16-bit mono PCM using Edge Neural TTS fallback."""
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return b""
    voice = EDGE_VOICE_MAP.get(language_code, "hi-IN-MadhurNeural")
    try:
        communicate = edge_tts.Communicate(cleaned, voice)
        mp3_data = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                mp3_data.extend(chunk.get("data", b""))

        if not mp3_data:
            return b""

        decoded = miniaudio.decode(
            bytes(mp3_data),
            nchannels=1,
            sample_rate=8000,
            output_format=miniaudio.SampleFormat.SIGNED16,
        )
        return decoded.samples.tobytes()
    except Exception as e:
        logger.error("Edge TTS generation failed for [%s]: %s", language_code, e)
        return b""


def stt_fallback_gemini(audio_bytes: bytes, language_code: str = "unknown") -> str:
    """Gemini multimodal audio transcription fallback."""
    if not gemini_client or not audio_bytes:
        return ""

    try:
        if not audio_bytes.startswith(b"RIFF"):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(audio_bytes)
            wav_bytes = buf.getvalue()
        else:
            wav_bytes = audio_bytes

        prompt = (
            "You are a speech-to-text system. Transcribe the spoken audio into text in the caller's spoken Indian language/script. "
            "Output ONLY the exact verbatim transcript. If there is only silence or background noise, return an empty string. "
            "Do not include timestamps, line numbers, or explanations."
        )
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=[
                    genai.types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    prompt,
                ],
                config={"temperature": 0.0},
            )
        except Exception:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[
                    genai.types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    prompt,
                ],
                config={"temperature": 0.0},
            )
        transcript = (response.text or "").strip()
        transcript = re.sub(r"^```[a-z]*\s*", "", transcript)
        transcript = re.sub(r"\s*```$", "", transcript).strip()
        # Clean up any timestamps like "00:00:00 - 00:00:02"
        transcript = re.sub(r"^\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}\s*", "", transcript)
        transcript = re.sub(r"^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\s*", "", transcript).strip()
        return transcript
    except Exception as e:
        logger.warning("Gemini STT fallback failed: %s", e)
        return ""


def stt(audio_bytes: bytes, language_code: str = "unknown") -> str:
    """
    Transcribe audio bytes (e.g. from Exotel audio stream) to text.
    Uses Sarvam STT if available and active, otherwise falls back instantly to Gemini STT.
    """
    if not audio_bytes:
        return ""

    if not audio_bytes.startswith(b"RIFF"):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(audio_bytes)
        wav_bytes = buf.getvalue()
    else:
        wav_bytes = audio_bytes

    if _is_sarvam_usable():
        try:
            audio_buffer = io.BytesIO(wav_bytes)
            audio_buffer.name = "recording.wav"

            response = sarvam_client.speech_to_text.transcribe(
                file=audio_buffer,
                model="saaras:v3",
                language_code=language_code if language_code != "unknown" else "unknown",
            )
            transcript = (response.transcript or "").strip()
            if transcript:
                return transcript
        except Exception as e:
            _disable_sarvam_temporarily(str(e))

    return stt_fallback_gemini(wav_bytes, language_code)


def tts_stream(text: str, language_code: str, speaker: str = "shubh") -> Generator[bytes, None, None]:
    """
    Stream audio chunks from Sarvam TTS with seamless Edge Neural TTS fallback.
    """
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return

    if _is_sarvam_usable():
        try:
            audio_stream = sarvam_client.text_to_speech.convert_stream(
                text=cleaned,
                language_code=language_code,
                model="bulbul:v3",
                speaker=speaker,
                speech_sample_rate=8000,
                output_audio_codec="wav",
            )
            for chunk in audio_stream:
                if chunk:
                    yield chunk
            return
        except Exception as e:
            _disable_sarvam_temporarily(str(e))

    try:
        pcm = asyncio.run(_generate_edge_tts_pcm(cleaned, language_code))
        if pcm:
            yield pcm
    except Exception as e:
        logger.error("Edge TTS fallback failed: %s", e)


async def tts_to_pcm(text: str, language_code: str = "hi-IN", speaker: str = "shubh") -> bytes:
    """
    Convert text to 8kHz 16-bit mono linear PCM bytes.
    Checks memory cache first, tries Sarvam TTS if enabled, falls back to Edge Neural TTS.
    """
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return b""

    cache_key = f"{language_code}:{cleaned}"
    if cache_key in AUDIO_CACHE:
        return AUDIO_CACHE[cache_key]

    if _is_sarvam_usable():
        try:
            raw_audio = await asyncio.to_thread(
                lambda: b"".join(
                    sarvam_client.text_to_speech.convert_stream(
                        text=cleaned,
                        language_code=language_code,
                        model="bulbul:v3",
                        speaker=speaker,
                        speech_sample_rate=8000,
                        output_audio_codec="wav",
                    )
                )
            )
            if raw_audio:
                pcm = wav_to_pcm(raw_audio)
                if pcm:
                    AUDIO_CACHE[cache_key] = pcm
                    return pcm
        except Exception as e:
            _disable_sarvam_temporarily(str(e))

    pcm = await _generate_edge_tts_pcm(cleaned, language_code)
    if pcm:
        AUDIO_CACHE[cache_key] = pcm
        return pcm

    return b""