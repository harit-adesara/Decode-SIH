import asyncio
import base64
import json
import logging
import math
import os
import struct
import time
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from agent.graph import workflow
from agent.speech import stt, tts_to_pcm, clean_text_for_speech
from agent.guardrails import check_input
from agent.nodes import send_sms_node, end_call_node
from agent.tools import hangup_exotel_call

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bharatswasthya.main")

app = FastAPI(
    title="BharatSwasthya AI",
    description="Multilingual Voice-First Healthcare Assistant for India (Exotel VoiceBot)",
    version="2.0.0",
)

# Audio format: 8 kHz, 16-bit, mono Linear PCM (Telephony standard)
SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2
CHANNELS = 1

# VAD (Voice Activity Detection) parameters tuned for natural conversational pauses (2-3s)
VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "260"))
SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "1800"))
MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "300"))
MAX_UTTERANCE_MS = int(os.getenv("VAD_MAX_UTTERANCE_MS", "12000"))

# Response timeouts & Inactivity watchdog
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "18.0"))
INACTIVITY_PROMPT_SECONDS = float(os.getenv("INACTIVITY_PROMPT_SECONDS", "25.0"))
INACTIVITY_HANGUP_SECONDS = float(os.getenv("INACTIVITY_HANGUP_SECONDS", "50.0"))
WATCHDOG_POLL_SECONDS = 1.0

WELCOME_TEXT = (
    "BharatSwasthya AI mein aapka swagat hai. "
    "Aapko jis bhasha mein baat karni hai, "
    "kripya us bhasha ka naam boliye jaise Hindi, Gujarati ya English."
)

LANGUAGE_RETRY_TEXT = (
    "Mujhe aapki awaaz sunai nahi di. "
    "Kripya apni bhasha ka naam boliye jaise Hindi, Gujarati, ya English."
)

RESUME_RETRY_TEXT = "Kripya apna sawal dobara boliye."

SUPPORTED_LANGUAGES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Gujarati": "gu-IN",
    "Bengali": "bn-IN",
    "Marathi": "mr-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
}

LANGUAGE_INTROS = {
    "gu-IN": "નમસ્તે! હું ભારત સ્વાસ્થ્ય AI છું. હું તમને સરકારી યોજનાઓ અને હોસ્પિટલો શોધવામાં મદદ કરી શકું છું. તમે શું જાણવા માંગો છો?",
    "hi-IN": "Namaste! Main BharatSwasthya AI hoon. Main aapko sarkari swasthya yojanaon aur aspatalon ke baare mein bata sakta hoon. Aap kya jaanna chahte hain?",
    "en-IN": "Hello! I am BharatSwasthya AI. I can assist you with government health schemes and nearby hospitals. What would you like to know?",
    "mr-IN": "Namaskar! Mi BharatSwasthya AI ahe. Mi tumhala sarkari arogya yojana ani rugnalayanbaddal mahiti deu shakto. Apan kay vicharu ichhita?",
    "bn-IN": "Nomoshkar! Aami BharatSwasthya AI. Aami apnake shorkari shastho prokoplpo ebong hashpatal somporke jante shahajjo korte pari. Apnar ki jante chan?",
    "ta-IN": "Vanakkam! Naan BharatSwasthya AI. Arasaanga maruthuva thittangal matrum maruthuvamanaigal patriya thagavalgalaip pera naan udhavugiren. Ungalukku enna thevai?",
    "te-IN": "Namaskaram! Nenu BharatSwasthya AI. Prabhuthva arogya pathakalu mariyu aashupatrula gurinchi meeku sahayam cheyagalanu. Meeku emi telusukovali?",
    "kn-IN": "Namaskara! Naanu BharatSwasthya AI. Sarkari arogya yojanegalu mattu aaspathregala bagge mahiti needalu sahaya maduttene. Nimage enu bekagide?",
    "ml-IN": "Namaskaram! Njan BharatSwasthya AI. Sarkkar arogya padhathikale kurichum aasupathrikale kurichum ariyikkan sahayikkam. Enthannu ariyendath?",
}

CLOSING_TEXTS = {
    "hi-IN": "BharatSwasthya AI se baat karne ke liye dhanyavaad. Apna khayal rakhiye. Namaste!",
    "en-IN": "Thank you for calling BharatSwasthya AI. Take care and stay healthy. Goodbye!",
    "gu-IN": "BharatSwasthya AI sathe vaat karva badal aabhar. Potanu dhyan rakhjo. Namaste!",
    "bn-IN": "BharatSwasthya AI-te call korar jonno dhonnobad. Bhalo thakben. Nomoshkar!",
    "mr-IN": "BharatSwasthya AI shi bollyabaddal dhanyavaad. Aplya arogyachi kalji ghya. Namaskar!",
    "ta-IN": "BharatSwasthya AI-kku azhaithadharku nandri. Udalnalathil gavanamaaga irungal. Vanakkam!",
    "te-IN": "BharatSwasthya AI ki call chesinanduku dhanyavadamulu. Mee arogyanni jagrattaga choosukondi. Namaskaram!",
    "kn-IN": "BharatSwasthya AI ge kare madiddakkagi dhanyavadagalu. Nimma arogyavannu nodikolli. Namaskara!",
    "ml-IN": "BharatSwasthya AI-yilekku vilichathinu nanni. Arogyam sradhikkuka. Namaskaram!",
}

AGENT_TIMEOUT_TEXTS = {
    "hi-IN": "Mujhe iska jawab dhoondhne mein thoda samay lag raha hai. Kripya apna sawal dobara boliye.",
    "en-IN": "I'm taking a little longer than usual to find that. Could you please repeat your question?",
    "gu-IN": "Aa mahiti sodhva mate mane thodo samay lagi rahyo chhe. Krupa karine tamaro prashn ferithi bolo.",
}

INACTIVITY_PROMPT_TEXTS = {
    "hi-IN": "Hello, kya aap wahin hain? Agar aapka koi aur sawal ho to kripya boliye.",
    "en-IN": "Hello, are you still there? Please go ahead if you have another question.",
    "gu-IN": "Hello, tame have pan line par cho? Koi sawal hoy to krupa karine bolo.",
}

INACTIVITY_GOODBYE_TEXTS = {
    "hi-IN": "Mujhe koi jawab nahi mila, isliye main call band kar raha hoon. Zaroorat padne par kripya phir se call kijiye. Namaste!",
    "en-IN": "Since I haven't heard a response, I'll end the call now. Please call again if you need help. Goodbye!",
    "gu-IN": "Mane koi javab malyo nathi, etle hu call band karu chhu. Jarur padye krupa karine ferithi call karo. Namaste!",
}





async def identify_language(user_text: str) -> str:
    """Identify caller's spoken language using Gemini LLM."""
    if not user_text or not user_text.strip():
        return "Hindi"

    from google import genai
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            prompt = (
                "You are an Indian language identification specialist for a voice healthcare assistant. "
                "Identify the primary language spoken in this utterance. "
                "Return ONLY the language name strictly from this list: "
                "Gujarati, Hindi, English, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam.\n\n"
                f"Utterance: \"{user_text}\"\nLanguage:"
            )
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-3.1-flash-lite",
                    contents=prompt,
                    config={"temperature": 0.0},
                )
            except Exception:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config={"temperature": 0.0},
                )
            lang = (response.text or "").strip()
            for valid in SUPPORTED_LANGUAGES.keys():
                if valid.lower() in lang.lower():
                    return valid
        except Exception as e:
            logger.warning("Gemini language identification fallback failed: %s", e)

    return "Hindi"


# ITU-T G.711 mu-law tables
MULAW_BIAS = 0x84
MULAW_CLIP = 32635
MULAW_DECODE_TABLE = []
for b in range(256):
    inv = ~b & 0xFF
    sign = inv & 0x80
    exponent = (inv >> 4) & 0x07
    mantissa = inv & 0x0F
    sample = ((mantissa << 3) + MULAW_BIAS) << exponent
    sample -= MULAW_BIAS
    if sign != 0:
        sample = -sample
    MULAW_DECODE_TABLE.append(sample)


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Decode 8-bit G.711 mu-law audio to 16-bit linear PCM."""
    if not mulaw_bytes:
        return b""
    samples = [MULAW_DECODE_TABLE[b] for b in mulaw_bytes]
    return struct.pack(f"<{len(samples)}h", *samples)


def pcm16_to_mulaw(pcm: bytes) -> bytes:
    """Encode 16-bit linear PCM audio to 8-bit G.711 mu-law."""
    if not pcm:
        return b""
    count = len(pcm) // 2
    if count == 0:
        return b""
    shorts = struct.unpack(f"<{count}h", pcm[: count * 2])
    out = bytearray(count)
    for i, sample in enumerate(shorts):
        sign = 0
        if sample < 0:
            sample = -sample
            sign = 0x80
        if sample > MULAW_CLIP:
            sample = MULAW_CLIP
        sample += MULAW_BIAS
        exponent = 7
        for exp in range(7):
            if sample <= (0x80 << (exp + 1)) - 1:
                exponent = exp
                break
        mantissa = (sample >> (exponent + 3)) & 0x0F
        out[i] = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return bytes(out)


def get_language_code(language: str) -> str:
    return SUPPORTED_LANGUAGES.get(language, "hi-IN")


def get_language_intro(language_code: str) -> str:
    return LANGUAGE_INTROS.get(language_code, LANGUAGE_INTROS["hi-IN"])


def get_closing_text(language_code: str) -> str:
    return CLOSING_TEXTS.get(language_code, CLOSING_TEXTS["hi-IN"])


def get_agent_timeout_text(language_code: str) -> str:
    return AGENT_TIMEOUT_TEXTS.get(language_code, AGENT_TIMEOUT_TEXTS["hi-IN"])


def get_inactivity_prompt_text(language_code: str) -> str:
    return INACTIVITY_PROMPT_TEXTS.get(language_code, INACTIVITY_PROMPT_TEXTS["hi-IN"])


def get_inactivity_goodbye_text(language_code: str) -> str:
    return INACTIVITY_GOODBYE_TEXTS.get(language_code, INACTIVITY_GOODBYE_TEXTS["hi-IN"])


def get_agent_response(result: dict) -> Optional[str]:
    """Extract latest assistant response text from LangGraph output."""
    if not isinstance(result, dict):
        return None

    messages = result.get("messages", [])
    if messages:
        for msg in reversed(messages):
            msg_type = getattr(msg, "type", "")
            is_ai = (
                msg_type in ("ai", "AIMessage")
                or type(msg).__name__ == "AIMessage"
                or (isinstance(msg, dict) and msg.get("role") in ("assistant", "ai"))
            )
            if is_ai:
                content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
                if content:
                    if isinstance(content, list):
                        text = "".join(
                            p if isinstance(p, str) else (p.get("text", "") if isinstance(p, dict) else str(p))
                            for p in content
                        ).strip()
                    else:
                        text = str(content).strip()
                    if text:
                        return text
    return None


def rms_level(pcm: bytes) -> int:
    """Pure-Python RMS calculation for voice activity detection."""
    if not pcm:
        return 0
    count = len(pcm) // SAMPLE_WIDTH
    if count == 0:
        return 0
    shorts = struct.unpack(f"<{count}h", pcm[: count * SAMPLE_WIDTH])
    sum_squares = sum(s * s for s in shorts)
    return int(math.isqrt(sum_squares // count))


class SpeechBuffer:
    """VAD buffer that tracks speech boundaries and caller activity."""

    def __init__(self):
        self.reset()
        self.prebuffer: list[bytes] = []
        self.last_voice_ts = time.monotonic()

    def reset(self):
        self.parts: list[bytes] = []
        self.in_speech = False
        self.speech_ms = 0
        self.silence_ms = 0
        self.total_ms = 0

    def clear_all(self):
        self.reset()
        self.prebuffer.clear()

    def add(self, pcm: bytes) -> Optional[bytes]:
        if not pcm:
            return None

        chunk_ms = max(1, int(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000))
        level = rms_level(pcm)
        voice = level >= VAD_RMS_THRESHOLD

        if voice:
            self.last_voice_ts = time.monotonic()

        if not self.in_speech:
            self.prebuffer.append(pcm)
            while sum(len(x) for x in self.prebuffer) > 1600:
                self.prebuffer.pop(0)

            if voice:
                self.in_speech = True
                self.parts = list(self.prebuffer)
                self.prebuffer.clear()
                self.speech_ms = chunk_ms
                self.silence_ms = 0
                self.total_ms = sum(
                    int(len(x) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000)
                    for x in self.parts
                )
                logger.debug("Speech detected (RMS: %d), listening...", level)
            return None

        self.parts.append(pcm)
        self.total_ms += chunk_ms

        if voice:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
        else:
            self.silence_ms += chunk_ms

        if self.speech_ms >= MIN_SPEECH_MS and self.silence_ms >= SILENCE_MS:
            audio = b"".join(self.parts)
            logger.info("Utterance captured: %d ms (%d bytes)", self.total_ms, len(audio))
            self.clear_all()
            return audio

        if self.total_ms >= MAX_UTTERANCE_MS:
            audio = b"".join(self.parts)
            logger.info("Max utterance reached: %d ms (%d bytes)", self.total_ms, len(audio))
            self.clear_all()
            return audio

        return None


AUDIO_CACHE: dict[str, bytes] = {}


async def send_exotel_media(
    websocket: WebSocket,
    stream_sid: str,
    pcm: bytes,
    encoding: str = "base64",
):
    """Stream audio chunks to Exotel with realistic real-time pacing."""
    if not pcm:
        return

    is_mulaw = "mulaw" in encoding.lower() or "ulaw" in encoding.lower() or "pcmu" in encoding.lower()

    if is_mulaw:
        payload_data = pcm16_to_mulaw(pcm)
        chunk_bytes = 800  # 100ms of 8-bit 8kHz audio = 800 bytes
    else:
        payload_data = pcm
        chunk_bytes = 1600  # 100ms of 16-bit 8kHz audio = 1600 bytes

    try:
        for offset in range(0, len(payload_data), chunk_bytes):
            chunk = payload_data[offset: offset + chunk_bytes]
            message = {
                "event": "media",
                "stream_sid": stream_sid,
                "streamSid": stream_sid,
                "media": {
                    "payload": base64.b64encode(chunk).decode("ascii"),
                },
            }
            await websocket.send_text(json.dumps(message))
            await asyncio.sleep(0.090)

        # Send mark event when bot finishes playing audio clip
        mark_msg = {
            "event": "mark",
            "stream_sid": stream_sid,
            "streamSid": stream_sid,
            "mark": {"name": "bot_playback_complete"},
        }
        await websocket.send_text(json.dumps(mark_msg))
    except Exception as e:
        logger.debug("Socket send finished or closed: %s", e)


async def speak(
    websocket: WebSocket,
    stream_sid: str,
    text: str,
    language_code: str,
    encoding: str = "base64",
):
    """Convert text to speech and stream audio to caller."""
    if not text or not text.strip():
        return

    cleaned_text = clean_text_for_speech(text)
    logger.info("Bot speaking [%s]: %s", language_code, cleaned_text)

    try:
        cache_key = f"{language_code}:{cleaned_text}"
        if cache_key in AUDIO_CACHE:
            pcm = AUDIO_CACHE[cache_key]
        else:
            pcm = await tts_to_pcm(cleaned_text, language_code)
            if pcm:
                AUDIO_CACHE[cache_key] = pcm

        if pcm:
            await send_exotel_media(websocket, stream_sid, pcm, encoding=encoding)
    except Exception as e:
        logger.error("Speak error for [%s]: %s", language_code, e)


async def select_language_turn(
    websocket: WebSocket,
    stream_sid: str,
    audio: bytes,
    encoding: str = "base64",
) -> tuple[str, str]:
    """Transcribe caller's language choice -> acknowledge and greet in that language."""
    try:
        user_text = await asyncio.to_thread(stt, audio, "unknown")
        user_text = (user_text or "").strip()
        logger.info("Caller language choice: %s", user_text)

        if not user_text:
            await speak(websocket, stream_sid, LANGUAGE_RETRY_TEXT, "hi-IN", encoding=encoding)
            return "Hindi", "hi-IN"

        language = await identify_language(user_text)
        language_code = get_language_code(language)
        logger.info("Selected language: %s (%s)", language, language_code)

        intro_text = get_language_intro(language_code)
        await speak(websocket, stream_sid, intro_text, language_code, encoding=encoding)
        return language, language_code

    except Exception:
        logger.exception("Language selection turn failed")
        await speak(websocket, stream_sid, LANGUAGE_INTROS["hi-IN"], "hi-IN", encoding=encoding)
        return "Hindi", "hi-IN"


async def conversation_turn(
    websocket: WebSocket,
    stream_sid: str,
    call_sid: str,
    phone_number: str,
    current_language: str,
    current_language_code: str,
    audio: bytes,
    turn_count: int,
    encoding: str = "base64",
) -> tuple[str, str, bool]:
    """Execute dynamic healthcare conversation turn with LangGraph workflow."""
    language = current_language or "Hindi"
    language_code = current_language_code or "hi-IN"

    try:
        user_text = await asyncio.to_thread(stt, audio, language_code)
        user_text = (user_text or "").strip()
        logger.info("Caller query [Turn %d]: %s", turn_count, user_text)

        if not user_text:
            await speak(websocket, stream_sid, RESUME_RETRY_TEXT, language_code, encoding=encoding)
            return language, language_code, False

        # LLM-based input safety & prompt injection check
        guard_result = await asyncio.to_thread(check_input, user_text)
        if guard_result.get("unsafe"):
            logger.warning("Caller input flagged by LLM safety guardrail: %s", guard_result.get("reason"))
            decline_text = (
                "હું આવા પ્રકારના પ્રશ્નોમાં મદદ કરી શકતો નથી. કૃપા કરીને સ્વાસ્થ્ય અથવા હોસ્પિટલ સંબંધિત પ્રશ્ન પૂછો."
                if language_code == "gu-IN"
                else "Main is prakar ke sawalon mein madad nahi kar sakta. Kripya swasthya ya hospital sambandhit sawal poochiye."
            )
            await speak(websocket, stream_sid, decline_text, language_code, encoding=encoding)
            return language, language_code, False

        # Dynamic LLM language detection if language switches
        if len(user_text.split()) >= 2:
            detected_lang = await identify_language(user_text)
            if detected_lang and detected_lang != language:
                language = detected_lang
                language_code = get_language_code(language)
                logger.info("Language dynamically updated via LLM: %s (%s)", language, language_code)

        config = {
            "configurable": {
                "thread_id": call_sid,
            }
        }

        turn_input = {
            "messages": [{"role": "user", "content": user_text}],
            "call_sid": call_sid,
            "phone_number": phone_number,
            "language": language,
            "is_emergency": False,
            "emergency_type": None,
            "sms_sent": False,
            "call_ended": False,
        }

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(workflow.invoke, turn_input, config=config),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("Agent exceeded %.0fs timeout | call_sid=%s", AGENT_TIMEOUT_SECONDS, call_sid)
            await speak(websocket, stream_sid, get_agent_timeout_text(language_code), language_code, encoding=encoding)
            return language, language_code, False

        response_text = get_agent_response(result)
        should_end = bool(result.get("call_ended", False))

        if response_text:
            if "CALL_TERMINATED:" in response_text or "CALL_ENDED:" in response_text:
                response_text = response_text.replace("CALL_TERMINATED:", "").replace("CALL_ENDED:", "").strip()
                should_end = True

            await speak(websocket, stream_sid, response_text, language_code, encoding=encoding)
        else:
            closing = get_closing_text(language_code)
            await speak(websocket, stream_sid, closing, language_code, encoding=encoding)
            should_end = True

        if should_end:
            await asyncio.sleep(1.2)
            return language, language_code, True

        return language, language_code, False

    except Exception:
        logger.exception("Conversation turn failed")
        await speak(websocket, stream_sid, RESUME_RETRY_TEXT, language_code, encoding=encoding)
        return language, language_code, False


async def finalize_call(call_sid: str, phone_number: str) -> None:
    """Send post-call summary SMS upon completion."""
    if not call_sid:
        return

    try:
        config = {"configurable": {"thread_id": call_sid}}
        snapshot = await asyncio.to_thread(workflow.get_state, config)
        final_state = dict(snapshot.values) if snapshot and snapshot.values else {}
        final_state["phone_number"] = final_state.get("phone_number") or phone_number

        final_state = await asyncio.to_thread(send_sms_node, final_state)
        final_state = end_call_node(final_state)

        if final_state.get("sms_sent"):
            logger.info("Post-call summary SMS sent | call_sid=%s", call_sid)
        else:
            logger.warning(
                "Post-call summary SMS not sent | call_sid=%s | reason=%s",
                call_sid,
                final_state.get("sms_error"),
            )
    except Exception:
        logger.exception("Post-call SMS/end-call handling failed | call_sid=%s", call_sid)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "BharatSwasthya AI",
        "telephony": "Exotel VoiceBot",
    }


@app.websocket("/exotel/media")
async def exotel_media(websocket: WebSocket):
    """Exotel VoiceBot bidirectional WebSocket endpoint."""
    await websocket.accept()
    logger.info("Exotel WebSocket connected")

    stream_sid = None
    call_sid = None
    phone_number = ""
    language = None
    language_code = None
    turn_count = 0
    encoding = "base64"

    speech_buffer = SpeechBuffer()
    is_speaking = False
    last_bot_speech_end = time.monotonic()
    incoming_queue = asyncio.Queue()
    call_ended_event = asyncio.Event()

    async def socket_reader():
        """Continuously read WebSocket frames to prevent TCP socket stalls."""
        try:
            while True:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)
                await incoming_queue.put(message)
        except (WebSocketDisconnect, asyncio.CancelledError):
            await incoming_queue.put(None)
        except Exception as e:
            logger.debug("Socket reader closed: %s", e)
            await incoming_queue.put(None)

    reader_task = asyncio.create_task(socket_reader())

    async def inactivity_watchdog():
        """Handles prolonged caller silence politely without prematurely dropping calls."""
        nonlocal is_speaking
        warned = False
        try:
            while not call_ended_event.is_set():
                await asyncio.sleep(WATCHDOG_POLL_SECONDS)

                if is_speaking or not stream_sid:
                    continue

                idle = time.monotonic() - speech_buffer.last_voice_ts
                lang_code = language_code or "hi-IN"

                if idle >= INACTIVITY_HANGUP_SECONDS:
                    logger.info("Caller silent for %.0fs, ending call | call_sid=%s", idle, call_sid)
                    is_speaking = True
                    try:
                        await speak(
                            websocket,
                            stream_sid,
                            get_inactivity_goodbye_text(lang_code),
                            lang_code,
                            encoding=encoding,
                        )
                    finally:
                        is_speaking = False
                        last_bot_speech_end = time.monotonic()

                    await finalize_call(call_sid, phone_number)
                    try:
                        await asyncio.to_thread(hangup_exotel_call, call_sid)
                    except Exception:
                        pass

                    await incoming_queue.put(None)
                    return

                if idle >= INACTIVITY_PROMPT_SECONDS and not warned:
                    warned = True
                    logger.info("Caller silent for %.0fs, checking in | call_sid=%s", idle, call_sid)
                    is_speaking = True
                    try:
                        await speak(
                            websocket,
                            stream_sid,
                            get_inactivity_prompt_text(lang_code),
                            lang_code,
                            encoding=encoding,
                        )
                    finally:
                        await asyncio.sleep(0.400)
                        is_speaking = False
                        last_bot_speech_end = time.monotonic()
                elif idle < INACTIVITY_PROMPT_SECONDS:
                    warned = False
        except asyncio.CancelledError:
            pass

    watchdog_task = asyncio.create_task(inactivity_watchdog())

    try:
        while True:
            message = await incoming_queue.get()
            if message is None:
                break

            event = message.get("event")

            if event == "connected":
                logger.info("Exotel 'connected' event received")

            elif event == "start":
                start_data = message.get("start", {})
                stream_sid = (
                    message.get("stream_sid")
                    or message.get("streamSid")
                    or message.get("stream sid")
                    or start_data.get("stream_sid")
                    or start_data.get("streamSid")
                    or start_data.get("stream sid")
                )
                call_sid = (
                    start_data.get("call_sid")
                    or start_data.get("callSid")
                    or start_data.get("call sid")
                    or message.get("call_sid")
                    or message.get("callSid")
                    or stream_sid
                )
                phone_number = (
                    start_data.get("from")
                    or start_data.get("From")
                    or message.get("from")
                    or ""
                )

                media_format = (
                    start_data.get("media_format")
                    or start_data.get("mediaFormat")
                    or start_data.get("media format")
                    or message.get("media_format")
                    or message.get("mediaFormat")
                    or {}
                )
                if isinstance(media_format, dict) and media_format.get("encoding"):
                    encoding = media_format.get("encoding")

                logger.info(
                    "Call started | call_sid=%s | stream_sid=%s | caller=%s | format=%s",
                    call_sid,
                    stream_sid,
                    phone_number,
                    media_format,
                )

                if not stream_sid:
                    stream_sid = call_sid or "exotel_stream"

                # Greet caller and ask for language
                is_speaking = True
                try:
                    await speak(websocket, stream_sid, WELCOME_TEXT, "hi-IN", encoding=encoding)
                finally:
                    await asyncio.sleep(0.350)
                    is_speaking = False
                    last_bot_speech_end = time.monotonic()
                    speech_buffer.clear_all()
                    speech_buffer.last_voice_ts = time.monotonic()

            elif event == "media":
                media_data = message.get("media", {})
                payload = media_data.get("payload")
                if not payload:
                    continue

                if not stream_sid:
                    stream_sid = message.get("stream_sid") or message.get("streamSid") or "exotel_stream"

                # Discard audio during bot playback AND cooldown period to prevent self-echo and buffer collisions
                if is_speaking or (time.monotonic() - last_bot_speech_end < 0.40):
                    continue

                try:
                    raw_audio = base64.b64decode(payload)
                    is_mulaw = "mulaw" in encoding.lower() or "ulaw" in encoding.lower() or "pcmu" in encoding.lower()
                    pcm = mulaw_to_pcm16(raw_audio) if is_mulaw else raw_audio

                    utterance = speech_buffer.add(pcm)
                except Exception:
                    logger.exception("Bad audio frame skipped | call_sid=%s", call_sid)
                    continue

                if utterance is None:
                    continue

                is_speaking = True
                try:
                    if language_code is None:
                        # Turn 0: Identify language and greet in chosen language
                        language, language_code = await select_language_turn(
                            websocket,
                            stream_sid,
                            utterance,
                            encoding=encoding,
                        )
                    else:
                        # Turn 1+: Direct healthcare consultation
                        turn_count += 1
                        language, language_code, should_end = await conversation_turn(
                            websocket,
                            stream_sid,
                            call_sid or "unknown_call",
                            phone_number,
                            language,
                            language_code,
                            utterance,
                            turn_count,
                            encoding=encoding,
                        )
                        if should_end:
                            logger.info("Call concluded normally with farewell | call_sid=%s", call_sid)
                            await finalize_call(call_sid, phone_number)
                            try:
                                await asyncio.to_thread(hangup_exotel_call, call_sid)
                            except Exception:
                                pass
                            break
                finally:
                    await asyncio.sleep(0.350)
                    is_speaking = False
                    last_bot_speech_end = time.monotonic()
                    speech_buffer.clear_all()
                    speech_buffer.last_voice_ts = time.monotonic()

            elif event == "dtmf":
                logger.info("DTMF digit received: %s", message.get("dtmf"))

            elif event == "mark":
                logger.debug("Mark event: %s", message.get("mark"))

            elif event == "stop":
                logger.info("Exotel stream stopped: %s", message.get("stop"))
                await finalize_call(call_sid, phone_number)
                break

    except WebSocketDisconnect:
        logger.info("Exotel WebSocket disconnected | call_sid=%s", call_sid)
    except Exception as e:
        logger.info("Exotel WebSocket session closed: %s", e)
    finally:
        call_ended_event.set()
        watchdog_task.cancel()
        reader_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)