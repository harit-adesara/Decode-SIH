# import os
# import time
# import uuid

# import requests
# from fastapi import FastAPI, Request
# from fastapi.responses import Response, StreamingResponse
# from twilio.twiml.voice_response import VoiceResponse
# from langgraph.graph import Command

# from agent.graph import workflow
# from agent.speech import stt as sarvam_stt, tts_stream as sarvam_tts_stream


# TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
# TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]

# # Public base URL of this FastAPI app (e.g. your Render URL), so Twilio
# # can fetch the audio clips we generate with Sarvam TTS.
# BASE_URL = os.environ["BASE_URL"].rstrip("/")

# app = FastAPI()

# # In-memory record of pending TTS requests, keyed by a random id.
# # Registering a request is instant - the actual Sarvam TTS call only
# # happens once Twilio fetches the audio URL, and is streamed straight
# # through as it's generated. Nothing is pre-generated, buffered in
# # full, or written to disk.
# TTS_REQUESTS: dict[str, tuple[str, str]] = {}


# def get_interrupt_message(result):

#     interrupts = result.get("__interrupt__")

#     if not interrupts:
#         return None

#     interrupt_value = interrupts[0].value

#     interrupt_type = interrupt_value.get("type")

#     if interrupt_type == "human_question":

#         return interrupt_value.get("question")

#     if interrupt_type == "final_response":

#         return interrupt_value.get("message")

#     return None

# SUPPORTED_LANGUAGES = {
#     "English": "en-IN",
#     "Hindi": "hi-IN",
#     "Gujarati": "gu-IN",
#     "Bengali": "bn-IN",
#     "Marathi": "mr-IN",
#     "Tamil": "ta-IN",
#     "Telugu": "te-IN",
#     "Kannada": "kn-IN",
#     "Malayalam": "ml-IN",
# }

# def get_language_code(language: str) -> str:
#     """
#     Maps a detected language name to its BCP-47 code. These codes are
#     used for both Sarvam STT/TTS and happen to match Twilio's own
#     language codes, so the same mapping serves both.
#     """

#     return SUPPORTED_LANGUAGES.get(
#         language,
#         "hi-IN"
#     )


# def get_audio_url(text: str, language_code: str) -> str:
#     """
#     Registers a pending TTS request and returns a URL for Twilio's
#     <Play> to fetch. This is instant - the Sarvam call itself only
#     happens when that URL is actually requested, in get_audio() below.
#     """

#     audio_id = uuid.uuid4().hex

#     TTS_REQUESTS[audio_id] = (text, language_code)

#     return f"{BASE_URL}/audio/{audio_id}"


# @app.get("/audio/{audio_id}")
# async def get_audio(audio_id: str):
#     """
#     Streams a Sarvam TTS clip to Twilio as it's generated. The
#     request is only looked up here, so generation starts the moment
#     Twilio asks for it, and playback can begin before the clip
#     finishes generating.
#     """

#     request_data = TTS_REQUESTS.pop(audio_id, None)

#     if request_data is None:
#         return Response(status_code=404)

#     text, language_code = request_data

#     return StreamingResponse(
#         sarvam_tts_stream(text, language_code),
#         media_type="audio/wav",
#     )


# def download_recording(recording_url: str, retries: int = 4, delay: float = 1.0) -> bytes:
#     """
#     Download a Twilio call recording as WAV bytes, entirely in
#     memory. Twilio recordings need Basic Auth, and the file can take
#     a moment to become available right after the recording action
#     fires, so this retries briefly before giving up.
#     """

#     audio_url = f"{recording_url}.wav"

#     last_error = None

#     for _ in range(retries):

#         try:
#             resp = requests.get(
#                 audio_url,
#                 auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
#                 timeout=15,
#             )

#             if resp.status_code == 200 and resp.content:
#                 return resp.content

#             last_error = f"status {resp.status_code}"

#         except requests.RequestException as e:

#             last_error = str(e)

#         time.sleep(delay)

#     raise RuntimeError(f"Could not download recording: {last_error}")


# def hangup_with_message(text: str, language_code: str) -> Response:

#     response = VoiceResponse()

#     response.play(get_audio_url(text, language_code))

#     response.hangup()

#     return Response(
#         content=str(response),
#         media_type="application/xml"
#     )


# def record_response(prompt_text: str, language_code: str, action: str) -> Response:
#     """
#     Build a TwiML response that plays `prompt_text` via Sarvam TTS and
#     then records the caller's reply for the given `action` webhook.
#     """

#     response = VoiceResponse()

#     response.play(get_audio_url(prompt_text, language_code))

#     response.record(
#         action=action,
#         method="POST",
#         max_length=15,
#         play_beep=False,
#         trim="trim-silence",
#     )

#     return Response(
#         content=str(response),
#         media_type="application/xml"
#     )


# async def detect_language_with_llm(user_text: str) -> str:

#     from google import genai

#     client = genai.Client(
#         api_key=os.environ["GEMINI_API_KEY"]
#     )

#     prompt = f"""
# Identify the language the caller wants to use.

# Caller said:
# {user_text}

# Return ONLY one language from:

# English
# Hindi
# Gujarati
# Bengali
# Marathi
# Tamil
# Telugu
# Kannada
# Malayalam

# Do not explain anything.
# """

#     response = client.models.generate_content(
#         model="gemini-3.1-flash-lite",
#         contents=prompt,
#     )

#     return response.text.strip()


# def run_turn(payload, config, language_code: str):
#     """
#     Invoke the LangGraph workflow, then speak either the next
#     interrupt message (and record the caller's reply) or the
#     closing message (and hang up).
#     """

#     try:

#         result = workflow.invoke(
#             payload,
#             config=config,
#         )

#         interrupt_message = get_interrupt_message(result)

#         if interrupt_message:

#             return record_response(
#                 interrupt_message,
#                 language_code,
#                 action=f"/resume?lang={language_code}",
#             )

#         return hangup_with_message(
#             "The conversation has ended. Thank you for calling BharatSwasthya AI.",
#             "en-IN",
#         )

#     except Exception as e:

#         print(f"⚠️ Workflow error: {e}")

#         return hangup_with_message(
#             "Sorry, something went wrong. Please try again later.",
#             language_code,
#         )


# @app.post("/call")
# async def incoming_call(request: Request):

#     return record_response(
#         "BharatSwasthya AI mein aapka swagat hai. "
#         "Aapko jis bhasha mein baat karni hai, "
#         "kripya us bhasha ka naam boliye.",
#         "hi-IN",
#         action="/language-selected",
#     )


# @app.post("/language-selected")
# async def language_selected(request: Request):

#     form = await request.form()

#     call_sid = form.get("CallSid")
#     phone_number = form.get("From")
#     recording_url = form.get("RecordingUrl")

#     if not call_sid or not recording_url:
#         return hangup_with_message(
#             "Sorry, I could not identify your call.",
#             "hi-IN",
#         )

#     try:
#         audio_bytes = download_recording(recording_url)
#         user_text = sarvam_stt(audio_bytes, language_code="unknown")
#     except Exception as e:
#         print(f"⚠️ STT failed: {e}")
#         user_text = ""

#     if not user_text:
#         return record_response(
#             "Mujhe aapki awaaz sunai nahi di. "
#             "Kripya apni bhasha ka naam dobara boliye.",
#             "hi-IN",
#             action="/language-selected",
#         )

#     language = await detect_language_with_llm(user_text)
#     language_code = get_language_code(language)

#     initial_state = {

#         "messages": [],

#         "call_sid": call_sid,

#         "phone_number": phone_number,

#         "language": language,

#         "emergency": False,

#         "emergency_type": "",

#         "sms_sent": False,

#         "call_ended": False,
#     }

#     config = {
#         "configurable": {
#             "thread_id": call_sid
#         }
#     }

#     return run_turn(initial_state, config, language_code)


# @app.post("/resume")
# async def resume_agent(request: Request):

#     form = await request.form()

#     call_sid = form.get("CallSid")
#     recording_url = form.get("RecordingUrl")

#     language_code = request.query_params.get("lang", "hi-IN")

#     if not call_sid or not recording_url:
#         return hangup_with_message(
#             "Sorry, something went wrong. Please try again later.",
#             language_code,
#         )

#     try:
#         audio_bytes = download_recording(recording_url)
#         user_text = sarvam_stt(audio_bytes, language_code=language_code)
#     except Exception as e:
#         print(f"⚠️ STT failed: {e}")
#         user_text = ""

#     if not user_text:
#         return record_response(
#             "Mujhe samajh nahi aaya, kripya dobara boliye.",
#             language_code,
#             action=f"/resume?lang={language_code}",
#         )

#     config = {
#         "configurable": {
#             "thread_id": call_sid
#         }
#     }

#     return run_turn(Command(resume=user_text), config, language_code)

import os
import time
import uuid

import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from twilio.twiml.voice_response import VoiceResponse
from langgraph.types import Command

from agent.graph import workflow
from agent.speech import stt as sarvam_stt, tts_stream as sarvam_tts_stream


TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]

# Public base URL of this FastAPI app (e.g. your Render URL), so Twilio
# can fetch the audio clips we generate with Sarvam TTS.
BASE_URL = os.environ["BASE_URL"].rstrip("/")

app = FastAPI()

# In-memory record of pending TTS requests, keyed by a random id.
# Registering a request is instant - the actual Sarvam TTS call only
# happens once Twilio fetches the audio URL, and is streamed straight
# through as it's generated. Nothing is pre-generated, buffered in
# full, or written to disk.
TTS_REQUESTS: dict[str, tuple[str, str]] = {}

# A handful of fixed system prompts get spoken on every single call
# (the welcome greeting, the "didn't catch that" retries, the closing
# line). Since their text never changes, we synthesize each one once
# at startup and keep the finished bytes in memory - repeat callers
# never pay a fresh Sarvam TTS round-trip for these. This is separate
# from TTS_REQUESTS/streaming, which is still used for every dynamic
# agent reply.
WELCOME_TEXT = (
    "BharatSwasthya AI mein aapka swagat hai. "
    "Aapko jis bhasha mein baat karni hai, "
    "kripya us bhasha ka naam boliye."
)
LANGUAGE_RETRY_TEXT = (
    "Mujhe aapki awaaz sunai nahi di. "
    "Kripya apni bhasha ka naam dobara boliye."
)
RESUME_RETRY_TEXT = "Mujhe samajh nahi aaya, kripya dobara boliye."
CLOSING_TEXT = "The conversation has ended. Thank you for calling BharatSwasthya AI."

STATIC_PROMPTS = {
    (WELCOME_TEXT, "hi-IN"),
    (LANGUAGE_RETRY_TEXT, "hi-IN"),
    (RESUME_RETRY_TEXT, "hi-IN"),
    (CLOSING_TEXT, "en-IN"),
}

STATIC_AUDIO_CACHE: dict[tuple[str, str], bytes] = {}


@app.on_event("startup")
async def warm_static_prompts():

    for text, language_code in STATIC_PROMPTS:

        try:
            STATIC_AUDIO_CACHE[(text, language_code)] = b"".join(
                sarvam_tts_stream(text, language_code)
            )
        except Exception as e:
            print(f"⚠️ Could not pre-warm prompt audio: {e}")


def get_interrupt_message(result):

    interrupts = result.get("__interrupt__")

    if not interrupts:
        return None

    interrupt_value = interrupts[0].value

    interrupt_type = interrupt_value.get("type")

    if interrupt_type == "human_question":

        return interrupt_value.get("question")

    if interrupt_type == "final_response":

        return interrupt_value.get("message")

    return None

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

def get_language_code(language: str) -> str:
    """
    Maps a detected language name to its BCP-47 code. These codes are
    used for both Sarvam STT/TTS and happen to match Twilio's own
    language codes, so the same mapping serves both.
    """

    return SUPPORTED_LANGUAGES.get(
        language,
        "hi-IN"
    )


def get_audio_url(text: str, language_code: str) -> str:
    """
    Registers a pending TTS request and returns a URL for Twilio's
    <Play> to fetch. This is instant - the Sarvam call itself only
    happens when that URL is actually requested, in get_audio() below
    (skipped entirely for cached static prompts).
    """

    audio_id = uuid.uuid4().hex

    TTS_REQUESTS[audio_id] = (text, language_code)

    return f"{BASE_URL}/audio/{audio_id}"


@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """
    Serves a TTS clip to Twilio. Known static prompts are served
    instantly from STATIC_AUDIO_CACHE (no Sarvam call). Everything
    else - the agent's actual dynamic replies - is streamed from
    Sarvam as it's generated, so playback can begin before the clip
    finishes.
    """

    request_data = TTS_REQUESTS.pop(audio_id, None)

    if request_data is None:
        return Response(status_code=404)

    text, language_code = request_data

    cached = STATIC_AUDIO_CACHE.get((text, language_code))

    if cached is not None:
        return Response(content=cached, media_type="audio/wav")

    return StreamingResponse(
        sarvam_tts_stream(text, language_code),
        media_type="audio/wav",
    )


def download_recording(recording_url: str, retries: int = 4, delay: float = 0.4) -> bytes:
    """
    Download a Twilio call recording as WAV bytes, entirely in
    memory. Twilio recordings need Basic Auth, and the file can take
    a moment to become available right after the recording action
    fires, so this retries briefly before giving up.
    """

    audio_url = f"{recording_url}.wav"

    last_error = None

    for _ in range(retries):

        try:
            resp = requests.get(
                audio_url,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                timeout=15,
            )

            if resp.status_code == 200 and resp.content:
                return resp.content

            last_error = f"status {resp.status_code}"

        except requests.RequestException as e:

            last_error = str(e)

        time.sleep(delay)

    raise RuntimeError(f"Could not download recording: {last_error}")


def hangup_with_message(text: str, language_code: str) -> Response:

    response = VoiceResponse()

    response.play(get_audio_url(text, language_code))

    response.hangup()

    return Response(
        content=str(response),
        media_type="application/xml"
    )


def record_response(prompt_text: str, language_code: str, action: str) -> Response:
    """
    Build a TwiML response that plays `prompt_text` via Sarvam TTS and
    then records the caller's reply for the given `action` webhook.
    """

    response = VoiceResponse()

    response.play(get_audio_url(prompt_text, language_code))

    response.record(
        action=action,
        method="POST",
        max_length=15,
        play_beep=False,
        trim="trim-silence",
        timeout=2,
    )

    return Response(
        content=str(response),
        media_type="application/xml"
    )


async def detect_language_with_llm(user_text: str) -> str:

    from google import genai

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    prompt = f"""
Identify the language the caller wants to use.

Caller said:
{user_text}

Return ONLY one language from:

English
Hindi
Gujarati
Bengali
Marathi
Tamil
Telugu
Kannada
Malayalam

Do not explain anything.
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    return response.text.strip()


def run_turn(payload, config, language_code: str):
    """
    Invoke the LangGraph workflow, then speak either the next
    interrupt message (and record the caller's reply) or the
    closing message (and hang up).
    """

    try:

        result = workflow.invoke(
            payload,
            config=config,
        )

        interrupt_message = get_interrupt_message(result)

        if interrupt_message:

            return record_response(
                interrupt_message,
                language_code,
                action=f"/resume?lang={language_code}",
            )

        return hangup_with_message(CLOSING_TEXT, "en-IN")

    except Exception as e:

        print(f"⚠️ Workflow error: {e}")

        return hangup_with_message(
            "Sorry, something went wrong. Please try again later.",
            language_code,
        )


@app.post("/call")
async def incoming_call(request: Request):

    return record_response(
        WELCOME_TEXT,
        "hi-IN",
        action="/language-selected",
    )


@app.post("/language-selected")
async def language_selected(request: Request):

    form = await request.form()

    call_sid = form.get("CallSid")
    phone_number = form.get("From")
    recording_url = form.get("RecordingUrl")

    if not call_sid or not recording_url:
        return hangup_with_message(
            "Sorry, I could not identify your call.",
            "hi-IN",
        )

    try:
        audio_bytes = download_recording(recording_url)
        user_text = sarvam_stt(audio_bytes, language_code="unknown")
    except Exception as e:
        print(f"⚠️ STT failed: {e}")
        user_text = ""

    if not user_text:
        return record_response(
            LANGUAGE_RETRY_TEXT,
            "hi-IN",
            action="/language-selected",
        )

    language = await detect_language_with_llm(user_text)
    language_code = get_language_code(language)

    initial_state = {

        "messages": [],

        "call_sid": call_sid,

        "phone_number": phone_number,

        "language": language,

        "emergency": False,

        "emergency_type": "",

        "sms_sent": False,

        "call_ended": False,
    }

    config = {
        "configurable": {
            "thread_id": call_sid
        }
    }

    return run_turn(initial_state, config, language_code)


@app.post("/resume")
async def resume_agent(request: Request):

    form = await request.form()

    call_sid = form.get("CallSid")
    recording_url = form.get("RecordingUrl")

    language_code = request.query_params.get("lang", "hi-IN")

    if not call_sid or not recording_url:
        return hangup_with_message(
            "Sorry, something went wrong. Please try again later.",
            language_code,
        )

    try:
        audio_bytes = download_recording(recording_url)
        user_text = sarvam_stt(audio_bytes, language_code=language_code)
    except Exception as e:
        print(f"⚠️ STT failed: {e}")
        user_text = ""

    if not user_text:
        return record_response(
            RESUME_RETRY_TEXT,
            language_code,
            action=f"/resume?lang={language_code}",
        )

    config = {
        "configurable": {
            "thread_id": call_sid
        }
    }

    return run_turn(Command(resume=user_text), config, language_code)