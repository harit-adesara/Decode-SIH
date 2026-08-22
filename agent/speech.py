# import os
# import io

# from sarvamai import SarvamAI
# from dotenv import load_dotenv

# load_dotenv()

# sarvam_client = SarvamAI(
#     api_subscription_key=os.environ["SARVAM_API_KEY"]
# )


# def stt(audio_bytes: bytes, language_code: str = "unknown") -> str:
#     """
#     Transcribe in-memory audio bytes (e.g. a downloaded Twilio
#     recording) to text using Sarvam's Speech-to-Text REST API.
#     Nothing is written to disk.

#     language_code:
#         BCP-47 code such as "hi-IN", "gu-IN", "en-IN".
#         Use "unknown" to let Sarvam auto-detect the spoken language
#         (used for the very first turn, before we know the caller's
#         chosen language).
#     """

#     audio_buffer = io.BytesIO(audio_bytes)
#     audio_buffer.name = "recording.wav"

#     response = sarvam_client.speech_to_text.transcribe(
#         file=audio_buffer,
#         model="saaras:v3",
#         language_code=language_code,
#     )

#     transcript = (response.transcript or "").strip()

#     return transcript


# def tts_stream(text: str, language_code: str, speaker: str = "shubh"):
#     """
#     Convert text to speech using Sarvam's streaming Text-to-Speech
#     REST endpoint and yield raw WAV audio chunks as they're
#     generated - no full clip is buffered in memory and nothing is
#     written to disk.

#     Output is fixed to 8kHz WAV to match Twilio's telephony audio
#     format, so no resampling is needed downstream.
#     """

#     audio_stream = sarvam_client.text_to_speech.convert_stream(
#         text=text,
#         language_code=language_code,
#         model="bulbul:v3",
#         speaker=speaker,
#         speech_sample_rate=8000,
#         output_audio_codec="wav",
#     )

#     for chunk in audio_stream:
#         if chunk:
#             yield chunk

import os
import io

from sarvamai import SarvamAI
from dotenv import load_dotenv

load_dotenv()

sarvam_client = SarvamAI(
    api_subscription_key=os.environ["SARVAM_API_KEY"]
)


def stt(audio_bytes: bytes, language_code: str = "unknown") -> str:
    """
    Transcribe in-memory audio bytes (e.g. a downloaded Twilio
    recording) to text using Sarvam's Speech-to-Text REST API.
    Nothing is written to disk.

    language_code:
        BCP-47 code such as "hi-IN", "gu-IN", "en-IN".
        Use "unknown" to let Sarvam auto-detect the spoken language
        (used for the very first turn, before we know the caller's
        chosen language).
    """

    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = "recording.wav"

    response = sarvam_client.speech_to_text.transcribe(
        file=audio_buffer,
        model="saaras:v3",
        language_code=language_code,
    )

    transcript = (response.transcript or "").strip()

    return transcript


def tts_stream(text: str, language_code: str, speaker: str = "shubh"):
    """
    Convert text to speech using Sarvam's streaming Text-to-Speech
    REST endpoint and yield raw WAV audio chunks as they're
    generated - no full clip is buffered in memory and nothing is
    written to disk.

    Output is fixed to 8kHz WAV to match Twilio's telephony audio
    format, so no resampling is needed downstream.
    """

    audio_stream = sarvam_client.text_to_speech.convert_stream(
        text=text,
        language_code=language_code,
        model="bulbul:v3",
        speaker=speaker,
        speech_sample_rate=8000,
        output_audio_codec="wav",
    )

    for chunk in audio_stream:
        if chunk:
            yield chunk