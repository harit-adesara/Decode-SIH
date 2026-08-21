import base64
import audioop

from google import genai
from google.genai import types

client = genai.Client(
    api_key="YOUR_GOOGLE_API_KEY"
)

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"


# ------------------------------------------------------------
# AUDIO CONVERSION
# ------------------------------------------------------------

def twilio_to_gemini(audio_bytes: bytes) -> bytes:

    pcm_8k = audioop.ulaw2lin(
        audio_bytes,
        2
    )

    pcm_16k, _ = audioop.ratecv(
        pcm_8k,
        2,
        1,
        8000,
        16000,
        None
    )

    return pcm_16k


def gemini_to_twilio(audio_bytes: bytes) -> bytes:

    pcm_8k, _ = audioop.ratecv(
        audio_bytes,
        2,
        1,
        24000,
        8000,
        None
    )

    return audioop.lin2ulaw(
        pcm_8k,
        2
    )

async def stt(
    session,
    twilio_audio: bytes
):
    """
    Twilio audio bytes
        ↓
    Gemini Live
        ↓
    text

    No .wav file.
    """

    pcm_audio = twilio_to_gemini(
        twilio_audio                                                                                                        
    )

    await session.send_realtime_input(
        audio=types.Blob(
            data=pcm_audio,
            mime_type="audio/pcm;rate=16000"
        )
    )

    async for response in session.receive():

        server_content = (
            response.server_content
        )

        if not server_content:
            continue

        # User's speech → text
        if server_content.input_transcription:

            text = (
                server_content
                .input_transcription
                .text
            )

            if text:
                return text

    return None

async def tts(
    session,
    text: str
):
    """
    text
      ↓
    Gemini Live
      ↓
    PCM audio chunks

    No .wav file.
    """

    # Give text to Gemini
    await session.send_client_content(

        turns=types.Content(
            role="user",
            parts=[
                types.Part(
                    text=text
                )
            ]
        ),

        turn_complete=True
    )

    # Receive generated audio
    async for response in session.receive():

        server_content = (
            response.server_content
        )

        if not server_content:
            continue

        if server_content.model_turn:

            for part in (
                server_content
                .model_turn
                .parts
            ):

                if not part.inline_data:
                    continue

                audio = (
                    part.inline_data.data
                )

                if isinstance(
                    audio,
                    str
                ):
                    audio = base64.b64decode(
                        audio
                    )

                # Convert Gemini audio
                # to Twilio format
                twilio_audio = (
                    gemini_to_twilio(
                        audio
                    )
                )

                yield twilio_audio