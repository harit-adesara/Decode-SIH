import json
import base64

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

app = FastAPI(title="BharatSwasthya AI")


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "BharatSwasthya AI"
    }


@app.post("/voice")
async def voice():
    """
    Twilio calls this endpoint when someone
    calls the Twilio phone number.
    """

    response = VoiceResponse()

    connect = Connect()

    stream = Stream(
        url="wss://YOUR_NGROK_DOMAIN.ngrok-free.app/media-stream"
    )

    connect.append(stream)
    response.append(connect)

    return Response(
        content=str(response),
        media_type="application/xml"
    )


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):

    await websocket.accept()

    print("================================")
    print("Twilio WebSocket connected")
    print("================================")

    try:

        while True:

            message = await websocket.receive_text()

            data = json.loads(message)

            event = data.get("event")

            # -------------------------
            # Twilio connected
            # -------------------------

            if event == "connected":

                print("Twilio connection established")

            # -------------------------
            # Call started
            # -------------------------

            elif event == "start":

                start = data["start"]

                stream_sid = start["streamSid"]
                call_sid = start["callSid"]

                print("Call SID   :", call_sid)
                print("Stream SID :", stream_sid)

                print(
                    "Media format:",
                    start["mediaFormat"]
                )

            # -------------------------
            # Audio received
            # -------------------------

            elif event == "media":

                payload = data["media"]["payload"]

                # Base64 → audio bytes
                audio_bytes = base64.b64decode(payload)

                print(
                    "Received audio:",
                    len(audio_bytes),
                    "bytes"
                )

                # NEXT:
                # audio_bytes → Google STT

            # -------------------------
            # Call ended
            # -------------------------

            elif event == "stop":

                print("================================")
                print("Call ended")
                print("================================")

                break

    except Exception as e:

        print("WebSocket error:", e)

    finally:

        print("WebSocket disconnected")