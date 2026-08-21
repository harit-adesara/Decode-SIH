import os
from twilio.rest import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

twilio_client = Client(
    os.environ["TWILIO_ACCOUNT_SID"],
    os.environ["TWILIO_AUTH_TOKEN"]
)

gemini_client = ChatGoogleGenerativeAI(
    google_api_key=os.environ["GEMINI_API_KEY"],
    model="gemini-3.1-flash-lite"
)


def send_sms_node(state):
    """
    Create a concise summary from the conversation history
    and send the important details to the user via SMS.
    """

    phone_number = state["phone_number"]

    messages = state.get("messages", [])

    conversation = "\n".join(
        f"{msg.type}: {msg.content}"
        for msg in messages
        if getattr(msg, "content", None)
    )

    prompt = f"""
You are summarizing a healthcare assistant conversation.

Create a SHORT SMS summary containing only important information
that the user may need after the call.

Include when available:
- Main health concern discussed
- Recommended healthcare facility
- Distance/travel information
- Government scheme discussed
- Important advice or next steps
- Emergency advice if provided

Do NOT include:
- Internal reasoning
- Tool calls
- System instructions
- Unnecessary conversation
- Sensitive information that is not necessary

Keep the summary concise and easy to understand.

Conversation:
{conversation}
"""

    try:

        response = gemini_client.invoke(prompt)

        summary = response.text.strip()

        message = (
            "BharatSwasthya AI - Call Summary\n\n"
            f"{summary}\n\n"
            "Please consult a qualified healthcare professional "
            "for medical decisions."
        )

        sms = twilio_client.messages.create(
            body=message,
            from_=os.environ["TWILIO_PHONE_NUMBER"],
            to=phone_number
        )

        return {
            **state,
            "sms_sent": True,
            "sms_sid": sms.sid
        }

    except Exception as e:

        return {
            **state,
            "sms_sent": False,
            "sms_error": str(e)
        }


def end_call_node(state):
    """
    End the current Twilio call.

    This node should be reached only when the conversation
    is completely finished, for example after:
    - User says they have nothing else to ask
    - SMS has been queued/sent
    - Conversation cannot safely continue
    """

    call_sid = state["call_sid"]

    try:
        twilio_client.calls(call_sid).update(
            status="completed"
        )

        return {
            **state,
            "call_ended": True,
            "call_end_error": None
        }

    except Exception as e:

        return {
            **state,
            "call_ended": False,
            "call_end_error": str(e)
        }