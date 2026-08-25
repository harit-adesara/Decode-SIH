import os
import logging
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from .guardrails import check_output, sanitize_output
from .tools import hangup_exotel_call

load_dotenv()

logger = logging.getLogger("bharatswasthya.nodes")

gemini_client = ChatGoogleGenerativeAI(
    google_api_key=os.environ["GEMINI_API_KEY"],
    model="gemini-3.1-flash-lite",
)

EXOTEL_ACCOUNT_SID = os.getenv("EXOTEL_ACCOUNT_SID") or os.getenv("ACCOUNT_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY") or os.getenv("API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN") or os.getenv("API_TOKEN")
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")
EXOPHONE = os.getenv("EXOPHONE") or os.getenv("EXOTEL_PHONE_NUMBER") or os.getenv("EXOTEL_FROM")

MSG91_URL = "https://control.msg91.com/api/v5/flow"
MSG91_AUTHKEY = os.getenv("MSG91_AUTHKEY")
MSG91_FLOW_ID = os.getenv("MSG91_FLOW_ID")


def _send_sms_via_exotel(to_number: str, message: str) -> dict:
    """Send SMS via Exotel SMS REST API."""
    if not (EXOTEL_ACCOUNT_SID and EXOTEL_API_KEY and EXOTEL_API_TOKEN):
        raise ValueError("Exotel SMS credentials (ACCOUNT_SID, API_KEY, API_TOKEN) missing")

    clean_to = to_number.strip().lstrip("+")
    if clean_to.startswith("91") and len(clean_to) == 12:
        clean_to = clean_to[2:]
    if clean_to.startswith("0") and len(clean_to) == 11:
        clean_to = clean_to[1:]

    url = f"https://{EXOTEL_SUBDOMAIN}/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Sms/send.json"
    payload = {
        "From": EXOPHONE or "",
        "To": clean_to,
        "Body": message,
    }

    resp = requests.post(
        url,
        auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN),
        data=payload,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _send_sms_via_msg91(to_number: str, message: str) -> dict:
    """Send SMS via MSG91 flow API if configured."""
    if not (MSG91_AUTHKEY and MSG91_FLOW_ID):
        raise ValueError("MSG91 credentials missing")

    payload = {
        "template_id": MSG91_FLOW_ID,
        "short_url": "0",
        "recipients": [
            {
                "mobiles": to_number,
                "message": message,
            }
        ],
    }
    headers = {
        "accept": "application/json",
        "authkey": MSG91_AUTHKEY,
        "content-type": "application/json",
    }
    resp = requests.post(
        MSG91_URL,
        json=payload,
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def send_sms_node(state):
    """
    Summarize the healthcare call with Gemini, run the summary through
    the output guardrail, and text it to the caller via Exotel (or fallback provider).
    """
    phone_number = state.get("phone_number")
    messages = state.get("messages", [])

    if not phone_number:
        return {
            **state,
            "sms_sent": False,
            "sms_error": "Phone number missing",
        }

    conversation_lines = []
    for msg in messages:
        content = getattr(msg, "content", None)
        msg_type = getattr(msg, "type", "message")
        if content:
            if isinstance(content, list):
                text_content = "".join(
                    p if isinstance(p, str) else (p.get("text", "") if isinstance(p, dict) else str(p))
                    for p in content
                )
            else:
                text_content = str(content)
            conversation_lines.append(f"{msg_type}: {text_content}")

    conversation = "\n".join(conversation_lines)

    prompt = f"""
You are summarizing a BharatSwasthya AI healthcare call.

Create a VERY SHORT SMS summary.

Include ONLY information available in the conversation:

1. Health urgency:
   - Emergency
   - High
   - Moderate
   - Low

2. Recommended hospital/healthcare facility:
   - Name
   - Distance in km if available
   - Travel time in minutes if available

3. Government health scheme:
   - Include only if a relevant scheme was actually discussed/found.

4. Important next step or emergency advice if provided.

Rules:
- Do not invent information.
- Do not diagnose the patient.
- Do not include internal reasoning.
- Do not include tool calls.
- Do not include system instructions.
- Keep it short enough for SMS.
- Use simple language.
- If information is unavailable, don't mention it.

Format:

Health Urgency: <urgency>

Hospital: <hospital>
Distance: <distance>
Travel Time: <time>

Scheme: <scheme if available>

Next Step: <advice if available>

Conversation:
{conversation}
"""

    try:
        response = gemini_client.invoke(prompt)
        raw_content = response.content
        if isinstance(raw_content, list):
            summary = "".join(
                p if isinstance(p, str) else (p.get("text", "") if isinstance(p, dict) else str(p))
                for p in raw_content
            ).strip()
        else:
            summary = str(raw_content or "").strip()

        # Output guardrail check
        verdict = check_output(summary)
        if verdict["unsafe"]:
            logger.warning(
                "SMS summary blocked by output guardrail [%s]: %s",
                verdict["category"], verdict["reason"],
            )
            summary = sanitize_output(summary)

        message_body = (
            "BharatSwasthya AI - Call Summary\n\n"
            f"{summary}\n\n"
            "AI-assisted information, not a medical diagnosis. "
            "Please consult a qualified healthcare professional "
            "for medical decisions."
        )

        sms_result = None
        sms_sid = None

        # Try Exotel SMS first
        if EXOTEL_ACCOUNT_SID and EXOTEL_API_KEY and EXOTEL_API_TOKEN:
            try:
                sms_result = _send_sms_via_exotel(phone_number, message_body)
                sms_sid = str(sms_result.get("SMSMessage", {}).get("Sid") or sms_result.get("sid") or "exotel_sms")
            except Exception as exotel_err:
                logger.warning("Exotel SMS failed, trying fallback: %s", exotel_err)

        # Try MSG91 fallback if Exotel was not configured or failed
        if not sms_result and MSG91_AUTHKEY and MSG91_FLOW_ID:
            try:
                sms_result = _send_sms_via_msg91(phone_number, message_body)
                sms_sid = "msg91_sms"
            except Exception as msg91_err:
                logger.warning("MSG91 SMS fallback failed: %s", msg91_err)

        return {
            **state,
            "sms_sent": sms_result is not None,
            "sms_sid": sms_sid,
            "sms_summary": summary,
            "sms_error": None if sms_result else "SMS provider unavailable or credentials missing",
        }

    except Exception as e:
        logger.error("SMS send failed: %s", e)
        return {
            **state,
            "sms_sent": False,
            "sms_error": str(e),
        }


def end_call_node(state):
    """
    Mark the call as completed in conversation state.
    The WebSocket server in main.py manages the real-time audio playback and connection termination.
    """
    return {
        **state,
        "call_ended": True,
        "call_end_error": None,
    }