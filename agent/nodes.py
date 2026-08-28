import os
import logging
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from .guardrails import check_output, sanitize_output

load_dotenv()

logger = logging.getLogger("bharatswasthya.nodes")

gemini_client = ChatGoogleGenerativeAI(
    google_api_key=os.environ["GEMINI_API_KEY"],
    model="gemini-3.1-flash-lite",
)

SMSMOBILEAPI_URL = (
    os.getenv("SMSMOBILEAPI_URL")
    or os.getenv("SMS_MOBILE_API_URL")
    or "https://api.smsmobileapi.com/sendsms/"
).strip().strip('"').strip("'")

SMSMOBILEAPI_KEY = (
    os.getenv("SMSMOBILEAPI_KEY")
    or os.getenv("SMSMOBILEAPI_API_KEY")
    or os.getenv("SMS_MOBILE_API_KEY")
    or ""
).strip().strip('"').strip("'")


def _send_sms_via_smsmobileapi(to_number: str, message: str) -> dict:
    """Send SMS via SMSMobileAPI gateway."""
    if not SMSMOBILEAPI_KEY:
        raise ValueError("SMSMobileAPI key (SMSMOBILEAPI_KEY) missing from environment")

    clean_to = to_number.strip()
    if not clean_to.startswith("+"):
        if len(clean_to) == 10:
            clean_to = f"+91{clean_to}"
        elif len(clean_to) == 11 and clean_to.startswith("0"):
            clean_to = f"+91{clean_to[1:]}"
        elif len(clean_to) == 12 and clean_to.startswith("91"):
            clean_to = f"+{clean_to}"
        else:
            clean_to = f"+{clean_to}"

    url = SMSMOBILEAPI_URL.rstrip("/") + "/"
    payload = {
        "apikey": SMSMOBILEAPI_KEY,
        "recipients": clean_to,
        "message": message,
    }

    logger.info("Sending SMS via SMSMobileAPI to %s", clean_to)
    resp = requests.post(
        url,
        data=payload,
        timeout=15,
    )
    resp.raise_for_status()

    try:
        data = resp.json()
    except Exception:
        data = {"response": resp.text}

    # Verify if SMSMobileAPI reported an error in the response payload
    if isinstance(data, dict):
        result_info = data.get("result")
        if isinstance(result_info, dict):
            if result_info.get("error") not in (0, "0", None, False):
                err_msg = result_info.get("message") or result_info.get("error") or "Failed to send SMS"
                raise RuntimeError(f"SMSMobileAPI error: {err_msg}")
        elif data.get("error") and data.get("error") not in (0, "0", False):
            raise RuntimeError(f"SMSMobileAPI error: {data.get('error')}")

    return data


def send_sms_node(state):
    """
    Summarize the healthcare call with Gemini, run the summary through
    the output guardrail, and text it to the caller via SMSMobileAPI.
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

        sms_result = _send_sms_via_smsmobileapi(phone_number, message_body)
        sms_sid = None
        if isinstance(sms_result, dict):
            result_info = sms_result.get("result", {})
            if isinstance(result_info, dict):
                sms_sid = str(
                    result_info.get("id")
                    or result_info.get("message_id")
                    or result_info.get("sent")
                    or "smsmobileapi_sms"
                )
            else:
                sms_sid = str(sms_result.get("id") or "smsmobileapi_sms")
        else:
            sms_sid = "smsmobileapi_sms"

        return {
            **state,
            "sms_sent": True,
            "sms_sid": sms_sid,
            "sms_summary": summary,
            "sms_error": None,
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