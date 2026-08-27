import asyncio
import base64
import json
import logging
import math
import os
import re
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
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "25.0"))
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

RESUME_RETRY_TEXTS = {
    "hi-IN": "Kripya apna sawal dobara boliye.",
    "en-IN": "Please ask your question again.",
    "gu-IN": "કૃપા કરીને તમારો પ્રશ્ન ફરીથી બોલો.",
    "mr-IN": "कृपया आपला प्रश्न पुन्हा विचारा.",
    "bn-IN": "দয়া করে আপনার প্রশ্নটি আবার বলুন।",
    "ta-IN": "தயவுசெய்து உங்கள் கேள்வியை மீண்டும் கேளுங்கள்.",
    "te-IN": "దయచేసి మీ ప్రశ్నను మళ్లీ అడగండి.",
    "kn-IN": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಮತ್ತೊಮ್ಮೆ ಕೇಳಿ.",
    "ml-IN": "ദയവായി നിങ്ങളുടെ ചോദ്യം വീണ്ടും ചോദിക്കുക.",
}

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
    "gu-IN": "આ માહિતી શોધવામાં થોડો સમય લાગી રહ્યો છે. કૃપા કરીને તમારો પ્રશ્ન ફરીથી બોલો.",
    "mr-IN": "माहिती शोधण्यासाठी मला थोडा वेळ लागत आहे. कृपया आपला प्रश्न पुन्हा विचारा.",
    "bn-IN": "তথ্যটি খুঁজতে কিছুটা সময় লাগছে। দয়া করে আপনার প্রশ্নটি আবার বলুন।",
    "ta-IN": "தகவலைத் தேட சிறிது நேரம் ஆகிறது. தயவுசெய்து உங்கள் கேள்வியை மீண்டும் கேளுங்கள்.",
    "te-IN": "సమాచారాన్ని కనుగొనడానికి కొంత సమయం పడుతోంది. దయచేసి మీ ప్రశ్నను మళ్లీ అడగండి.",
    "kn-IN": "ಮಾಹಿತಿಯನ್ನು ಹುಡುಕಲು ಸ್ವಲ್ಪ ಸಮಯ ತೆಗೆದುಕೊಳ್ಳುತ್ತಿದೆ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಮತ್ತೊಮ್ಮೆ ಕೇಳಿ.",
    "ml-IN": "വിവരം കണ്ടെത്താൻ അൽപ്പം സമയമെടുക്കുന്നു. ദയവായി നിങ്ങളുടെ ചോദ്യം വീണ്ടും ചോദിക്കുക.",
}

INACTIVITY_PROMPT_TEXTS = {
    "hi-IN": "Hello, kya aap wahin hain? Agar aapka koi aur sawal ho to kripya boliye.",
    "en-IN": "Hello, are you still there? Please go ahead if you have another question.",
    "gu-IN": "હેલો, તમે હજી લાઇન પર છો? જો તમારો કોઈ બીજો પ્રશ્ન હોય તો કૃપા કરીને બોલો.",
    "mr-IN": "हॅलो, आपण तिथे आहात का? जर आपला काही प्रश्न असेल तर कृपया बोला.",
    "bn-IN": "হ্যালো, আপনি কি লাইনে আছেন? আপনার যদি অন্য কোনো প্রশ্ন থাকে তবে দয়া করে বলুন।",
    "ta-IN": "ஹலோ, இணைப்பில் இருக்கிறீர்களா? வேறு ஏதேனும் கேள்வி இருந்தால் தயவுசெய்து சொல்லுங்கள்.",
    "te-IN": "హలో, మీరు లైన్ లో ఉన్నారా? మీకు ఏదైనా ప్రశ్న ఉంటే దయచేసి మాట్లాడండి.",
    "kn-IN": "ಹಲೋ, ನೀವು ಲೈನ್‌ನಲ್ಲಿದ್ದೀರಾ? ನಿಮಗೆ ಬೇರೆ ಯಾವುದೇ ಪ್ರಶ್ನೆ ಇದ್ದರೆ ದಯವಿಟ್ಟು ಮಾತನಾಡಿ.",
    "ml-IN": "ഹലോ, നിങ്ങൾ ലൈനിലുണ്ടോ? എന്തെങ്കിലും ചോദ്യമുണ്ടെങ്കിൽ ദയവായി സംസാരിക്കുക.",
}

INACTIVITY_GOODBYE_TEXTS = {
    "hi-IN": "Mujhe koi jawab nahi mila, isliye main call band kar raha hoon. Zaroorat padne par kripya phir se call kijiye. Namaste!",
    "en-IN": "Since I haven't heard a response, I'll end the call now. Please call again if you need help. Goodbye!",
    "gu-IN": "મને કોઈ જવાબ ન મળ્યો, તેથી હું કોલ સમાપ્ત કરું છું. જરૂર પડે તો ફરીથી કોલ કરજો. નમસ્તે!",
    "mr-IN": "मला कोणताही प्रतिसाद मिळाला नाही, म्हणून मी कॉल समाप्त करत आहे. गरज असल्यास पुन्हा कॉल करा. नमस्कार!",
    "bn-IN": "কোনো সাড়া না পাওয়ায় আমি কলটি শেষ করছি। প্রয়োজন হলে আবার কল করবেন। নমস্কার!",
    "ta-IN": "பதில் எதுவும் வராததால் அழைப்பை முடிக்கிறேன். தேவைப்பட்டால் மீண்டும் அழைக்கவும். வணக்கம்!",
    "te-IN": "ఎటువంటి సమాధానం రాకపోవడంతో కాల్ ముగిస్తున్నాను. అవసరమైతే మళ్లీ కాల్ చేయండి. నమస్కారం!",
    "kn-IN": "ಯಾವುದೇ ಪ್ರತಿಕ್ರಿಯೆ ಬಾರದ ಕಾರಣ ಕರೆಯನ್ನು ಕೊನೆಗೊಳಿಸುತ್ತಿದ್ದೇನೆ. ಅಗತ್ಯವಿದ್ದರೆ ಮತ್ತೆ ಕರೆ ಮಾಡಿ. ನಮಸ್ಕಾರ!",
    "ml-IN": "പ്രതികരണമൊന്നും ലഭിക്കാത്തതിനാൽ കോൾ അവസാനിപ്പിക്കുന്നു. ആവശ്യമെങ്കിൽ വീണ്ടും വിളിക്കുക. നമസ്കാരം!",
}

SAFETY_DECLINE_TEXTS = {
    "hi-IN": "Main is prakar ke sawalon mein madad nahi kar sakta. Kripya swasthya ya hospital sambandhit sawal poochiye.",
    "en-IN": "I cannot assist with this type of request. Please ask a healthcare, hospital, or government scheme related question.",
    "gu-IN": "હું આવા પ્રકારના પ્રશ્નોમાં મદદ કરી શકતો નથી. કૃપા કરીને સ્વાસ્થ્ય અથવા હોસ્પિટલ સંબંધિત પ્રશ્ન પૂછો.",
    "mr-IN": "मी अशा प्रकारच्या प्रश्नांमध्ये मदत करू शकत नाही. कृपया आरोग्य किंवा रुग्णालयाशी संबंधित प्रश्न विचारा.",
    "bn-IN": "আমি এই ধরণের অনুরোধে সাহায্য করতে পারি না। দয়া করে স্বাস্থ্য বা হাসপাতাল সম্পর্কিত প্রশ্ন জিজ্ঞাসা করুন।",
    "ta-IN": "இந்த வகையான கேள்விகளுக்கு என்னால் உதவ முடியாது. தயவுசெய்து உடல்நலம் அல்லது மருத்துவமனை தொடர்பான கேள்வியைக் கேளுங்கள்.",
    "te-IN": "నేను ఈ రకమైన ప్రశ్నలకు సహాయం చేయలేను. దయచేసి ఆరోగ్యం లేదా ఆసుపత్రికి సంబంధించిన ప్రశ్నను అడగండి.",
    "kn-IN": "ನಾನು ಈ ರೀತಿಯ ಪ್ರಶ್ನೆಗಳಿಗೆ ಸಹಾಯ ಮಾಡಲು ಸಾಧ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಆರೋಗ್ಯ ಅಥವಾ ಆಸ್ಪತ್ರೆಗೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ.",
    "ml-IN": "ഇത്തരം ചോദ്യങ്ങളിൽ എനിക്ക് സഹായിക്കാനാകില്ല. ദയവായി ആരോഗ്യ അല്ലെങ്കിൽ ആശുപത്രി സംബന്ധമായ ചോദ്യങ്ങൾ ചോദിക്കുക.",
}


PROCESSING_TEXTS = {
    "hi-IN": "Kripya thoda intezar kijiye, main jankari dekh raha hoon.",
    "en-IN": "Please wait a moment while I look that up for you.",
    "gu-IN": "કૃપા કરીને થોડી રાહ જુઓ, હું માહિતી શોધી રહ્યો છું.",
    "mr-IN": "कृपया थोडा वेळ थांबा, मी माहिती शोधत आहे.",
    "bn-IN": "অনুগ্রহ করে একটু অপেক্ষা করুন, আমি তথ্যটি খুঁজছি।",
    "ta-IN": "தயவுசெய்து சிறிது காத்திருங்கள், தகவலைப் பார்க்கிறேன்.",
    "te-IN": "దయచేసి కొద్దిసేపు వేచి ఉండండి, సమాచారాన్ని చూస్తున్నాను.",
    "kn-IN": "ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಕಾಯಿರಿ, ನಾನು ಮಾಹಿತಿಯನ್ನು ಹುಡುಕುತ್ತಿದ್ದೇನೆ.",
    "ml-IN": "ദയവായി അല്പം കാത്തിരിക്കുക, വിവരങ്ങൾ പരിശോധിക്കുകയാണ്.",
}


from google import genai
from google.genai import types
from agent.schemas import LanguageClassification, CallClosingClassification

_gemini_key = os.getenv("GEMINI_API_KEY")
_gemini_client = genai.Client(api_key=_gemini_key) if _gemini_key else None
_lang_config = types.GenerateContentConfig(
    temperature=0.0,
    response_mime_type="application/json",
    response_schema=LanguageClassification,
)
_closing_config = types.GenerateContentConfig(
    temperature=0.0,
    response_mime_type="application/json",
    response_schema=CallClosingClassification,
)


async def identify_language(user_text: str) -> str:
    """Identify caller's spoken language using Gemini LLM with Pydantic classification schema."""
    if not user_text or not user_text.strip():
        return "Hindi"

    if _gemini_client:
        prompt = (
            "You are an Indian language identification specialist for a voice healthcare assistant. "
            "Identify the primary language spoken in this utterance strictly from the supported list: "
            "Gujarati, Hindi, English, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam.\n\n"
            f"Utterance: \"{user_text}\""
        )
        try:
            try:
                response = await asyncio.to_thread(
                    _gemini_client.models.generate_content,
                    model="gemini-3.1-flash-lite",
                    contents=prompt,
                    config=_lang_config,
                )
            except Exception:
                response = await asyncio.to_thread(
                    _gemini_client.models.generate_content,
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=_lang_config,
                )
            resp_text = (response.text or "").strip()
            result = LanguageClassification.model_validate_json(resp_text)
            return result.language.value
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


def get_resume_retry_text(language_code: str) -> str:
    return RESUME_RETRY_TEXTS.get(language_code, RESUME_RETRY_TEXTS["hi-IN"])


def get_safety_decline_text(language_code: str) -> str:
    return SAFETY_DECLINE_TEXTS.get(language_code, SAFETY_DECLINE_TEXTS["hi-IN"])


def get_processing_text(language_code: str) -> str:
    return PROCESSING_TEXTS.get(language_code, PROCESSING_TEXTS["hi-IN"])


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


# Primary sentence & clause delimiters for streaming speech
PRIMARY_DELIMS = re.compile(
    r'(?:[\?\!\n\u0964\u0965]+|(?<=[^\d\w\.\/][\w\d]{2})\.(?:\s+|$)|(?<=[^\d\W])\.(?:\s+|$))'
)
SECONDARY_DELIMS = re.compile(r'[,;:]+(?:\s+|$)')


class SentenceStreamer:
    """Buffers streaming token deltas and extracts clean, speakable sentences / clauses in real time."""

    def __init__(self, min_first_clause_len: int = 20, min_clause_len: int = 35):
        self.buffer = ""
        self.min_first_clause_len = min_first_clause_len
        self.min_clause_len = min_clause_len
        self.first_sentence_emitted = False

    def _is_too_short_or_list_prefix(self, text: str) -> bool:
        cleaned = text.strip(" \t\r\n.,;:!॥।")
        if not cleaned:
            return True
        if cleaned.isdigit() or len(cleaned) <= 2:
            return True
        return False

    def push(self, token: str) -> list[str]:
        if not token:
            return []
        self.buffer += token
        sentences = []
        while True:
            split_idx = self._find_split_point()
            if split_idx == -1:
                break
            raw_sentence = self.buffer[:split_idx].strip()
            if self._is_too_short_or_list_prefix(raw_sentence):
                break
            self.buffer = self.buffer[split_idx:].lstrip(" \t\r\n.,;:!॥।")
            sentence = clean_text_for_speech(raw_sentence)
            if sentence:
                sentences.append(sentence)
                self.first_sentence_emitted = True
        return sentences

    def flush(self) -> list[str]:
        rem = self.buffer.strip(" \t\r\n.,;:!॥।")
        self.buffer = ""
        sentence = clean_text_for_speech(rem)
        if sentence:
            return [sentence]
        return []

    def _find_split_point(self) -> int:
        match = PRIMARY_DELIMS.search(self.buffer)
        if match:
            # Avoid cutting off inside domain names e.g. 'pmjay.'
            if match.end() == len(self.buffer) and match.group().endswith("."):
                return -1
            cand = self.buffer[:match.end()].strip()
            if self._is_too_short_or_list_prefix(cand):
                match2 = PRIMARY_DELIMS.search(self.buffer, match.end())
                if match2:
                    return match2.end()
                return -1
            return match.end()

        # For fast initial speech, allow comma/semicolon split on early clauses
        min_len = self.min_first_clause_len if not self.first_sentence_emitted else self.min_clause_len
        if len(self.buffer) >= min_len:
            match = SECONDARY_DELIMS.search(self.buffer, min_len - 10)
            if match:
                return match.end()

        # Fallback if sentence is very long without punctuation
        if len(self.buffer) >= 120:
            last_space = self.buffer.rfind(" ", min_len, 120)
            if last_space != -1:
                return last_space + 1
        return -1


async def send_exotel_media_pcm(
    websocket: WebSocket,
    stream_sid: str,
    pcm: bytes,
    encoding: str = "base64",
):
    """Send PCM audio frames to Exotel over WebSocket in 100ms packetized chunks."""
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
    except Exception as e:
        logger.debug("Socket media send finished or closed: %s", e)


async def send_exotel_mark(
    websocket: WebSocket,
    stream_sid: str,
    mark_name: str = "bot_playback_complete",
):
    """Send playback completion mark event."""
    try:
        mark_msg = {
            "event": "mark",
            "stream_sid": stream_sid,
            "streamSid": stream_sid,
            "mark": {"name": mark_name},
        }
        await websocket.send_text(json.dumps(mark_msg))
    except Exception as e:
        logger.debug("Mark event error: %s", e)


async def send_exotel_media(
    websocket: WebSocket,
    stream_sid: str,
    pcm: bytes,
    encoding: str = "base64",
):
    """Stream audio chunks to Exotel and send completion mark."""
    await send_exotel_media_pcm(websocket, stream_sid, pcm, encoding=encoding)
    await send_exotel_mark(websocket, stream_sid, "bot_playback_complete")


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
            await send_exotel_media_pcm(websocket, stream_sid, pcm, encoding=encoding)
            await send_exotel_mark(websocket, stream_sid, "bot_playback_complete")
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


PROCESSING_FILLER_DELAY_SECONDS = float(os.getenv("PROCESSING_FILLER_DELAY_SECONDS", "5.0"))

FAREWELL_USER_WORDS = {
    # English
    "no", "nope", "nah", "nothing", "nothing else", "no more", "bye", "goodbye", "bye bye",
    "thanks", "thank you", "done", "stop", "thats all", "that is all", "no thanks",
    # Hindi / Hinglish
    "nahi", "nahin", "na", "na na", "kuch nahi", "kuch nahin", "bas", "alvida", "shukriya",
    "dhanyawad", "dhanyavad", "kuch nahi chahiye", "koi sawal nahi", "kuch aur nahi",
    "नहीं", "ना", "ना ना", "कुछ नहीं", "बस", "अलविदा", "शुक्रिया", "धन्यवाद", "कोई सवाल नहीं",
    # Gujarati / Gujlish
    "kai nahi", "kai nathi", "aavjo", "avjo", "aabhar", "kashu nahi", "kashu nathi", "nathi joi tu",
    "ના", "ના ના", "નાના", "કંઈ નહીં", "કંઈ નહિ", "કશું નહીં", "આવજો", "આભાર", "કંઈ નથી", "નથી",
    "તમારો આભાર", "હવે કંઈ નથી", "હવે કઈ નથી",
    # Marathi
    "kahi nahi", "nako", "namaskar", "काही नाही", "नको", "नमस्कार",
    # Bengali
    "kichu na", "dhonnobad", "aar kichu na", "কিছু না", "ধন্যবাদ", "আর কিছু না",
    # South Indian
    "illai", "vendam", "nandri", "இல்லை", "நன்றி",
    "ledu", "em ledu", "vaddu", "లేదు", "ఏం లేదు",
    "illa", "enu illa", "beda", "ಇಲ್ಲ", "ಏನು ಇಲ್ಲ",
    "onnum illa", "nanni", "venda", "ഇല്ല", "ഒന്നുമില്ല", "നന്ദി",
}

FAREWELL_BOT_KEYWORDS = [
    # Gujarati
    "આવજો", "ધ્યાન રાખજો", "આનંદ થયો", "કોલ કરવા બદલ આભાર", "વાત કરવા બદલ આભાર", "નમસ્તે", "સાથે વાત કરીને", "શુભ દિવસ",
    # Hindi / Hinglish
    "ख्याल रखिए", "ख्याल रखना", "अलविदा", "धन्यवाद", "शुभ दिन", "नमस्ते", "बात करने के लिए धन्यवाद",
    "khayal rakhiye", "khayal rakhna", "alvida", "dhanyavaad", "dhanyavad", "shubh din", "baat karne ke liye dhanyavaad",
    # English
    "goodbye", "take care", "stay healthy", "thank you for calling", "have a great day", "have a nice day",
    # Regional
    "bhalo thakben", "gavanamaaga", "jagrattaga", "nodikolli", "sradhikkuka", "aabhar", "potanu dhyan",
]


def is_closing_intent_heuristic(text: str) -> bool:
    """Fast rule-based/regex fallback for detecting if caller is concluding the call."""
    if not text:
        return False
    cleaned = re.sub(r"[\.\?\!\,\:\;\-\_]", "", text.strip().lower()).strip()
    cleaned_no_spaces = re.sub(r"\s+", " ", cleaned)
    if cleaned_no_spaces in FAREWELL_USER_WORDS or text.strip() in FAREWELL_USER_WORDS:
        return True
    for word in FAREWELL_USER_WORDS:
        if cleaned_no_spaces == word or cleaned_no_spaces.startswith(word + " ") or cleaned_no_spaces.endswith(" " + word):
            if len(cleaned_no_spaces.split()) <= 4:
                return True
    return False


async def is_closing_intent_llm(user_text: str, current_language: str = "Hindi") -> bool:
    """
    LLM-based structured closing intent classification across all 9 Indian languages and English.
    Determines if caller wants to conclude the conversation, declines further help, or says goodbye.
    Falls back gracefully to fast heuristic if Gemini API is unreachable.
    """
    if not user_text or not user_text.strip():
        return False

    cleaned_quick = re.sub(r"[\.\?\!\,\:\;\-\_]", "", user_text.strip().lower()).strip()
    # Immediate shortcut for single-word universally unambiguous closing words
    if cleaned_quick in ("bye", "goodbye", "alvida", "aavjo", "avjo", "namaste", "dhanyavaad", "dhanyavad", "shukriya", "aabhar", "thanks", "thank you", "stop"):
        return True

    if _gemini_client:
        prompt = (
            "You are an intent classification expert for BharatSwasthya AI, a multilingual Indian voice healthcare assistant.\n"
            "Determine if the caller's spoken utterance indicates they are concluding the phone call, declining further help, saying thank you/goodbye, or stating they have no further questions.\n\n"
            f"Active Language Context: {current_language}\n"
            "Supported languages: Gujarati, Hindi, English, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam (both native scripts and Latin transliteration e.g. Hinglish, Gujlish).\n\n"
            "Classification Rules:\n"
            "- Set is_closing to true if the caller expresses satisfaction, says no more questions, says thanks/bye to finish, or clearly indicates the call is done.\n"
            "- Set is_closing to false if the caller is asking a medical/health question, describing symptoms, asking about hospital or government schemes, or asking any follow-up question.\n\n"
            "Examples of CLOSING intent (is_closing: true):\n"
            "- 'Nahi, ab kuch nahi chahiye, dhanyawaad'\n"
            "- 'No thank you, that is all'\n"
            "- 'Have kai nathi joi tu, aabhar'\n"
            "- 'ના હવે કંઈ નથી પૂછવું, તમારો આભાર'\n"
            "- 'Bas itna hi tha, alvida'\n"
            "- 'Kahi nako, namaskar'\n"
            "- 'Aar kichu na, dhonnobad'\n"
            "- 'Illai, nandri'\n"
            "- 'Em ledu, thanks'\n"
            "- 'Enu beda, dhanyavada'\n"
            "- 'Onnum illa, nanni'\n\n"
            "Examples of NOT closing intent (is_closing: false):\n"
            "- 'Mujhe kal se bukhar hai'\n"
            "- 'Civil hospital kitni door hai?'\n"
            "- 'Ayushman card kaise banwaye?'\n"
            "- 'Mane pet ma dard thay che'\n"
            "- 'Doctor kahan milenge?'\n"
            "- 'Nahi mere paas Aadhar card hai, aage kya karein?' (This is clarifying a scheme question, not closing)\n\n"
            f"Caller Utterance: \"{user_text}\""
        )
        try:
            try:
                response = await asyncio.to_thread(
                    _gemini_client.models.generate_content,
                    model="gemini-3.1-flash-lite",
                    contents=prompt,
                    config=_closing_config,
                )
            except Exception:
                response = await asyncio.to_thread(
                    _gemini_client.models.generate_content,
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=_closing_config,
                )
            resp_text = (response.text or "").strip()
            result = CallClosingClassification.model_validate_json(resp_text)
            logger.info("LLM closing intent detection: %s (reason: %s) for input: '%s'", result.is_closing, result.reason, user_text)
            return result.is_closing
        except Exception as e:
            logger.warning("LLM closing intent check failed (%s), using heuristic fallback", e)

    return is_closing_intent_heuristic(user_text)


# Backwards compatibility alias
is_closing_intent = is_closing_intent_heuristic


def contains_closing_phrase(text: str) -> bool:
    """Detect if bot response contains farewell/call-ending phrases."""
    if not text:
        return False
    t_lower = text.lower()
    for phrase in FAREWELL_BOT_KEYWORDS:
        if phrase.lower() in t_lower:
            return True
    return False


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
    """
    Execute low-latency streaming healthcare conversation turn:
    Tokens stream from LangGraph -> split to sentences -> pipelined TTS -> streamed to Exotel WebSocket in real time.
    """
    language = current_language or "Hindi"
    language_code = current_language_code or "hi-IN"

    try:
        user_text = await asyncio.to_thread(stt, audio, language_code)
        user_text = (user_text or "").strip()
        logger.info("Caller query [Turn %d]: %s", turn_count, user_text)

        if not user_text:
            await speak(websocket, stream_sid, get_resume_retry_text(language_code), language_code, encoding=encoding)
            return language, language_code, False

        # Concurrent LLM safety guardrail & dynamic language detection (low-latency parallel execution)
        guard_task = asyncio.to_thread(check_input, user_text)
        lang_task = identify_language(user_text) if len(user_text.split()) >= 2 else None

        if lang_task:
            guard_result, detected_lang = await asyncio.gather(guard_task, lang_task)
            if detected_lang and detected_lang != language:
                language = detected_lang
                language_code = get_language_code(language)
                logger.info("Language dynamically updated via LLM: %s (%s)", language, language_code)
        else:
            guard_result = await guard_task

        if guard_result.get("unsafe"):
            logger.warning("Caller input flagged by LLM safety guardrail: %s", guard_result.get("reason"))
            decline_text = get_safety_decline_text(language_code)
            await speak(websocket, stream_sid, decline_text, language_code, encoding=encoding)
            return language, language_code, False

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

        # The ReAct Agent inside LangGraph natively detects closing intent and invokes end_call_tool.
        # Fast 0ms in-memory heuristic check used only to suppress filler audio if user says a quick farewell token.
        is_quick_farewell = is_closing_intent_heuristic(user_text)
        should_end = False

        sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        streamer = SentenceStreamer()
        full_tokens: list[str] = []
        first_token_event = asyncio.Event()

        # Adaptive delayed processing filler:
        # Only triggers if computation/tool execution takes > PROCESSING_FILLER_DELAY_SECONDS.
        # If the LLM generates tokens before 2.0s, this task is cancelled and plays nothing.
        async def adaptive_filler_worker():
            if is_quick_farewell:
                return
            try:
                await asyncio.sleep(PROCESSING_FILLER_DELAY_SECONDS)
                if not first_token_event.is_set():
                    logger.info("Agent tool computation taking > %.1fs, playing filler [%s]", PROCESSING_FILLER_DELAY_SECONDS, language_code)
                    filler_pcm = await tts_to_pcm(get_processing_text(language_code), language_code)
                    if filler_pcm and not first_token_event.is_set():
                        await send_exotel_media_pcm(websocket, stream_sid, filler_pcm, encoding=encoding)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug("Adaptive filler error: %s", e)

        filler_task = asyncio.create_task(adaptive_filler_worker())

        async def llm_token_producer():
            nonlocal should_end
            try:
                async for event in workflow.astream_events(turn_input, config=config, version="v2"):
                    kind = event.get("event")
                    if kind == "on_chat_model_stream":
                        # Only stream conversational agent tokens, ignore SMS summary & call finalization nodes
                        node = event.get("metadata", {}).get("langgraph_node")
                        if node in ("send_sms", "end_call"):
                            continue

                        chunk = event.get("data", {}).get("chunk")
                        if not chunk:
                            continue
                        # Skip tool call argument streaming chunks
                        if getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None):
                            continue
                        content = getattr(chunk, "content", "")
                        if not content:
                            continue
                        if isinstance(content, list):
                            text = "".join(
                                p.get("text", "") if isinstance(p, dict) else str(p)
                                for p in content
                            )
                        else:
                            text = str(content)
                        if not text:
                            continue

                        if not first_token_event.is_set():
                            first_token_event.set()
                            if not filler_task.done():
                                filler_task.cancel()

                        full_tokens.append(text)
                        new_sentences = streamer.push(text)
                        for s in new_sentences:
                            if any(k in s for k in ("CALL_TERMINATED:", "CALL_ENDED:", "CALL TERMINATED:", "CALL ENDED:")) or contains_closing_phrase(s):
                                should_end = True
                            clean_s = re.sub(r"CALL[_\s](?:TERMINATED|ENDED):?", "", s, flags=re.IGNORECASE).strip()
                            if clean_s:
                                await sentence_queue.put(clean_s)

                # Flush remaining buffer at the end of LLM generation
                remaining_sentences = streamer.flush()
                for s in remaining_sentences:
                    if any(k in s for k in ("CALL_TERMINATED:", "CALL_ENDED:", "CALL TERMINATED:", "CALL ENDED:")) or contains_closing_phrase(s):
                        should_end = True
                    clean_s = re.sub(r"CALL[_\s](?:TERMINATED|ENDED):?", "", s, flags=re.IGNORECASE).strip()
                    if clean_s:
                        await sentence_queue.put(clean_s)

            except Exception as e:
                logger.exception("LLM token streaming error: %s", e)
            finally:
                if not filler_task.done():
                    filler_task.cancel()
                await sentence_queue.put(None)

        async def tts_worker():
            try:
                while True:
                    sentence = await sentence_queue.get()
                    if sentence is None:
                        sentence_queue.task_done()
                        break
                    logger.info("Streaming TTS synthesizing [%s]: %s", language_code, sentence)
                    pcm = await tts_to_pcm(sentence, language_code)
                    if pcm:
                        await audio_queue.put(pcm)
                    sentence_queue.task_done()
            except Exception as e:
                logger.error("Streaming TTS worker failed: %s", e)
            finally:
                await audio_queue.put(None)

        async def audio_player_worker():
            total_audio_bytes = 0
            try:
                while True:
                    pcm = await audio_queue.get()
                    if pcm is None:
                        audio_queue.task_done()
                        break
                    total_audio_bytes += len(pcm)
                    await send_exotel_media_pcm(websocket, stream_sid, pcm, encoding=encoding)
                    audio_queue.task_done()
                if total_audio_bytes > 0:
                    await send_exotel_mark(websocket, stream_sid, "bot_playback_complete")
            except Exception as e:
                logger.debug("Audio playback worker closed: %s", e)

        # 1. Start LLM token producer with generation timeout
        producer_task = asyncio.create_task(llm_token_producer())
        tts_task = asyncio.create_task(tts_worker())
        player_task = asyncio.create_task(audio_player_worker())

        try:
            # Timeout only guards the LLM token generation phase
            await asyncio.wait_for(producer_task, timeout=AGENT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Agent LLM generation exceeded %.0fs timeout | call_sid=%s", AGENT_TIMEOUT_SECONDS, call_sid)
            producer_task.cancel()
            if not filler_task.done():
                filler_task.cancel()
            await sentence_queue.put(None)
            if not full_tokens:
                tts_task.cancel()
                player_task.cancel()
                await speak(websocket, stream_sid, get_agent_timeout_text(language_code), language_code, encoding=encoding)
                return language, language_code, False

        # 2. Wait for TTS synthesis and WebSocket audio playback to complete naturally
        await asyncio.gather(tts_task, player_task)

        # Verify state from checkpoint if call ended
        try:
            snapshot = workflow.get_state(config)
            if snapshot and snapshot.values and snapshot.values.get("call_ended"):
                should_end = True
        except Exception:
            pass

        # If no tokens were generated at all, play fallback closing
        if not full_tokens:
            closing = get_closing_text(language_code)
            await speak(websocket, stream_sid, closing, language_code, encoding=encoding)
            should_end = True

        if should_end:
            await asyncio.sleep(0.8)
            return language, language_code, True

        return language, language_code, False

    except Exception:
        logger.exception("Conversation turn failed")
        await speak(websocket, stream_sid, get_resume_retry_text(language_code), language_code, encoding=encoding)
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