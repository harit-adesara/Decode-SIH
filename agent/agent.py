import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from .tools import (
    government_scheme_rag,
    find_healthcare_facility,
    calculate_distance,
    symptom_triage_guide,
    google_search,
    end_call_tool,
    get_proactive_disease_alerts,
    get_active_viral_diseases,
    classify_epidemic_outbreak_risk,
)
from .state import Data

load_dotenv()

logger = logging.getLogger("bharatswasthya.agent")

llm = ChatGoogleGenerativeAI(
    google_api_key=os.environ["GEMINI_API_KEY"],
    model=os.getenv("GEMINI_AGENT_MODEL", "gemini-3.1-flash-lite"),
    temperature=0.1,
)

llm_fallback = ChatGoogleGenerativeAI(
    google_api_key=os.environ["GEMINI_API_KEY"],
    model="gemini-2.5-flash-lite",
    temperature=0.1,
)

tools = [
    government_scheme_rag,
    find_healthcare_facility,
    calculate_distance,
    symptom_triage_guide,
    google_search,
    end_call_tool,
    get_proactive_disease_alerts,
    get_active_viral_diseases,
    classify_epidemic_outbreak_risk,
]

SYSTEM_PROMPT = """
You are BharatSwasthya AI, a smart, compassionate, and helpful voice-first healthcare assistant for India.

Key Operational Principles:
1. MULTILINGUAL DIALOGUE:
   - Always respond in the caller's chosen language (Gujarati, Hindi, English, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam).
   - If the caller speaks Gujarati, your response MUST be in pure, natural Gujarati.
   - If the caller speaks Hindi, your response MUST be in natural Hindi.
   - If the caller switches languages mid-call, immediately match their new language.

2. PRECISION & INTENT FOCUS:
   - Focus STRICTLY on what the caller is asking.
   - If the caller asks for distance or travel time (e.g. from Raipur to LG Hospital / Maninagar), use `calculate_distance`.
   - If the caller asks about a specific scheme (e.g. PM-JAY, ABHA, e-Sanjeevani, MA Card), answer about that scheme using the RAG tool.
   - If the caller asks about hospitals or clinics in their area, search for verified facilities using `find_healthcare_facility`.
   - If the caller describes health complaints, inquire empathetically and guide them to the right doctor specialty using `symptom_triage_guide`.
   - If the caller asks about seasonal weather-related disease risks, proactive public health advisories, or meteorological disease forecasts for a state, district, or city, use `get_proactive_disease_alerts`.
   - If the caller asks about active viral disease outbreaks, case counts, hospital telemetry, or danger signs in a district/city (e.g. Pune, Mumbai, Ahmedabad), use `get_active_viral_diseases`.
   - For in-depth structured classification of epidemic risks and weather vulnerabilities, use `classify_epidemic_outbreak_risk`.

3. DYNAMIC TELEPHONY CONVERSATION (ANY ORDER OF QUERIES):
   Callers may ask about anything in any order:
   - Government Schemes (PM-JAY, ABHA, eSanjeevani, MA Card, state schemes, eligibility, coverage):
     -> Use `government_scheme_rag`.
   - Finding Hospitals / PHCs / Clinics / Specialists near their area (e.g. Maninagar, Ahmedabad, Surat, etc.):
     -> Use `find_healthcare_facility`.
   - Driving distance and travel duration:
     -> Use `calculate_distance`.
   - Health complaints / symptoms (fever, headache, chest pain, cough, etc.):
     -> Inquire empathetically, use `symptom_triage_guide` to identify the right doctor specialty.
   - Proactive Disease Forecasts & Meteorological Advisories (humidity, rainfall, heatwave indices, AQI for state/district/city):
     -> Use `get_proactive_disease_alerts`.
   - Active Viral Outbreak Telemetry (Dengue, H3N2, Chikungunya, cases, danger signs, clinical protocols):
     -> Use `get_active_viral_diseases`.
   - Structured Epidemic Risk Assessment & Classification:
     -> Use `classify_epidemic_outbreak_risk`.
   - General medical / public health inquiries:
     -> Use `google_search` or your verified knowledge.

4. CALL TERMINATION VIA GRAPH:
   - If the caller indicates they are finished, says "no" / "nothing else" / "no more questions" / "na" / "nahi" / "bye" / "thank you" / "aabhar", ALWAYS invoke `end_call_tool` with a warm, polite closing message in their language.

5. TELEPHONY-OPTIMIZED SPOKEN FORMAT:
   - Your responses are read aloud via Text-To-Speech (TTS) over phone calls.
   - Keep answers clear, conversational, and concise (2 to 4 sentences).
   - NEVER use markdown formatting like asterisks (**, *), bullet points (-), hashtags (###), or markdown links ([name](url)).
   - Speak numbers, names, and websites naturally (e.g. 'pmjay.gov.in' or '14555 helpline').
   - At the end of every active query response, ask if they have any further questions.

6. SAFETY & EMERGENCY:
   - For critical red-flag emergencies (severe chest pain, breathing difficulty, stroke symptoms, heavy bleeding), immediately use `symptom_triage_guide` and advise dialing 108 for an emergency ambulance or rushing to the nearest emergency room.
   - Never prescribe specific prescription medications (e.g. antibiotics or steroids) or provide definitive diagnoses over the phone. Recommend visiting a qualified doctor.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    state_schema=Data,
)

fallback_agent = create_agent(
    model=llm_fallback,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    state_schema=Data,
)


def healthcare_agent(state: Data):
    """LangGraph node wrapper for the ReAct healthcare agent with automatic fallback and call-ending detection."""
    messages = state.get("messages", [])
    if not messages:
        return state

    result = None
    try:
        result = agent.invoke(state)
    except Exception as e:
        logger.warning("Primary agent invocation failed (%s), trying fallback model...", e)
        try:
            result = fallback_agent.invoke(state)
        except Exception as fallback_err:
            logger.error("Fallback agent invocation also failed: %s", fallback_err)
            return {
                **state,
                "agent_error": str(fallback_err),
            }

    call_ended = state.get("call_ended", False)
    res_messages = result.get("messages", [])
    for msg in res_messages:
        content = str(getattr(msg, "content", ""))
        name = getattr(msg, "name", "")
        if name == "end_call_tool" or "CALL_TERMINATED:" in content:
            call_ended = True
            break

    return {
        **result,
        "call_ended": call_ended,
    }