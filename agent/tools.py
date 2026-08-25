import os
import logging
import requests
from dotenv import load_dotenv
from google import genai
from langchain_core.tools import tool
from tavily import TavilyClient

from .rag import retrieve_and_rerank
from .guardrails import wrap_untrusted


load_dotenv()

logger = logging.getLogger("bharatswasthya.tools")

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

EXOTEL_ACCOUNT_SID = os.getenv("EXOTEL_ACCOUNT_SID") or os.getenv("ACCOUNT_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY") or os.getenv("API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN") or os.getenv("API_TOKEN")
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")


def hangup_exotel_call(call_sid: str) -> bool:
    """Attempt to terminate an ongoing Exotel call via REST API."""
    if not (EXOTEL_ACCOUNT_SID and EXOTEL_API_KEY and EXOTEL_API_TOKEN and call_sid):
        return True

    url = f"https://{EXOTEL_SUBDOMAIN}/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls/{call_sid}"
    try:
        response = requests.post(
            url,
            auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN),
            data={"Status": "completed"},
            timeout=10,
        )
        return response.status_code in (200, 201, 204)
    except Exception as e:
        logger.warning("Exotel call hangup request failed: %s", e)
        return False


@tool
def government_scheme_rag(query: str) -> str:
    """
    Retrieve accurate information about Indian government healthcare schemes from the BharatSwasthya knowledge base.

    Use this tool when the caller asks about:
    - Ayushman Bharat PM-JAY, ABHA health ID, eSanjeevani telemedicine, Jan Aushadhi, JSy, PMSSY
    - State government health schemes (e.g. MA Card Gujarat, Arogyasri, Swasthya Sathi, Mahatma Jyotirao Phule)
    - Eligibility criteria, covered hospital treatments, financial benefits (e.g. 5 lakh cover)
    - Required documents (Aadhaar, Ration Card) and registration/application steps
    """
    try:
        context = retrieve_and_rerank(query)
        return wrap_untrusted("GOVERNMENT SCHEME KNOWLEDGE BASE", context)
    except Exception as e:
        logger.error("Government scheme RAG query failed: %s", e)
        return f"Unable to query scheme knowledge base: {e}"


@tool
def find_healthcare_facility(
    location: str,
    facility_type: str = "hospital",
) -> str:
    """
    Find hospitals, clinics, Primary Health Centers (PHC), Community Health Centers (CHC),
    or specialized medical centers near a specific location/area in India.

    Parameters:
    - location: The city, town, or neighborhood (e.g. 'Maninagar, Ahmedabad', 'Surat', 'Andheri, Mumbai')
    - facility_type: Type of facility (e.g. 'hospital', 'clinic', 'PHC', 'eye hospital', 'maternity center')
    """
    query = f"top verified {facility_type} in {location} India address phone number"

    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

        results = []
        if response.get("answer"):
            results.append(f"Summary: {response['answer']}")

        for item in response.get("results", []):
            title = item.get("title", "Healthcare Facility")
            content = item.get("content", "")[:350]
            url = item.get("url", "")
            results.append(f"Facility Name: {title}\nDetails: {content}\nSource: {url}")

        if not results:
            return f"No specific facilities found for {facility_type} in {location}."

        return wrap_untrusted("HEALTHCARE FACILITY SEARCH", "\n\n".join(results))

    except Exception as e:
        logger.error("Healthcare facility search failed: %s", e)
        return f"Failed to search healthcare facilities: {str(e)}"


@tool
def calculate_distance(
    origin: str,
    destination: str,
) -> str:
    """
    Calculate driving distance and estimated travel time between the user's location and a healthcare facility.

    Parameters:
    - origin: Starting address or area (e.g. 'Maninagar, Ahmedabad')
    - destination: Target hospital or clinic (e.g. 'Civil Hospital, Asarwa, Ahmedabad')
    """
    if not GOOGLE_MAPS_API_KEY:
        try:
            res = tavily_client.search(
                query=f"distance and driving time from {origin} to {destination} Ahmedabad India",
                max_results=2,
                include_answer=True,
            )
            answer = res.get("answer")
            if answer:
                return f"Estimated distance information: {answer}"
        except Exception:
            pass
        return f"Distance between {origin} and {destination} is typically within local driving distance (approx 5 to 15 km depending on traffic)."

    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration",
    }
    body = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("routes"):
            return "Could not calculate exact driving route."

        route = data["routes"][0]
        distance_km = route["distanceMeters"] / 1000
        duration_min = round(float(route["duration"].rstrip("s")) / 60)

        return f"Driving Distance: {distance_km:.1f} km. Estimated Travel Time: {duration_min} minutes."

    except Exception as e:
        logger.warning("Google Routes API failed: %s", e)
        return f"Estimated driving distance from {origin} to {destination} is approximately 5-15 km."


@tool
def end_call_tool(closing_message: str) -> str:
    """
    Use this tool when the caller indicates they are done, says 'no' / 'no more questions' / 'thank you' / 'bye' / 'goodbye', or wants to end the phone call.

    Parameters:
    - closing_message: A warm, polite closing greeting in the caller's spoken language (e.g. Gujarati: 'તમારી સાથે વાત કરીને આનંદ થયો. પોતાનું ધ્યાન રાખજો, આવજો!', Hindi: 'BharatSwasthya AI se baat karne ke liye dhanyavaad. Apna khayal rakhiye. Namaste!').
    """
    return f"CALL_TERMINATED: {closing_message}"


@tool
def symptom_triage_guide(
    symptoms: str,
    duration_or_context: str = "",
) -> str:
    """
    Provide LLM-powered clinical triage guidance for symptoms reported by the caller.

    Parameters:
    - symptoms: Description of the physical complaints (e.g. 'chest discomfort', 'fever and headache', 'knee pain', 'cough for 2 weeks')
    - duration_or_context: Duration or patient background (e.g. '3 days', 'diabetic', 'elderly person')

    Returns clinical urgency (EMERGENCY / URGENT / ROUTINE / HOME_CARE), recommended doctor specialty, facility type, and immediate guidance.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            prompt = (
                "You are an expert emergency medical triage evaluator for BharatSwasthya AI. "
                "Analyze the caller's reported symptoms and patient context.\n\n"
                "Determine:\n"
                "1. Urgency Level: (CRITICAL_EMERGENCY / URGENT / ROUTINE / HOME_CARE)\n"
                "2. Recommended Medical Specialty (e.g. Cardiology, Neurology, Orthopedics, Pediatrics, OB/GYN, ENT, Ophthalmology, Dermatology, General Physician)\n"
                "3. Recommended facility type (e.g. 108 Emergency Ambulance / Emergency Room, PHC, CHC, Specialist Clinic)\n"
                "4. Immediate lifesaving action (if emergency: emphasize calling 108 ambulance immediately) and follow-up clinical questions.\n\n"
                f"Caller Symptoms: {symptoms}\n"
                f"Context/Duration: {duration_or_context}\n\n"
                "Triage Assessment:"
            )
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt,
                    config={"temperature": 0.1},
                )
            except Exception:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config={"temperature": 0.1},
                )
            return (response.text or "").strip()
        except Exception as e:
            logger.warning("LLM triage call failed (%s), using fallback triage rules", e)

    s = (symptoms or "").lower()
    if any(e in s for e in ["chest", "breathing", "stroke", "bleeding", "heart", "chhati"]):
        return "TRIAGE: CRITICAL EMERGENCY. Action: Call 108 Emergency Ambulance or rush to nearest Emergency Room immediately. Specialty: Cardiology / Emergency Medicine."
    return "TRIAGE: Routine Consultation. Recommended Specialty: General Physician / PHC."


@tool
def google_search(query: str) -> str:
    """
    Search the internet via Tavily for general healthcare, public health, hospital timings, or current updates.
    """
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=3,
            include_answer=True,
        )

        output = []
        if response.get("answer"):
            output.append(f"Summary:\n{response['answer']}")

        for item in response.get("results", []):
            output.append(f"Title: {item.get('title')}\nDetails: {item.get('content')[:300]}")

        return wrap_untrusted("WEB SEARCH", "\n\n".join(output)) if output else "No results found."

    except Exception as e:
        return f"Search failed: {str(e)}"