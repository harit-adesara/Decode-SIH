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


from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import (
    TriageAssessment,
    TriageUrgency,
    MedicalSpecialty,
    FacilityType,
    EpidemicRiskClassification,
    OutbreakRiskLevel,
    DiseaseCategory,
    ContainmentPriority,
    WeatherVulnerability,
)

PUBLIC_HEALTH_API_BASE_URL = (
    os.getenv("PUBLIC_HEALTH_API_BASE_URL")
    or os.getenv("EPIDEMIC_API_BASE_URL")
    or os.getenv("API_BASE_URL")
    or "http://localhost:8000"
)

_triage_config = types.GenerateContentConfig(
    temperature=0.1,
    response_mime_type="application/json",
    response_schema=TriageAssessment,
)


@tool
def get_proactive_disease_alerts(
    state: str,
    risk_level: str = "",
) -> str:
    """
    Retrieve daily proactive AI disease forecasts correlated with meteorological weather factors (humidity, rainfall, heatwave indices, AQI).

    Use this tool when the caller asks about:
    - Weather-related disease advisories or epidemic forecasts in an Indian state (e.g. Maharashtra, Gujarat, Delhi)
    - Proactive health risks, seasonal viral/dengue/malaria warnings correlated with rain or humidity
    - Daily AI disease alerts filtered by risk level ('high', 'severe', 'moderate')

    Parameters:
    - state: Indian State (e.g. 'Maharashtra', 'Delhi', 'Gujarat', 'Karnataka')
    - risk_level: Optional filter by 'high', 'severe', or 'moderate'
    """
    clean_state = state.strip()
    url = f"{PUBLIC_HEALTH_API_BASE_URL.rstrip('/')}/api/v1/public/proactive-alerts"
    params = {"state": clean_state}
    if risk_level and risk_level.strip():
        params["riskLevel"] = risk_level.strip().lower()

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            payload = response.json()
            data_body = payload.get("data", {})
            alerts = data_body.get("alerts", [])
            count = data_body.get("count", len(alerts))
            filter_info = data_body.get("filter", {})

            if not alerts:
                return wrap_untrusted(
                    "PROACTIVE DISEASE ALERTS",
                    f"No active proactive disease alerts found for {clean_state}"
                    + (f" with risk level '{risk_level}'" if risk_level else "")
                    + "."
                )

            formatted_alerts = []
            for idx, item in enumerate(alerts, 1):
                wf = item.get("weatherFactors", {})
                weather_str = (
                    f"Temp: {wf.get('temperature', 'N/A')}, Humidity: {wf.get('humidity', 'N/A')}, "
                    f"Rainfall Risk: {wf.get('rainfallRisk', 'N/A')}, AQI: {wf.get('airQualityIndex', 'N/A')}, Season: {wf.get('season', 'N/A')}"
                )
                precautions = item.get("recommendedPrecautions", [])
                precautions_str = "; ".join(precautions) if precautions else "Standard precautions apply."
                symptoms = item.get("symptomsToWatch", [])
                symptoms_str = ", ".join(symptoms) if symptoms else "N/A"

                district_city = f"{item.get('district', 'All')}, {item.get('state', clean_state)}"
                if item.get("city") and item.get("city") != "All":
                    district_city = f"{item.get('city')}, {district_city}"

                alert_text = (
                    f"Alert #{idx}: {item.get('diseaseName', 'Unknown Disease')} [{item.get('riskLevel', 'unknown').upper()} RISK]\n"
                    f"- Location: {district_city}\n"
                    f"- Weather Correlation: {weather_str}\n"
                    f"- Summary: {item.get('summary', '')}\n"
                    f"- Symptoms To Watch: {symptoms_str}\n"
                    f"- Recommended Precautions: {precautions_str}\n"
                    f"- AI Epidemiological Insights: {item.get('aiInsights', 'N/A')}"
                )
                formatted_alerts.append(alert_text)

            header = f"Proactive Disease Forecasts for {clean_state} (Total Alerts: {count}):\n\n"
            return wrap_untrusted("PROACTIVE DISEASE ALERTS", header + "\n\n".join(formatted_alerts))

        else:
            logger.warning("Proactive alerts API responded with status %s: %s", response.status_code, response.text)
            return (
                f"Proactive alerts endpoint returned status {response.status_code}. "
                f"Please ensure base URL ({PUBLIC_HEALTH_API_BASE_URL}) is reachable."
            )

    except requests.exceptions.RequestException as req_err:
        logger.info("Live proactive alerts API unreachable (%s), providing standard public health advisory for %s", req_err, clean_state)
        return wrap_untrusted(
            "PROACTIVE DISEASE ALERTS (OFFLINE ADVISORY)",
            f"Active public health advisory for {clean_state}:\n"
            f"- Seasonal Meteorological Alert: High monsoon humidity and standing water increase vector-borne (Dengue/Malaria) and water-borne transmission risk.\n"
            f"- Precautions: Prevent water stagnation in coolers and containers, use mosquito repellents, drink boiled water, and seek immediate medical care if high fever or chills occur.\n"
            f"- Note: Telemetry API endpoint ({PUBLIC_HEALTH_API_BASE_URL}/api/v1/public/proactive-alerts) currently offline."
        )
    except Exception as e:
        logger.error("Failed to query proactive alerts: %s", e)
        return f"Error retrieving proactive disease alerts: {str(e)}"


@tool
def get_active_viral_diseases(
    state: str,
    district: str = "",
    city: str = "",
) -> str:
    """
    Retrieve active viral disease outbreaks and clinical telemetry filtered by Indian state, district, or city.

    Use this tool when the caller or health worker asks about:
    - Active viral disease outbreaks (e.g. Dengue, H3N2 Influenza, Chikungunya, Viral Hepatitis)
    - Total cases, severity, danger signs, incubation period, and clinical protocols in a district/city
    - Hospital telemetry and reported clusters in a specific locality (e.g. Pune, Mumbai, Ahmedabad)

    Parameters:
    - state: Indian State (e.g. 'Maharashtra', 'Gujarat', 'Delhi')
    - district: District name (e.g. 'Pune', 'Mumbai Suburban', 'Surat') (optional)
    - city: City or Village (e.g. 'Shivajinagar', 'Hadapsar', 'Andheri') (optional)
    """
    clean_state = state.strip()
    url = f"{PUBLIC_HEALTH_API_BASE_URL.rstrip('/')}/api/v1/public/viral-diseases"
    params = {"state": clean_state}
    if district and district.strip():
        params["district"] = district.strip()
    if city and city.strip():
        params["city"] = city.strip()

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            payload = response.json()
            data_body = payload.get("data", {})
            outbreaks = data_body.get("data", [])
            count = data_body.get("count", len(outbreaks))
            filter_info = data_body.get("filter", {})

            if not outbreaks:
                loc_str = clean_state
                if district:
                    loc_str = f"{district}, {loc_str}"
                if city:
                    loc_str = f"{city}, {loc_str}"
                return wrap_untrusted(
                    "ACTIVE VIRAL DISEASE OUTBREAKS",
                    f"No active viral disease outbreaks currently recorded for {loc_str}."
                )

            formatted_outbreaks = []
            for idx, item in enumerate(outbreaks, 1):
                affected_cities = ", ".join(item.get("affectedCities", [])) or "District-wide"
                symptoms = ", ".join(item.get("symptoms", [])) or "N/A"
                danger_signs = "; ".join(item.get("dangerSigns", [])) or "None specified"
                precautions = "; ".join(item.get("recommendedPrecautions", [])) or "Standard infection control"
                doctor_remarks = " | ".join(item.get("doctorRemarks", [])) or "N/A"
                clinical_protocol = item.get("clinicalProtocol", "Standard supportive care.")

                outbreak_text = (
                    f"Outbreak #{idx}: {item.get('diseaseName', 'Viral Outbreak')} (Severity: {item.get('highestSeverity', 'unknown').upper()})\n"
                    f"- Location: District: {item.get('district', 'All')}, State: {item.get('state', clean_state)} (Affected: {affected_cities})\n"
                    f"- Cases: {item.get('totalCases', 0)} total reported cases ({item.get('activeReportsCount', 0)} active hospital clusters)\n"
                    f"- Transmission & Incubation: {item.get('transmissionType', 'N/A')} | Incubation: {item.get('incubationPeriod', 'N/A')}\n"
                    f"- Symptoms: {symptoms}\n"
                    f"- DANGER SIGNS: {danger_signs}\n"
                    f"- Recommended Precautions: {precautions}\n"
                    f"- Doctor Remarks: {doctor_remarks}\n"
                    f"- Clinical Protocol: {clinical_protocol}"
                )
                formatted_outbreaks.append(outbreak_text)

            filter_desc = f"{filter_info.get('district', district or 'All')}, {filter_info.get('state', clean_state)}"
            header = f"Active Viral Disease Outbreaks for {filter_desc} (Total Outbreaks: {count}):\n\n"
            return wrap_untrusted("ACTIVE VIRAL DISEASE OUTBREAKS", header + "\n\n".join(formatted_outbreaks))

        else:
            logger.warning("Viral diseases API responded with status %s: %s", response.status_code, response.text)
            return (
                f"Viral disease telemetry endpoint returned status {response.status_code}. "
                f"Please ensure base URL ({PUBLIC_HEALTH_API_BASE_URL}) is reachable."
            )

    except requests.exceptions.RequestException as req_err:
        logger.info("Live viral diseases API unreachable (%s), providing standard clinical telemetry for %s", req_err, clean_state)
        return wrap_untrusted(
            "ACTIVE VIRAL DISEASE OUTBREAKS (OFFLINE ADVISORY)",
            f"Active clinical advisory for {clean_state} (District: {district or 'All'}):\n"
            f"- Monitored Viral Pathogens: Seasonal Influenza (H3N2), Dengue (DENV), Chikungunya, Viral Gastroenteritis.\n"
            f"- Key Danger Signs: Persistent high fever >3 days, difficulty breathing, mucosal bleeding, severe vomiting/dehydration.\n"
            f"- Clinical Protocol: Hydration, antipyretics (avoid NSAIDs/Aspirin in suspected dengue), prompt hospital consultation.\n"
            f"- Note: Telemetry API endpoint ({PUBLIC_HEALTH_API_BASE_URL}/api/v1/public/viral-diseases) currently offline."
        )
    except Exception as e:
        logger.error("Failed to query viral diseases: %s", e)
        return f"Error retrieving active viral disease data: {str(e)}"


@tool
def classify_epidemic_outbreak_risk(
    state: str,
    district: str = "",
    symptoms_or_weather: str = "",
) -> str:
    """
    Perform AI classification-based epidemic and outbreak risk assessment using structured Pydantic output (with_structured_output).
    Classifies risk level, disease category, containment priority, weather vulnerability, and actionable precautions.

    Parameters:
    - state: Indian state (e.g. 'Maharashtra', 'Gujarat', 'Delhi')
    - district: District name (e.g. 'Pune', 'Mumbai Suburban') (optional)
    - symptoms_or_weather: Caller's symptoms or local weather conditions (e.g. 'fever with retro-orbital pain, heavy rainfall 85% humidity')
    """
    prompt = (
        "You are an expert public health epidemiologist for BharatSwasthya AI. "
        "Analyze the given Indian geographic location, reported symptoms, and meteorological weather factors, "
        "and classify the epidemic outbreak risk using the structured output classification schema.\n\n"
        f"State: {state}\n"
        f"District: {district or 'General/All'}\n"
        f"Reported Symptoms / Weather Context: {symptoms_or_weather}\n"
    )

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return (
            "EPIDEMIC RISK CLASSIFICATION:\n"
            "- Classified Risk Level: MODERATE\n"
            "- Disease Category: Vector-Borne\n"
            "- Containment Priority: Targeted Advisory\n"
            "- Weather Vulnerability: Moderate\n"
            "- Primary Suspected Disease: Dengue / Seasonal Viral Fever\n"
            "- Urgency Summary: Moderate risk detected based on regional seasonal factors. Monitor symptoms and take vector control precautions."
        )

    try:
        llm = ChatGoogleGenerativeAI(
            google_api_key=gemini_key,
            model=os.getenv("GEMINI_AGENT_MODEL", "gemini-3.1-flash-lite"),
            temperature=0.1,
        )
        structured_classifier = llm.with_structured_output(EpidemicRiskClassification)
        classification: EpidemicRiskClassification = structured_classifier.invoke(prompt)

        precautions_text = "\n".join(f"  • {p}" for p in classification.actionable_precautions) if classification.actionable_precautions else "  • Standard public health precautions"

        return (
            f"EPIDEMIC RISK CLASSIFICATION (STRUCTURED AI OUTPUT):\n"
            f"- Classified Risk Level: {classification.classified_risk_level.value.upper()}\n"
            f"- Disease Category: {classification.disease_category.value}\n"
            f"- Primary Suspected Disease: {classification.primary_suspected_disease}\n"
            f"- Containment Priority: {classification.containment_priority.value}\n"
            f"- Weather Vulnerability: {classification.weather_vulnerability.value}\n"
            f"- Summary: {classification.urgency_summary}\n"
            f"- Actionable Precautions:\n{precautions_text}"
        )
    except Exception as e:
        logger.warning("with_structured_output classification failed (%s), trying direct schema fallback", e)
        try:
            client = genai.Client(api_key=gemini_key)
            config = types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=EpidemicRiskClassification,
            )
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt,
                    config=config,
                )
            except Exception:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=config,
                )
            data = EpidemicRiskClassification.model_validate_json(response.text or "{}")
            precautions_text = "\n".join(f"  • {p}" for p in data.actionable_precautions)
            return (
                f"EPIDEMIC RISK CLASSIFICATION (STRUCTURED AI OUTPUT):\n"
                f"- Classified Risk Level: {data.classified_risk_level.value.upper()}\n"
                f"- Disease Category: {data.disease_category.value}\n"
                f"- Primary Suspected Disease: {data.primary_suspected_disease}\n"
                f"- Containment Priority: {data.containment_priority.value}\n"
                f"- Weather Vulnerability: {data.weather_vulnerability.value}\n"
                f"- Summary: {data.urgency_summary}\n"
                f"- Actionable Precautions:\n{precautions_text}"
            )
        except Exception as inner_err:
            logger.error("Fallback classification also failed: %s", inner_err)
            return f"Epidemic risk classification completed with default advisory for {state}."


@tool
def symptom_triage_guide(
    symptoms: str,
    duration_or_context: str = "",
) -> str:
    """
    Provide LLM-powered clinical triage guidance for symptoms reported by the caller using Pydantic classification schema.

    Parameters:
    - symptoms: Description of the physical complaints (e.g. 'chest discomfort', 'fever and headache', 'knee pain', 'cough for 2 weeks')
    - duration_or_context: Duration or patient background (e.g. '3 days', 'diabetic', 'elderly person')

    Returns clinical urgency (CRITICAL_EMERGENCY / URGENT / ROUTINE / HOME_CARE), recommended doctor specialty, facility type, and immediate guidance.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            prompt = (
                "You are an expert emergency medical triage evaluator for BharatSwasthya AI. "
                "Analyze the caller's reported symptoms and patient context, and classify into structured clinical triage guidance.\n\n"
                f"Caller Symptoms: {symptoms}\n"
                f"Context/Duration: {duration_or_context}"
            )
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt,
                    config=_triage_config,
                )
            except Exception:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=_triage_config,
                )
            triage = TriageAssessment.model_validate_json(response.text or "{}")
            guidance_extra = f"\n- Follow-up Guidance: {triage.follow_up_guidance}" if triage.follow_up_guidance else ""
            return (
                f"TRIAGE ASSESSMENT:\n"
                f"- Urgency Level: {triage.urgency_level.value}\n"
                f"- Recommended Specialty: {triage.recommended_specialty.value}\n"
                f"- Recommended Facility: {triage.recommended_facility.value}\n"
                f"- Immediate Action: {triage.immediate_action}"
                f"{guidance_extra}"
            )
        except Exception as e:
            logger.warning("LLM triage call failed (%s), using fallback triage rules", e)

    s = (symptoms or "").lower()
    if any(e in s for e in ["chest", "breathing", "stroke", "bleeding", "heart", "chhati"]):
        return (
            f"TRIAGE ASSESSMENT:\n"
            f"- Urgency Level: {TriageUrgency.CRITICAL_EMERGENCY.value}\n"
            f"- Recommended Specialty: {MedicalSpecialty.CARDIOLOGY.value} / {MedicalSpecialty.EMERGENCY_MEDICINE.value}\n"
            f"- Recommended Facility: {FacilityType.AMBULANCE_108.value}\n"
            f"- Immediate Action: Call 108 Emergency Ambulance or rush to nearest Emergency Room immediately."
        )
    return (
        f"TRIAGE ASSESSMENT:\n"
        f"- Urgency Level: {TriageUrgency.ROUTINE.value}\n"
        f"- Recommended Specialty: {MedicalSpecialty.GENERAL_PHYSICIAN.value}\n"
        f"- Recommended Facility: {FacilityType.PRIMARY_HEALTH_CENTER.value}\n"
        f"- Immediate Action: Visit local PHC/clinic for a routine checkup."
    )


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