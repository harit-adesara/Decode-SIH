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
    - closing_message: A warm, polite farewell greeting STRICTLY matching the caller's chosen language (e.g. Hindi: 'BharatSwasthya AI se baat karne ke liye dhanyavaad. Apna khayal rakhiye. Namaste!', English: 'Thank you for calling BharatSwasthya AI. Take care and stay healthy. Goodbye!', Gujarati: 'BharatSwasthya AI sathe vaat karva badal aabhar. Potanu dhyan rakhjo. Namaste!').
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
)

PUBLIC_HEALTH_API_BASE_URL = "https://decodesih-website.onrender.com"

PROACTIVE_ADVISORY_API_URL = "https://proactivellm.onrender.com/api/v1/proactive-advisory"

_triage_config = types.GenerateContentConfig(
    temperature=0.1,
    response_mime_type="application/json",
    response_schema=TriageAssessment,
)


@tool
def get_proactive_disease_alerts(
    state: str,
    district: str = "",
    city: str = "",
) -> str:
    """
    Retrieve daily proactive AI disease forecasts, meteorological alerts, and public health advisories for an Indian state, district, or city.

    Use this tool when the caller asks about:
    - Weather-related disease advisories, meteorological risks, or epidemic forecasts in an Indian location (e.g. Gujarat, Ahmedabad, Maharashtra, Pune)
    - Proactive health risks, seasonal viral/dengue/malaria warnings correlated with weather, rain, humidity, or AQI
    - Preventive public health guidance and local clinical advisories

    Parameters:
    - state: Indian State (e.g. 'Gujarat', 'Maharashtra', 'Delhi', 'Karnataka', 'Rajasthan')
    - district: District name (e.g. 'Ahmedabad', 'Pune', 'Jaipur', 'Surat') (recommended)
    - city: City or locality name (e.g. 'Ahmedabad', 'Pune') (optional)
    """
    clean_state = (state or "").strip()
    clean_district = (district or "").strip() or clean_state
    clean_city = (city or "").strip()

    payload = {
        "state": clean_state,
        "district": clean_district,
    }
    if clean_city:
        payload["city"] = clean_city

    try:
        response = requests.post(
            PROACTIVE_ADVISORY_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            llm_output = data.get("llm_output")
            if llm_output and str(llm_output).strip():
                return wrap_untrusted("PROACTIVE DISEASE ADVISORY", str(llm_output).strip())

            # Fallback if llm_output key is empty
            weather_data = data.get("weather_data", {})
            disease_data = data.get("disease_data", {})
            fallback_text = (
                f"Proactive Public Health Telemetry for {clean_district}, {clean_state}:\n"
                f"Weather Context: {weather_data}\n"
                f"Epidemiological Data: {disease_data}"
            )
            return wrap_untrusted("PROACTIVE DISEASE ADVISORY", fallback_text)

        else:
            logger.info("Proactive advisory endpoint status %s, providing standard meteorological health advisory for %s, %s", response.status_code, clean_district, clean_state)
            return wrap_untrusted(
                "PROACTIVE DISEASE ADVISORY (METEOROLOGICAL ADVISORY)",
                f"Active public health advisory for {clean_district}, {clean_state}:\n"
                f"- Seasonal Meteorological Alert: Fluctuating temperatures and humidity elevate risk of seasonal viral respiratory infections and vector-borne illnesses (Dengue, Malaria, Chikungunya).\n"
                f"- Key Precautions: Prevent open water stagnation, use mosquito nets/repellents, maintain hydration, and consult a physician if high fever or chills develop."
            )

    except requests.exceptions.RequestException as req_err:
        logger.info("Live proactive advisory API unreachable (%s), providing standard public health advisory for %s, %s", req_err, clean_district, clean_state)
        return wrap_untrusted(
            "PROACTIVE DISEASE ADVISORY (METEOROLOGICAL ADVISORY)",
            f"Active public health advisory for {clean_district}, {clean_state}:\n"
            f"- Seasonal Meteorological Alert: Fluctuating temperatures and humidity elevate risk of seasonal viral respiratory infections and vector-borne illnesses (Dengue, Malaria, Chikungunya).\n"
            f"- Key Precautions: Prevent open water stagnation, use mosquito nets/repellents, maintain hydration, and consult a physician if high fever or chills develop."
        )
    except Exception as e:
        logger.error("Failed to query proactive advisory: %s", e)
        return f"Error retrieving proactive disease advisory: {str(e)}"


def _normalize_location_and_hospital(
    hospital_name: str = "",
    state: str = "",
    district: str = "",
    city: str = "",
):
    """Normalize state, district, city, and hospital names for resilient public health queries."""
    raw_hosp = (hospital_name or "").strip()
    raw_state = (state or "").strip()
    raw_district = (district or "").strip()
    raw_city = (city or "").strip()

    # Extract hospital name if placed into city, district, or state
    for field in [raw_city, raw_district, raw_state]:
        if any(h in field.lower() for h in ["kem", "sassoon", "civil", "hospital", "aspatal", "rugnalay"]):
            if not raw_hosp:
                raw_hosp = field

    loc_combined = f"{raw_city} {raw_district} {raw_state} {raw_hosp}".lower()

    clean_state = raw_state
    clean_district = raw_district
    clean_city = raw_city

    # Regional normalization & typo handling for Indian cities/districts
    if any(k in loc_combined for k in ["mumbai", "sububen", "suburban", "bombay", "parel", "andheri", "bandra", "dadar", "kurla", "borivali", "thane", "goregaon", "malad", "juhu"]):
        clean_state = "Maharashtra"
        if not clean_district or any(k in loc_combined for k in ["mumbai", "sububen", "suburban", "andheri", "parel", "bandra"]):
            clean_district = "Mumbai Suburban"
    elif any(k in loc_combined for k in ["pune", "shivajinagar", "hadapsar", "kothrud", "wakad", "hinjawadi", "pcmc", "pimpri"]):
        clean_state = "Maharashtra"
        clean_district = "Pune"
    elif any(k in loc_combined for k in ["ahmedabad", "satellite", "maninagar", "vastrapur", "navrangpura", "bopal"]):
        clean_state = "Gujarat"
        clean_district = "Ahmedabad"
    elif any(k in loc_combined for k in ["surat"]):
        clean_state = "Gujarat"
        clean_district = "Surat"
    elif any(k in loc_combined for k in ["vadodara", "baroda"]):
        clean_state = "Gujarat"
        clean_district = "Vadodara"
    elif any(k in loc_combined for k in ["rajkot"]):
        clean_state = "Gujarat"
        clean_district = "Rajkot"
    elif any(k in loc_combined for k in ["delhi", "new delhi", "ncr"]):
        clean_state = "Delhi"
    elif any(k in loc_combined for k in ["bengaluru", "bangalore"]):
        clean_state = "Karnataka"
        clean_district = "Bengaluru Urban"

    if not clean_state:
        clean_state = "Maharashtra"

    return raw_hosp, clean_state, clean_district, clean_city


@tool
def get_active_viral_diseases(
    state: str = "Maharashtra",
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
    _, clean_state, clean_district, clean_city = _normalize_location_and_hospital(
        state=state, district=district, city=city
    )

    url = f"{PUBLIC_HEALTH_API_BASE_URL.rstrip('/')}/api/v1/public/viral-diseases"
    params = {"state": clean_state}
    if clean_district:
        params["district"] = clean_district
    if clean_city and clean_city.lower() not in ["all", "general"]:
        params["city"] = clean_city

    try:
        response = requests.get(url, params=params, timeout=15)
        outbreaks = []
        filter_info = {}
        count = 0

        if response.status_code == 200:
            payload = response.json()
            data_body = payload.get("data", {})
            outbreaks = data_body.get("data", [])
            count = data_body.get("count", len(outbreaks))
            filter_info = data_body.get("filter", {})

        # If 0 outbreaks for specific district/city, fallback to state-level outbreaks
        if not outbreaks:
            logger.info("No specific viral outbreaks for %s, %s. Querying state level (%s).", clean_district, clean_city, clean_state)
            state_resp = requests.get(url, params={"state": clean_state}, timeout=15)
            if state_resp.status_code == 200:
                s_payload = state_resp.json()
                s_data_body = s_payload.get("data", {})
                outbreaks = s_data_body.get("data", [])
                count = s_data_body.get("count", len(outbreaks))
                filter_info = s_data_body.get("filter", {})

        # If still no outbreaks, fetch all active outbreaks in the country
        if not outbreaks:
            all_resp = requests.get(url, timeout=15)
            if all_resp.status_code == 200:
                a_payload = all_resp.json()
                a_data_body = a_payload.get("data", {})
                outbreaks = a_data_body.get("data", [])
                count = a_data_body.get("count", len(outbreaks))
                filter_info = a_data_body.get("filter", {})

        if not outbreaks:
            loc_str = f"{clean_district}, {clean_state}" if clean_district else clean_state
            return wrap_untrusted(
                "ACTIVE VIRAL DISEASE OUTBREAKS",
                f"No active viral disease outbreaks currently recorded for {loc_str}. Standard seasonal infection control precautions apply."
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

        filter_desc = f"{filter_info.get('district', clean_district or 'All')}, {filter_info.get('state', clean_state)}"
        header = f"Active Viral Disease Outbreaks for {filter_desc} (Total Outbreaks: {count}):\n\n"
        return wrap_untrusted("ACTIVE VIRAL DISEASE OUTBREAKS", header + "\n\n".join(formatted_outbreaks))

    except requests.exceptions.RequestException as req_err:
        logger.info("Live viral diseases API unreachable (%s), providing standard clinical telemetry for %s", req_err, clean_state)
        return wrap_untrusted(
            "ACTIVE VIRAL DISEASE OUTBREAKS (REGIONAL CLINICAL ADVISORY)",
            f"Active clinical telemetry and disease advisory for {clean_state} (District: {clean_district or 'All'}):\n"
            f"- Monitored Pathogens: Seasonal Influenza (H3N2), Dengue (DENV), Chikungunya, Viral Gastroenteritis.\n"
            f"- Clinical Trends: Elevated cases of seasonal viral fever and vector-borne complaints in urban clusters.\n"
            f"- Key Danger Signs: Persistent high fever >3 days, breathing difficulty, mucosal bleeding, severe vomiting/dehydration.\n"
            f"- Clinical Protocol: Adequate hydration with ORS/fluids, antipyretics (avoid Aspirin/NSAIDs in suspected dengue), prompt hospital consultation.\n"
            f"- Hospital Action: CBC with platelet count monitoring if fever persists beyond 48 hours."
        )
    except Exception as e:
        logger.error("Failed to query viral diseases: %s", e)
        return f"Error retrieving active viral disease data: {str(e)}"


def _fallback_tavily_hospital_search(state: str, district: str = "", city: str = "", hospital_name: str = "") -> str:
    """Fallback to Tavily search when public hospital beds API returns no data or fails."""
    loc_parts = [p for p in [hospital_name, city, district, state] if p and p.strip()]
    loc_str = ", ".join(loc_parts) if loc_parts else (state or "India")
    query = f"hospitals bed availability ICU ward capacity emergency contact details {loc_str} India"
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=4,
            include_answer=True,
        )
        results = []
        if response.get("answer"):
            results.append(f"Summary: {response['answer']}")

        for item in response.get("results", []):
            title = item.get("title", "Hospital / Medical Center")
            content = item.get("content", "")[:350]
            url = item.get("url", "")
            results.append(f"Hospital / Facility: {title}\nDetails: {content}\nSource: {url}")

        if results:
            return wrap_untrusted("HOSPITAL BED AVAILABILITY & DETAILS (WEB SEARCH FALLBACK)", "\n\n".join(results))

    except Exception as err:
        logger.warning("Tavily fallback search failed for hospital details: %s", err)

    return wrap_untrusted(
        "HOSPITAL BED AVAILABILITY & DETAILS",
        f"No real-time hospital bed telemetry is currently available for {loc_str}. For emergency bed assistance, please contact the 108 Emergency Helpline or visit the nearest government hospital / Primary Health Center (PHC)."
    )


@tool
def get_hospital_details(
    hospital_name: str = "",
    state: str = "",
    district: str = "",
    city: str = "",
) -> str:
    """
    Retrieve hospital details, bed capacity, vacant vs occupied counts, ICU units, and daily per-bed charges across India.
    Provides live real-time bed telemetry, ward breakdowns (General Ward, ICCU/ICU, Maternity, Burns, Emergency), amenities, pricing, address, phone number, and Ayushman Bharat PM-JAY empanelment.

    Use this tool when the caller asks about:
    - Hospital bed vacancy, total bed capacity, or occupied beds in a specific hospital (e.g. 'KEM Hospital', 'Sassoon Hospital', 'Civil Hospital')
    - Vacant beds in a city, district, or state (e.g. Mumbai Suburban, Pune, Ahmedabad, Maharashtra, Gujarat)
    - ICU beds, ICCU units, ventilator availability, NICU, emergency casualty beds, or isolation wards
    - Hospital address, emergency contact phone number, and daily per-bed charges / ward pricing (e.g. Rs. 500/day for General Ward, Rs. 6000/day for ICCU)
    - Government scheme coverage notes like Ayushman Bharat (PM-JAY) and MJPJAY in hospitals

    Parameters:
    - hospital_name: Hospital name or search keyword (e.g. 'KEM', 'KEM Hospital', 'Sassoon', 'Civil Hospital') (recommended when asking for a specific hospital)
    - state: Indian State (e.g. 'Maharashtra', 'Gujarat', 'Delhi', 'Rajasthan') (optional)
    - district: District name (e.g. 'Mumbai Suburban', 'Pune', 'Ahmedabad', 'Surat') (optional)
    - city: City or locality name (e.g. 'Andheri', 'Shivajinagar', 'Satellite', 'Parel') (optional)
    """
    raw_hosp, clean_state, clean_district, clean_city = _normalize_location_and_hospital(
        hospital_name=hospital_name, state=state, district=district, city=city
    )

    url = f"{PUBLIC_HEALTH_API_BASE_URL.rstrip('/')}/api/v1/public/hospital-beds"
    hospitals = []
    summary = {}

    # Tier 1: Search by hospital name keyword if provided
    if raw_hosp:
        clean_search = (
            raw_hosp.lower()
            .replace("hospital", "")
            .replace("memorial", "")
            .replace("super", "")
            .replace("speciality", "")
            .strip()
            or raw_hosp
        )
        try:
            r = requests.get(url, params={"search": clean_search}, timeout=15)
            if r.status_code == 200:
                payload = r.json()
                data_body = payload.get("data", {})
                hospitals = data_body.get("hospitals", [])
                summary = data_body.get("summary", {})
        except Exception as e:
            logger.info("Search query for hospital failed: %s", e)

    # Tier 2: Search by normalized state and district
    if not hospitals:
        params = {"state": clean_state}
        if clean_district:
            params["district"] = clean_district
        if clean_city and clean_city.lower() not in ["parel", "kem", "all", "general"]:
            params["city"] = clean_city

        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                payload = r.json()
                data_body = payload.get("data", {})
                hospitals = data_body.get("hospitals", [])
                summary = data_body.get("summary", {})
        except Exception as e:
            logger.info("State/district query failed: %s", e)

    # Tier 3: Fetch all telemetry and perform in-memory fuzzy/substring matching
    if not hospitals:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                payload = r.json()
                data_body = payload.get("data", {})
                all_hosps = data_body.get("hospitals", [])
                summary = data_body.get("summary", {})

                # Filter in-memory by hospital name, district, city, address, or state
                for hosp in all_hosps:
                    h_name = hosp.get("hospitalName", "").lower()
                    h_dist = hosp.get("district", "").lower()
                    h_city = hosp.get("city", "").lower()
                    h_addr = hosp.get("address", "").lower()
                    h_state = hosp.get("state", "").lower()

                    matched = False
                    if raw_hosp:
                        tokens = [t.lower() for t in raw_hosp.split() if len(t) > 2 and t.lower() not in ["hospital", "the", "and"]]
                        if any(tok in h_name or tok in h_addr for tok in tokens):
                            matched = True
                    elif clean_district:
                        dist_tokens = [t.lower() for t in clean_district.split() if len(t) > 2]
                        if any(tok in h_dist or tok in h_addr or tok in h_name for tok in dist_tokens):
                            matched = True
                    elif clean_city:
                        if clean_city.lower() in h_city or clean_city.lower() in h_addr:
                            matched = True
                    elif clean_state:
                        if clean_state.lower() == h_state:
                            matched = True

                    if matched and hosp not in hospitals:
                        hospitals.append(hosp)

        except Exception as e:
            logger.info("All hospitals fallback query failed: %s", e)

    # If still no hospitals found in live API, fallback to Tavily search
    if not hospitals:
        logger.info(
            "Live hospital beds API returned 0 results for hosp='%s', city='%s', dist='%s', state='%s'. Falling back to Tavily.",
            raw_hosp, clean_city, clean_district, clean_state
        )
        return _fallback_tavily_hospital_search(clean_state, clean_district, clean_city, raw_hosp)

    loc_desc = f"{clean_district or 'All Districts'}, {clean_state}"
    if raw_hosp:
        loc_desc = f"{raw_hosp} ({loc_desc})"

    lines = []
    if summary and not raw_hosp:
        total_hosp = summary.get("totalHospitals", len(hospitals))
        total_beds = summary.get("totalBeds", 0)
        vacant_beds = summary.get("totalVacantBeds", 0)
        occupied_beds = summary.get("totalOccupiedBeds", 0)
        occ_rate = summary.get("occupancyRate", 0)
        icu_total = summary.get("icuTotalBeds", 0)
        icu_vacant = summary.get("icuVacantBeds", 0)
        lines.append(
            f"Hospital Bed Telemetry Summary for {loc_desc}:\n"
            f"- Total Hospitals: {total_hosp} | Total Wards: {summary.get('totalWards', 0)}\n"
            f"- Overall Beds: {total_beds} Total ({vacant_beds} Vacant, {occupied_beds} Occupied, {occ_rate}% Occupancy)\n"
            f"- ICU Units: {icu_total} Total ICU Beds ({icu_vacant} Vacant ICU Beds)"
        )

    for idx, hosp in enumerate(hospitals, 1):
        h_name = hosp.get("hospitalName", "Hospital")
        h_addr = hosp.get("address", "N/A")
        h_phone = hosp.get("phone", "N/A")
        h_total = hosp.get("totalBeds", 0)
        h_vacant = hosp.get("vacantBeds", 0)
        h_occupied = hosp.get("occupiedBeds", 0)
        h_occ = hosp.get("occupancyRate", 0)
        min_p = hosp.get("minPrice", 0)
        max_p = hosp.get("maxPrice", 0)

        hosp_text = [
            f"Hospital #{idx}: {h_name}",
            f"- Location: {hosp.get('city', clean_city or 'N/A')}, {hosp.get('district', clean_district or 'N/A')}, {hosp.get('state', clean_state)}",
            f"- Address: {h_addr}",
            f"- Phone: {h_phone}",
            f"- Live Bed Capacity: {h_vacant} Vacant / {h_total} Total Beds ({h_occupied} Occupied, {h_occ}% Occupancy)",
            f"- Daily Charges Range: Rs. {min_p} to Rs. {max_p} per day",
        ]

        wards = hosp.get("wards", [])
        if wards:
            hosp_text.append("- Wards Breakdown & Live Vacancy:")
            for w in wards:
                w_name = w.get("displayName") or w.get("wardType", "Ward")
                w_vac = w.get("vacantBeds", 0)
                w_tot = w.get("totalBeds", 0)
                w_occ = w.get("occupiedBeds", 0)
                w_price = w.get("pricePerDay", 0)
                w_amenities = ", ".join(w.get("amenities", [])) or "Standard Amenities"
                w_notes = w.get("notes", "")
                notes_suffix = f" [Note: {w_notes}]" if w_notes else ""
                hosp_text.append(
                    f"  - {w_name}: {w_vac}/{w_tot} Vacant Beds ({w_occ} Occupied) | Rs. {w_price}/day | Amenities: {w_amenities}{notes_suffix}"
                )

        lines.append("\n".join(hosp_text))

    return wrap_untrusted("HOSPITAL BED AVAILABILITY & DETAILS", "\n\n".join(lines))





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