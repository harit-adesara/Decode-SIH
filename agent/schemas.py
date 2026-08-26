"""
schemas.py

Pydantic schemas and Enums for classification-based LLM outputs across BharatSwasthya AI.
Ensures strict type-safety, guaranteed choice adherence, and protection against prompt injection.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SupportedLanguage(str, Enum):
    GUJARATI = "Gujarati"
    HINDI = "Hindi"
    ENGLISH = "English"
    MARATHI = "Marathi"
    BENGALI = "Bengali"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    KANNADA = "Kannada"
    MALAYALAM = "Malayalam"


class LanguageClassification(BaseModel):
    language: SupportedLanguage = Field(
        description="The primary Indian language or English spoken in the utterance chosen strictly from the supported list"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Short reason for the classification"
    )


class SafetyCategory(str, Enum):
    NONE = "none"   
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    SYSTEM_OVERRIDE = "system_override"
    MALICIOUS_INTENT = "malicious_intent"
    SENSITIVE_DATA_LEAK = "sensitive_data_leak"
    UNSAFE_DIAGNOSIS = "unsafe_diagnosis"
    INAPPROPRIATE = "inappropriate"
    OTHER = "other"


class GuardrailClassification(BaseModel):
    unsafe: bool = Field(
        description="True if the text violates safety policies, attempts prompt injection/jailbreak, or leaks sensitive data"
    )
    category: SafetyCategory = Field(
        default=SafetyCategory.NONE,
        description="Specific safety violation category classification"
    )
    reason: str = Field(
        default="pass",
        description="Brief explanation justifying the safety classification"
    )


class TriageUrgency(str, Enum):
    CRITICAL_EMERGENCY = "CRITICAL_EMERGENCY"
    URGENT = "URGENT"
    ROUTINE = "ROUTINE"
    HOME_CARE = "HOME_CARE"


class MedicalSpecialty(str, Enum):
    CARDIOLOGY = "Cardiology"
    NEUROLOGY = "Neurology"
    ORTHOPEDICS = "Orthopedics"
    PEDIATRICS = "Pediatrics"
    OBSTETRICS_GYNECOLOGY = "Obstetrics & Gynecology"
    ENT = "ENT"
    OPHTHALMOLOGY = "Ophthalmology"
    DERMATOLOGY = "Dermatology"
    PULMONOLOGY = "Pulmonology"
    GASTROENTEROLOGY = "Gastroenterology"
    NEPHROLOGY_UROLOGY = "Nephrology / Urology"
    PSYCHIATRY = "Psychiatry / Mental Health"
    DENTAL = "Dental / Dentist"
    ONCOLOGY = "Oncology"
    GENERAL_PHYSICIAN = "General Physician"
    EMERGENCY_MEDICINE = "Emergency Medicine"
    OTHER = "Other Specialist"


class FacilityType(str, Enum):
    AMBULANCE_108 = "108 Emergency Ambulance"
    EMERGENCY_ROOM = "Hospital Emergency Room"
    COMMUNITY_HEALTH_CENTER = "Community Health Center (CHC)"
    PRIMARY_HEALTH_CENTER = "Primary Health Center (PHC)"
    SPECIALIST_CLINIC = "Specialist Clinic"
    TELEMEDICINE = "e-Sanjeevani / Teleconsultation"
    AYUSH_CENTER = "AYUSH / Wellness Center"
    DIAGNOSTIC_CENTER = "Diagnostic / Pathology Lab"
    HOME_CARE = "Home Care"
    OTHER = "Other Healthcare Facility"


class TriageAssessment(BaseModel):
    urgency_level: TriageUrgency = Field(
        description="Clinical urgency classification level"
    )
    recommended_specialty: MedicalSpecialty = Field(
        description="Recommended medical doctor specialty"
    )
    recommended_facility: FacilityType = Field(
        description="Recommended healthcare facility type"
    )
    immediate_action: str = Field(
        description="Immediate lifesaving advice or recommended next step for the caller"
    )
    follow_up_guidance: Optional[str] = Field(
        default=None,
        description="Key follow-up guidance or warning signs to watch for"
    )


class OutbreakRiskLevel(str, Enum):
    CRITICAL = "critical"
    SEVERE = "severe"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class DiseaseCategory(str, Enum):
    VIRAL_RESPIRATORY = "Viral Respiratory"
    VECTOR_BORNE = "Vector-Borne"
    WATER_FOOD_BORNE = "Water/Food-Borne"
    ZOONOTIC = "Zoonotic"
    BACTERIAL = "Bacterial"
    OTHER = "Other"


class ContainmentPriority(str, Enum):
    EMERGENCY_MOBILIZATION = "Emergency Mobilization"
    ACTIVE_CONTAINMENT = "Active Containment"
    TARGETED_ADVISORY = "Targeted Advisory"
    ROUTINE_MONITORING = "Routine Monitoring"


class WeatherVulnerability(str, Enum):
    EXTREME = "Extreme"
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"


class WeatherFactors(BaseModel):
    temperature: Optional[str] = Field(default=None, description="Current ambient temperature (e.g. 31°C)")
    humidity: Optional[str] = Field(default=None, description="Relative humidity level (e.g. 86%)")
    rainfall_risk: Optional[str] = Field(default=None, description="Rainfall or inundation risk description")
    air_quality_index: Optional[str] = Field(default=None, description="Air Quality Index assessment")
    season: Optional[str] = Field(default=None, description="Current climate season (e.g. Monsoon, Winter, Summer)")


class EpidemicRiskClassification(BaseModel):
    classified_risk_level: OutbreakRiskLevel = Field(
        description="Overall epidemic outbreak risk level classification (critical, severe, high, moderate, low)"
    )
    disease_category: DiseaseCategory = Field(
        description="Primary category classification of the outbreak or disease concern"
    )
    containment_priority: ContainmentPriority = Field(
        description="Public health containment and advisory urgency priority"
    )
    weather_vulnerability: WeatherVulnerability = Field(
        description="Meteorological weather risk vulnerability (humidity, rain, heatwave impact)"
    )
    primary_suspected_disease: str = Field(
        description="Most probable disease or outbreak identified from the forecast or symptoms"
    )
    actionable_precautions: list[str] = Field(
        default_factory=list,
        description="Top actionable precautions for the caller or local community"
    )
    urgency_summary: str = Field(
        description="Concise conversational summary of risk level and next immediate actions"
    )

