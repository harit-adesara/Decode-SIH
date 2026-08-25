from typing import Optional
from langchain.agents.middleware import AgentState

class Data(AgentState, total=False):
    call_sid: str
    phone_number: str
    language: str
    location: Optional[str]
    user_intent: Optional[str]
    symptoms: Optional[str]
    facility_name: Optional[str]
    facility_address: Optional[str]
    facility_type: Optional[str]
    facility_distance: Optional[str]
    facility_travel_time: Optional[str]
    scheme_name: Optional[str]
    is_emergency: bool
    emergency_type: Optional[str]
    emergency_location: Optional[str]
    sms_sent: bool
    sms_sid: Optional[str]
    sms_error: Optional[str]
    call_ended: bool
    call_end_error: Optional[str]