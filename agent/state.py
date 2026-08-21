from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class Data(TypedDict, total=False):
    messages: Annotated[list, add_messages]
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