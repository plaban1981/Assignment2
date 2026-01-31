"""
Pydantic models for structured output.
This ensures type-safe responses from the LLM for the Internal Ops Desk.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class RequestRouting(BaseModel):
    """Routing decision from the Supervisor agent."""

    specialist: Literal['it', 'hr', 'facilities', 'escalate'] = Field(
        description="The specialist to route the request to"
    )
    reasoning: str = Field(
        description="The reasoning for the routing decision"
    )
    confidence: float = Field(
        description="The confidence in the routing decision (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )


class SpecialistResponse(BaseModel):
    """Structured response from specialist agents (IT, HR, Facilities, Escalate)."""

    reasoning: str = Field(
        description="Your step-by-step reasoning for this decision"
    )
    action: str = Field(
        description="Action to take: provide_info|create_ticket|schedule_service|escalate_to_human|grant_access|reset_credentials"
    )
    confidence: float = Field(
        description="Confidence in this decision (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    message: str = Field(
        description="Message to send to the employee"
    )
    requires_approval: bool = Field(
        description="Does this action require manager/HR approval?"
    )
    ticket_type: Optional[str] = Field(
        default=None,
        description="Type of ticket if created: incident|service_request|change_request|access_request"
    )
