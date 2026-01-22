"""
Models - Task 2: Structured Output with Pydantic

Pydantic models for structured LLM output to replace string-based decision logic.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from enum import Enum


class UserTier(str, Enum):
    """User tier classification."""
    VIP = "vip"
    STANDARD = "standard"


class IssuePriority(str, Enum):
    """Issue priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(str, Enum):
    """Types of customer support issues."""
    ORDER_STATUS = "order_status"
    BILLING = "billing"
    TECHNICAL = "technical"
    COMPLAINT = "complaint"
    GENERAL = "general"
    REFUND = "refund"


class RoutingDecision(BaseModel):
    """
    Structured output for routing decisions.
    Used to determine which path a customer request should take.
    """
    user_tier: UserTier = Field(
        description="The tier of the user (vip or standard)"
    )
    reasoning: str = Field(
        description="Brief explanation for the routing decision"
    )


class IssueAnalysis(BaseModel):
    """
    Structured output for analyzing customer issues.
    Provides detailed classification of the customer's problem.
    """
    issue_type: IssueType = Field(
        description="The category of the customer's issue"
    )
    priority: IssuePriority = Field(
        description="Priority level based on issue severity"
    )
    summary: str = Field(
        description="Brief summary of the customer's issue"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Key terms extracted from the issue"
    )
    requires_tool: bool = Field(
        default=False,
        description="Whether this issue requires using a tool"
    )
    suggested_tool: Optional[str] = Field(
        default=None,
        description="The tool to use if requires_tool is True"
    )


class EscalationDecision(BaseModel):
    """
    Structured output for escalation decisions.
    Determines whether an issue should be escalated to senior support.
    """
    should_escalate: bool = Field(
        description="Whether the issue should be escalated"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for escalation if should_escalate is True"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for this decision (0-1)"
    )
    recommended_action: str = Field(
        description="Recommended next action for the support agent"
    )


class SupportResponse(BaseModel):
    """
    Structured output for the final support response.
    Ensures consistent response format from the agent.
    """
    message: str = Field(
        description="The response message to the customer"
    )
    actions_taken: List[str] = Field(
        default_factory=list,
        description="List of actions taken to resolve the issue"
    )
    ticket_created: bool = Field(
        default=False,
        description="Whether a support ticket was created"
    )
    ticket_id: Optional[str] = Field(
        default=None,
        description="The ticket ID if a ticket was created"
    )
    follow_up_required: bool = Field(
        default=False,
        description="Whether follow-up is needed"
    )


class TierClassification(BaseModel):
    """
    Structured output for classifying user tier from message content.
    Used in check_user_tier_node to make routing decisions.
    """
    detected_tier: UserTier = Field(
        description="The detected user tier based on message analysis"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in tier detection (0-1)"
    )
    tier_indicators: List[str] = Field(
        default_factory=list,
        description="Words or phrases that indicated the tier"
    )
    reasoning: str = Field(
        description="Explanation for the tier classification"
    )


class InputValidation(BaseModel):
    """
    Structured output for input validation results.
    Used to validate user input before processing.
    """
    is_valid: bool = Field(
        description="Whether the input is valid and safe to process"
    )
    sanitized_input: str = Field(
        description="The sanitized version of the input"
    )
    detected_issues: List[str] = Field(
        default_factory=list,
        description="Any issues detected in the input"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Risk level of the input"
    )


# Example usage for with_structured_output()
if __name__ == "__main__":
    # Example of creating instances
    routing = RoutingDecision(
        user_tier=UserTier.VIP,
        reasoning="User mentioned VIP status in message"
    )
    print(f"Routing: {routing.model_dump_json(indent=2)}")

    analysis = IssueAnalysis(
        issue_type=IssueType.ORDER_STATUS,
        priority=IssuePriority.MEDIUM,
        summary="Customer wants to check order status",
        keywords=["order", "status", "delivery"],
        requires_tool=True,
        suggested_tool="check_order_status"
    )
    print(f"\nAnalysis: {analysis.model_dump_json(indent=2)}")

    escalation = EscalationDecision(
        should_escalate=False,
        confidence=0.95,
        recommended_action="Check order status and provide update to customer"
    )
    print(f"\nEscalation: {escalation.model_dump_json(indent=2)}")
