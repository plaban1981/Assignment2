"""
Tools - Customer Support Agent Tools

Contains the tools available to the customer support agent.
These are the same tools from Week 1, organized in a separate module.
"""

from langchain.tools import tool
from typing import Optional


@tool
def check_order_status(order_id: str) -> dict:
    """
    Check the status of an order.

    Args:
        order_id: The order ID to check (e.g., 'ORD123')

    Returns:
        Dictionary with order status information
    """
    # Mock implementation - in production, this would query a database
    mock_orders = {
        "ORD123": {"status": "shipped", "eta": "2024-01-20", "carrier": "FedEx"},
        "ORD456": {"status": "processing", "eta": "2024-01-22", "carrier": "pending"},
        "ORD789": {"status": "delivered", "eta": "2024-01-15", "carrier": "UPS"},
    }

    if order_id in mock_orders:
        order_info = mock_orders[order_id]
        return {
            "order_id": order_id,
            "status": order_info["status"],
            "eta": order_info["eta"],
            "carrier": order_info["carrier"]
        }
    else:
        return {
            "order_id": order_id,
            "status": "not_found",
            "message": f"Order {order_id} not found in system"
        }


@tool
def create_ticket(issue: str, priority: str = "medium") -> dict:
    """
    Create a support ticket for an issue.

    Args:
        issue: Description of the issue
        priority: Priority level (low, medium, high, critical)

    Returns:
        Dictionary with ticket information
    """
    import random
    import string

    # Generate a mock ticket ID
    ticket_id = "TKT" + "".join(random.choices(string.digits, k=5))

    # Validate priority
    valid_priorities = ["low", "medium", "high", "critical"]
    if priority.lower() not in valid_priorities:
        priority = "medium"

    return {
        "ticket_id": ticket_id,
        "issue": issue,
        "priority": priority.lower(),
        "status": "open",
        "message": f"Ticket {ticket_id} created successfully with {priority} priority"
    }


@tool
def escalate_issue(issue: str, user_tier: str, reason: Optional[str] = None) -> dict:
    """
    Escalate an issue to senior support.

    Args:
        issue: Description of the issue to escalate
        user_tier: The user's tier (vip or standard)
        reason: Optional reason for escalation

    Returns:
        Dictionary with escalation status
    """
    if user_tier.lower() == "vip":
        return {
            "escalation_status": "escalated_to_senior_support",
            "issue": issue,
            "priority": "high",
            "response_time": "15 minutes",
            "reason": reason or "VIP customer priority escalation"
        }
    else:
        return {
            "escalation_status": "standard_escalation",
            "issue": issue,
            "priority": "medium",
            "response_time": "2-4 hours",
            "reason": reason or "Standard escalation process"
        }


@tool
def get_refund_status(order_id: str) -> dict:
    """
    Check the refund status for an order.

    Args:
        order_id: The order ID to check refund status for

    Returns:
        Dictionary with refund status information
    """
    # Mock implementation
    mock_refunds = {
        "ORD123": {"status": "not_requested", "amount": None},
        "ORD456": {"status": "pending", "amount": 49.99},
        "ORD789": {"status": "completed", "amount": 29.99},
    }

    if order_id in mock_refunds:
        refund_info = mock_refunds[order_id]
        return {
            "order_id": order_id,
            "refund_status": refund_info["status"],
            "refund_amount": refund_info["amount"]
        }
    else:
        return {
            "order_id": order_id,
            "refund_status": "no_order_found",
            "message": f"No refund information found for order {order_id}"
        }


# Export all tools as a list
tools = [check_order_status, create_ticket, escalate_issue, get_refund_status]


if __name__ == "__main__":
    # Test the tools
    print("Testing tools:")
    print(f"check_order_status('ORD123'): {check_order_status.invoke({'order_id': 'ORD123'})}")
    print(f"create_ticket('Test issue', 'high'): {create_ticket.invoke({'issue': 'Test issue', 'priority': 'high'})}")
    print(f"escalate_issue('Complex issue', 'vip'): {escalate_issue.invoke({'issue': 'Complex issue', 'user_tier': 'vip'})}")
    print(f"get_refund_status('ORD456'): {get_refund_status.invoke({'order_id': 'ORD456'})}")
