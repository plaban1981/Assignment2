"""
Cost Tracker - Task 4: Token Usage and Cost Monitoring

Implements cost tracking for LLM calls with per-request and daily totals,
budget limits, and pricing for multiple models.
"""

import logging
from datetime import datetime, date
from typing import Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Represents token usage for a single LLM call."""
    input_tokens: int
    output_tokens: int
    model: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostRecord:
    """Represents a cost record for a single LLM call."""
    usage: TokenUsage
    cost: float
    user_id: Optional[str] = None
    request_id: Optional[str] = None


class CostTracker:
    """
    Tracks token usage and costs for LLM API calls.

    Features:
    - Per-request cost calculation
    - Daily cost totals per user and system-wide
    - Budget limits and warnings
    - Multiple model pricing support
    """

    # Pricing per 1M tokens (as of 2024)
    MODEL_PRICING = {
        # OpenAI models
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},  # Example pricing

        # Anthropic models (for reference)
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    def __init__(
        self,
        budget_per_request: float = 1.00,
        budget_per_user_daily: float = 10.00,
        budget_system_daily: float = 100.00
    ):
        """
        Initialize the CostTracker.

        Args:
            budget_per_request: Maximum cost allowed per single request
            budget_per_user_daily: Maximum daily cost per user
            budget_system_daily: Maximum daily cost system-wide
        """
        self.budget_per_request = budget_per_request
        self.budget_per_user_daily = budget_per_user_daily
        self.budget_system_daily = budget_system_daily

        # Cost tracking storage
        self._request_costs: list[CostRecord] = []
        self._daily_costs: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._user_daily_costs: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate the cost for a single LLM call.

        Args:
            model: The model name (e.g., 'gpt-4o', 'gpt-4o-mini')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        pricing = self.MODEL_PRICING.get(model)

        if pricing is None:
            logger.warning(f"Unknown model '{model}', using gpt-4o-mini pricing")
            pricing = self.MODEL_PRICING["gpt-4o-mini"]

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return round(total_cost, 6)

    def record_cost(
        self,
        user_id: str,
        cost: float,
        model: str = "gpt-4o",
        input_tokens: int = 0,
        output_tokens: int = 0,
        request_id: Optional[str] = None
    ) -> None:
        """
        Record a cost for tracking.

        Args:
            user_id: Identifier for the user
            cost: The cost to record
            model: Model used for the call
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_id: Optional request identifier
        """
        today = date.today().isoformat()

        # Create usage and record
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model
        )
        record = CostRecord(
            usage=usage,
            cost=cost,
            user_id=user_id,
            request_id=request_id
        )

        self._request_costs.append(record)

        # Update daily totals
        self._daily_costs[today]["total"] += cost
        self._daily_costs[today]["requests"] += 1
        self._daily_costs[today][model] += cost

        # Update per-user daily totals
        self._user_daily_costs[user_id][today] += cost

        logger.info(
            f"Cost recorded: ${cost:.6f} for user {user_id} "
            f"(model: {model}, tokens: {input_tokens}+{output_tokens})"
        )

    def check_budget(
        self,
        user_id: str,
        estimated_cost: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a request is within budget limits.

        Args:
            user_id: User identifier
            estimated_cost: Estimated cost of the request

        Returns:
            Tuple of (is_allowed, reason_if_blocked)
        """
        today = date.today().isoformat()

        # Check per-request budget
        if estimated_cost > self.budget_per_request:
            return False, f"Request cost ${estimated_cost:.4f} exceeds per-request limit ${self.budget_per_request:.2f}"

        # Check user daily budget
        user_daily_total = self._user_daily_costs[user_id].get(today, 0)
        if user_daily_total + estimated_cost > self.budget_per_user_daily:
            return False, f"User daily budget exceeded (current: ${user_daily_total:.4f}, limit: ${self.budget_per_user_daily:.2f})"

        # Check system daily budget
        system_daily_total = self._daily_costs[today].get("total", 0)
        if system_daily_total + estimated_cost > self.budget_system_daily:
            return False, f"System daily budget exceeded (current: ${system_daily_total:.4f}, limit: ${self.budget_system_daily:.2f})"

        return True, None

    def get_daily_total(self, target_date: Optional[date] = None) -> Dict:
        """
        Get the daily cost total.

        Args:
            target_date: Date to get totals for (defaults to today)

        Returns:
            Dictionary with cost breakdown
        """
        if target_date is None:
            target_date = date.today()

        date_key = target_date.isoformat()
        daily_data = self._daily_costs.get(date_key, {})

        return {
            "date": date_key,
            "total_cost": daily_data.get("total", 0),
            "total_requests": int(daily_data.get("requests", 0)),
            "by_model": {
                k: v for k, v in daily_data.items()
                if k not in ["total", "requests"]
            }
        }

    def get_user_daily_total(
        self,
        user_id: str,
        target_date: Optional[date] = None
    ) -> float:
        """
        Get daily cost total for a specific user.

        Args:
            user_id: User identifier
            target_date: Date to get totals for (defaults to today)

        Returns:
            Total cost for the user on that day
        """
        if target_date is None:
            target_date = date.today()

        date_key = target_date.isoformat()
        return self._user_daily_costs[user_id].get(date_key, 0)

    def get_request_summary(self) -> Dict:
        """
        Get a summary of all recorded requests.

        Returns:
            Summary dictionary with statistics
        """
        if not self._request_costs:
            return {
                "total_requests": 0,
                "total_cost": 0,
                "average_cost": 0,
                "by_model": {}
            }

        total_cost = sum(r.cost for r in self._request_costs)
        by_model = defaultdict(lambda: {"count": 0, "cost": 0, "tokens": 0})

        for record in self._request_costs:
            model = record.usage.model
            by_model[model]["count"] += 1
            by_model[model]["cost"] += record.cost
            by_model[model]["tokens"] += record.usage.total_tokens

        return {
            "total_requests": len(self._request_costs),
            "total_cost": round(total_cost, 6),
            "average_cost": round(total_cost / len(self._request_costs), 6),
            "by_model": dict(by_model)
        }

    def estimate_cost(
        self,
        model: str,
        input_text: str,
        estimated_output_tokens: int = 500
    ) -> float:
        """
        Estimate cost for a request before making it.

        Args:
            model: Model to use
            input_text: The input text
            estimated_output_tokens: Estimated output tokens

        Returns:
            Estimated cost in USD
        """
        # Rough estimation: ~4 chars per token for English
        estimated_input_tokens = len(input_text) // 4
        return self.calculate_cost(model, estimated_input_tokens, estimated_output_tokens)

    def log_cost_summary(self) -> None:
        """Log a summary of costs to the logger."""
        summary = self.get_request_summary()
        daily = self.get_daily_total()

        logger.info(
            f"Cost Summary - "
            f"Today: ${daily['total_cost']:.4f} ({daily['total_requests']} requests), "
            f"All-time: ${summary['total_cost']:.4f} ({summary['total_requests']} requests)"
        )


# Convenience function for quick cost calculation
def calculate_llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Quick helper function to calculate LLM cost.

    Args:
        model: Model name
        input_tokens: Input token count
        output_tokens: Output token count

    Returns:
        Cost in USD
    """
    tracker = CostTracker()
    return tracker.calculate_cost(model, input_tokens, output_tokens)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create tracker
    tracker = CostTracker(
        budget_per_request=0.50,
        budget_per_user_daily=5.00,
        budget_system_daily=100.00
    )

    # Simulate some LLM calls
    print("=== Cost Tracking Demo ===\n")

    # Calculate cost for different models
    models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"]
    for model in models:
        cost = tracker.calculate_cost(model, 1000, 500)
        print(f"{model}: 1000 input + 500 output tokens = ${cost:.6f}")

    print()

    # Record some costs
    tracker.record_cost("user_123", 0.001, "gpt-4o-mini", 500, 200)
    tracker.record_cost("user_123", 0.002, "gpt-4o-mini", 800, 300)
    tracker.record_cost("user_456", 0.005, "gpt-4o", 1000, 400)

    # Check budget
    allowed, reason = tracker.check_budget("user_123", 0.10)
    print(f"Budget check: allowed={allowed}, reason={reason}")

    # Get summaries
    print(f"\nDaily total: {tracker.get_daily_total()}")
    print(f"User 123 daily: ${tracker.get_user_daily_total('user_123'):.6f}")
    print(f"Request summary: {tracker.get_request_summary()}")
