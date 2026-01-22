"""
Error Handling - Task 3: Retry Logic and Exception Handling

Implements retry decorator with exponential backoff, custom exceptions,
and graceful error handling for production-ready agents.
"""

import time
import random
import functools
import logging
from typing import Callable, Any, Optional, Type, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class AgentError(Exception):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


class RateLimitError(AgentError):
    """Raised when API rate limits are exceeded."""

    def __init__(self, message: str = "Rate limit exceeded",
                 retry_after: Optional[int] = None):
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after


class MaxIterationsError(AgentError):
    """Raised when maximum iterations are exceeded in a loop."""

    def __init__(self, message: str = "Maximum iterations exceeded",
                 iterations: int = 0, limit: int = 0):
        super().__init__(message, {"iterations": iterations, "limit": limit})
        self.iterations = iterations
        self.limit = limit


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails."""

    def __init__(self, message: str, tool_name: str,
                 original_error: Optional[Exception] = None):
        super().__init__(message, {
            "tool_name": tool_name,
            "original_error": str(original_error) if original_error else None
        })
        self.tool_name = tool_name
        self.original_error = original_error


class LLMError(AgentError):
    """Raised when LLM API call fails."""

    def __init__(self, message: str, model: Optional[str] = None,
                 status_code: Optional[int] = None):
        super().__init__(message, {"model": model, "status_code": status_code})
        self.model = model
        self.status_code = status_code


class ValidationError(AgentError):
    """Raised when input/output validation fails."""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None):
        super().__init__(message, {"field": field, "value": str(value)})
        self.field = field
        self.value = value


class BudgetExceededError(AgentError):
    """Raised when cost budget is exceeded."""

    def __init__(self, message: str = "Budget exceeded",
                 current_cost: float = 0, budget_limit: float = 0):
        super().__init__(message, {
            "current_cost": current_cost,
            "budget_limit": budget_limit
        })
        self.current_cost = current_cost
        self.budget_limit = budget_limit


class SecurityError(AgentError):
    """Raised when a security violation is detected."""

    def __init__(self, message: str, threat_type: Optional[str] = None):
        super().__init__(message, {"threat_type": threat_type})
        self.threat_type = threat_type


# ============================================================================
# Retry Decorator with Exponential Backoff
# ============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        RateLimitError, LLMError, ConnectionError, TimeoutError
    ),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exceptions that should trigger retry
        on_retry: Optional callback function called on each retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                            f"Last error: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    # Handle rate limit with retry-after header
                    if isinstance(e, RateLimitError) and e.retry_after:
                        delay = max(delay, e.retry_after)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} for {func.__name__} "
                        f"failed with {type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

                except Exception as e:
                    # Non-retryable exception - raise immediately
                    logger.error(
                        f"Non-retryable error in {func.__name__}: {type(e).__name__}: {e}"
                    )
                    raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


# ============================================================================
# Tool Execution Wrapper
# ============================================================================

def safe_tool_execution(tool_func: Callable, tool_name: str, **kwargs) -> dict:
    """
    Safely execute a tool with error handling and feedback.

    Args:
        tool_func: The tool function to execute
        tool_name: Name of the tool for error reporting
        **kwargs: Arguments to pass to the tool

    Returns:
        Tool result or error feedback dictionary
    """
    try:
        result = tool_func(**kwargs)
        return {
            "success": True,
            "tool_name": tool_name,
            "result": result
        }
    except Exception as e:
        logger.error(f"Tool execution failed for {tool_name}: {e}")
        return {
            "success": False,
            "tool_name": tool_name,
            "error": str(e),
            "error_type": type(e).__name__,
            "feedback": f"Tool {tool_name} failed: {e}. Please try a different approach."
        }


# ============================================================================
# Iteration Limiter
# ============================================================================

class IterationLimiter:
    """
    Tracks and limits iterations to prevent infinite loops.
    """

    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def increment(self) -> None:
        """Increment the iteration counter and check limit."""
        self.current_iteration += 1
        if self.current_iteration > self.max_iterations:
            raise MaxIterationsError(
                f"Exceeded maximum iterations limit of {self.max_iterations}",
                iterations=self.current_iteration,
                limit=self.max_iterations
            )

    def reset(self) -> None:
        """Reset the iteration counter."""
        self.current_iteration = 0

    def get_remaining(self) -> int:
        """Get remaining iterations."""
        return max(0, self.max_iterations - self.current_iteration)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()
        return False


# ============================================================================
# Error Handler for Graceful Degradation
# ============================================================================

class ErrorHandler:
    """
    Centralized error handler for the agent with graceful degradation.
    """

    def __init__(self):
        self.error_count = 0
        self.error_history = []

    def handle_error(self, error: Exception, context: str = "") -> dict:
        """
        Handle an error and return a user-friendly response.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred

        Returns:
            Dictionary with error info and user-friendly message
        """
        self.error_count += 1
        self.error_history.append({
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context
        })

        # Map errors to user-friendly messages
        user_messages = {
            RateLimitError: "Our service is experiencing high demand. Please try again in a moment.",
            MaxIterationsError: "Your request is taking longer than expected. Please simplify your request.",
            ToolExecutionError: "We encountered an issue processing your request. Please try again.",
            LLMError: "Our AI service is temporarily unavailable. Please try again shortly.",
            BudgetExceededError: "Service usage limit reached. Please contact support.",
            SecurityError: "Your request could not be processed for security reasons.",
            ValidationError: "There was an issue with the input provided. Please check and try again."
        }

        user_message = user_messages.get(
            type(error),
            "An unexpected error occurred. Please try again or contact support."
        )

        return {
            "error": True,
            "error_type": type(error).__name__,
            "user_message": user_message,
            "technical_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

    def get_error_stats(self) -> dict:
        """Get error statistics."""
        return {
            "total_errors": self.error_count,
            "recent_errors": self.error_history[-10:],  # Last 10 errors
            "error_types": {}
        }


# Global error handler instance
error_handler = ErrorHandler()


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)

    # Test retry decorator
    @retry_with_backoff(max_retries=3, base_delay=0.1)
    def flaky_function():
        if random.random() < 0.7:  # 70% chance of failure
            raise LLMError("API temporarily unavailable", model="gpt-4")
        return "Success!"

    try:
        result = flaky_function()
        print(f"Result: {result}")
    except LLMError as e:
        print(f"Failed after retries: {e}")

    # Test iteration limiter
    print("\nTesting iteration limiter:")
    limiter = IterationLimiter(max_iterations=5)
    try:
        for i in range(10):
            limiter.increment()
            print(f"Iteration {limiter.current_iteration}, remaining: {limiter.get_remaining()}")
    except MaxIterationsError as e:
        print(f"Caught: {e}")

    # Test error handler
    print("\nTesting error handler:")
    error_info = error_handler.handle_error(
        RateLimitError("Rate limit exceeded", retry_after=30),
        context="vip_agent_node"
    )
    print(f"User message: {error_info['user_message']}")
