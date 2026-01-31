"""
Internal Ops Desk - Multi-Agent System with Production Features
Routes employee requests to IT, HR, or Facilities specialists.
Integrates: logging, input sanitization, rate limiting, prompt management,
AB testing, error handling, cost tracking, circuit breaker, and output validation.
"""

import time
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional, List
from pathlib import Path

# Import all production components
from logging_config import StructuredLogger
from input_sanitizer import InputSanitizer
from rate_limiter import RateLimiter
from prompt_manager import PromptManager
from ab_test_manager import ABTestManager
from error_handling import retry_with_backoff
from cost_tracker import CostTracker
from output_validator import OutputValidator
from circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from models import RequestRouting, SpecialistResponse

load_dotenv()

# Initialize production components
logger = StructuredLogger("internal_ops_desk")
input_sanitizer = InputSanitizer()
rate_limiter = RateLimiter()
prompt_manager = PromptManager(prompts_dir=str(Path(__file__).parent / "prompts"))
ab_test_manager = ABTestManager()
cost_tracker = CostTracker()
output_validator = OutputValidator()

# Initialize circuit breakers for each agent
supervisor_breaker = CircuitBreaker(max_failures=3, timeout=60)
it_breaker = CircuitBreaker(max_failures=3, timeout=60)
hr_breaker = CircuitBreaker(max_failures=3, timeout=60)
facilities_breaker = CircuitBreaker(max_failures=3, timeout=60)
escalate_breaker = CircuitBreaker(max_failures=3, timeout=60)


def load_prompt_with_fallback(agent_name: str, user_id: str) -> tuple[dict, str]:
    """
    Load prompt with fallback to v1.0.0 if current doesn't exist.

    Returns:
        Tuple of (prompt_data, version_used)
    """
    prompt_version = ab_test_manager.get_prompt_version(agent_name, user_id)

    if prompt_version == "current":
        try:
            prompt_data = prompt_manager.load_prompt(agent_name, "current")
            return prompt_data, "current"
        except ValueError:
            prompt_version = "v1.0.0"
            prompt_data = prompt_manager.load_prompt(agent_name, prompt_version)
            return prompt_data, prompt_version
    else:
        prompt_data = prompt_manager.load_prompt(agent_name, prompt_version)
        return prompt_data, prompt_version


# Define the State
class OpsState(TypedDict):
    """State contract for the Internal Ops Desk workflow."""
    request_text: str
    employee_id: str
    employee_email: Optional[str]
    chosen_specialist: Literal['it', 'hr', 'facilities', 'escalate']
    confidence: float
    reasoning: str
    response_message: str
    specialist_used: str
    iteration_count: int
    error: Optional[str]
    log_trace: List[dict]
    sanitized_request: Optional[str]
    escalation_reason: Optional[str]
    prompt_version: Optional[str]


# Agent Classes
class SupervisorAgent:
    """Supervisor Agent - Routes requests to appropriate specialists."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("API_KEY")
        ).with_structured_output(RequestRouting)

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def route(self, request: str, employee_id: str) -> RequestRouting:
        """Route the request to the appropriate specialist."""
        prompt_data, prompt_version = load_prompt_with_fallback("supervisor", employee_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, request)

        start_time = time.time()

        try:
            messages = [
                SystemMessage(content=compiled_prompt),
                HumanMessage(content=request)
            ]
            routing = supervisor_breaker.call(lambda: self.llm.invoke(messages))

            latency_ms = (time.time() - start_time) * 1000

            logger.log_agent_call(
                user_id=employee_id,
                agent_name="supervisor",
                prompt_version=prompt_version,
                user_message=request,
                response=routing,
                tokens_used=0,
                latency_ms=latency_ms,
                success=True
            )

            return routing
        except CircuitBreakerOpen as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=employee_id,
                agent_name="supervisor",
                prompt_version=prompt_version,
                user_message=request,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=f"Circuit breaker open: {str(e)}"
            )
            raise
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=employee_id,
                agent_name="supervisor",
                prompt_version=prompt_version,
                user_message=request,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise


class ITAgent:
    """IT Agent - Handles IT support requests."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("API_KEY")
        ).with_structured_output(SpecialistResponse)

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def handle(self, request: str, employee_id: str) -> SpecialistResponse:
        """Handle IT support request."""
        prompt_data, prompt_version = load_prompt_with_fallback("it", employee_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, request)

        start_time = time.time()

        try:
            messages = [
                SystemMessage(content=compiled_prompt),
                HumanMessage(content=request)
            ]
            response = it_breaker.call(lambda: self.llm.invoke(messages))

            latency_ms = (time.time() - start_time) * 1000

            logger.log_agent_call(
                user_id=employee_id,
                agent_name="it",
                prompt_version=prompt_version,
                user_message=request,
                response=response,
                tokens_used=0,
                latency_ms=latency_ms,
                success=True
            )

            return response
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=employee_id,
                agent_name="it",
                prompt_version=prompt_version,
                user_message=request,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise


class HRAgent:
    """HR Agent - Handles HR requests."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("API_KEY")
        ).with_structured_output(SpecialistResponse)

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def handle(self, request: str, employee_id: str) -> SpecialistResponse:
        """Handle HR request."""
        prompt_data, prompt_version = load_prompt_with_fallback("hr", employee_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, request)

        start_time = time.time()

        try:
            messages = [
                SystemMessage(content=compiled_prompt),
                HumanMessage(content=request)
            ]
            response = hr_breaker.call(lambda: self.llm.invoke(messages))

            latency_ms = (time.time() - start_time) * 1000

            logger.log_agent_call(
                user_id=employee_id,
                agent_name="hr",
                prompt_version=prompt_version,
                user_message=request,
                response=response,
                tokens_used=0,
                latency_ms=latency_ms,
                success=True
            )

            return response
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=employee_id,
                agent_name="hr",
                prompt_version=prompt_version,
                user_message=request,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise


class FacilitiesAgent:
    """Facilities Agent - Handles facilities requests."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("API_KEY")
        ).with_structured_output(SpecialistResponse)

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def handle(self, request: str, employee_id: str) -> SpecialistResponse:
        """Handle facilities request."""
        prompt_data, prompt_version = load_prompt_with_fallback("facilities", employee_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, request)

        start_time = time.time()

        try:
            messages = [
                SystemMessage(content=compiled_prompt),
                HumanMessage(content=request)
            ]
            response = facilities_breaker.call(lambda: self.llm.invoke(messages))

            latency_ms = (time.time() - start_time) * 1000

            logger.log_agent_call(
                user_id=employee_id,
                agent_name="facilities",
                prompt_version=prompt_version,
                user_message=request,
                response=response,
                tokens_used=0,
                latency_ms=latency_ms,
                success=True
            )

            return response
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=employee_id,
                agent_name="facilities",
                prompt_version=prompt_version,
                user_message=request,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise


class EscalateAgent:
    """Escalate Agent - Handles escalated requests."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3
        ).with_structured_output(SpecialistResponse)

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def handle(self, request: str, employee_id: str, escalation_reason: str = None) -> SpecialistResponse:
        """Handle escalated request."""
        prompt_data, prompt_version = load_prompt_with_fallback("escalate", employee_id)

        # Add escalation context to the request if available
        if escalation_reason:
            request = f"[Escalation Reason: {escalation_reason}]\n\nOriginal Request: {request}"

        compiled_prompt = prompt_manager.compile_prompt(prompt_data, request)

        start_time = time.time()

        try:
            messages = [
                SystemMessage(content=compiled_prompt),
                HumanMessage(content=request)
            ]
            response = escalate_breaker.call(lambda: self.llm.invoke(messages))

            latency_ms = (time.time() - start_time) * 1000

            logger.log_agent_call(
                user_id=employee_id,
                agent_name="escalate",
                prompt_version=prompt_version,
                user_message=request,
                response=response,
                tokens_used=0,
                latency_ms=latency_ms,
                success=True
            )

            return response
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=employee_id,
                agent_name="escalate",
                prompt_version=prompt_version,
                user_message=request,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise


def create_ops_graph(supervisor, it_agent, hr_agent, facilities_agent, escalate_agent):
    """
    Create the LangGraph workflow for the Internal Ops Desk.
    """
    workflow = StateGraph(OpsState)

    # Supervisor Node
    def supervisor_node(state: OpsState) -> dict:
        """Supervisor Node - Routes requests."""
        try:
            # Check rate limit
            allowed, retry_after = rate_limiter.check_rate_limit(
                state["employee_id"],
                max_requests=10,
                window_seconds=60
            )
            if not allowed:
                return {
                    "error": f"Rate limit exceeded. Please try again in {retry_after} seconds.",
                    "escalation_reason": "rate_limit_exceeded"
                }

            # Check budget
            if not cost_tracker.check_budget(state["employee_id"], daily_limit=1.0):
                return {
                    "error": "Daily request limit reached. Please try again tomorrow.",
                    "escalation_reason": "budget_exceeded"
                }

            # Sanitize input
            sanitized, is_suspicious = input_sanitizer.sanitize(state["request_text"])

            if is_suspicious:
                logger.logger.warning(f"Injection attempt detected for employee {state['employee_id']}")
                # Route suspicious requests to escalate
                current_log_trace = state.get("log_trace", [])
                return {
                    "chosen_specialist": "escalate",
                    "confidence": 0.0,
                    "reasoning": "Suspicious input detected - routing to escalation",
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "sanitized_request": sanitized,
                    "escalation_reason": "injection_attempt_detected",
                    "log_trace": current_log_trace + [{
                        "agent": "supervisor",
                        "action": "routing",
                        "specialist": "escalate",
                        "confidence": 0.0,
                        "reasoning": "Suspicious input detected"
                    }]
                }

            # Route the request
            routing = supervisor.route(sanitized, state["employee_id"])

            # Check confidence threshold - route to escalate if below 0.5
            if routing.confidence < 0.5:
                current_log_trace = state.get("log_trace", [])
                return {
                    "chosen_specialist": "escalate",
                    "confidence": routing.confidence,
                    "reasoning": routing.reasoning,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "sanitized_request": sanitized,
                    "escalation_reason": f"low_confidence_{routing.confidence:.2f}",
                    "log_trace": current_log_trace + [{
                        "agent": "supervisor",
                        "action": "routing",
                        "specialist": "escalate",
                        "confidence": routing.confidence,
                        "reasoning": f"Low confidence routing: {routing.reasoning}"
                    }]
                }

            current_log_trace = state.get("log_trace", [])
            return {
                "chosen_specialist": routing.specialist,
                "confidence": routing.confidence,
                "reasoning": routing.reasoning,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "sanitized_request": sanitized,
                "log_trace": current_log_trace + [{
                    "agent": "supervisor",
                    "action": "routing",
                    "specialist": routing.specialist,
                    "confidence": routing.confidence,
                    "reasoning": routing.reasoning
                }]
            }
        except CircuitBreakerOpen as e:
            return {
                "error": "System temporarily unavailable. Please try again later.",
                "escalation_reason": "circuit_breaker_open"
            }
        except Exception as e:
            logger.logger.error(f"Supervisor error: {e}", exc_info=True)
            return {"error": str(e)}

    workflow.add_node("supervisor", supervisor_node)

    # IT Node
    def it_node(state: OpsState) -> dict:
        """IT Node - Handles IT requests."""
        try:
            request = state.get("sanitized_request") or state["request_text"]
            response_obj = it_agent.handle(request, state["employee_id"])

            # Validate output
            employee_email = state.get("employee_email", "")
            is_valid, error_type = output_validator.validate(
                response_obj.message,
                employee_email,
                action=response_obj.action,
                requires_approval=response_obj.requires_approval,
                ticket_type=response_obj.ticket_type
            )

            if not is_valid:
                return {
                    "error": f"Output validation failed: {error_type}",
                    "escalation_reason": f"validation_failed_{error_type}"
                }

            current_log_trace = state.get("log_trace", [])
            return {
                "response_message": response_obj.message,
                "specialist_used": "it",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": current_log_trace + [{
                    "agent": "it",
                    "action": response_obj.action,
                    "requires_approval": response_obj.requires_approval,
                    "confidence": response_obj.confidence
                }]
            }
        except Exception as e:
            logger.logger.error(f"IT error: {e}", exc_info=True)
            return {"error": str(e)}

    workflow.add_node("it", it_node)

    # HR Node
    def hr_node(state: OpsState) -> dict:
        """HR Node - Handles HR requests."""
        try:
            request = state.get("sanitized_request") or state["request_text"]
            response_obj = hr_agent.handle(request, state["employee_id"])

            # Validate output
            employee_email = state.get("employee_email", "")
            is_valid, error_type = output_validator.validate(
                response_obj.message,
                employee_email,
                action=response_obj.action,
                requires_approval=response_obj.requires_approval,
                ticket_type=response_obj.ticket_type
            )

            if not is_valid:
                return {
                    "error": f"Output validation failed: {error_type}",
                    "escalation_reason": f"validation_failed_{error_type}"
                }

            current_log_trace = state.get("log_trace", [])
            return {
                "response_message": response_obj.message,
                "specialist_used": "hr",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": current_log_trace + [{
                    "agent": "hr",
                    "action": response_obj.action,
                    "requires_approval": response_obj.requires_approval,
                    "confidence": response_obj.confidence
                }]
            }
        except Exception as e:
            logger.logger.error(f"HR error: {e}", exc_info=True)
            return {"error": str(e)}

    workflow.add_node("hr", hr_node)

    # Facilities Node
    def facilities_node(state: OpsState) -> dict:
        """Facilities Node - Handles facilities requests."""
        try:
            request = state.get("sanitized_request") or state["request_text"]
            response_obj = facilities_agent.handle(request, state["employee_id"])

            # Validate output
            employee_email = state.get("employee_email", "")
            is_valid, error_type = output_validator.validate(
                response_obj.message,
                employee_email,
                action=response_obj.action,
                requires_approval=response_obj.requires_approval,
                ticket_type=response_obj.ticket_type
            )

            if not is_valid:
                return {
                    "error": f"Output validation failed: {error_type}",
                    "escalation_reason": f"validation_failed_{error_type}"
                }

            current_log_trace = state.get("log_trace", [])
            return {
                "response_message": response_obj.message,
                "specialist_used": "facilities",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": current_log_trace + [{
                    "agent": "facilities",
                    "action": response_obj.action,
                    "requires_approval": response_obj.requires_approval,
                    "confidence": response_obj.confidence
                }]
            }
        except Exception as e:
            logger.logger.error(f"Facilities error: {e}", exc_info=True)
            return {"error": str(e)}

    workflow.add_node("facilities", facilities_node)

    # Escalate Node
    def escalate_node(state: OpsState) -> dict:
        """Escalate Node - Handles escalated requests."""
        try:
            request = state.get("sanitized_request") or state["request_text"]
            escalation_reason = state.get("escalation_reason")
            response_obj = escalate_agent.handle(request, state["employee_id"], escalation_reason)

            # Validate output
            employee_email = state.get("employee_email", "")
            is_valid, error_type = output_validator.validate(
                response_obj.message,
                employee_email,
                action=response_obj.action,
                requires_approval=response_obj.requires_approval,
                ticket_type=response_obj.ticket_type
            )

            if not is_valid:
                return {"error": f"Output validation failed: {error_type}"}

            current_log_trace = state.get("log_trace", [])
            return {
                "response_message": response_obj.message,
                "specialist_used": "escalate",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": current_log_trace + [{
                    "agent": "escalate",
                    "action": response_obj.action,
                    "requires_approval": response_obj.requires_approval,
                    "confidence": response_obj.confidence,
                    "escalation_reason": escalation_reason
                }]
            }
        except Exception as e:
            logger.logger.error(f"Escalate error: {e}", exc_info=True)
            return {"error": str(e)}

    workflow.add_node("escalate", escalate_node)

    # Build Graph Structure
    workflow.add_edge(START, "supervisor")

    def routing_decision(state: OpsState) -> str:
        """Routing Decision - Routes based on specialist."""
        if state.get("error"):
            return "escalate"  # Route errors to escalate
        return state["chosen_specialist"]

    workflow.add_conditional_edges("supervisor", routing_decision, {
        "it": "it",
        "hr": "hr",
        "facilities": "facilities",
        "escalate": "escalate"
    })

    # Connect every specialist to END
    workflow.add_edge("it", END)
    workflow.add_edge("hr", END)
    workflow.add_edge("facilities", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


def process_request(graph, request_text: str, employee_id: str, employee_email: str = None) -> dict:
    """
    Process a single employee request through the Internal Ops Desk.

    Args:
        graph: Compiled LangGraph workflow
        request_text: Employee's request text
        employee_id: Employee identifier
        employee_email: Employee's email (optional)

    Returns:
        Final state after processing
    """
    initial_state = {
        "request_text": request_text,
        "employee_id": employee_id,
        "employee_email": employee_email,
        "chosen_specialist": "escalate",  # default
        "confidence": 0.0,
        "reasoning": "",
        "response_message": "",
        "specialist_used": "",
        "iteration_count": 0,
        "error": None,
        "log_trace": [],
        "sanitized_request": None,
        "escalation_reason": None,
        "prompt_version": None
    }

    result = graph.invoke(initial_state)
    return result


def main():
    """Main function - Runs the Internal Ops Desk with sample requests."""
    # Initialize agents
    supervisor = SupervisorAgent()
    it_agent = ITAgent()
    hr_agent = HRAgent()
    facilities_agent = FacilitiesAgent()
    escalate_agent = EscalateAgent()

    # Create the workflow graph
    graph = create_ops_graph(supervisor, it_agent, hr_agent, facilities_agent, escalate_agent)
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="internal_ops_desk_workflow.png")
    print("\n" + "=" * 100) 
    print("INTERNAL OPS DESK - Multi-Agent System Demo")
    print("=" * 100)

    # Sample requests for each specialist
    test_cases = [
        {
            "request": "VPN keeps disconnecting on macOS after sleep. Any fix?",
            "employee_id": "emp_001",
            "employee_email": "john.doe@company.com",
            "expected_specialist": "it"
        },
        {
            "request": "When does health insurance coverage start after joining?",
            "employee_id": "emp_002",
            "employee_email": "jane.smith@company.com",
            "expected_specialist": "hr"
        },
        {
            "request": "My badge stopped working at the 3rd floor door.",
            "employee_id": "emp_003",
            "employee_email": "bob.wilson@company.com",
            "expected_specialist": "facilities"
        },
        {
            "request": "I forgot my password and can't log in to the HR portal",
            "employee_id": "emp_004",
            "employee_email": "alice.brown@company.com",
            "expected_specialist": "it"
        },
        {
            "request": "How many PTO days do I have left this year?",
            "employee_id": "emp_005",
            "employee_email": "charlie.davis@company.com",
            "expected_specialist": "hr"
        },
    ]

    results = []
    for test_case in test_cases:
        print(f"\n{'-' * 100}")
        print(f"Request: {test_case['request']}")
        print(f"Employee: {test_case['employee_id']}")
        print(f"Expected Specialist: {test_case['expected_specialist']}")
        print(f"{'-' * 100}")

        result = process_request(
            graph,
            test_case["request"],
            test_case["employee_id"],
            test_case["employee_email"]
        )
        results.append(result)

        specialist_match = "CORRECT" if result.get("specialist_used") == test_case["expected_specialist"] else "MISMATCH"

        print(f"\nResult:")
        print(f"  Specialist Used: {result.get('specialist_used', 'N/A')} [{specialist_match}]")
        print(f"  Routing Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')[:80]}...")
        print(f"  Response: {result.get('response_message', 'N/A')[:150]}...")
        if result.get('error'):
            print(f"  Error: {result['error']}")
        print(f"  Log Trace Entries: {len(result.get('log_trace', []))}")

    # Summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"Total requests processed: {len(results)}")
    print(f"Successful: {len([r for r in results if not r.get('error')])}")
    print(f"Failed: {len([r for r in results if r.get('error')])}")

    specialist_counts = {}
    for r in results:
        specialist = r.get('specialist_used', 'unknown')
        specialist_counts[specialist] = specialist_counts.get(specialist, 0) + 1

    print(f"\nRouting Distribution:")
    for specialist, count in sorted(specialist_counts.items()):
        print(f"  {specialist}: {count}")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
