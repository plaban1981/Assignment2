from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Optional
from prompt_manager import PromptManager
from ab_test_manager import ABTestManager
from pathlib import Path
from models import TicketRouting, SupportResponse
from error_handling import retry_with_backoff
from circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from logging_config import StructuredLogger
from cost_tracker import CostTracker
from rate_limiter import RateLimiter
from input_sanitizer import InputSanitizer
from output_validator import OutputValidator
import time
load_dotenv()

prompt_manager = PromptManager(prompts_dir=str(Path(__file__).parent / "prompts"))
ab_test_manager = ABTestManager()
cost_tracker = CostTracker()
rate_limiter = RateLimiter()
logger = StructuredLogger("customer_support_agent")
input_sanitizer = InputSanitizer()
output_validator = OutputValidator(allowed_emails=["support@techcorp.com"])
# initialize the circuit breaker for each agent

supervisor_circuit_breaker = CircuitBreaker(max_failures=3, timeout=60)
billing_circuit_breaker = CircuitBreaker(max_failures=3, timeout=60)
technical_circuit_breaker = CircuitBreaker(max_failures=3, timeout=60)
general_circuit_breaker = CircuitBreaker(max_failures=3, timeout=60)
escalate_circuit_breaker = CircuitBreaker(max_failures=3, timeout=60)

#Load the Prompt 
def load_prompt_with_fallback(agent_name:str, user_id: str):
    """ Load the prompt with fallback to v1.0.0 if current version is not available"""
    prompt_version = ab_test_manager.get_prompt_version(agent_name, user_id)

    if prompt_version == "current":
        try:
            prompt_data = prompt_manager.load_prompt(agent_name, prompt_version)
            return prompt_data, prompt_version
        except ValueError:
            prompt_version = "v1.0.0"
            prompt_data = prompt_manager.load_prompt(agent_name, prompt_version)
            return prompt_data, prompt_version
    else:
        prompt_data = prompt_manager.load_prompt(agent_name, prompt_version)
        return prompt_data, prompt_version

# Define the State

class AgentState(TypedDict):
    ticket: str
    user_id: str
    user_email: str
    specialist: Literal['billing', 'technical', 'general','escalate']
    routing_confidence: float
    routing_reasoning: str
    response: str
    specialist_used: str
    iteration_count: int
    log_trace: list
    sanitized_ticket: str
    prompt_version: str
    tokens_used: Optional[int]
    cost: Optional[float]
    error: Optional[str]

# Bad Practice: Updating the state directly
# state["user_id"] = "123456"  # not recommmended
# return {"user_id": "123456"} # this is the correct way to update the state
# Agents

# Define each class for new agents - 
# The reason, 
# The response is different for each agent - so we need to define a new class for each agent
# The specialist is different for each agent - so we need to define a new class for each agent
# The iteration count is different for each agent - so we need to define a new class for each agent
# The log trace is different for each agent - so we need to define a new class for each agent

class SupervisorAgent:
    """ Supervisor Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.6).with_structured_output(TicketRouting)
    # Use the cheapest model for this agent

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def route(self, ticket: str, user_id: str) :
        """ Route the ticket to the appropriate specialist """
        prompt_data, prompt_version = load_prompt_with_fallback("supervisor", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        start_time = time.time() # track the time taken to route the ticket
        try:
            messages = [
                SystemMessage(content=compiled_prompt),
                HumanMessage(content=ticket)
            ]
            
            routing = supervisor_circuit_breaker.call(lambda: self.llm.invoke(messages))
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="supervisor",
                prompt_version=prompt_version,
                user_message=ticket,
                response=routing,
                tokens_used=0, # we don't know the tokens used yet because we are using the structured output
                latency_ms=latency_ms,
                success=True,
                error=None
            )
            return routing
        except CircuitBreakerOpen as e:
            print(f"Circuit Breaker Open: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="supervisor",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise 
        
        except Exception as e:
            print(f"Error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="supervisor",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise


class BillingAgent:
    """ Billing Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6).with_structured_output(SupportResponse)
    # Use the cheapest model for this agent

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def handle(self, ticket: str, user_id: str) -> str : # Enhance the readability of the code by using the return type
        """ Handle the Billing related ticket"""
        prompt_data, prompt_version = load_prompt_with_fallback("billing", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        start_time = time.time()
        try:
            messages = [
                SystemMessage(content=compiled_prompt),
                HumanMessage(content=ticket)
            ]

            response = billing_circuit_breaker.call(lambda: self.llm.invoke(messages))
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="billing",
                prompt_version=prompt_version,
                user_message=ticket,
                response=response,
                tokens_used=0,
                latency_ms=latency_ms,
                success=True,
                error=None
            )
            return response
        except CircuitBreakerOpen as e:
            print(f"Circuit Breaker Open: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="billing",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise 
        
        except Exception as e:
            print(f"Error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="billing",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise


class TechnicalAgent:
    """ Technical Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6).with_structured_output(SupportResponse)

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def handle(self, ticket: str, user_id: str) -> str :
        """ Handle the Technical related ticket"""
        prompt_data, prompt_version = load_prompt_with_fallback("technical", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        start_time = time.time()

        try:
            messages = [
            SystemMessage(content=compiled_prompt),
            HumanMessage(content=ticket)
        ]
            response = technical_circuit_breaker.call(lambda: self.llm.invoke(messages))
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="technical",
                prompt_version=prompt_version,
                user_message=ticket,
                response=response,
                tokens_used=0,
                latency_ms=latency_ms,
                success=True,
                error=None
            )
            return response
        except CircuitBreakerOpen as e:
            print(f"Circuit Breaker Open: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="technical",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise 
        
        except Exception as e:
            print(f"Error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="technical",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise

class GeneralAgent:
    """ General Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.6).with_structured_output(SupportResponse)

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def handle(self, ticket: str,user_id: str) -> str :
        """ Handle the General related ticket"""
        prompt_data, prompt_version = load_prompt_with_fallback("general", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        start_time = time.time()
        try:
            messages = [
                SystemMessage(content=compiled_prompt),
                HumanMessage(content=ticket)
            ]
            response = general_circuit_breaker.call(lambda: self.llm.invoke(messages))
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="general",
                prompt_version=prompt_version,
                user_message=ticket,
                response=response,
                tokens_used=0,
                latency_ms=latency_ms,
                success=True,
                error=None
            )
            return response
        except CircuitBreakerOpen as e:
            print(f"Circuit Breaker Open: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="general",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise 
        
        except Exception as e:
            print(f"Error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="general",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise

class EscalateAgent:
    """ Escalate Agent """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.6).with_structured_output(SupportResponse)

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def handle(self, ticket: str,user_id: str) -> str :
        """ Handle the Escalate related ticket"""
        prompt_data, prompt_version = load_prompt_with_fallback("escalate", user_id)
        compiled_prompt = prompt_manager.compile_prompt(prompt_data, ticket)
        start_time = time.time()
        try:
            messages = [
                SystemMessage(content=compiled_prompt),
                HumanMessage(content=ticket)
            ]
            response = escalate_circuit_breaker.call(lambda: self.llm.invoke(messages))
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="escalate",
                prompt_version=prompt_version,
                user_message=ticket,
                response=response,
                tokens_used=0,
                latency_ms=latency_ms,
                success=True,
                error=None
            )
            return response
        except CircuitBreakerOpen as e:
            print(f"Circuit Breaker Open: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="escalate",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise 
        
        except Exception as e:
            print(f"Error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            logger.log_agent_call(
                user_id=user_id,
                agent_name="escalate",
                prompt_version=prompt_version,
                user_message=ticket,
                response=None,
                tokens_used=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            raise
# Steps to create the graph:
# 1. Initialize the StateGraph
# 2. Add the nodes for each agent
# 3. Add the edges for each agent - based on the routing decision
# 4. Compile the graph
# 5. Run the graph

def create_simple_graph(supervisor, billing, technical, general, escalate):
    """
    Create a simple graph for the ticket routing system
    Args:
        supervisor: SupervisorAgent
        billing: BillingAgent
        technical: TechnicalAgent
        general: GeneralAgent
        escalate: EscalateAgent
    Returns:
        workflow: StateGraph

    """
    # Initialize the StateGraph
    workflow = StateGraph(AgentState)

    # Supervisor Node
    def supervisor_node(state: AgentState) -> AgentState:
        """ Supervisor Node """
        print("Supervisor Node: Analyzing the ticket")
        try:
            # check the Rate Limit
            start_time = time.time()
            allowed, retry_after = rate_limiter.check_rate_limit(state["user_id"],
            max_requests=10,
            window_seconds=60
            )

            if not allowed:
                return {"error": f"Rate limit exceeded. Retry after {retry_after}s"}
            
            if not cost_tracker.check_budget(state["user_id"], daily_limit=1.0):
                return {"error": "Daily budget exceeded"}

            # Sanitize the input
            sanitized_ticket, is_suspicious = input_sanitizer.sanitize(state["ticket"])
            if is_suspicious:
                logger.logger.warning(f"Suspicious input from {state['user_id']}: {state['ticket'][:100]}")
            


            routing = supervisor.route(sanitized_ticket,state["user_id"])
            print(f"Routed to {routing.specialist} with confidence {routing.confidence} and reasoning {routing.reasoning}")
            return {
                "specialist": routing.specialist,
                "routing_confidence": routing.confidence,
                "routing_reasoning": routing.reasoning,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": state.get("log_trace", []) + [{
                    "agent": "supervisor",
                    "action": "routing",
                    "specialist": routing.specialist,
                    "confidence": routing.confidence,
                    "reasoning": routing.reasoning
                }]
            }
        except Exception as e:
            logger.logger.error(f"Supervisor Node Error: {e}")
            return {"error": str(e)}

    workflow.add_node("supervisor", supervisor_node)

    # Billing Node
    def billing_node(state: AgentState) -> AgentState:
        """ Billing Node """
        print("Billing Node: Handling the ticket")
        try:
            response = billing.handle(state["ticket"],state["user_id"])
            #print(f"Handled the ticket with response {state['response']}")
            # validate the output
            is_valid, error = output_validator.validate(response.message, state["user_email"],action=response.action, requires_approval=response.requires_approval)
            if not is_valid:
                logger.logger.error(f"Billing Node Error, Output validation failed: {error}")
                return {"error": error}
            # update the state
            return {
                "response": response.message,
                "specialist_used": "billing",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": state.get("log_trace", []) + [{
                    "agent": "billing",
                    "action": "handling",
                    "response": response.message
                }]
            }
        except Exception as e:
            logger.logger.error(f"Billing Node Error: {e}")
            return {"error": str(e)}

    workflow.add_node("billing", billing_node)

    # Technical Node
    def technical_node(state: AgentState) -> AgentState:
        """ Technical Node """
        print("Technical Node: Handling the ticket")
        try:
            response = technical.handle(state["ticket"],state["user_id"])
            # validate the output
            is_valid, error = output_validator.validate(response.message, state["user_email"],action=response.action, requires_approval=response.requires_approval)
            if not is_valid:
                logger.logger.error(f"Technical Node Error, Output validation failed: {error}")
                return {"error": error}
            # update the state
            return {
                "response": response.message,
                "specialist_used": "technical",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": state.get("log_trace", []) + [{
                    "agent": "technical",
                    "action": "handling",
                    "response": response.message
                }]
            }
        except Exception as e:
            logger.logger.error(f"Technical Node Error: {e}")
            return {"error": str(e)}
    workflow.add_node("technical", technical_node)

    # General Node
    def general_node(state: AgentState) -> AgentState:
        """ General Node """
        print("General Node: Handling the ticket")
        try:
            response = general.handle(state["ticket"],state["user_id"])
            # validate the output
            is_valid, error = output_validator.validate(response.message, state["user_email"],action=response.action, requires_approval=response.requires_approval)
            if not is_valid:
                logger.logger.error(f"General Node Error, Output validation failed: {error}")
                return {"error": error}
            # update the state
            return {
                "response": response.message,
                "specialist_used": "general",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": state.get("log_trace", []) + [{
                    "agent": "general",
                    "action": "handling",
                    "response": response.message
                }]
            }
        except Exception as e:
            logger.logger.error(f"General Node Error: {e}")
            return {"error": str(e)}

    workflow.add_node("general", general_node)

    # Escalate Node
    def escalate_node(state: AgentState) -> AgentState:
        """ Escalate Node """
        print("Escalate Node: Escalating the ticket")
        try:
            response = escalate.handle(state["ticket"],state["user_id"])
            # validate the output
            is_valid, error = output_validator.validate(response.message, state["user_email"],action=response.action, requires_approval=response.requires_approval)
            if not is_valid:
                logger.logger.error(f"Escalate Node Error, Output validation failed: {error}")
                return {"error": error}
            # update the state
            return {
                "response": response.message,
                "specialist_used": "escalate",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "log_trace": state.get("log_trace", []) + [{
                    "agent": "escalate",
                    "action": "handling",
                    "response": response.message
                }]
            }
        except Exception as e:
            logger.logger.error(f"Escalate Node Error: {e}")
            return {"error": str(e)}

    workflow.add_node("escalate", escalate_node)

    # Add the edges for each agent - based on the routing decision
    # Build Graph Structure
    workflow.add_edge(START, "supervisor")

    def routing_decision(state: AgentState) -> str:
        """ Routing Decision """
        return state["specialist"]

    workflow.add_conditional_edges("supervisor", routing_decision, {
        "billing": "billing",
        "technical": "technical",
        "general": "general",
        "escalate": "escalate"
    })
    # Connect every specialist to the END node
    workflow.add_edge("billing", END)
    workflow.add_edge("technical", END)
    workflow.add_edge("general", END)
    workflow.add_edge("escalate", END)
    return workflow.compile()

def main():
    """ Main function """
    supervisor = SupervisorAgent()
    billing = BillingAgent()
    technical = TechnicalAgent()
    general = GeneralAgent()
    escalate = EscalateAgent()
    graph = create_simple_graph(supervisor, billing, technical, general, escalate)
    
    test_cases = [
        "I was charged twice for my order# 12345",
        "How do I integrate my website with your payment gateway?",
        "I forgot my password, can you help me reset it?",
        "My Subscription is not working, can you help me?",
        "API is returning 429 error",
        "I am going to sue you for $1000000"
    ]
    results = []
    for test_case in test_cases:
        print(f"Testing: {test_case}")
        initial_state = {
            "ticket": test_case,
            "user_id": "1234567890",
            "specialist": "general", # default specialist
            "routing_confidence": 0.0,
            "routing_reasoning": "",
            "response": "",
            "specialist_used": "",
            "iteration_count": 0,
            "log_trace": []
        }
        result = graph.invoke(initial_state)
        results.append(result)
        # print(f"Result: {result}")
        print(f"Specialist used: {result['specialist_used']}")
        print(f"Response: {result['response']}")
        print(f"Routing confidence: {result['routing_confidence']}")
        print(f"Routing reasoning: {result['routing_reasoning']}")
        print(f"Iteration count: {result['iteration_count']}")
        print(f"Log trace: {result['log_trace']}")
        print("-"*100)
    print(result)

if __name__ == "__main__":
    main()