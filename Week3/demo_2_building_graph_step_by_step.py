"""
Demo 2: Building LangGraph Step-by-Step
=======================================

This intermediate demo shows how to build a LangGraph incrementally:
- Start with empty graph
- Add nodes one by one
- Add edges
- Add conditional routing
- Compile and test

Perfect for students learning how to build graphs from scratch.

"""

import sys
from pathlib import Path
from typing import TypedDict, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END


# ==================== STATE DEFINITION ====================

class SimpleState(TypedDict):
    """Simple state for step-by-step tutorial"""
    ticket: str
    specialist: Literal['billing', 'technical', 'general']
    response: str


# ==================== STEP-BY-STEP BUILDING ====================

def step_1_create_graph():
    """Step 1: Create an empty graph"""
    print("\n" + "="*70)
    print("STEP 1: CREATE EMPTY GRAPH")
    print("="*70 + "\n")
    
    print("Code:")
    print("  workflow = StateGraph(SimpleState)")
    print()
    
    workflow = StateGraph(SimpleState)
    
    print("✅ Graph created! Currently empty (no nodes, no edges)")
    print(f"   Graph type: {type(workflow)}")
    print()
    
    return workflow


def step_2_add_supervisor_node(workflow):
    """Step 2: Add supervisor node"""
    print("\n" + "="*70)
    print("STEP 2: ADD SUPERVISOR NODE")
    print("="*70 + "\n")
    
    def supervisor_node(state: SimpleState) -> SimpleState:
        """Supervisor routes the ticket"""
        print(f"   [Supervisor] Processing ticket: {state['ticket']}")
        
        # Simple routing logic
        ticket_lower = state['ticket'].lower()
        if any(word in ticket_lower for word in ['charge', 'refund', 'payment']):
            state['specialist'] = 'billing'
        elif any(word in ticket_lower for word in ['bug', 'error', 'api']):
            state['specialist'] = 'technical'
        else:
            state['specialist'] = 'general'
        
        print(f"   [Supervisor] Routed to: {state['specialist']}")
        return state
    
    print("Code:")
    print("  def supervisor_node(state: SimpleState) -> SimpleState:")
    print("      # Routing logic here")
    print("      return state")
    print()
    print("  workflow.add_node('supervisor', supervisor_node)")
    print()
    
    workflow.add_node('supervisor', supervisor_node)
    
    print("✅ Supervisor node added!")
    print(f"   Nodes in graph: {list(workflow.nodes.keys())}")
    print()
    
    return workflow


def step_3_add_specialist_nodes(workflow):
    """Step 3: Add specialist nodes"""
    print("\n" + "="*70)
    print("STEP 3: ADD SPECIALIST NODES")
    print("="*70 + "\n")
    
    def billing_node(state: SimpleState) -> SimpleState:
        print(f"   [Billing] Handling: {state['ticket']}")
        state['response'] = f"Billing response for: {state['ticket']}"
        return state
    
    def technical_node(state: SimpleState) -> SimpleState:
        print(f"   [Technical] Handling: {state['ticket']}")
        state['response'] = f"Technical response for: {state['ticket']}"
        return state
    
    def general_node(state: SimpleState) -> SimpleState:
        print(f"   [General] Handling: {state['ticket']}")
        state['response'] = f"General response for: {state['ticket']}"
        return state
    
    print("Code:")
    print("  workflow.add_node('billing', billing_node)")
    print("  workflow.add_node('technical', technical_node)")
    print("  workflow.add_node('general', general_node)")
    print()
    
    workflow.add_node('billing', billing_node)
    workflow.add_node('technical', technical_node)
    workflow.add_node('general', general_node)
    
    print("✅ All specialist nodes added!")
    print(f"   Nodes in graph: {list(workflow.nodes.keys())}")
    print()
    
    return workflow


def step_4_set_entry_point(workflow):
    """Step 4: Set entry point"""
    print("\n" + "="*70)
    print("STEP 4: SET ENTRY POINT")
    print("="*70 + "\n")
    
    print("Code:")
    print("  workflow.set_entry_point('supervisor')")
    print()
    print("This tells the graph: 'Start execution at the supervisor node'")
    print()
    
    workflow.set_entry_point('supervisor')
    
    print("✅ Entry point set to 'supervisor'")
    print("   Graph will always start here")
    print()
    
    return workflow


def step_5_add_conditional_routing(workflow):
    """Step 5: Add conditional routing"""
    print("\n" + "="*70)
    print("STEP 5: ADD CONDITIONAL ROUTING")
    print("="*70 + "\n")
    
    def route_to_specialist(state: SimpleState) -> str:
        """Routing function - returns which node to go to next"""
        specialist = state['specialist']
        print(f"   [Routing] Routing to: {specialist}")
        return specialist
    
    print("Code:")
    print("  def route_to_specialist(state: SimpleState) -> str:")
    print("      return state['specialist']  # Returns 'billing', 'technical', or 'general'")
    print()
    print("  workflow.add_conditional_edges(")
    print("      'supervisor',")
    print("      route_to_specialist,")
    print("      {")
    print("          'billing': 'billing',")
    print("          'technical': 'technical',")
    print("          'general': 'general'")
    print("      }")
    print("  )")
    print()
    
    workflow.add_conditional_edges(
        'supervisor',
        route_to_specialist,
        {
            'billing': 'billing',
            'technical': 'technical',
            'general': 'general'
        }
    )
    
    print("✅ Conditional routing added!")
    print("   After supervisor, graph will route based on state['specialist']")
    print()
    
    return workflow


def step_6_add_edges_to_end(workflow):
    """Step 6: Add edges to END"""
    print("\n" + "="*70)
    print("STEP 6: ADD EDGES TO END")
    print("="*70 + "\n")
    
    print("Code:")
    print("  workflow.add_edge('billing', END)")
    print("  workflow.add_edge('technical', END)")
    print("  workflow.add_edge('general', END)")
    print()
    print("This tells the graph: 'After each specialist, end execution'")
    print()
    
    workflow.add_edge('billing', END)
    workflow.add_edge('technical', END)
    workflow.add_edge('general', END)
    
    print("✅ All edges to END added!")
    print("   Graph structure complete")
    print()
    
    return workflow


def step_7_compile_and_test(workflow):
    """Step 7: Compile and test"""
    print("\n" + "="*70)
    print("STEP 7: COMPILE AND TEST")
    print("="*70 + "\n")
    
    print("Code:")
    print("  graph = workflow.compile()")
    print()
    print("This compiles the graph. Now it's ready to use!")
    print()
    
    graph = workflow.compile()
    
    print("✅ Graph compiled!")
    print()
    
    # Test the graph
    print("Testing the graph...")
    print()
    
    test_cases = [
        "I was charged twice",
        "The API is broken",
        "I need help"
    ]
    
    for ticket in test_cases:
        print(f"Test: {ticket}")
        initial_state: SimpleState = {
            'ticket': ticket,
            'specialist': 'general',
            'response': ''
        }
        
        result = graph.invoke(initial_state)
        print(f"   Result: {result['response']}")
        print()
    
    return graph


# ==================== COMPLETE TUTORIAL ====================

def main():
    """Run the step-by-step tutorial"""
    
    print("\n" + "="*70)
    print("🏗️  DEMO 9: Building LangGraph Step-by-Step")
    print("="*70)
    print("\nThis tutorial shows you how to build a LangGraph from scratch.")
    print("We'll add one piece at a time and see how it works.\n")
    
    input("Press Enter to start...")
    
    # Step 1: Create graph
    workflow = step_1_create_graph()
    input("Press Enter to continue to Step 2...")
    
    # Step 2: Add supervisor
    workflow = step_2_add_supervisor_node(workflow)
    input("Press Enter to continue to Step 3...")
    
    # Step 3: Add specialists
    workflow = step_3_add_specialist_nodes(workflow)
    input("Press Enter to continue to Step 4...")
    
    # Step 4: Set entry point
    workflow = step_4_set_entry_point(workflow)
    input("Press Enter to continue to Step 5...")
    
    # Step 5: Add conditional routing
    workflow = step_5_add_conditional_routing(workflow)
    input("Press Enter to continue to Step 6...")
    
    # Step 6: Add edges
    workflow = step_6_add_edges_to_end(workflow)
    input("Press Enter to continue to Step 7...")
    
    # Step 7: Compile and test
    graph = step_7_compile_and_test(workflow)
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY: Graph Building Steps")
    print("="*70 + "\n")
    
    steps = [
        "1. Create graph: StateGraph(State)",
        "2. Add nodes: workflow.add_node('name', function)",
        "3. Set entry point: workflow.set_entry_point('node')",
        "4. Add conditional routing: workflow.add_conditional_edges(...)",
        "5. Add edges: workflow.add_edge('from', 'to')",
        "6. Compile: graph = workflow.compile()",
        "7. Invoke: result = graph.invoke(state)"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print("\n" + "="*70)
    print("✅ Tutorial Complete!")
    print("="*70 + "\n")
    
    print("💡 Key Takeaways:")
    print("  • Build graph incrementally")
    print("  • Add nodes first")
    print("  • Then add routing")
    print("  • Finally add edges")
    print("  • Compile before using")
    print("  • Test with simple state\n")


if __name__ == "__main__":
    main()
