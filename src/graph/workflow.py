from langgraph.graph import StateGraph, END
from src.graph.state import US4AIState
from src.graph.nodes import risk_identification_node, requirement_derivation_node

def build_us4ai_graph():
    workflow = StateGraph(US4AIState)

    workflow.add_node("identify_risks", risk_identification_node)
    workflow.add_node("derive_requirements", requirement_derivation_node)

    workflow.set_entry_point("identify_risks")

    workflow.add_edge("identify_risks", "derive_requirements")
    workflow.add_edge("derive_requirements", END)

    return workflow.compile()
