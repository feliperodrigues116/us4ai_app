from langgraph.graph import StateGraph, END
from src.graph.state import US4AIState
from src.graph.nodes import detect_ai_node, risk_identification_node, requirement_derivation_node

def build_us4ai_graph():
    workflow = StateGraph(US4AIState)

    workflow.add_node("detect_ai", detect_ai_node)
    workflow.add_node("identify_risks", risk_identification_node)
    workflow.add_node("derive_requirements", requirement_derivation_node)

    workflow.set_entry_point("detect_ai")

    # Se não for de IA, ele encerra o fluxo e pula os nós caros de RAG/Requisitos.
    workflow.add_conditional_edges(
        "detect_ai",
        lambda state: "identify_risks" if state["is_ai_related"] else END
    )

    workflow.add_edge("identify_risks", "derive_requirements")
    workflow.add_edge("derive_requirements", END)

    return workflow.compile()
