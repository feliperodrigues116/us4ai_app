"""NIST evidence to contextual risk to requirement orchestration."""

from functools import partial

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import derive_ai_requirements, infer_contextual_risks, retrieve_nist_evidence, retrieve_treatment_evidence
from src.graph.state import US4AIState
from src.retriever import NistVectorRetriever


def build_us4ai_graph(*, retriever=None, client=None):
    """Compile the graph; injected dependencies support isolated offline tests."""
    retriever = retriever if retriever is not None else NistVectorRetriever()
    workflow = StateGraph(US4AIState)
    workflow.add_node("retrieve_nist_evidence", partial(retrieve_nist_evidence, retriever=retriever))
    workflow.add_node("infer_contextual_risks", partial(infer_contextual_risks, client=client))
    workflow.add_node("retrieve_treatment_evidence", partial(retrieve_treatment_evidence, retriever=retriever))
    workflow.add_node("derive_ai_requirements", partial(derive_ai_requirements, client=client))
    workflow.add_edge(START, "retrieve_nist_evidence")
    workflow.add_conditional_edges(
        "retrieve_nist_evidence",
        lambda state: "infer_contextual_risks" if state["retrieved_evidence"] else END,
        {"infer_contextual_risks": "infer_contextual_risks", END: END},
    )
    workflow.add_conditional_edges(
        "infer_contextual_risks",
        lambda state: "retrieve_treatment_evidence" if state["contextual_risks"] else END,
        {"retrieve_treatment_evidence": "retrieve_treatment_evidence", END: END},
    )
    workflow.add_conditional_edges(
        "retrieve_treatment_evidence",
        lambda state: "derive_ai_requirements" if state["treatment_evidence"] else END,
        {"derive_ai_requirements": "derive_ai_requirements", END: END},
    )
    workflow.add_edge("derive_ai_requirements", END)
    return workflow.compile()
