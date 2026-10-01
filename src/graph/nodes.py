"""Retrieval, contextual inference, and requirement derivation nodes."""

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.graph.prompts import build_requirement_messages, build_risk_messages
from src.graph.state import US4AIState
from src.retriever import NistVectorRetriever, build_retrieval_query, build_treatment_query
from src.schemas import AIRequirementsOutput, ContextualRisksOutput, US4AIAnalysisInput
from src.traceability import reference_ai_tasks, validate_ai_requirements, validate_contextual_risks


def create_generation_client():
    """Create an Instructor client only when a generation stage needs it."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for contextual generation.")
    import instructor
    from openai import OpenAI

    return instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY, max_retries=0))


def retrieve_nist_evidence(state: US4AIState, *, retriever=None) -> dict:
    """Retrieve unchanged NIST evidence and resolve its complete source records."""
    scenario = US4AIAnalysisInput.model_validate(state["analysis_input"])
    query = build_retrieval_query(scenario)
    retriever = retriever if retriever is not None else NistVectorRetriever()
    # Keep diagnostics request-local: the cached retriever may serve multiple sessions.
    diagnostics = None
    if isinstance(retriever, NistVectorRetriever):
        diagnostics = {}
        evidence = retriever.retrieve(scenario, diagnostics=diagnostics)
    else:
        evidence = retriever.retrieve(scenario)
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise ValueError("Duplicate retrieved NIST evidence IDs are not allowed.")
    records = {item.evidence_id: item.resolve_record() for item in evidence}
    return {
        "analysis_input": scenario,
        "ai_task_references": reference_ai_tasks(scenario),
        "retrieval_query": query,
        "retrieval_diagnostics": diagnostics,
        "retrieved_evidence": evidence,
        "nist_records": records,
        "contextual_risks": [],
        "ai_requirements": [],
        "treatment_evidence": [],
        "treatment_records": {},
        "treatment_diagnostics": None,
        "status": "evidence_retrieved" if evidence else "no_evidence",
        "status_message": "NIST evidence retrieved." if evidence else
            "No NIST evidence was retrieved; generation was skipped. No risk-free conclusion can be drawn.",
    }


def infer_contextual_risks(state: US4AIState, *, client=None) -> dict:
    """Generate contextual inferences, then enforce reference integrity."""
    if not state["retrieved_evidence"]:
        raise ValueError("Contextual risk inference requires retrieved NIST evidence.")
    messages = build_risk_messages(state["analysis_input"], state["retrieved_evidence"], state["nist_records"])
    client = client if client is not None else create_generation_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0, max_retries=1,
        response_model=ContextualRisksOutput, messages=messages,
    )
    output = ContextualRisksOutput.model_validate(response)
    risks = output.contextual_risks
    validate_contextual_risks(risks, state["analysis_input"], state["retrieved_evidence"])
    return {
        "contextual_risks": risks,
        "status": "risks_inferred" if risks else "no_supported_risks",
        "status_message": "Contextual risks inferred from scenario and NIST evidence." if risks else
            "No sufficiently supported contextual risks were inferred. This does not mean the system is risk-free.",
    }


def retrieve_treatment_evidence(state: US4AIState, *, retriever=None) -> dict:
    """Retrieve once for all validated risks using the same NIST index and ranking."""
    risks = state["contextual_risks"]
    if not risks:
        raise ValueError("Treatment retrieval requires contextual risks.")
    validate_contextual_risks(risks, state["analysis_input"], state["retrieved_evidence"])
    query = build_treatment_query(state["analysis_input"], risks)
    retriever = retriever if retriever is not None else NistVectorRetriever()
    diagnostics = None
    if isinstance(retriever, NistVectorRetriever):
        diagnostics = {}
        evidence = retriever.retrieve(query, candidate_k=15, top_k=5, diagnostics=diagnostics)
    else:
        evidence = retriever.retrieve(query, candidate_k=15, top_k=5)
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise ValueError("Duplicate treatment NIST evidence IDs are not allowed.")
    return {
        "treatment_query": query,
        "treatment_evidence": evidence,
        "treatment_records": {item.evidence_id: item.resolve_record() for item in evidence},
        "treatment_diagnostics": diagnostics,
        "status": "treatment_retrieved" if evidence else "no_treatment_evidence",
        "status_message": "Treatment evidence retrieved." if evidence else
            "No treatment evidence was retrieved; requirement generation was skipped.",
    }


def derive_ai_requirements(state: US4AIState, *, client=None) -> dict:
    """Derive requirements from scenario, inferred risks, and original evidence."""
    risks = state["contextual_risks"]
    if not risks:
        raise ValueError("Requirement derivation requires contextual risks.")
    validate_contextual_risks(risks, state["analysis_input"], state["retrieved_evidence"])
    if not state.get("treatment_evidence"):
        raise ValueError("Requirement derivation requires treatment evidence.")
    messages = build_requirement_messages(
        state["analysis_input"], state["treatment_evidence"], state["treatment_records"], risks,
    )
    client = client if client is not None else create_generation_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0, max_retries=1,
        response_model=AIRequirementsOutput, messages=messages,
    )
    requirements = AIRequirementsOutput.model_validate(response).ai_requirements
    validate_ai_requirements(requirements, risks, state["analysis_input"], state["retrieved_evidence"],
                             treatment_evidence=state["treatment_evidence"])
    return {
        "ai_requirements": requirements,
        "status": "complete" if requirements else "no_requirements",
        "status_message": "Requirements derived with validated references." if requirements else
            "Contextual risks were inferred, but no justified requirements were derived. Review is needed.",
    }
