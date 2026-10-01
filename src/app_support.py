"""Input, result presentation, and export helpers for the Streamlit application."""

import csv
import hashlib
import io
import json
from pathlib import Path

from src.config import (
    CHROMA_PATH, NIST_COLLECTION_NAME, NIST_EMBEDDING_MODEL,
    NIST_EMBEDDING_MAX_LENGTH, NIST_DOCUMENT_EMBEDDING_TYPE,
    NIST_MODEL_THREADS, NIST_RERANKER_MODEL, OPENAI_MODEL,
)
from src.graph.workflow import build_us4ai_graph
from src.nist_index import build_nist_index, open_nist_store, validate_nist_index
from src.nist_playbook import PLAYBOOK_PATH
from src.retriever import NistVectorRetriever
from src.schemas import US4AIAnalysisInput
from src.failure_diagnostics import sanitize_error_message
from src.traceability import reference_ai_tasks, validate_ai_requirements


STATUS_MESSAGES = {
    "no_treatment_evidence": "No treatment evidence was retrieved. Risks are preserved; requirement generation was skipped.",
    "no_evidence": "No NIST evidence was retrieved for this analysis. Generation could not proceed.",
    "no_supported_risks": "No sufficiently supported contextual risks were produced from the retrieved evidence. This does not establish that the scenario is risk-free.",
    "no_requirements": "Contextual risks were inferred, but no justified requirements were produced. Review the preserved risks and evidence.",
    "complete": "Analysis completed with validated references. Researcher review is still required.",
    "failed": "Analysis stopped because a stage failed. Previously validated artifacts are preserved below; this is an incomplete result.",
}


def input_from_form(system_purpose: str, user_story: str, criteria_text: str,
                    task_rows: list[dict]) -> US4AIAnalysisInput:
    """Use the existing input contract; ignore only entirely blank editor rows."""
    tasks = []
    for row in task_rows:
        category, task = row.get("category") or "", row.get("task") or ""
        if category.strip() or task.strip():
            tasks.append({"category": category, "task": task})
    return US4AIAnalysisInput(
        system_purpose=system_purpose, user_story=user_story,
        acceptance_criteria=[line.strip() for line in criteria_text.splitlines() if line.strip()],
        ai_tasks=tasks,
    )


def initialize_analysis_graph():
    """Build only an absent index; validate existing data without rebuilding it."""
    if not Path(CHROMA_PATH).exists():
        build_nist_index()
    store = open_nist_store()
    validate_nist_index(store)
    return build_us4ai_graph(retriever=NistVectorRetriever(vector_store=store))


class AnalysisExecutionError(RuntimeError):
    """Expose a safe error category and validated partial state, not API details."""

    def __init__(self, partial_state, error_type):
        super().__init__("Analysis execution failed.")
        self.partial_state = partial_state
        self.error_type = error_type


def run_analysis(graph, scenario: US4AIAnalysisInput) -> dict:
    """Stream graph updates to preserve validated earlier stages if a later one fails."""
    result = {
        "analysis_input": scenario, "ai_task_references": reference_ai_tasks(scenario),
        "retrieved_evidence": [], "nist_records": {}, "contextual_risks": [],
        "ai_requirements": [],
    }
    active_stage = "risk_retrieval"
    next_stage = {
        "retrieve_nist_evidence": "risk_generation",
        "infer_contextual_risks": "treatment_retrieval",
        "retrieve_treatment_evidence": "requirement_generation",
    }
    try:
        for update in graph.stream({"analysis_input": scenario}, stream_mode="updates"):
            for node, values in update.items():
                if values:
                    result.update(values)
                active_stage = next_stage.get(node, active_stage)
    except Exception as error:
        result["failed_stage"] = active_stage
        result["error_type"] = type(error).__name__
        result["error_message"] = sanitize_error_message(error)
        result["status"] = "failed"
        result["status_message"] = STATUS_MESSAGES["failed"]
        raise AnalysisExecutionError(result, type(error).__name__) from None
    return result


def traceability_rows(result: dict) -> list[dict[str, str]]:
    """Render only explicit directed references; never infer task-evidence pairs."""
    scenario = result["analysis_input"]
    risks, requirements = result.get("contextual_risks", []), result.get("ai_requirements", [])
    validate_ai_requirements(requirements, risks, scenario, result.get("retrieved_evidence", []),
                             treatment_evidence=result.get("treatment_evidence"))
    rows = []

    def add(source_type, source_id, relation, target_type, target_id):
        row = {"source_type": source_type, "source_id": source_id, "relation": relation,
               "target_type": target_type, "target_id": target_id}
        if row not in rows:
            rows.append(row)

    for risk in risks:
        for task in risk.ai_task_ids:
            add("AI Task", task, "associated with", "Contextual AI Risk", risk.risk_id)
        for evidence in risk.evidence_ids:
            add("NIST Evidence", evidence, "supports inference", "Contextual AI Risk", risk.risk_id)
    for requirement in requirements:
        for risk in requirement.risk_ids:
            add("Contextual AI Risk", risk, "addressed by", "AI-specific Requirement", requirement.requirement_id)
        for task in requirement.ai_task_ids:
            add("AI Task", task, "associated with", "AI-specific Requirement", requirement.requirement_id)
        for evidence in requirement.evidence_ids:
            add("NIST Evidence", evidence, "supports derivation", "AI-specific Requirement", requirement.requirement_id)
    return rows


def analysis_export(result: dict) -> str:
    """Export an allowlisted research artifact without secrets or environment state."""
    payload = {
        "schema_version": "1.0",
        "analysis_input": result["analysis_input"].model_dump(),
        "ai_task_references": {key: task.model_dump() for key, task in reference_ai_tasks(result["analysis_input"]).items()},
        "retrieval_query": result.get("retrieval_query", ""),
        "retrieved_evidence": [item.model_dump() for item in result.get("retrieved_evidence", [])],
        "nist_records": {key: record.model_dump(by_alias=True) for key, record in result.get("nist_records", {}).items()},
        "contextual_risks": [risk.model_dump() for risk in result.get("contextual_risks", [])],
        "ai_requirements": [item.model_dump() for item in result.get("ai_requirements", [])],
        "traceability": traceability_rows(result),
        "status": result.get("status", "failed"),
        "status_message": STATUS_MESSAGES.get(result.get("status"), STATUS_MESSAGES["failed"]),
        "pipeline_metadata": {
            "source": "catalogs/nist_ai_rmf_playbook.json",
            "source_sha256": hashlib.sha256(PLAYBOOK_PATH.read_bytes()).hexdigest(),
            "collection": NIST_COLLECTION_NAME, "embedding_model": NIST_EMBEDDING_MODEL,
            "embedding_max_length": NIST_EMBEDDING_MAX_LENGTH,
            "document_embedding_type": NIST_DOCUMENT_EMBEDDING_TYPE,
            "reranker_model": NIST_RERANKER_MODEL, "model_threads": NIST_MODEL_THREADS,
            "generation_model": OPENAI_MODEL, "generation_temperature": 0,
            "candidate_k": 15, "top_k": 5,
        },
    }
    if "treatment_evidence" in result:
        # Additive fields keep risk evidence semantics and resolve treatment-only citations.
        payload["treatment_evidence"] = [item.model_dump() for item in result["treatment_evidence"]]
        payload["treatment_records"] = {key: record.model_dump(by_alias=True)
                                        for key, record in result.get("treatment_records", {}).items()}
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def traceability_csv(result: dict) -> str:
    """Export the same direct relationships shown in the table."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["source_type", "source_id", "relation", "target_type", "target_id"])
    writer.writeheader()
    writer.writerows(traceability_rows(result))
    return output.getvalue()
