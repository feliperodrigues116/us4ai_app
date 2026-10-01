"""Deterministic reference validation for generated analysis artifacts."""

import re

from src.nist_evidence import RetrievedNistEvidence
from src.schemas import AITask, AISpecificRequirement, ContextualAIRisk, US4AIAnalysisInput


def reference_ai_tasks(scenario: US4AIAnalysisInput) -> dict[str, AITask]:
    """Assign stable references within this analysis using original input order."""
    return {f"TASK-{index:02d}": task for index, task in enumerate(scenario.ai_tasks, start=1)}


def _unique(values: list[str], label: str) -> set[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} IDs are not allowed.")
    return set(values)


def _references(values: list[str], allowed: set[str], label: str) -> None:
    if not values:
        raise ValueError(f"{label} references must not be empty.")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} references are not allowed.")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} references: {', '.join(sorted(unknown))}")


def validate_contextual_risks(
    risks: list[ContextualAIRisk], scenario: US4AIAnalysisInput,
    evidence: list[RetrievedNistEvidence],
) -> None:
    """Reject duplicate risk IDs and unresolved task/evidence references."""
    _unique([risk.risk_id for risk in risks], "risk")
    evidence_ids = _unique([item.evidence_id for item in evidence], "evidence")
    task_ids = set(reference_ai_tasks(scenario))
    for risk in risks:
        _references(risk.ai_task_ids, task_ids, "AI task")
        _references(risk.evidence_ids, evidence_ids, "NIST evidence")


def validate_ai_requirements(
    requirements: list[AISpecificRequirement], risks: list[ContextualAIRisk],
    scenario: US4AIAnalysisInput, evidence: list[RetrievedNistEvidence],
    *, treatment_evidence: list[RetrievedNistEvidence] | None = None,
) -> None:
    """Validate requirements against the actual scenario, evidence, and risks."""
    validate_contextual_risks(risks, scenario, evidence)
    _unique([requirement.requirement_id for requirement in requirements], "requirement")
    risk_ids = {risk.risk_id for risk in risks}
    task_ids = set(reference_ai_tasks(scenario))
    # Omitted treatment evidence supports validation of legacy single-stage artifacts.
    allowed = evidence if treatment_evidence is None else treatment_evidence
    evidence_ids = _unique([item.evidence_id for item in allowed], "treatment evidence")
    for requirement in requirements:
        _references(requirement.risk_ids, risk_ids, "contextual risk")
        _references(requirement.ai_task_ids, task_ids, "AI task")
        _references(requirement.evidence_ids, evidence_ids, "NIST evidence")

        addressed_tasks = {
            task for risk in risks if risk.risk_id in requirement.risk_ids
            for task in risk.ai_task_ids
        }
        if not set(requirement.ai_task_ids).issubset(addressed_tasks):
            raise ValueError("Requirement AI tasks must be associated with its addressed contextual risks.")
        if re.match(
            r"^\s*(?:the\s+)?(?:organization|organisation|project team|stakeholders|management|developers)\b",
            requirement.statement, re.IGNORECASE,
        ):
            raise ValueError("Requirement statement must specify a system-level control, not an organizational recommendation.")
