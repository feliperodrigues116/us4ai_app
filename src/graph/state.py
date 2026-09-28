"""Typed artifacts retained throughout a NIST-grounded analysis."""

from typing import Literal, Required, TypedDict

from src.nist_evidence import RetrievedNistEvidence
from src.nist_playbook import NistPlaybookRecord
from src.schemas import AITask, AISpecificRequirement, ContextualAIRisk, US4AIAnalysisInput


class US4AIState(TypedDict, total=False):
    analysis_input: Required[US4AIAnalysisInput]
    ai_task_references: dict[str, AITask]
    retrieval_query: str
    retrieved_evidence: list[RetrievedNistEvidence]
    nist_records: dict[str, NistPlaybookRecord]
    contextual_risks: list[ContextualAIRisk]
    ai_requirements: list[AISpecificRequirement]
    status: Literal["evidence_retrieved", "risks_inferred", "complete", "no_evidence", "no_supported_risks", "no_requirements"]
    status_message: str
