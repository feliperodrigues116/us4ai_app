"""Scenario inputs and distinct contextual-risk and requirement contracts."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyInputText = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]


class AITask(BaseModel):
    """An AI task described structurally, without imposing a taxonomy."""

    model_config = ConfigDict(strict=True)

    category: NonEmptyInputText
    task: NonEmptyInputText


class US4AIAnalysisInput(BaseModel):
    """Input contract for a future Playbook-grounded analysis."""

    model_config = ConfigDict(strict=True)

    system_purpose: NonEmptyInputText
    user_story: NonEmptyInputText
    acceptance_criteria: list[str] = Field(default_factory=list)
    ai_tasks: list[AITask] = Field(min_length=1)


RiskId = Annotated[str, StringConstraints(pattern=r"^R-[0-9]{2,}$")]
RequirementId = Annotated[str, StringConstraints(pattern=r"^REQ-[0-9]{2,}$")]


class ContextualAIRisk(BaseModel):
    """A scenario-specific inference supported by NIST, not a NIST statement."""

    model_config = ConfigDict(extra="forbid")

    risk_id: RiskId
    title: NonEmptyInputText
    description: NonEmptyInputText
    ai_task_ids: list[NonEmptyInputText] = Field(min_length=1)
    evidence_ids: list[NonEmptyInputText] = Field(min_length=1)


class AISpecificRequirement(BaseModel):
    """An actionable contextual requirement addressing inferred AI risks."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: RequirementId
    statement: NonEmptyInputText
    risk_ids: list[RiskId] = Field(min_length=1)
    ai_task_ids: list[NonEmptyInputText] = Field(min_length=1)
    evidence_ids: list[NonEmptyInputText] = Field(min_length=1)
    rationale: NonEmptyInputText


class ContextualRisksOutput(BaseModel):
    """An empty list means no sufficiently supported risks were inferred."""

    model_config = ConfigDict(extra="forbid")
    contextual_risks: list[ContextualAIRisk]


class AIRequirementsOutput(BaseModel):
    """An empty list means no justified requirements were derived."""

    model_config = ConfigDict(extra="forbid")
    ai_requirements: list[AISpecificRequirement]
