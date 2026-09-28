from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated, List

class KnowledgeSource(BaseModel):
    knowledge_item_id: str
    source_name: str
    source_section: str
    category: str

class IdentifiedRisk(BaseModel):
    risk_id: str
    risk_label: str
    description: str
    grounded_in: List[KnowledgeSource] = Field(default_factory=list)

class DerivedRequirement(BaseModel):
    requirement_id: str
    title: str
    statement: str = Field(description="Must start with 'The system shall...'")
    rationale: str
    category: str
    mitigates_risks: List[str] = Field(description="List of risk_ids this requirement mitigates")
    sources: List[KnowledgeSource] = Field(default_factory=list)

class OutputRequirements(BaseModel):
    story_id: str
    context_summary: str
    derived_ai_requirements: List[DerivedRequirement] = Field(default_factory=list)

class OutputRisks(BaseModel):
    story_id: str
    identified_risks: List[IdentifiedRisk] = Field(default_factory=list)


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
