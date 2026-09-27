from pydantic import BaseModel, Field
from typing import List

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
