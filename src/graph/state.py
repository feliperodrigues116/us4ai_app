from typing import TypedDict, List, Dict, Any

class US4AIState(TypedDict):
    story_id: str
    description: str
    acceptance_criteria: List[str]
    is_ai_related: bool
    ai_reason: str
    retrieved_docs: List[Dict[str, Any]]
    identified_risks: List[Dict[str, Any]]
    derived_requirements: List[Dict[str, Any]]
