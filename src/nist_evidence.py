"""Retrieved source evidence, separate from any generated contextual risk."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.nist_playbook import NistPlaybookRecord, NonEmptyText, load_nist_playbook


def resolve_nist_record(evidence_id: str) -> NistPlaybookRecord:
    """Resolve a stable original title using only the authoritative loader."""
    for record in load_nist_playbook():
        if record.title == evidence_id:
            return record
    raise ValueError(f"Unknown NIST evidence ID: {evidence_id}")


class RetrievedNistEvidence(BaseModel):
    """Source content and scores; vector distance is lower-is-better.

    Reranking scores are raw cross-encoder scores, not probabilities.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    evidence_id: NonEmptyText
    type: Literal["Govern", "Map", "Measure", "Manage"]
    category: NonEmptyText
    retrieval_text: NonEmptyText
    vector_distance: float | None = None
    reranking_score: float | None = None

    def resolve_record(self) -> NistPlaybookRecord:
        """Return the complete unchanged authoritative source record."""
        return resolve_nist_record(self.evidence_id)
