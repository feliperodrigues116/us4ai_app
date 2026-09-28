"""Validate original NIST Playbook records and render retrieval text."""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


PLAYBOOK_PATH = Path(__file__).resolve().parents[1] / "catalogs" / "nist_ai_rmf_playbook.json"
EXPECTED_RECORD_COUNT = 72
NonEmptyText = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]


class NistPlaybookRecord(BaseModel):
    """An original record, including source text and original JSON aliases."""

    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)

    type: Literal["Govern", "Map", "Measure", "Manage"]
    title: NonEmptyText
    category: NonEmptyText
    description: NonEmptyText
    section_about: NonEmptyText
    section_actions: NonEmptyText
    section_doc: NonEmptyText
    section_ref: NonEmptyText
    ai_actors: list[str] = Field(alias="AI Actors")
    topic: list[str] = Field(alias="Topic", min_length=1)

    def retrieval_text(self) -> str:
        """Render selected fields without rewriting source text or topic order."""
        return "\n\n".join(
            (
                f"Title: {self.title}",
                f"Description: {self.description}",
                f"About: {self.section_about}",
                f"Suggested Actions: {self.section_actions}",
                f"Topics: {', '.join(self.topic)}",
            )
        )


def load_nist_playbook(path: str | Path = PLAYBOOK_PATH) -> list[NistPlaybookRecord]:
    """Load and validate 72 uniquely titled records, preserving source order."""
    with Path(path).open("r", encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, list):
        raise ValueError("The NIST Playbook must be a JSON array.")
    if len(data) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"The NIST Playbook must contain exactly {EXPECTED_RECORD_COUNT} records; "
            f"found {len(data)}."
        )
    records = [NistPlaybookRecord.model_validate(item) for item in data]
    if len({record.title for record in records}) != len(records):
        raise ValueError("NIST Playbook titles must be unique.")
    return records
