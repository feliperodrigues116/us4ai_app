"""Deterministic scenario queries and NIST vector retrieval with reranking."""

import math

from src.config import NIST_MODEL_THREADS, NIST_RERANKER_MODEL
from src.nist_evidence import RetrievedNistEvidence
from src.nist_index import open_nist_store, validate_nist_index
from src.nist_playbook import load_nist_playbook
from src.schemas import US4AIAnalysisInput


def build_retrieval_query(scenario: US4AIAnalysisInput) -> str:
    """Render the validated input in its original order without LLM rewriting."""
    sections = [
        f"System Purpose:\n{scenario.system_purpose}",
        f"User Story:\n{scenario.user_story}",
    ]
    if scenario.acceptance_criteria:
        sections.append("Acceptance Criteria:\n" + "\n".join(
            f"- {criterion}" for criterion in scenario.acceptance_criteria
        ))
    sections.append("AI Tasks:\n" + "\n".join(
        f"- Category: {task.category} | Task: {task.task}" for task in scenario.ai_tasks
    ))
    return "\n\n".join(sections)


class NistVectorRetriever:
    """Retrieve vector candidates, rerank their text, and retain source evidence."""

    def __init__(self, *, vector_store=None, reranker=None):
        self.vector_store = vector_store if vector_store is not None else open_nist_store()
        validate_nist_index(self.vector_store)
        self.records = {record.title: record for record in load_nist_playbook()}
        if reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            reranker = TextCrossEncoder(model_name=NIST_RERANKER_MODEL, threads=NIST_MODEL_THREADS)
        self.reranker = reranker

    def retrieve(self, scenario: US4AIAnalysisInput | str, *, top_k: int = 5,
                 candidate_k: int = 15) -> list[RetrievedNistEvidence]:
        """Return reranked evidence with raw vector distances and encoder scores."""
        if not 1 <= top_k <= candidate_k <= len(self.records):
            raise ValueError("Require 1 <= top_k <= candidate_k <= 72.")
        query = build_retrieval_query(scenario) if isinstance(scenario, US4AIAnalysisInput) else scenario
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The retrieval query must be a non-empty string.")
        candidates = self.vector_store.similarity_search_with_score(query, k=candidate_k)
        if not candidates:
            return []
        seen = set()
        for document, _ in candidates:
            identity = document.id
            if identity not in self.records or identity in seen:
                raise ValueError(f"Invalid or duplicate NIST candidate identity: {identity}")
            seen.add(identity)
            record = self.records[identity]
            if document.page_content != record.retrieval_text() or document.metadata != {
                "title": record.title, "type": record.type, "category": record.category,
            }:
                raise ValueError(f"NIST candidate content mismatch: {identity}")
        scores = list(self.reranker.rerank(query, [doc.page_content for doc, _ in candidates]))
        if len(scores) != len(candidates) or any(not math.isfinite(float(score)) for score in scores):
            raise ValueError("Reranker must return one finite score per candidate.")
        evidence = [
            RetrievedNistEvidence(
                evidence_id=doc.id,
                type=doc.metadata["type"],
                category=doc.metadata["category"],
                retrieval_text=doc.page_content,
                vector_distance=float(distance),
                reranking_score=float(score),
            )
            for (doc, distance), score in zip(candidates, scores, strict=True)
        ]
        return sorted(evidence, key=lambda item: (-item.reranking_score, item.evidence_id))[:top_k]
