"""Prepare authoritative NIST documents without writing a vector index."""

from langchain_core.documents import Document

from src.nist_playbook import load_nist_playbook


def prepare_nist_documents() -> list[Document]:
    """Return validated Playbook documents with stable original-title identities."""
    return [
        Document(
            id=record.title,
            page_content=record.retrieval_text(),
            metadata={
                "title": record.title,
                "type": record.type,
                "category": record.category,
            },
        )
        for record in load_nist_playbook()
    ]
