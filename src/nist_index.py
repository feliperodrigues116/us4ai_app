"""Build and validate an isolated authoritative NIST vector corpus."""

import hashlib
import json
from pathlib import Path

from src.config import (
    CHROMA_PATH, NIST_COLLECTION_NAME, NIST_DOCUMENT_EMBEDDING_TYPE,
    NIST_EMBEDDING_MAX_LENGTH, NIST_EMBEDDING_MODEL, NIST_MODEL_THREADS,
)
from src.ingestion import prepare_nist_documents


def corpus_metadata(documents):
    """Fingerprint ordered documents and the shared embedding configuration."""
    payload = {
        "documents": [(doc.id, doc.page_content, doc.metadata) for doc in documents],
        "model": NIST_EMBEDDING_MODEL,
        "max_length": NIST_EMBEDDING_MAX_LENGTH,
        "document_embedding_type": NIST_DOCUMENT_EMBEDDING_TYPE,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"corpus_sha256": digest, "embedding_model": NIST_EMBEDDING_MODEL}


def create_nist_embeddings():
    """Construct the same FastEmbed configuration for indexing and queries."""
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    return FastEmbedEmbeddings(
        model_name=NIST_EMBEDDING_MODEL,
        max_length=NIST_EMBEDDING_MAX_LENGTH,
        doc_embed_type=NIST_DOCUMENT_EMBEDDING_TYPE,
        threads=NIST_MODEL_THREADS,
    )


def open_nist_store(*, create=False):
    """Open the explicit collection, creating it only for an indexing request."""
    if not create and not (Path(CHROMA_PATH) / "chroma.sqlite3").is_file():
        raise ValueError("The NIST index does not exist. Build it before retrieval.")

    # Match the existing application's SQLite compatibility support.
    import sqlite3
    import sys
    if sqlite3.sqlite_version_info < (3, 35, 0):
        import pysqlite3
        sys.modules["sqlite3"] = pysqlite3
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=NIST_COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=create_nist_embeddings(),
        collection_metadata=corpus_metadata(prepare_nist_documents()) if create else None,
        create_collection_if_not_exists=create,
    )


def validate_nist_index(store) -> int:
    """Reject wrong identities, stale text, metadata, or embedding provenance."""
    documents = prepare_nist_documents()
    expected = {doc.id: doc for doc in documents}
    collection = store._collection
    if collection.name != NIST_COLLECTION_NAME:
        raise ValueError("NIST index mismatch: unexpected collection name.")
    stored = store.get(include=["documents", "metadatas"])
    ids = stored["ids"]
    if collection.count() != len(expected) or len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError("NIST index mismatch: expected exactly 72 authoritative NIST titles.")
    if any((collection.metadata or {}).get(key) != value
           for key, value in corpus_metadata(documents).items()):
        raise ValueError("NIST index mismatch: corpus or embedding configuration changed.")
    texts, metadata = stored["documents"], stored["metadatas"]
    if texts is None or metadata is None or len(texts) != len(ids) or len(metadata) != len(ids):
        raise ValueError("NIST index mismatch: missing stored content or metadata.")
    for identity, text, fields in zip(ids, texts, metadata, strict=True):
        document = expected[identity]
        if text != document.page_content or fields != document.metadata:
            raise ValueError(f"NIST index mismatch: changed content or metadata for {identity}.")
    return len(ids)


def build_nist_index() -> int:
    """Create an empty index once; validate existing indexes without rewriting.

    A mismatched or partially populated index fails rather than being erased.
    """
    store = open_nist_store(create=True)
    if store._collection.count() == 0:
        documents = prepare_nist_documents()
        if store._collection.metadata != corpus_metadata(documents):
            raise ValueError("NIST index mismatch: empty collection has unexpected provenance.")
        store.add_documents(documents=documents, ids=[doc.id for doc in documents])
    return validate_nist_index(store)


if __name__ == "__main__":
    print(f"Validated {build_nist_index()} documents in {NIST_COLLECTION_NAME} at {CHROMA_PATH}.")
