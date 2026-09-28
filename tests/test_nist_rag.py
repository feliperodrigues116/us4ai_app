"""Offline contract tests for the NIST RAG foundation."""

import copy
import hashlib
import unittest
from unittest.mock import Mock, patch

from src.config import NIST_COLLECTION_NAME
from src.ingestion import prepare_nist_documents
from src.nist_evidence import RetrievedNistEvidence, resolve_nist_record
from src.nist_index import build_nist_index, corpus_metadata, open_nist_store, validate_nist_index
from src.nist_playbook import PLAYBOOK_PATH, load_nist_playbook
from src.retriever import NistVectorRetriever, build_retrieval_query
from src.schemas import US4AIAnalysisInput


def scenario(criteria=None):
    return US4AIAnalysisInput(
        system_purpose="Support customer-service agents.",
        user_story="As an agent, I want suggested answers to customer questions.",
        acceptance_criteria=criteria or [],
        ai_tasks=[
            {"category": "Language", "task": "Generate answers"},
            {"category": "Classification", "task": "Route requests"},
        ],
    )


def populated_store():
    documents = prepare_nist_documents()
    store = Mock()
    store._collection.name = NIST_COLLECTION_NAME
    store._collection.metadata = corpus_metadata(documents)
    store._collection.count.return_value = 72
    store.get.return_value = {
        "ids": [doc.id for doc in documents],
        "documents": [doc.page_content for doc in documents],
        "metadatas": [doc.metadata for doc in documents],
    }
    return store


class QueryTests(unittest.TestCase):
    def test_all_fields_and_tasks_are_preserved(self):
        data = scenario(["Require agent review.", "Protect personal information."])
        query = build_retrieval_query(data)
        self.assertEqual(query, (
            "System Purpose:\nSupport customer-service agents.\n\n"
            "User Story:\nAs an agent, I want suggested answers to customer questions.\n\n"
            "Acceptance Criteria:\n- Require agent review.\n- Protect personal information.\n\n"
            "AI Tasks:\n- Category: Language | Task: Generate answers\n"
            "- Category: Classification | Task: Route requests"
        ))
        self.assertEqual(build_retrieval_query(data), query)

    def test_empty_criteria_omit_only_that_section(self):
        query = build_retrieval_query(scenario())
        self.assertNotIn("Acceptance Criteria:", query)
        for label in ("System Purpose:", "User Story:", "AI Tasks:"):
            self.assertIn(label, query)


class IndexTests(unittest.TestCase):
    def test_complete_corpus_and_explicit_collection(self):
        store = populated_store()
        self.assertEqual(validate_nist_index(store), 72)
        self.assertEqual(len(set(store.get.return_value["ids"])), 72)
        self.assertEqual(set(store.get.return_value["ids"]), {r.title for r in load_nist_playbook()})

    def test_mismatches_fail_without_writing(self):
        for mismatch in ("count", "foreign", "duplicate", "collection", "text", "metadata", "fingerprint"):
            with self.subTest(mismatch=mismatch):
                store = populated_store()
                stored = store.get.return_value
                if mismatch == "count":
                    store._collection.count.return_value = 71
                elif mismatch == "foreign":
                    stored["ids"][0] = "FOREIGN 1"
                elif mismatch == "duplicate":
                    stored["ids"][0] = stored["ids"][1]
                elif mismatch == "collection":
                    store._collection.name = "default"
                elif mismatch == "text":
                    stored["documents"][0] = "Stale text"
                elif mismatch == "metadata":
                    stored["metadatas"][0] = {"title": "Wrong title"}
                else:
                    store._collection.metadata["corpus_sha256"] = "wrong"
                with patch("src.nist_index.open_nist_store", return_value=store):
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        build_nist_index()
                store.add_documents.assert_not_called()

    def test_existing_valid_index_is_not_rewritten(self):
        store = populated_store()
        with patch("src.nist_index.open_nist_store", return_value=store) as factory:
            self.assertEqual(build_nist_index(), 72)
            self.assertEqual(build_nist_index(), 72)
        factory.assert_called_with(create=True)
        store.add_documents.assert_not_called()

    def test_empty_index_receives_exact_authoritative_documents(self):
        store = populated_store()
        store._collection.count.side_effect = [0, 72]
        with patch("src.nist_index.open_nist_store", return_value=store):
            self.assertEqual(build_nist_index(), 72)
        store.add_documents.assert_called_once()
        arguments = store.add_documents.call_args.kwargs
        self.assertEqual(arguments["documents"], prepare_nist_documents())
        self.assertEqual(arguments["ids"], [r.title for r in load_nist_playbook()])

    def test_missing_index_fails_before_model_or_database_creation(self):
        with patch("src.nist_index.Path.is_file", return_value=False), \
             patch("src.nist_index.create_nist_embeddings") as embeddings:
            with self.assertRaisesRegex(ValueError, "does not exist"):
                open_nist_store()
        embeddings.assert_not_called()


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.store = populated_store()
        self.documents = prepare_nist_documents()[:3]
        self.store.similarity_search_with_score.return_value = list(zip(self.documents, [0.1, 0.2, 0.3]))
        self.reranker = Mock()
        self.reranker.rerank.return_value = [0.2, 0.9, 0.5]
        self.retriever = NistVectorRetriever(vector_store=self.store, reranker=self.reranker)

    def test_reranking_preserves_text_identity_and_both_scores(self):
        result = self.retriever.retrieve(scenario(), top_k=2, candidate_k=3)
        self.assertTrue(all(isinstance(item, RetrievedNistEvidence) for item in result))
        self.assertEqual([item.evidence_id for item in result], ["GOVERN 1.2", "GOVERN 1.3"])
        self.assertEqual([item.vector_distance for item in result], [0.2, 0.3])
        self.assertEqual([item.reranking_score for item in result], [0.9, 0.5])
        for item, document in zip(result, self.documents[1:], strict=True):
            self.assertEqual(item.retrieval_text, document.page_content)
            self.assertEqual(item.type, document.metadata["type"])
            self.assertEqual(item.category, document.metadata["category"])
            self.assertEqual(item.resolve_record().title, item.evidence_id)
            self.assertNotIn("knowledge_item_id", item.model_dump())
        query = build_retrieval_query(scenario())
        self.store.similarity_search_with_score.assert_called_once_with(query, k=3)
        self.reranker.rerank.assert_called_once_with(query, [doc.page_content for doc in self.documents])

    def test_query_string_and_deterministic_ties(self):
        self.reranker.rerank.return_value = [0.5, 0.5, 0.5]
        result = self.retriever.retrieve("Customer service", top_k=3, candidate_k=3)
        self.assertEqual([item.evidence_id for item in result], sorted(doc.id for doc in self.documents))

    def test_empty_candidates_do_not_call_reranker(self):
        self.store.similarity_search_with_score.return_value = []
        self.assertEqual(self.retriever.retrieve(scenario()), [])
        self.reranker.rerank.assert_not_called()

    def test_unknown_candidate_fails(self):
        self.documents[0].id = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "candidate identity"):
            self.retriever.retrieve(scenario())
        self.reranker.rerank.assert_not_called()

    def test_changed_candidate_text_fails(self):
        self.documents[0].page_content = "Altered text"
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            self.retriever.retrieve(scenario())

    def test_wrong_reranker_score_count_and_nonfinite_scores_fail(self):
        for scores in ([0.5], [0.5, float("nan"), 0.2]):
            self.reranker.rerank.return_value = scores
            with self.assertRaisesRegex(ValueError, "finite score"):
                self.retriever.retrieve(scenario())

    def test_invalid_query_and_limits_fail(self):
        for arguments in ({"top_k": 0}, {"top_k": 16, "candidate_k": 15}, {"candidate_k": 73}):
            with self.assertRaises(ValueError):
                self.retriever.retrieve(scenario(), **arguments)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.retriever.retrieve("  ")


class ResolutionAndSafetyTests(unittest.TestCase):
    def test_every_identity_resolves_to_exact_original_record(self):
        for record in load_nist_playbook():
            self.assertEqual(resolve_nist_record(record.title).model_dump(by_alias=True),
                             record.model_dump(by_alias=True))
        with self.assertRaisesRegex(ValueError, "Unknown NIST evidence ID"):
            resolve_nist_record("UNKNOWN")

    def test_offline_pipeline_preserves_source_and_uses_only_nist(self):
        before = hashlib.sha256(PLAYBOOK_PATH.read_bytes()).digest()
        store = populated_store()
        store.similarity_search_with_score.return_value = [(prepare_nist_documents()[0], 0.2)]
        reranker = Mock()
        reranker.rerank.return_value = [0.8]
        # Any accidental import of a model/API dependency fails in this offline path.
        with patch.dict("sys.modules", {"openai": None, "fastembed": None, "chromadb": None}):
            evidence = NistVectorRetriever(vector_store=store, reranker=reranker).retrieve(scenario())
            for item in evidence:
                self.assertIn(item.evidence_id, {r.title for r in load_nist_playbook()})
                item.resolve_record()
        self.assertEqual(hashlib.sha256(PLAYBOOK_PATH.read_bytes()).digest(), before)
