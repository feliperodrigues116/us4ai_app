"""Verify NIST document preparation without vector writes or source changes."""

import builtins
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ingestion import prepare_nist_documents
from src.nist_playbook import PLAYBOOK_PATH, load_nist_playbook


class NistDocumentPreparationTests(unittest.TestCase):
    def test_document_count_and_stable_unique_identities(self):
        documents = prepare_nist_documents()
        titles = [record.title for record in load_nist_playbook()]
        self.assertEqual(len(documents), 72)
        self.assertEqual([document.id for document in documents], titles)
        self.assertEqual(len({document.id for document in documents}), 72)
        self.assertEqual([document.id for document in prepare_nist_documents()], titles)

    def test_content_and_metadata_match_original_records(self):
        for document, record in zip(prepare_nist_documents(), load_nist_playbook(), strict=True):
            with self.subTest(title=record.title):
                self.assertEqual(document.page_content, record.retrieval_text())
                self.assertEqual(document.metadata, {
                    "title": record.title,
                    "type": record.type,
                    "category": record.category,
                })
                self.assertNotIn("Problem:", document.page_content)
                self.assertNotIn("Control:", document.page_content)

    def test_only_authoritative_loader_is_used(self):
        with patch("src.ingestion.load_nist_playbook", wraps=load_nist_playbook) as loader:
            documents = prepare_nist_documents()
        loader.assert_called_once_with()
        self.assertEqual(len(documents), 72)

    def test_other_catalogs_and_working_directory_do_not_affect_preparation(self):
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            catalogs = Path(directory) / "catalogs"
            catalogs.mkdir()
            (catalogs / "unrelated.json").write_text("not valid JSON", encoding="utf-8")
            try:
                os.chdir(directory)
                self.assertEqual(len(prepare_nist_documents()), 72)
                self.assertFalse((Path(directory) / "data").exists())
            finally:
                os.chdir(previous_directory)

    def test_preparation_has_no_vector_dependencies_or_directory_writes(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"chromadb", "langchain_chroma", "fastembed"}:
                raise AssertionError("Document preparation must not import vector runtime dependencies.")
            return original_import(name, *args, **kwargs)

        index_path = PLAYBOOK_PATH.parents[1] / "data" / "chroma_nist"
        existed_before = index_path.exists()
        with patch("builtins.__import__", side_effect=guarded_import), \
             patch("os.mkdir", side_effect=AssertionError("Preparation must not create directories.")):
            self.assertEqual(len(prepare_nist_documents()), 72)
        self.assertEqual(index_path.exists(), existed_before)

    def test_original_json_is_unchanged(self):
        before = hashlib.sha256(PLAYBOOK_PATH.read_bytes()).digest()
        prepare_nist_documents()
        self.assertEqual(hashlib.sha256(PLAYBOOK_PATH.read_bytes()).digest(), before)


if __name__ == "__main__":
    unittest.main()
