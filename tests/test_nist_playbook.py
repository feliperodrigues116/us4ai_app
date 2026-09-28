"""Focused tests for Playbook validation, preservation, and retrieval text."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from src.nist_playbook import PLAYBOOK_PATH, NistPlaybookRecord, load_nist_playbook


class NistPlaybookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_bytes = PLAYBOOK_PATH.read_bytes()
        cls.raw_records = json.loads(cls.source_bytes)

    def test_complete_dataset_and_functions(self):
        records = load_nist_playbook()
        self.assertEqual(len(records), 72)
        self.assertEqual(len({record.title for record in records}), 72)
        self.assertEqual({record.type for record in records}, {"Govern", "Map", "Measure", "Manage"})

    def test_all_original_fields_and_order_are_preserved(self):
        records = load_nist_playbook()
        self.assertEqual([record.model_dump(by_alias=True) for record in records], self.raw_records)
        self.assertEqual(records[0].title, "GOVERN 1.1")
        self.assertEqual(
            records[0].description,
            "Legal and regulatory requirements involving AI are understood, managed, and documented.",
        )
        by_title = {record.title: record for record in records}
        self.assertIn("\n-Establish", by_title["MANAGE 2.2"].section_actions)
        self.assertEqual(by_title["MAP 3.2"].ai_actors.count("AI Design"), 2)
        self.assertEqual(
            [record.title for record in records if not record.ai_actors],
            [f"MAP 1.{number}" for number in range(1, 7)] + ["MAP 2.1", "MAP 2.2"],
        )
        self.assertTrue(all(isinstance(record.topic, list) for record in records))

    def test_default_path_is_independent_of_working_directory(self):
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(len(load_nist_playbook()), 72)
            finally:
                os.chdir(previous_directory)

    def test_retrieval_text_selects_only_requested_fields(self):
        data = copy.deepcopy(self.raw_records[0])
        data.update(
            title="title sentinel",
            description="description sentinel",
            section_about="about sentinel\nwith source spacing  ",
            section_actions="-actions sentinel\n\t- nested sentinel",
            section_doc="documentation exclusion sentinel",
            section_ref="references exclusion sentinel",
        )
        data["AI Actors"] = ["actor exclusion sentinel"]
        data["Topic"] = ["first topic", "second topic"]
        record = NistPlaybookRecord.model_validate(data)
        expected = (
            "Title: title sentinel\n\n"
            "Description: description sentinel\n\n"
            "About: about sentinel\nwith source spacing  \n\n"
            "Suggested Actions: -actions sentinel\n\t- nested sentinel\n\n"
            "Topics: first topic, second topic"
        )
        self.assertEqual(record.retrieval_text(), expected)
        self.assertEqual(record.retrieval_text(), record.retrieval_text())
        for excluded in (record.section_doc, record.section_ref, record.ai_actors[0]):
            self.assertNotIn(excluded, record.retrieval_text())
        self.assertEqual(record.model_dump(by_alias=True), data)

    def test_python_names_and_original_aliases(self):
        data = copy.deepcopy(self.raw_records[0])
        data["ai_actors"] = data.pop("AI Actors")
        data["topic"] = data.pop("Topic")
        record = NistPlaybookRecord.model_validate(data)
        self.assertEqual(record.model_dump(by_alias=True), self.raw_records[0])

    def test_required_fields_reject_empty_and_missing_values(self):
        for field in self.raw_records[0]:
            with self.subTest(field=field, case="missing"):
                data = copy.deepcopy(self.raw_records[0])
                del data[field]
                with self.assertRaises(ValidationError):
                    NistPlaybookRecord.model_validate(data)
        for field in ("type", "title", "category", "description", "section_about",
                      "section_actions", "section_doc", "section_ref", "Topic"):
            values = ([], None) if field == "Topic" else ("", " \n\t", None)
            for value in values:
                with self.subTest(field=field, value=value):
                    data = copy.deepcopy(self.raw_records[0])
                    data[field] = value
                    with self.assertRaises(ValidationError):
                        NistPlaybookRecord.model_validate(data)

    def test_invalid_types_and_function_are_rejected(self):
        for field, value in (("type", "GOVERN"), ("description", 123),
                             ("Topic", "Governance"), ("Topic", [123]),
                             ("AI Actors", "TEVV"), ("AI Actors", [123])):
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.raw_records[0])
                data[field] = value
                with self.assertRaises(ValidationError):
                    NistPlaybookRecord.model_validate(data)

    def test_loader_rejects_invalid_collections_and_validates_last_record(self):
        duplicate = copy.deepcopy(self.raw_records)
        duplicate[-1]["title"] = duplicate[0]["title"]
        invalid = copy.deepcopy(self.raw_records)
        invalid[-1]["description"] = ""
        cases = (
            ({}, ValueError, "JSON array"),
            (self.raw_records[:-1], ValueError, "exactly 72"),
            (self.raw_records + [self.raw_records[0]], ValueError, "exactly 72"),
            (duplicate, ValueError, "unique"),
            (invalid, ValidationError, "description"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for data, error, message in cases:
                with self.subTest(message=message, count=len(data)):
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaisesRegex(error, message):
                        load_nist_playbook(path)

    def test_source_json_is_not_modified(self):
        before = hashlib.sha256(self.source_bytes).digest()
        for record in load_nist_playbook():
            record.retrieval_text()
            record.model_dump(by_alias=True)
        self.assertEqual(hashlib.sha256(PLAYBOOK_PATH.read_bytes()).digest(), before)


if __name__ == "__main__":
    unittest.main()
