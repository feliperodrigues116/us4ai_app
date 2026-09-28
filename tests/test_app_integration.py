"""Offline integration tests without remote calls or real model initialization."""

import ast
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from src.app_support import (
    AnalysisExecutionError, STATUS_MESSAGES, analysis_export, initialize_analysis_graph,
    input_from_form, run_analysis, traceability_csv, traceability_rows,
)
from src.graph.workflow import build_us4ai_graph
from src.nist_playbook import PLAYBOOK_PATH
from src.schemas import AIRequirementsOutput, ContextualRisksOutput
from tests.test_generative_core import evidence, requirement, risk, scenario


def result():
    retriever = Mock()
    retriever.retrieve.return_value = [evidence()]
    client = Mock()
    client.chat.completions.create.side_effect = [
        ContextualRisksOutput(contextual_risks=[risk()]),
        AIRequirementsOutput(ai_requirements=[requirement()]),
    ]
    return run_analysis(build_us4ai_graph(retriever=retriever, client=client), scenario())


class InputIntegrationTests(unittest.TestCase):
    def test_complete_input_uses_existing_model(self):
        expected = scenario()
        actual = input_from_form(expected.system_purpose, expected.user_story,
            "\n".join(expected.acceptance_criteria), [task.model_dump() for task in expected.ai_tasks])
        self.assertEqual(actual, expected)

    def test_empty_criteria_and_blank_editor_rows(self):
        actual = input_from_form("Purpose", "Story", " \n", [
            {"category": "", "task": None}, {"category": "Custom", "task": "Task"},
        ])
        self.assertEqual(actual.acceptance_criteria, [])
        self.assertEqual(len(actual.ai_tasks), 1)

    def test_missing_or_partial_input_is_rejected(self):
        for purpose, story, rows in (
            ("", "Story", [{"category": "C", "task": "T"}]),
            ("Purpose", "", [{"category": "C", "task": "T"}]),
            ("Purpose", "Story", []),
            ("Purpose", "Story", [{"category": "C", "task": ""}]),
        ):
            with self.assertRaises(ValidationError):
                input_from_form(purpose, story, "", rows)


class IndexStartupTests(unittest.TestCase):
    def test_missing_directory_builds_then_opens_and_validates(self):
        with patch("src.app_support.Path.exists", return_value=False), \
             patch("src.app_support.build_nist_index") as build, \
             patch("src.app_support.open_nist_store") as open_store, \
             patch("src.app_support.validate_nist_index") as validate, \
             patch("src.app_support.NistVectorRetriever") as retriever, \
             patch("src.app_support.build_us4ai_graph") as graph:
            self.assertIs(initialize_analysis_graph(), graph.return_value)
            build.assert_called_once_with()
            open_store.assert_called_once_with()
            validate.assert_called_once_with(open_store.return_value)
            retriever.assert_called_once_with(vector_store=open_store.return_value)

    def test_existing_invalid_index_is_not_rebuilt(self):
        with patch("src.app_support.Path.exists", return_value=True), \
             patch("src.app_support.build_nist_index") as build, \
             patch("src.app_support.open_nist_store") as open_store, \
             patch("src.app_support.validate_nist_index", side_effect=ValueError("Index mismatch")):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                initialize_analysis_graph()
            build.assert_not_called()
            open_store.assert_called_once_with()


class ResultIntegrationTests(unittest.TestCase):
    def test_traceability_contains_only_explicit_edges(self):
        data = result()
        rows = traceability_rows(data)
        self.assertEqual(len(rows), 5)
        self.assertIn({"source_type": "Contextual AI Risk", "source_id": "R-01",
            "relation": "addressed by", "target_type": "AI-specific Requirement", "target_id": "REQ-01"}, rows)
        self.assertFalse(any(row["source_type"] == "AI Task" and row["target_type"] == "NIST Evidence" for row in rows))
        self.assertFalse(any(row["source_id"] == "TASK-02" for row in rows))

    def test_invalid_references_cannot_be_exported(self):
        data = result()
        data["contextual_risks"] = [risk().model_copy(update={"evidence_ids": ["UNKNOWN"]})]
        with self.assertRaisesRegex(ValueError, "Unknown"):
            analysis_export(data)

    def test_json_export_preserves_artifacts_and_excludes_unapproved_fields(self):
        data = result()
        data["secret"] = "not-for-export"
        exported = json.loads(analysis_export(data))
        self.assertEqual(exported["analysis_input"], scenario().model_dump())
        self.assertEqual(exported["retrieved_evidence"], [evidence().model_dump()])
        self.assertEqual(exported["contextual_risks"], [risk().model_dump()])
        self.assertEqual(exported["ai_requirements"], [requirement().model_dump()])
        self.assertEqual(exported["nist_records"]["GOVERN 1.1"], evidence().resolve_record().model_dump(by_alias=True))
        self.assertEqual(exported["pipeline_metadata"]["generation_model"], "gpt-4o-mini")
        self.assertNotIn("not-for-export", analysis_export(data))
        self.assertNotIn("OPENAI_API_KEY", analysis_export(data))
        self.assertIn("R-01,addressed by", traceability_csv(data))

    def test_empty_statuses_preserve_available_artifacts(self):
        for status, risks in (("no_evidence", []), ("no_supported_risks", []), ("no_requirements", [risk()])):
            data = result()
            data.update(status=status, contextual_risks=risks, ai_requirements=[])
            if status == "no_evidence":
                data.update(retrieved_evidence=[], nist_records={})
            exported = json.loads(analysis_export(data))
            self.assertEqual(exported["status"], status)
            self.assertEqual(exported["contextual_risks"], [item.model_dump() for item in risks])
            self.assertEqual(exported["status_message"], STATUS_MESSAGES[status])

    def test_later_generation_failure_preserves_validated_risks(self):
        retriever = Mock()
        retriever.retrieve.return_value = [evidence()]
        client = Mock()
        client.chat.completions.create.side_effect = [ContextualRisksOutput(contextual_risks=[risk()]), RuntimeError("private API details")]
        graph = build_us4ai_graph(retriever=retriever, client=client)
        with self.assertRaises(AnalysisExecutionError) as caught:
            run_analysis(graph, scenario())
        partial = caught.exception.partial_state
        self.assertEqual(partial["contextual_risks"], [risk()])
        self.assertEqual(partial["status"], "failed")
        self.assertNotIn("private API details", analysis_export(partial))

    def test_source_and_runtime_safety(self):
        before = hashlib.sha256(PLAYBOOK_PATH.read_bytes()).digest()
        analysis_export(result())
        self.assertEqual(hashlib.sha256(PLAYBOOK_PATH.read_bytes()).digest(), before)
        root = Path(__file__).resolve().parents[1]
        for path in [root / "app.py", *(root / "src").rglob("*.py")]:
            text = path.read_text()
            ast.parse(text)
            for legacy in ("HybridRetriever", "detect_ai", "knowledge_item_id", "OWASP", "patterns_catalog"):
                self.assertNotIn(legacy, text)


class StreamlitIntegrationTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()

    def test_app_renders_form_and_reuses_cached_resources(self):
        from streamlit.testing.v1 import AppTest
        with patch("src.app_support.initialize_analysis_graph", return_value=Mock()) as initialize, \
             patch("src.config.OPENAI_API_KEY", "offline-test-placeholder"):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual([item.label for item in app.text_area], ["System Purpose", "User Story", "Acceptance Criteria (optional, one per line)"])
            app.run()
            initialize.assert_called_once()
            app.button(key="run_analysis").click().run()
            self.assertGreater(len(app.error), 0)
            self.assertEqual(len(app.exception), 0)

    def test_app_displays_and_exports_graph_results_without_real_services(self):
        from streamlit.testing.v1 import AppTest
        retriever, client = Mock(), Mock()
        retriever.retrieve.return_value = [evidence()]
        client.chat.completions.create.side_effect = [ContextualRisksOutput(contextual_risks=[risk()]), AIRequirementsOutput(ai_requirements=[requirement()])]
        graph = build_us4ai_graph(retriever=retriever, client=client)
        with patch("src.app_support.initialize_analysis_graph", return_value=graph), \
             patch("src.app_support.input_from_form", return_value=scenario()), \
             patch("src.config.OPENAI_API_KEY", "offline-test-placeholder"):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
            app.selectbox(key="task_group_0").select("Natural Language Processing").run()
            app.selectbox(key="task_value_0_Natural Language Processing").select("Text Generation").run()
            app.button(key="run_analysis").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 4)
            self.assertEqual(app.session_state["analysis_result"]["status"], "complete")
            app.run()
            self.assertEqual(client.chat.completions.create.call_count, 2)
