"""Offline tests for conservative generation, references, and orchestration."""

import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from src.config import OPENAI_MODEL
from src.graph.nodes import derive_ai_requirements, infer_contextual_risks, retrieve_nist_evidence
from src.graph.prompts import build_requirement_messages, build_risk_messages
from src.graph.workflow import build_us4ai_graph
from src.nist_evidence import RetrievedNistEvidence
from src.nist_playbook import PLAYBOOK_PATH, load_nist_playbook
from src.schemas import (
    AIRequirementsOutput, AISpecificRequirement, ContextualAIRisk,
    ContextualRisksOutput, US4AIAnalysisInput,
)
from src.traceability import reference_ai_tasks, validate_ai_requirements, validate_contextual_risks


def scenario():
    return US4AIAnalysisInput(
        system_purpose="Assist billing support while protecting customer information.",
        user_story="As an agent, I want draft answers that I can review before sending.",
        acceptance_criteria=["Require agent approval.", "Do not expose another customer's information."],
        ai_tasks=[
            {"category": "Generation", "task": "Draft billing responses"},
            {"category": "Classification", "task": "Route billing requests"},
        ],
    )


def evidence():
    record = load_nist_playbook()[0]
    return RetrievedNistEvidence(
        evidence_id=record.title, type=record.type, category=record.category,
        retrieval_text=record.retrieval_text(), vector_distance=0.2, reranking_score=0.8,
    )


def risk():
    return ContextualAIRisk(
        risk_id="R-01", title="Disclosure of customer information",
        description="Draft answers may expose customer information without appropriate review.",
        ai_task_ids=["TASK-01"], evidence_ids=["GOVERN 1.1"],
    )


def requirement():
    return AISpecificRequirement(
        requirement_id="REQ-01",
        statement="The system shall require agent approval before sending AI-generated billing responses.",
        risk_ids=["R-01"], ai_task_ids=["TASK-01"], evidence_ids=["GOVERN 1.1"],
        rationale="The scenario involves customer information; documenting applicable obligations supports managing the inferred disclosure risk.",
    )


def state():
    item = evidence()
    return {
        "analysis_input": scenario(), "retrieved_evidence": [item],
        "nist_records": {item.evidence_id: item.resolve_record()},
        "contextual_risks": [risk()], "ai_requirements": [],
    }


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.state = state()

    def test_risk_node_receives_complete_input_and_substantive_evidence(self):
        self.client.chat.completions.create.return_value = ContextualRisksOutput(contextual_risks=[risk()])
        result = infer_contextual_risks(self.state, client=self.client)
        self.assertEqual(result["contextual_risks"], [risk()])
        arguments = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(arguments["model"], OPENAI_MODEL)
        self.assertEqual(arguments["temperature"], 0)
        self.assertIs(arguments["response_model"], ContextualRisksOutput)
        context = json.loads(arguments["messages"][1]["content"])
        self.assertEqual(context["scenario"], scenario().model_dump())
        self.assertEqual(context["ai_task_references"]["TASK-02"], scenario().ai_tasks[1].model_dump())
        source = context["nist_source_evidence"][0]
        record = self.state["nist_records"]["GOVERN 1.1"]
        for field in ("title", "type", "category", "description", "section_about", "section_actions"):
            self.assertEqual(source[field], getattr(record, field))
        self.assertIn("Organizations can document", source["documentation_prompts"])
        self.assertNotIn("AI Transparency Resources", source["documentation_prompts"])
        self.assertNotIn("section_ref", source)
        self.assertNotIn("retrieval_text", source)
        self.assertIn("contextual inferences", arguments["messages"][0]["content"])

    def test_requirement_node_receives_scenario_risks_and_evidence(self):
        self.client.chat.completions.create.return_value = AIRequirementsOutput(ai_requirements=[requirement()])
        result = derive_ai_requirements(self.state, client=self.client)
        self.assertEqual(result["ai_requirements"], [requirement()])
        messages = self.client.chat.completions.create.call_args.kwargs["messages"]
        context = json.loads(messages[1]["content"])
        self.assertEqual(context["scenario"], scenario().model_dump())
        self.assertEqual(context["contextual_risk_inferences"], [risk().model_dump()])
        self.assertTrue(context["nist_source_evidence"][0]["section_actions"])
        self.assertIn("system-level control", messages[0]["content"])
        self.assertIn("Do not invent technologies", messages[0]["content"])

    def test_risk_unknown_references_and_duplicate_ids_fail_after_generation(self):
        for field, value, message in (
            ("evidence_ids", ["UNKNOWN"], "Unknown NIST evidence"),
            ("ai_task_ids", ["TASK-99"], "Unknown AI task"),
        ):
            with self.subTest(field=field):
                changed = {**risk().model_dump(), field: value}
                self.client.chat.completions.create.return_value = {"contextual_risks": [changed]}
                with self.assertRaisesRegex(ValueError, message):
                    infer_contextual_risks(self.state, client=self.client)
        self.client.chat.completions.create.return_value = ContextualRisksOutput(contextual_risks=[risk(), risk()])
        with self.assertRaisesRegex(ValueError, "Duplicate risk"):
            infer_contextual_risks(self.state, client=self.client)

    def test_requirement_unknown_references_and_duplicate_ids_fail_after_generation(self):
        for field, value, message in (
            ("risk_ids", ["R-99"], "Unknown contextual risk"),
            ("evidence_ids", ["UNKNOWN"], "Unknown NIST evidence"),
            ("ai_task_ids", ["TASK-99"], "Unknown AI task"),
        ):
            with self.subTest(field=field):
                changed = {**requirement().model_dump(), field: value}
                self.client.chat.completions.create.return_value = {"ai_requirements": [changed]}
                with self.assertRaisesRegex(ValueError, message):
                    derive_ai_requirements(self.state, client=self.client)
        self.client.chat.completions.create.return_value = AIRequirementsOutput(ai_requirements=[requirement(), requirement()])
        with self.assertRaisesRegex(ValueError, "Duplicate requirement"):
            derive_ai_requirements(self.state, client=self.client)

    def test_missing_evidence_and_task_references_rejected_for_risks(self):
        for field in ("evidence_ids", "ai_task_ids"):
            with self.subTest(field=field):
                self.client.chat.completions.create.return_value = {"contextual_risks": [{**risk().model_dump(), field: []}]}
                with self.assertRaises(ValidationError):
                    infer_contextual_risks(self.state, client=self.client)

    def test_missing_risk_evidence_and_task_references_rejected_for_requirements(self):
        for field in ("risk_ids", "evidence_ids", "ai_task_ids"):
            with self.subTest(field=field):
                self.client.chat.completions.create.return_value = {"ai_requirements": [{**requirement().model_dump(), field: []}]}
                with self.assertRaises(ValidationError):
                    derive_ai_requirements(self.state, client=self.client)

    def test_no_generation_without_required_artifacts(self):
        self.state["retrieved_evidence"] = []
        with self.assertRaisesRegex(ValueError, "requires retrieved"):
            infer_contextual_risks(self.state, client=self.client)
        self.state["contextual_risks"] = []
        with self.assertRaisesRegex(ValueError, "requires contextual"):
            derive_ai_requirements(self.state, client=self.client)
        self.client.chat.completions.create.assert_not_called()

    def test_prompts_are_deterministic_and_mismatched_source_is_rejected(self):
        args = (scenario(), self.state["retrieved_evidence"], self.state["nist_records"])
        self.assertEqual(build_risk_messages(*args), build_risk_messages(*args))
        self.assertEqual(build_requirement_messages(*args, [risk()]), build_requirement_messages(*args, [risk()]))
        self.state["nist_records"]["GOVERN 1.1"] = self.state["nist_records"]["GOVERN 1.1"].model_copy(update={"section_actions": "Changed source"})
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_risk_messages(*args)


class TraceabilityTests(unittest.TestCase):
    def test_valid_traceability_chain_and_original_task_order(self):
        data = scenario()
        tasks = reference_ai_tasks(data)
        self.assertEqual(list(tasks), ["TASK-01", "TASK-02"])
        self.assertEqual(list(tasks.values()), data.ai_tasks)
        validate_contextual_risks([risk()], data, [evidence()])
        validate_ai_requirements([requirement()], [risk()], data, [evidence()])
        req = requirement()
        referenced_risk = {risk().risk_id: risk()}[req.risk_ids[0]]
        self.assertEqual(tasks[referenced_risk.ai_task_ids[0]], data.ai_tasks[0])
        self.assertEqual(referenced_risk.evidence_ids, req.evidence_ids)
        self.assertEqual(evidence().resolve_record().title, req.evidence_ids[0])

    def test_deterministic_validation_rejects_empty_references_even_if_schema_bypassed(self):
        for field in ("ai_task_ids", "evidence_ids"):
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                validate_contextual_risks([risk().model_copy(update={field: []})], scenario(), [evidence()])
        for field in ("risk_ids", "ai_task_ids", "evidence_ids"):
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                validate_ai_requirements([requirement().model_copy(update={field: []})], [risk()], scenario(), [evidence()])

    def test_blank_generated_content_and_malformed_ids_are_rejected(self):
        for model, data, fields in (
            (ContextualAIRisk, risk().model_dump(), ["risk_id", "title", "description"]),
            (AISpecificRequirement, requirement().model_dump(), ["requirement_id", "statement", "rationale"]),
        ):
            for field in fields:
                with self.subTest(field=field):
                    with self.assertRaises(ValidationError):
                        model.model_validate({**data, field: " "})


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.retriever.retrieve.return_value = [evidence()]
        self.client = Mock()
        self.client.chat.completions.create.side_effect = [
            ContextualRisksOutput(contextual_risks=[risk()]),
            AIRequirementsOutput(ai_requirements=[requirement()]),
        ]

    def run_graph(self):
        with patch("src.graph.nodes.create_generation_client", side_effect=AssertionError("No real API client in unit tests.")):
            graph = build_us4ai_graph(retriever=self.retriever, client=self.client)
            return graph.invoke({"analysis_input": scenario()})

    def test_graph_retrieves_first_and_preserves_all_typed_artifacts(self):
        events = []
        self.retriever.retrieve.side_effect = lambda data: events.append("retrieve") or [evidence()]
        outputs = iter([ContextualRisksOutput(contextual_risks=[risk()]), AIRequirementsOutput(ai_requirements=[requirement()])])
        def generate(**kwargs):
            events.append(kwargs["response_model"].__name__)
            return next(outputs)
        self.client.chat.completions.create.side_effect = generate
        result = self.run_graph()
        self.assertEqual(events, ["retrieve", "ContextualRisksOutput", "AIRequirementsOutput"])
        self.assertEqual(result["analysis_input"], scenario())
        self.assertEqual(result["retrieved_evidence"], [evidence()])
        self.assertEqual(result["contextual_risks"], [risk()])
        self.assertEqual(result["ai_requirements"], [requirement()])
        self.assertEqual(result["nist_records"]["GOVERN 1.1"], evidence().resolve_record())
        self.assertEqual(result["ai_task_references"], reference_ai_tasks(scenario()))
        self.assertEqual(result["status"], "complete")
        self.retriever.retrieve.assert_called_once_with(scenario())

    def test_no_evidence_skips_both_generation_stages(self):
        self.retriever.retrieve.return_value = []
        result = self.run_graph()
        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(result["contextual_risks"], [])
        self.assertEqual(result["ai_requirements"], [])
        self.client.chat.completions.create.assert_not_called()

    def test_no_supported_risks_is_not_a_risk_free_conclusion(self):
        self.client.chat.completions.create.side_effect = [ContextualRisksOutput(contextual_risks=[])]
        result = self.run_graph()
        self.assertEqual(result["status"], "no_supported_risks")
        self.assertIn("does not mean the system is risk-free", result["status_message"])
        self.assertEqual(result["retrieved_evidence"], [evidence()])
        self.assertEqual(self.client.chat.completions.create.call_count, 1)

    def test_no_requirements_preserves_risks_for_review(self):
        self.client.chat.completions.create.side_effect = [ContextualRisksOutput(contextual_risks=[risk()]), AIRequirementsOutput(ai_requirements=[])]
        result = self.run_graph()
        self.assertEqual(result["status"], "no_requirements")
        self.assertEqual(result["contextual_risks"], [risk()])
        self.assertEqual(result["ai_requirements"], [])

    def test_invalid_output_fails_instead_of_being_silently_repaired(self):
        self.client.chat.completions.create.side_effect = [ContextualRisksOutput(contextual_risks=[risk().model_copy(update={"evidence_ids": ["UNKNOWN"]})])]
        with self.assertRaisesRegex(ValueError, "Unknown NIST evidence"):
            self.run_graph()
        self.assertEqual(self.client.chat.completions.create.call_count, 1)

    def test_complete_input_required_before_retrieval(self):
        with self.assertRaises(ValidationError):
            retrieve_nist_evidence({"analysis_input": {"user_story": "Incomplete"}}, retriever=self.retriever)
        self.retriever.retrieve.assert_not_called()

    def test_source_safety_and_no_legacy_graph_dependencies(self):
        before = hashlib.sha256(PLAYBOOK_PATH.read_bytes()).digest()
        self.run_graph()
        self.assertEqual(hashlib.sha256(PLAYBOOK_PATH.read_bytes()).digest(), before)
        graph_dir = Path(__file__).resolve().parents[1] / "src" / "graph"
        for path in graph_dir.glob("*.py"):
            text = path.read_text()
            for legacy in ("HybridRetriever", "OWASP", "Design Patterns", "detect_ai", "knowledge_item_id"):
                self.assertNotIn(legacy, text)
