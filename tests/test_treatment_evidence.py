"""Offline coverage for independent treatment evidence within the final Top-K."""

import json
import unittest
from unittest.mock import Mock

from src.app_support import analysis_export, run_analysis, traceability_csv, traceability_rows
from src.graph.prompts import REQUIREMENT_INSTRUCTIONS
from src.graph.workflow import build_us4ai_graph
from src.ingestion import prepare_nist_documents
from src.retrieval_diagnostics import diagnostic_report
from src.retriever import NistVectorRetriever
from src.schemas import AIRequirementsOutput, ContextualRisksOutput
from src.traceability import validate_ai_requirements, validate_contextual_risks
from tests.test_generative_core import requirement, risk, scenario
from tests.test_nist_rag import populated_store


class TreatmentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.docs = prepare_nist_documents()[:15]
        self.store = populated_store()
        self.store.similarity_search_with_score.return_value = [(doc, i / 100) for i, doc in enumerate(self.docs)]
        self.reranker = Mock()
        self.reranker.rerank.return_value = list(range(15, 0, -1))
        self.retriever = NistVectorRetriever(vector_store=self.store, reranker=self.reranker)
        self.ids = [doc.id for doc in self.docs]
        # Structural fixtures test reference independence, not semantic endorsement.
        self.risk = risk().model_copy(update={'evidence_ids': [self.ids[0], self.ids[2]]})
        self.requirements = [
            requirement().model_copy(update={'evidence_ids': [self.ids[1]]}),
            requirement().model_copy(update={
                'requirement_id': 'REQ-02', 'evidence_ids': [self.ids[1], self.ids[2]],
                'statement': 'The draft store shall restrict draft access to the assigned agent.',
            }),
        ]
        self.client = Mock()
        self.client.chat.completions.create.side_effect = [
            ContextualRisksOutput(contextual_risks=[self.risk]),
            AIRequirementsOutput(ai_requirements=self.requirements),
        ]

    def execute(self):
        return run_analysis(build_us4ai_graph(retriever=self.retriever, client=self.client), scenario())

    def test_complete_top_k_and_substantive_fields_reach_treatment_generation(self):
        result = self.execute()
        calls = self.client.chat.completions.create.call_args_list
        context = json.loads(calls[1].kwargs['messages'][1]['content'])
        self.assertEqual(context['scenario'], scenario().model_dump())
        self.assertEqual(context['contextual_risk_inferences'], [self.risk.model_dump()])
        self.assertEqual([s['evidence_id'] for s in context['nist_source_evidence']], self.ids[:5])
        for source in context['nist_source_evidence']:
            record = self.retriever.records[source['evidence_id']]
            for field in ('title', 'type', 'category', 'description', 'section_about', 'section_actions'):
                self.assertEqual(source[field], getattr(record, field))
            self.assertEqual(source['topics'], record.topic)
        risk_context = json.loads(calls[0].kwargs['messages'][1]['content'])
        self.assertTrue(all('topics' not in s for s in risk_context['nist_source_evidence']))
        self.assertEqual(result['ai_requirements'], self.requirements)
        self.assertEqual(self.store.similarity_search_with_score.call_count, 2)
        self.assertEqual(self.reranker.rerank.call_count, 2)
        self.assertEqual(len(calls), 2)

    def test_disjoint_reused_and_multiple_evidence_preserve_independent_edges_and_usage(self):
        result = self.execute()
        rows = traceability_rows(result)
        inference = {(r['source_id'], r['target_id']) for r in rows if r['relation'] == 'supports inference'}
        derivation = {(r['source_id'], r['target_id']) for r in rows if r['relation'] == 'supports derivation'}
        self.assertEqual(inference, {(self.ids[0], 'R-01'), (self.ids[2], 'R-01')})
        self.assertEqual(derivation, {(self.ids[1], 'REQ-01'), (self.ids[1], 'REQ-02'), (self.ids[2], 'REQ-02')})
        report = diagnostic_report(result)
        self.assertEqual([r['usage'] for r in report['evidence_usage']],
                         ['used_by_risk_only', 'used_by_requirement_only', 'used_by_both', 'unused', 'unused'])
        self.assertEqual(report['unused_final_evidence_ids'], self.ids[3:5])
        without = {k: v for k, v in result.items() if k != 'retrieval_diagnostics'}
        self.assertEqual(analysis_export(result), analysis_export(without))
        self.assertEqual(traceability_csv(result), traceability_csv(without))
        exported = json.loads(analysis_export(result))
        self.assertEqual(exported['ai_requirements'][0]['evidence_ids'], [self.ids[1]])
        self.assertEqual(exported['traceability'], rows)

    def test_vector_candidates_outside_top_k_and_unknown_ids_are_rejected(self):
        result = self.execute()
        evidence = result['retrieved_evidence']
        for invalid in (self.ids[5], 'UNKNOWN'):
            with self.subTest(invalid=invalid):
                req = self.requirements[0].model_copy(update={'evidence_ids': [invalid]})
                with self.assertRaisesRegex(ValueError, 'Unknown NIST evidence'):
                    validate_ai_requirements([req], [self.risk], scenario(), evidence)
                changed_risk = self.risk.model_copy(update={'evidence_ids': [invalid]})
                with self.assertRaisesRegex(ValueError, 'Unknown NIST evidence'):
                    validate_contextual_risks([changed_risk], scenario(), evidence)
        reused = self.requirements[0].model_copy(update={'evidence_ids': self.risk.evidence_ids})
        validate_ai_requirements([reused], [self.risk], scenario(), evidence)

    def test_prompt_explicitly_separates_inference_and_treatment_support(self):
        prompt = ' '.join(REQUIREMENT_INSTRUCTIONS.split())
        for clause in (
            'Select treatment evidence independently', 'overlap is not required',
            'Reuse is valid', 'do not force different evidence for novelty',
            'review the COMPLETE treatment Top-K NIST evidence set',
            'including items unused by risk inference',
            'Do not require requirement evidence_ids to be a subset of associated risk evidence_ids',
            'Requirement evidence_ids MUST belong to the treatment retrieved Top-K set',
            'Do not force every treatment Top-K item to be used',
            'One strong item is sufficient', 'multiple items may be cited when genuinely needed',
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, prompt)
