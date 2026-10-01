"""Controlled offline experiment for separate inference and treatment retrieval."""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.app_support import analysis_export, run_analysis, traceability_rows
from src.graph.nodes import derive_ai_requirements
from src.graph.workflow import build_us4ai_graph
from src.ingestion import prepare_nist_documents
from src.retrieval_diagnostics import diagnostic_report
from src.retriever import NistVectorRetriever, build_retrieval_query, build_treatment_query
from src.schemas import AIRequirementsOutput, ContextualRisksOutput
from src.traceability import validate_ai_requirements, validate_contextual_risks
from tests.test_generative_core import requirement, risk, scenario
from tests.test_nist_rag import populated_store


class TwoStageRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = prepare_nist_documents()
        self.store = populated_store()
        self.reranker = Mock()
        self.reranker.rerank.return_value = list(range(15, 0, -1))
        self.retriever = NistVectorRetriever(vector_store=self.store, reranker=self.reranker)
        self.risks = [risk(), risk().model_copy(update={'risk_id': 'R-02', 'title': 'Altered drafts'})]
        self.client = Mock()

    def configure(self, start=10, empty=False, no_risks=False):
        self.store.similarity_search_with_score.side_effect = [
            [(d, i / 100) for i, d in enumerate(self.docs[:15])],
            [] if empty else [(d, i / 100) for i, d in enumerate(self.docs[start:start + 15])],
        ]
        req = requirement().model_copy(update={'evidence_ids': [self.docs[start].id]})
        self.client.chat.completions.create.side_effect = [
            ContextualRisksOutput(contextual_risks=[] if no_risks else self.risks),
            AIRequirementsOutput(ai_requirements=[req]),
        ]
        return build_us4ai_graph(retriever=self.retriever, client=self.client)

    def test_query_is_deterministic_complete_and_ignores_risk_citations(self):
        data = scenario()
        query = build_treatment_query(data, self.risks)
        self.assertIn(build_retrieval_query(data), query)
        self.assertIn('system-level controls', query)
        for r in self.risks:
            for value in (r.risk_id, r.title, r.description):
                self.assertIn(value, query)
        self.assertIn('TASK-01', query)
        changed = [r.model_copy(update={'evidence_ids': ['UNRELATED-ID']}) for r in self.risks]
        self.assertEqual(query, build_treatment_query(data, changed))
        self.assertNotIn('UNRELATED-ID', query)
        self.assertNotIn(self.risks[0].evidence_ids[0], query)

    def test_disjoint_stages_use_one_shared_retriever_and_two_generation_calls(self):
        result = run_analysis(self.configure(), scenario())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(self.store.similarity_search_with_score.call_count, 2)
        self.assertEqual(self.reranker.rerank.call_count, 2)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        queries = [call.args[0] for call in self.store.similarity_search_with_score.call_args_list]
        self.assertEqual(queries, [build_retrieval_query(scenario()), build_treatment_query(scenario(), self.risks)])
        for call in self.store.similarity_search_with_score.call_args_list:
            self.assertEqual(call.kwargs, {'k': 15})
        risk_ids = [d.id for d in self.docs[:5]]
        treatment_ids = [d.id for d in self.docs[10:15]]
        self.assertEqual([e.evidence_id for e in result['retrieved_evidence']], risk_ids)
        self.assertEqual([e.evidence_id for e in result['treatment_evidence']], treatment_ids)
        calls = self.client.chat.completions.create.call_args_list
        for call, expected in zip(calls, (risk_ids, treatment_ids)):
            context = json.loads(call.kwargs['messages'][1]['content'])
            self.assertEqual([e['evidence_id'] for e in context['nist_source_evidence']], expected)
            self.assertEqual(call.kwargs['temperature'], 0)
        report = diagnostic_report(result)
        self.assertEqual(report['cross_stage_summary']['risk_top_k_only'], risk_ids)
        self.assertEqual(report['cross_stage_summary']['treatment_top_k_only'], treatment_ids)
        self.assertEqual(report['cross_stage_summary']['both_top_k'], [])
        for key in ('risk_retrieval', 'treatment_retrieval'):
            stage = report[key]
            self.assertEqual((stage['candidate_k'], stage['top_k']), (15, 5))
            self.assertEqual(len(stage['vector_candidates']), 15)
            self.assertEqual(len(stage['reranked_candidates']), 15)
            self.assertEqual(len(stage['used_evidence_ids']), 1)
            self.assertEqual(len(stage['unused_evidence_ids']), 4)
        rows = traceability_rows(result)
        self.assertFalse(any(r['source_id'] in treatment_ids and r['relation'] == 'supports inference' for r in rows))
        self.assertFalse(any(r['source_id'] in risk_ids and r['relation'] == 'supports derivation' for r in rows))
        exported = json.loads(analysis_export(result))
        self.assertEqual([e['evidence_id'] for e in exported['retrieved_evidence']], risk_ids)
        self.assertEqual([e['evidence_id'] for e in exported['treatment_evidence']], treatment_ids)
        self.assertEqual(set(exported['nist_records']), set(risk_ids))
        self.assertEqual(set(exported['treatment_records']), set(treatment_ids))
        self.assertNotIn('retrieval_diagnostics', exported)

    def test_stage_specific_validation_rejects_cross_stage_and_accepts_overlap(self):
        result = run_analysis(self.configure(), scenario())
        risk_evidence, treatment = result['retrieved_evidence'], result['treatment_evidence']
        with self.assertRaisesRegex(ValueError, 'Unknown NIST evidence'):
            validate_ai_requirements([requirement()], self.risks, scenario(), risk_evidence, treatment_evidence=treatment)
        with self.assertRaisesRegex(ValueError, 'Unknown NIST evidence'):
            validate_contextual_risks([risk().model_copy(update={'evidence_ids': [treatment[0].evidence_id]})], scenario(), risk_evidence)
        with self.assertRaisesRegex(ValueError, 'Unknown NIST evidence'):
            validate_ai_requirements([requirement()], self.risks, scenario(), risk_evidence, treatment_evidence=[])
        validate_ai_requirements([requirement()], self.risks, scenario(), risk_evidence, treatment_evidence=[risk_evidence[0]])

    def test_partial_overlap_membership_is_independent_of_usage(self):
        result = run_analysis(self.configure(start=3), scenario())
        summary = diagnostic_report(result)['cross_stage_summary']
        self.assertEqual(summary['both_top_k'], [d.id for d in self.docs[3:5]])
        self.assertEqual(summary['risk_top_k_only'], [d.id for d in self.docs[:3]])
        self.assertEqual(summary['treatment_top_k_only'], [d.id for d in self.docs[5:8]])
        self.assertEqual(summary['used_for_risk_inference'], [self.docs[0].id])
        self.assertEqual(summary['used_for_requirement_derivation'], [self.docs[3].id])

    def test_no_risks_skip_second_retrieval_and_requirements(self):
        result = run_analysis(self.configure(no_risks=True), scenario())
        self.assertEqual(result['status'], 'no_supported_risks')
        self.store.similarity_search_with_score.assert_called_once()
        self.reranker.rerank.assert_called_once()
        self.assertEqual(self.client.chat.completions.create.call_count, 1)
        self.assertIsNone(diagnostic_report(result)['treatment_retrieval'])

    def test_empty_treatment_never_falls_back(self):
        result = run_analysis(self.configure(empty=True), scenario())
        self.assertEqual(result['status'], 'no_treatment_evidence')
        self.assertEqual(result['contextual_risks'], self.risks)
        self.assertEqual(result['ai_requirements'], [])
        self.assertEqual(self.store.similarity_search_with_score.call_count, 2)
        self.reranker.rerank.assert_called_once()
        self.assertEqual(self.client.chat.completions.create.call_count, 1)
        self.assertEqual(diagnostic_report(result)['treatment_retrieval']['final_evidence_ids'], [])
        with self.assertRaisesRegex(ValueError, 'requires treatment evidence'):
            derive_ai_requirements(result, client=self.client)

    def test_streamlit_retains_both_stages_and_download_on_rerun(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        with patch('src.app_support.initialize_analysis_graph', return_value=self.configure()), \
             patch('src.config.OPENAI_API_KEY', 'offline-placeholder'):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
            app.selectbox(key='task_group_0').select('Natural Language Processing').run()
            app.selectbox(key='task_value_0_Natural Language Processing').select('Text Generation').run()
            app.text_area[0].input('Support review').run()
            app.text_area[1].input('Prepare drafts').run()
            app.button(key='run_analysis').click().run()
            self.assertEqual(len(app.exception), 0)
            report = diagnostic_report(app.session_state['analysis_result'])
            self.assertIsNotNone(report['treatment_retrieval'])
            app.run()
            self.assertEqual(diagnostic_report(app.session_state['analysis_result']), report)
            self.assertEqual(json.loads(json.dumps(report)), report)
            self.assertIn('Download retrieval diagnostics JSON', [b.label for b in app.get('download_button')])
            self.assertEqual(self.store.similarity_search_with_score.call_count, 2)
            self.assertEqual(self.reranker.rerank.call_count, 2)
            self.assertEqual(self.client.chat.completions.create.call_count, 2)
