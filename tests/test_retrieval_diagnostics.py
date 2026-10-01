"""Offline checks for passive ranking capture and separate diagnostic presentation."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from src.app_support import AnalysisExecutionError, analysis_export, run_analysis, traceability_csv
from src.graph.workflow import build_us4ai_graph
from src.ingestion import prepare_nist_documents
from src.retrieval_diagnostics import diagnostic_report
from src.retriever import NistVectorRetriever, build_retrieval_query
from src.schemas import AIRequirementsOutput, ContextualRisksOutput
from tests.test_generative_core import evidence, requirement, risk, scenario
from tests.test_nist_rag import populated_store


class RetrievalDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.docs = prepare_nist_documents()[:15]
        self.store = populated_store()
        self.candidates = [(doc, (rank + 1) / 100) for rank, doc in enumerate(self.docs)]
        self.store.similarity_search_with_score.return_value = self.candidates
        self.reranker = Mock()
        self.scores = [float(15 - i) for i in range(15)]
        self.scores[11] = 20.0
        self.reranker.rerank.return_value = self.scores
        self.retriever = NistVectorRetriever(vector_store=self.store, reranker=self.reranker)
        self.client = Mock()
        self.client.chat.completions.create.side_effect = [
            ContextualRisksOutput(contextual_risks=[risk()]),
            AIRequirementsOutput(ai_requirements=[requirement()]),
        ]

    def graph(self):
        return build_us4ai_graph(retriever=self.retriever, client=self.client)

    def test_actual_vector_order_is_captured_before_reranking(self):
        captured = {}
        def rerank(query, texts):
            self.assertEqual(len(captured['vector_candidates']), 15)
            self.assertEqual([row['evidence_id'] for row in captured['vector_candidates']],
                             [doc.id for doc in self.docs])
            self.assertEqual(texts, [doc.page_content for doc in self.docs])
            return self.scores
        self.reranker.rerank.side_effect = rerank
        evidence = self.retriever.retrieve(scenario(), diagnostics=captured)
        self.store.similarity_search_with_score.assert_called_once_with(build_retrieval_query(scenario()), k=15)
        self.reranker.rerank.assert_called_once()
        self.assertEqual(captured['retrieval_query'], build_retrieval_query(scenario()))
        self.assertEqual((captured['candidate_k'], captured['top_k']), (15, 5))
        self.assertEqual([row['vector_rank'] for row in captured['vector_candidates']], list(range(1, 16)))
        self.assertEqual([row['vector_distance'] for row in captured['vector_candidates']],
                         [distance for _, distance in self.candidates])
        self.assertEqual([row['rerank_rank'] for row in captured['reranked_candidates']], list(range(1, 16)))
        first = captured['reranked_candidates'][0]
        self.assertEqual((first['vector_rank'], first['rerank_rank'], first['reranker_score']), (12, 1, 20.0))
        for row in captured['reranked_candidates']:
            self.assertEqual(row['reranker_score'], self.scores[row['vector_rank'] - 1])
            self.assertEqual(row['vector_distance'], self.candidates[row['vector_rank'] - 1][1])
        self.assertEqual(captured['final_evidence_ids'], [item.evidence_id for item in evidence])
        for row in captured['vector_candidates']:
            record = self.retriever.records[row['evidence_id']]
            self.assertEqual((row['type'], row['category'], row['topics']),
                             (record.type, record.category, record.topic))

    def test_capture_preserves_results_including_ties(self):
        for scores in (self.scores, [0.5] * 15):
            self.reranker.rerank.return_value = scores
            plain = self.retriever.retrieve(scenario())
            self.store.similarity_search_with_score.reset_mock()
            self.reranker.rerank.reset_mock()
            captured = {}
            observed = self.retriever.retrieve(scenario(), diagnostics=captured)
            self.assertEqual(observed, plain)
            expected = sorted(zip(self.docs, scores), key=lambda pair: (-pair[1], pair[0].id))[:5]
            self.assertEqual([item.evidence_id for item in observed], [doc.id for doc, _ in expected])
            self.store.similarity_search_with_score.assert_called_once()
            self.reranker.rerank.assert_called_once()

    def test_graph_preserves_diagnostics_without_extra_calls_or_export_changes(self):
        result = run_analysis(self.graph(), scenario())
        report = diagnostic_report(result)
        self.assertEqual(report['final_evidence_ids'], [item.evidence_id for item in result['retrieved_evidence']])
        for call in self.client.chat.completions.create.call_args_list:
            context = json.loads(call.kwargs['messages'][1]['content'])
            self.assertEqual([item['evidence_id'] for item in context['nist_source_evidence']], report['final_evidence_ids'])
            self.assertNotIn('retrieval_diagnostics', context)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        self.assertEqual(self.store.similarity_search_with_score.call_count, 2)
        self.assertEqual(self.reranker.rerank.call_count, 2)
        without = {key: value for key, value in result.items() if key != 'retrieval_diagnostics'}
        self.assertEqual(analysis_export(result), analysis_export(without))
        self.assertEqual(traceability_csv(result), traceability_csv(without))
        self.assertNotIn('retrieval_diagnostics', json.loads(analysis_export(result)))
        self.assertNotIn('retrieval_text', json.dumps(report))

    def test_usage_is_deterministic_and_does_not_mutate_capture(self):
        captured = {'final_evidence_ids': ['A', 'B', 'C', 'D']}
        result = {'retrieval_diagnostics': captured,
                  'contextual_risks': [risk().model_copy(update={'evidence_ids': ['A', 'B', 'A']})],
                  'ai_requirements': [requirement().model_copy(update={'evidence_ids': ['B', 'C']})]}
        original = copy.deepcopy(captured)
        report = diagnostic_report(result)
        self.assertEqual(report, diagnostic_report(result))
        self.assertEqual(captured, original)
        self.assertEqual(report['risk_used_evidence_ids'], ['A', 'B'])
        self.assertEqual(report['requirement_used_evidence_ids'], ['B', 'C'])
        self.assertEqual(report['unused_final_evidence_ids'], ['D'])
        self.assertEqual([row['usage'] for row in report['evidence_usage']],
                         ['used_by_risk_only', 'used_by_both', 'used_by_requirement_only', 'unused'])

    def test_empty_retrieval_and_sequential_calls_do_not_leak_diagnostics(self):
        first = {}
        self.retriever.retrieve(scenario(), diagnostics=first)
        original = copy.deepcopy(first)
        self.store.similarity_search_with_score.return_value = []
        self.reranker.rerank.reset_mock()
        result = run_analysis(self.graph(), scenario())
        report = diagnostic_report(result)
        self.assertEqual(report['vector_candidates'], [])
        self.assertEqual(report['reranked_candidates'], [])
        self.assertEqual(report['final_evidence_ids'], [])
        self.assertEqual(first, original)
        self.reranker.rerank.assert_not_called()
        self.client.chat.completions.create.assert_not_called()
        self.assertIsNone(diagnostic_report({}))

    def test_partial_generation_retains_rankings_and_validated_usage(self):
        self.client.chat.completions.create.side_effect = [
            ContextualRisksOutput(contextual_risks=[risk()]), ValueError('Invalid generation')]
        with self.assertRaises(AnalysisExecutionError) as caught:
            run_analysis(self.graph(), scenario())
        report = diagnostic_report(caught.exception.partial_state)
        self.assertEqual(report['analysis_status'], 'failed')
        self.assertEqual(report['risk_used_evidence_ids'], ['GOVERN 1.1'])
        self.assertEqual(report['requirement_used_evidence_ids'], [])
        self.assertEqual(len(report['vector_candidates']), 15)

    def test_streamlit_exposes_separate_diagnostics_without_rerunning_analysis(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        with patch('src.app_support.initialize_analysis_graph', return_value=self.graph()), \
             patch('src.config.OPENAI_API_KEY', 'offline-placeholder'):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
            app.selectbox(key='task_group_0').select('Natural Language Processing').run()
            app.selectbox(key='task_value_0_Natural Language Processing').select('Text Generation').run()
            app.text_area[0].input('Support billing').run()
            app.text_area[1].input('Issue a bill').run()
            app.button(key='run_analysis').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn('Retrieval Diagnostics', [item.label for item in app.expander])
            self.assertIn('Download retrieval diagnostics JSON', [item.label for item in app.get('download_button')])
            self.assertTrue(any(len(table.value) == 15 for table in app.dataframe))
            app.run()
            self.assertEqual(self.store.similarity_search_with_score.call_count, 2)
            self.assertEqual(self.reranker.rerank.call_count, 2)
            self.assertEqual(self.client.chat.completions.create.call_count, 2)

    def test_cached_pre_diagnostics_graph_is_replaced_before_next_analysis(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        app_path = Path(__file__).resolve().parents[1] / 'app.py'
        source = app_path.read_text()
        # Reproduce the cache entry created by the application before diagnostics.
        legacy_source = source.replace(
            'def get_analysis_graph(diagnostics_version):', 'def get_analysis_graph():'
        ).replace('get_analysis_graph(diagnostics_version=2)', 'get_analysis_graph()')
        legacy_retriever = Mock()
        legacy_retriever.retrieve.return_value = [evidence()]
        legacy_client = Mock()
        legacy_client.chat.completions.create.side_effect = [
            ContextualRisksOutput(contextual_risks=[risk()]),
            AIRequirementsOutput(ai_requirements=[requirement()]),
        ]
        legacy_graph = build_us4ai_graph(retriever=legacy_retriever, client=legacy_client)
        legacy_result = run_analysis(legacy_graph, scenario())
        self.assertEqual(legacy_result['status'], 'complete')
        self.assertIsNone(diagnostic_report(legacy_result))
        current_graph = self.graph()
        with patch('src.app_support.initialize_analysis_graph', side_effect=[legacy_graph, current_graph]) as initialize, \
             patch('src.config.OPENAI_API_KEY', 'offline-placeholder'):
            old_app = AppTest.from_string(legacy_source, default_timeout=30).run()
            self.assertEqual(len(old_app.exception), 0)
            app = AppTest.from_file(str(app_path), default_timeout=30).run()
            self.assertEqual(initialize.call_count, 2, 'A pre-diagnostics graph must not survive the cache contract change.')
            app.selectbox(key='task_group_0').select('Natural Language Processing').run()
            app.selectbox(key='task_value_0_Natural Language Processing').select('Text Generation').run()
            app.text_area[0].input('Support billing').run()
            app.text_area[1].input('Issue a bill').run()
            app.button(key='run_analysis').click().run()
            self.assertEqual(len(app.exception), 0)
            result = app.session_state['analysis_result']
            self.assertEqual(result['status'], 'complete')
            report = diagnostic_report(result)
            self.assertIsNotNone(report)
            self.assertEqual(report['retrieval_query'], result['retrieval_query'])
            self.assertEqual(len(report['vector_candidates']), 15)
            self.assertEqual(len(report['reranked_candidates']), 15)
            self.assertEqual(report['final_evidence_ids'], [item.evidence_id for item in result['retrieved_evidence']])
            self.assertEqual(report['risk_used_evidence_ids'], ['GOVERN 1.1'])
            self.assertEqual(report['requirement_used_evidence_ids'], ['GOVERN 1.1'])
            self.assertEqual(json.loads(json.dumps(report)), report)
            app.run()
            self.assertEqual(diagnostic_report(app.session_state['analysis_result']), report)
            self.assertFalse(any('diagnostics are unavailable' in item.value for item in app.info))
            self.assertIn('Download retrieval diagnostics JSON', [item.label for item in app.get('download_button')])
            self.assertEqual(initialize.call_count, 2)
            self.assertEqual(legacy_retriever.retrieve.call_count, 2)
            self.assertEqual(self.store.similarity_search_with_score.call_count, 2)
            self.assertEqual(self.reranker.rerank.call_count, 2)
            self.assertEqual(self.client.chat.completions.create.call_count, 2)

    def test_diagnostics_survive_final_graph_state(self):
        result = self.graph().invoke({'analysis_input': scenario()})
        report = diagnostic_report(result)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(report['vector_candidates']), 15)
        self.assertEqual(len(report['reranked_candidates']), 15)
        self.assertEqual(report['final_evidence_ids'], [item.evidence_id for item in result['retrieved_evidence']])
        self.assertEqual(self.store.similarity_search_with_score.call_count, 2)
        self.assertEqual(self.reranker.rerank.call_count, 2)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
