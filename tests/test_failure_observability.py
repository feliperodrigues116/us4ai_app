"""Offline failure metadata and redaction checks without changing graph behavior."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app_support import AnalysisExecutionError, analysis_export, run_analysis, traceability_csv
from src.failure_diagnostics import sanitize_error_message
from src.retrieval_diagnostics import diagnostic_report
from tests import test_two_stage_retrieval
from tests.test_generative_core import scenario


class FailureObservabilityTests(unittest.TestCase):
    def fixture(self, stage):
        f = test_two_stage_retrieval.TwoStageRetrievalTests()
        f.setUp()
        graph = f.configure()
        message = 'Reranker must return one finite score per candidate.'
        error = ValueError(message)
        if stage == 'risk_retrieval':
            f.store.similarity_search_with_score.side_effect = error
        elif stage == 'treatment_retrieval':
            first = [(d, i / 100) for i, d in enumerate(f.docs[:15])]
            f.store.similarity_search_with_score.side_effect = [first, error]
        else:
            outputs = list(f.client.chat.completions.create.side_effect)
            if stage == 'risk_generation':
                outputs[0].contextual_risks[0].evidence_ids = ['MAP 999.1']
            else:
                outputs[1].ai_requirements[0].evidence_ids = ['MAP 999.1']
            f.client.chat.completions.create.side_effect = outputs
            message = 'Unknown NIST evidence references: MAP 999.1'
        return f, graph, message

    def test_all_failure_stages_preserve_only_completed_artifacts(self):
        for stage in ('risk_retrieval', 'risk_generation', 'treatment_retrieval', 'requirement_generation'):
            with self.subTest(stage=stage):
                f, graph, message = self.fixture(stage)
                with self.assertRaises(AnalysisExecutionError) as caught:
                    run_analysis(graph, scenario())
                result = caught.exception.partial_state
                self.assertEqual(result['failed_stage'], stage)
                self.assertEqual(result['error_type'], 'ValueError')
                self.assertEqual(result['error_message'], message)
                self.assertEqual(caught.exception.error_type, 'ValueError')
                report = diagnostic_report(result)
                self.assertEqual(report['analysis_status'], 'failed')
                for key in ('failed_stage', 'error_type', 'error_message'):
                    self.assertEqual(report[key], result[key])
                self.assertEqual(report['risk_retrieval'] is not None, stage != 'risk_retrieval')
                self.assertEqual(report['treatment_retrieval'] is not None, stage == 'requirement_generation')
                self.assertEqual(bool(result['contextual_risks']), stage in ('treatment_retrieval', 'requirement_generation'))
                self.assertEqual(result['ai_requirements'], [])
                if stage == 'requirement_generation':
                    self.assertEqual(len(report['treatment_retrieval']['vector_candidates']), 15)
                    self.assertEqual(len(result['treatment_evidence']), 5)
                self.assertEqual(json.loads(json.dumps(report)), report)
                exported = json.loads(analysis_export(result))
                self.assertNotIn('error_message', exported)
                self.assertNotIn('failed_stage', exported)
                self.assertEqual(traceability_csv(result).splitlines()[0], 'source_type,source_id,relation,target_type,target_id')

    def test_sanitizer_withholds_payloads_tokens_and_unsafe_reference_values(self):
        with patch('src.config.OPENAI_API_KEY', 'test-secret-key'), patch.dict('os.environ', {'OPENAI_API_KEY': 'env-secret-key'}):
            for text in ('Authorization: Bearer token-value', 'prompt: private-user-input',
                         'response: private-model-output', 'environment: PRIVATE=value',
                         'test-secret-key', 'env-secret-key',
                         'Unknown NIST evidence references: Authorization: Bearer token-value'):
                safe = sanitize_error_message(ValueError(text))
                self.assertNotIn(text, safe)
                for secret in ('token-value', 'private-user-input', 'private-model-output', 'PRIVATE=value', 'test-secret-key', 'env-secret-key'):
                    self.assertNotIn(secret, safe)
        self.assertEqual(sanitize_error_message(ValueError('Unknown NIST evidence references: MAP 2.10')),
                         'Unknown NIST evidence references: MAP 2.10')
        self.assertIn('withheld', sanitize_error_message(RuntimeError('raw response')))

    def test_streamlit_failure_survives_rerun_without_false_reference_claim(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        f, graph, message = self.fixture('treatment_retrieval')
        with patch('src.app_support.initialize_analysis_graph', return_value=graph), patch('src.config.OPENAI_API_KEY', 'offline-placeholder'):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
            app.selectbox(key='task_group_0').select('Natural Language Processing').run()
            app.selectbox(key='task_value_0_Natural Language Processing').select('Text Generation').run()
            app.text_area[0].input('Review drafts').run()
            app.text_area[1].input('Prepare drafts').run()
            app.button(key='run_analysis').click().run()
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('Treatment Retrieval' in e.value for e in app.error))
            self.assertIn('Error: ValueError', [t.value for t in app.text])
            self.assertIn('Reason: ' + message, [t.value for t in app.text])
            self.assertFalse(any('Invalid generated references' in e.value for e in app.caption))
            self.assertIn('Download retrieval diagnostics JSON', [b.label for b in app.get('download_button')])
            self.assertEqual(f.store.similarity_search_with_score.call_count, 2)
            self.assertEqual(f.client.chat.completions.create.call_count, 1)
