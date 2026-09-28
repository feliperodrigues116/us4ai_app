"""Offline tests for the study vocabulary and generation discipline."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from src.ai_task_catalog import AI_TASK_CATALOG, tasks_for_group, validate_catalog_selection
from src.app_support import input_from_form, traceability_rows
from src.graph.prompts import RISK_INSTRUCTIONS, REQUIREMENT_INSTRUCTIONS
from src.schemas import AITask
from src.traceability import reference_ai_tasks, validate_ai_requirements, validate_contextual_risks
from tests.test_generative_core import scenario, evidence, risk, requirement


class CatalogTests(unittest.TestCase):
    def test_exact_catalog(self):
        expected = {
            'Multimodal': 'Audio-Text-to-Text|Image-Text-to-Text|Image-Text-to-Image|Image-Text-to-Video|Visual Question Answering|Document Question Answering|Video-Text-to-Text|Visual Document Retrieval|Any-to-Any',
            'Computer Vision': 'Depth Estimation|Image Classification|Object Detection|Image Segmentation|Text-to-Image|Image-to-Text|Image-to-Image|Image-to-Video|Unconditional Image Generation|Video Classification|Text-to-Video|Zero-Shot Image Classification|Mask Generation|Zero-Shot Object Detection|Text-to-3D|Image-to-3D|Image Feature Extraction|Keypoint Detection|Video-to-Video',
            'Natural Language Processing': 'Text Classification|Token Classification|Table Question Answering|Question Answering|Zero-Shot Classification|Translation|Summarization|Feature Extraction|Text Generation|Fill-Mask|Sentence Similarity|Text Ranking',
            'Audio': 'Text-to-Speech|Text-to-Audio|Automatic Speech Recognition|Audio-to-Audio|Audio Classification|Voice Activity Detection|Tabular',
            'Tabular Classification': 'Tabular Regression|Time Series Forecasting',
            'Reinforcement Learning': 'Reinforcement Learning|Robotics',
            'Other': 'Graph Machine Learning',
        }
        self.assertEqual(AI_TASK_CATALOG, {key: tuple(value.split('|')) for key, value in expected.items()})
        self.assertEqual(list(AI_TASK_CATALOG), list(expected))

    def test_dependent_options(self):
        self.assertEqual(tasks_for_group(None), ())
        self.assertEqual(tasks_for_group('Other'), ('Graph Machine Learning',))
        with self.assertRaises(ValueError):
            validate_catalog_selection([{'category': 'Audio', 'task': 'Text Generation'}])

    def test_multiple_tasks_and_order(self):
        rows = [{'category': 'Natural Language Processing', 'task': item}
                for item in ('Text Generation', 'Text Classification')]
        validate_catalog_selection(rows)
        data = input_from_form('Purpose', 'Story', '', rows)
        self.assertEqual([task.model_dump() for task in data.ai_tasks], rows)
        self.assertEqual(reference_ai_tasks(data)['TASK-01'].task, 'Text Generation')
        self.assertEqual(reference_ai_tasks(data)['TASK-02'].task, 'Text Classification')

    def test_duplicates_empty_and_incomplete_rejected(self):
        row = {'category': 'Other', 'task': 'Graph Machine Learning'}
        for rows in ([], [row, row], [{'category': 'Other', 'task': None}]):
            with self.assertRaises(ValueError):
                validate_catalog_selection(rows)

    def test_backend_stays_open(self):
        self.assertEqual(AITask(category='Custom research group', task='Custom task').task, 'Custom task')


class SemanticDisciplineTests(unittest.TestCase):
    def test_risk_prompt_requires_joint_grounding_and_selectivity(self):
        prompt = ' '.join(RISK_INSTRUCTIONS.split())
        for phrase in ('BOTH concrete scenario grounding AND', 'retrieved evidence does not automatically imply',
                       'Retrieved evidence may remain unused', 'Do not associate every risk with every AI Task',
                       'Prefer fewer strongly contextualized risks', 'Do not assume architecture',
                       'Do not automatically require human oversight'):
            self.assertIn(phrase, prompt)

    def test_requirements_are_system_controls_not_governance_paraphrases(self):
        prompt = ' '.join(REQUIREMENT_INSTRUCTIONS.split())
        for phrase in ('system-level control, constraint, mechanism, or behavior',
                       'Do not use the organization, project team, stakeholders, management, or developers',
                       'Do not simply paraphrase NIST Suggested Actions', 'Keep rationale separate',
                       'return an empty ai_requirements list'):
            self.assertIn(phrase, prompt)

    def test_duplicate_references_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate AI task'):
            validate_contextual_risks([risk().model_copy(update={'ai_task_ids': ['TASK-01', 'TASK-01']})], scenario(), [evidence()])
        with self.assertRaisesRegex(ValueError, 'Duplicate NIST evidence'):
            validate_contextual_risks([risk().model_copy(update={'evidence_ids': ['GOVERN 1.1', 'GOVERN 1.1']})], scenario(), [evidence()])

    def test_requirement_cannot_add_task_unrelated_to_addressed_risk(self):
        changed = requirement().model_copy(update={'ai_task_ids': ['TASK-02']})
        with self.assertRaisesRegex(ValueError, 'associated with its addressed'):
            validate_ai_requirements([changed], [risk()], scenario(), [evidence()])

    def test_organizational_subject_rejected_but_technical_subject_accepted(self):
        for subject in ('The organization', 'The project team', 'Stakeholders', 'Management', 'Developers'):
            changed = requirement().model_copy(update={'statement': subject + ' shall review policies.'})
            with self.assertRaisesRegex(ValueError, 'system-level control'):
                validate_ai_requirements([changed], [risk()], scenario(), [evidence()])
        changed = requirement().model_copy(update={'statement': 'The response gateway shall require agent approval before sending drafts.'})
        validate_ai_requirements([changed], [risk()], scenario(), [evidence()])

    def test_traceability_never_adds_unreferenced_tasks(self):
        rows = traceability_rows({'analysis_input': scenario(), 'contextual_risks': [risk()],
                                  'ai_requirements': [requirement()], 'retrieved_evidence': [evidence()]})
        self.assertFalse(any(row['source_id'] == 'TASK-02' for row in rows))
        self.assertFalse(any(row['target_type'] == 'NIST Evidence' for row in rows))


class CatalogUITests(unittest.TestCase):
    def test_dynamic_dependent_rows_and_duplicate_validation_offline(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        with patch('src.app_support.initialize_analysis_graph', return_value=Mock()), \
             patch('src.config.OPENAI_API_KEY', 'offline-placeholder'):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
            self.assertTrue(app.selectbox(key='task_value_0_None').disabled)
            self.assertTrue(app.button(key='remove_task_0').disabled)
            app.selectbox(key='task_group_0').select('Natural Language Processing').run()
            app.selectbox(key='task_value_0_Natural Language Processing').select('Text Generation').run()
            app.button(key='add_ai_task').click().run()
            app.selectbox(key='task_group_1').select('Natural Language Processing').run()
            app.selectbox(key='task_value_1_Natural Language Processing').select('Text Generation').run()
            app.button(key='run_analysis').click().run()
            self.assertTrue(any('duplicate' in item.value for item in app.error))
            app.selectbox(key='task_group_1').select('Other').run()
            self.assertEqual(app.selectbox(key='task_value_1_Other').options, ['Graph Machine Learning'])
            app.button(key='remove_task_0').click().run()
            self.assertEqual(app.session_state['task_rows'], [1])
            self.assertTrue(app.button(key='remove_task_1').disabled)
            self.assertEqual(len(app.exception), 0)
