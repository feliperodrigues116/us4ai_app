"""Offline regression checks for the delivered generation prompt contract.

These checks protect instructions, not the semantic quality of model outputs.
"""

import json
import unittest

from src.graph.prompts import build_requirement_messages, build_risk_messages
from src.app_support import traceability_rows
from src.schemas import AIRequirementsOutput, ContextualAIRisk
from src.traceability import validate_ai_requirements
from tests.test_generative_core import evidence, requirement, risk, scenario, state


class PromptContractTests(unittest.TestCase):
    def setUp(self):
        data = state()
        args = (scenario(), data['retrieved_evidence'], data['nist_records'])
        self.risk_prompt = ' '.join(build_risk_messages(*args)[0]['content'].split())
        self.requirement_prompt = ' '.join(
            build_requirement_messages(*args, [risk()])[0]['content'].split()
        )

    def assert_contract(self, prompt, clauses):
        for clause in clauses:
            with self.subTest(clause=clause):
                self.assertIn(clause, prompt)

    def test_risks_require_additional_ai_failure_mechanisms(self):
        self.assert_contract(self.risk_prompt, (
            'ACCEPTANCE CRITERIA ARE EXISTING REQUIREMENTS, NOT RISKS.',
            'Do not derive a contextual risk by simply negating, reversing, or describing non-compliance with an Acceptance Criterion.',
            'scenario + relevant AI Task + retrieved NIST evidence -> plausible AI-related failure mechanism -> contextual AI risk',
            "referenced AI Task's behavior contributes to that mechanism",
            'retrieved evidence does not automatically imply a contextual risk',
            'An AI Task does not automatically imply a risk.',
            'An Acceptance Criterion does not automatically imply a risk.',
            'Prefer fewer strongly contextualized risks over many generic risks.',
            'Zero risks is valid',
            'Return an empty contextual_risks list',
            'These examples are not a fixed AI Task-to-risk mapping.',
            'use them only when the actual scenario and evidence justify them',
            'WEAK / INVALID risk: "The chatbot may fail to confirm two customer attributes before issuing the bill."',
        ))

    def test_requirements_add_semantic_mitigation_information(self):
        self.assert_contract(self.requirement_prompt, (
            'DERIVED REQUIREMENTS MUST ADD RISK-MITIGATION INFORMATION.',
            'Mere paraphrase or generalization without meaningful risk-treatment information is insufficient',
            'Technical terminology changes or changes of subject or verb do not make an existing obligation new.',
            'Lexical novelty is not semantic novelty.',
            'internally compare the contextual risk against System Purpose, User Story, and every Acceptance Criterion',
            'Does this candidate merely restate an existing requirement, or does it add meaningful AI-specific risk-treatment information?',
            'Reject a mere restatement',
            'INVALID derived requirement: "The AI chatbot shall implement an authentication mechanism requiring at least two customer attributes before accessing billing information."',
        ))

    def test_controls_must_be_concrete_supported_and_optional(self):
        self.assert_contract(self.requirement_prompt, (
            'Generic safeguards without concrete behavior are insufficient',
            'what the system restricts, separates, validates, filters, gates, retrieves, logs, monitors, abstains from, or otherwise enforces',
            'do not force a technology unsupported by scenario and evidence',
            'Do not invent arbitrary controls for novelty or creativity.',
            'It addresses at least one generated Contextual AI Risk.',
            'That risk is supported by the supplied scenario.',
            'That risk is relevant to the referenced AI Task(s).',
            'The derivation is grounded in retrieved NIST evidence.',
            'It specifies an implementable system-level control',
            'It adds meaningful AI-specific risk-treatment information to the known baseline',
            'It does not depend on facts neither supplied nor defensibly inferred.',
            'If any condition fails, omit that candidate; assess other distinct controls for the risk.',
            'Prefer no requirement over a redundant, generic, speculative, or unsupported requirement.',
            'return an empty ai_requirements list',
            'Do not simply paraphrase NIST Suggested Actions or mechanically turn each bullet into a requirement.',
            'Do not present the control as a NIST requirement or the contextual risk as a verbatim NIST statement.',
        ))

    def test_examples_are_conditional_not_preferred_controls(self):
        self.assert_contract(self.requirement_prompt, (
            'prevent the text-generation component from accessing customer-specific billing information until',
            'prevent the language model from independently generating or modifying those values',
            'beyond merely restating "verify identity", provided that restriction is not already required',
            'Do not copy, reproduce, or preferentially generate either control unless supported by the actual scenario, generated contextual risk, supplied AI Tasks, and retrieved NIST evidence',
            'not evidence that these components, data flows, or failure modes exist in this scenario',
        ))

    def test_evidence_identifiers_are_exact_in_structured_and_free_text(self):
        for prompt in (self.risk_prompt, self.requirement_prompt):
            self.assert_contract(prompt, (
                'Copy evidence identifiers exactly from the supplied evidence objects.',
                'Never invent, transform, abbreviate, or reconstruct a NIST identifier.',
            ))
        self.assert_contract(self.risk_prompt, (
            "it must exactly match one of that risk's evidence_ids",
        ))
        self.assert_contract(self.requirement_prompt, (
            "If an evidence ID appears in rationale text, it must exactly match one of that requirement's structured evidence_ids.",
            'Prefer rationale wording that does not unnecessarily repeat identifiers',
        ))

    def test_technical_subjects_do_not_require_a_literal_prefix(self):
        self.assert_contract(self.requirement_prompt, (
            '"The system shall..." is a preferred style, not a mandatory prefix.',
            'The AI chatbot shall...',
            'The retrieval component shall...',
            'The generation component shall...',
            'The authorization layer shall...',
            'Do not use the organization, project team, stakeholders, management, or developers as the primary subject',
        ))

    def test_risk_evidence_requires_substantive_support_for_each_citation(self):
        self.assert_contract(self.risk_prompt, (
            'REFERENCE VALIDITY means the evidence ID exists among retrieved NIST evidence',
            'SEMANTIC SUPPORT means its substantive content provides a defensible basis',
            'Reference validity is not equivalent to semantic support; both are required',
            'Do not cite a retrieved NIST item merely because it is generally related to trustworthy AI',
            'For every evidence item attached to a risk, internally ask:',
            'What specific concept, concern, condition, or guidance',
            'supports this particular risk?',
            'If no concrete answer is possible, do not attach that evidence',
            'If no retrieved evidence substantively supports the candidate risk, do not output that risk',
            'Use actual evidence content, not only its title, function, topic, or retrieval rank',
        ))

    def test_topic_mismatch_is_not_justified_by_shared_scenario(self):
        self.assert_contract(self.risk_prompt, (
            'Avoid evidence topic mismatch',
            'privacy risk, confidentiality, data minimization, sensitive-information inference, or access control',
            'does not support inaccurate payment-code generation merely because both occur in the same chatbot',
            'Fairness evidence must actually connect to an accuracy risk',
            'human-oversight evidence does not support every generation risk',
            'generic context-of-use evidence does not automatically support every contextual risk',
            'documentation guidance does not automatically justify a runtime technical failure',
        ))

    def test_minimal_evidence_applies_to_both_generation_stages(self):
        for prompt in (self.risk_prompt, self.requirement_prompt):
            self.assert_contract(prompt, (
                'MINIMAL EVIDENCE:',
                'One strongly relevant evidence item is preferable to several weakly related items',
                'Do not maximize citations',
                'Retrieved evidence may remain unused',
            ))
        self.assertIn('Attach only items that materially support the specific risk', self.risk_prompt)
        self.assertIn('Every requirement evidence item must materially support derivation of the control', self.requirement_prompt)

    def test_acceptance_criteria_allow_grounded_baseline_enrichment(self):
        self.assert_contract(self.requirement_prompt, (
            'Acceptance Criteria define the known behavioral baseline',
            'baseline, not a prohibition boundary',
            'may complement, refine, specialize, or strengthen',
            'Semantic overlap alone is not a reason to reject',
            'Do not require independence from Acceptance Criteria',
            'existing mitigation obligations',
            'not proof that those obligations have already been implemented',
            'including any residual exposure',
            'Reject a mere restatement',
            'accept enrichment only when grounded',
            'Do not use lexical difference as evidence of novelty or add artificial detail',
        ))
        self.assertNotIn('If every Acceptance Criterion were implemented', self.requirement_prompt)
        self.assertNotIn('residual-value test', self.requirement_prompt)

    def test_control_targets_mechanism_without_prescribing_unsupported_solutions(self):
        self.assert_contract(self.requirement_prompt, (
            'CONTROL THE FAILURE MECHANISM, NOT JUST THE FINAL OUTCOME.',
            'Text Classification incorrectly classifying ambiguous or variable customer-provided attributes',
            'merely repeats the final business rule',
            'It does not control classification uncertainty or misclassification',
            'These are possibilities, not required mechanisms',
            'Do not assume confidence scores, fallback models, human review, or deterministic validation',
            'unless the actual scenario and evidence support such a control',
        ))

    def test_requirement_evidence_supports_control_not_only_risk(self):
        self.assert_contract(self.requirement_prompt, (
            'A valid retrieved reference alone is insufficient',
            'For every requirement evidence item, internally ask:',
            'What specific content in this NIST evidence supports deriving this type of control for this contextual risk?',
            'If it supports only the broad existence of the risk',
            'does not reasonably support the mitigation/control, do not use it as requirement evidence',
            'privacy guidance does not justify payment-code accuracy controls',
            'If no retrieved evidence substantively supports an additional defensible control, output no requirement',
            'weakly grounded evidence is insufficient',
            'Do not expose internal step-by-step reasoning',
            'return only the existing structured schema',
        ))

    def test_control_count_is_driven_by_distinct_supported_opportunities(self):
        self.assert_contract(self.requirement_prompt, (
            'zero, one, or multiple AI-specific requirements',
            'One requirement may address multiple risks when justified',
            'examine the supported control space',
            'optional reasoning dimensions, not a mandatory checklist',
            'Do not assume a dimension applies',
            'Generate as many distinct AI-specific requirements as are substantively justified, and no more',
            'no minimum or target requirement count',
            'A single strong requirement is preferable',
            'Do not stop after the first valid requirement',
            'multiple requirements must represent distinct controls',
            'Do not split or paraphrase the same control to inflate the count',
        ))

    def test_quality_goals_need_operational_behavior(self):
        self.assert_contract(self.requirement_prompt, (
            'QUALITY GOALS ARE NOT CONTROLS',
            'Classify accurately',
            'ensure accuracy',
            'insufficient without operational system behavior',
            'must do, restrict, validate, detect, prevent, monitor, route, record, compare',
            'not unsupported implementation-level design details',
            'otherwise return none for that aspect',
            'Do not limit mitigation to identity verification',
        ))

    def test_substantive_evidence_reaches_requirement_generation(self):
        data = state()
        messages = build_requirement_messages(
            scenario(), data['retrieved_evidence'], data['nist_records'], [risk()]
        )
        source = json.loads(messages[1]['content'])['nist_source_evidence'][0]
        record = data['nist_records'][source['evidence_id']]
        for field in ('description', 'section_about', 'section_actions'):
            with self.subTest(field=field):
                self.assertTrue(source[field])
                self.assertEqual(source[field], getattr(record, field))
                self.assertIn(field, self.requirement_prompt)

    def test_existing_models_validators_and_edges_support_many_to_many(self):
        # Structural fixtures demonstrate cardinality, not semantic NIST endorsement.
        first_risk = risk()
        second_risk = ContextualAIRisk.model_validate({
            **first_risk.model_dump(), 'risk_id': 'R-02',
            'title': 'Disclosure through retained generated drafts',
            'description': 'Retained generated drafts may expose customer information.',
        })
        first = requirement().model_dump()
        second = {
            **first, 'requirement_id': 'REQ-02', 'risk_ids': ['R-01', 'R-02'],
            'statement': 'The draft store shall restrict draft access to the assigned agent.',
        }
        output = AIRequirementsOutput.model_validate({'ai_requirements': [first, second]})
        risks = [first_risk, second_risk]
        for requirements in ([], output.ai_requirements[:1], output.ai_requirements):
            with self.subTest(count=len(requirements)):
                validate_ai_requirements(requirements, risks, scenario(), [evidence()])
        rows = traceability_rows({
            'analysis_input': scenario(), 'contextual_risks': risks,
            'ai_requirements': output.ai_requirements, 'retrieved_evidence': [evidence()],
        })
        edges = {(row['source_id'], row['target_id']) for row in rows
                 if row['relation'] == 'addressed by'}
        self.assertEqual(edges, {('R-01', 'REQ-01'), ('R-01', 'REQ-02'), ('R-02', 'REQ-02')})
