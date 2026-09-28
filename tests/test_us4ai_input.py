"""Tests for the future US4AI analysis input contract."""

import unittest

from pydantic import ValidationError

from src.schemas import AITask, US4AIAnalysisInput


class US4AIAnalysisInputTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "system_purpose": "Help users organize documents.",
            "user_story": "As a user, I want documents grouped by subject.",
            "ai_tasks": [{"category": "Custom category", "task": "Custom task"}],
        }

    def test_valid_input_without_acceptance_criteria_and_with_one_task(self):
        result = US4AIAnalysisInput.model_validate(self.data)
        self.assertEqual(result.acceptance_criteria, [])
        self.assertEqual(len(result.ai_tasks), 1)
        self.assertIsInstance(result.ai_tasks[0], AITask)
        self.assertEqual(result.ai_tasks[0].category, "Custom category")
        self.assertEqual(result.ai_tasks[0].task, "Custom task")
        self.data["acceptance_criteria"] = []
        self.assertEqual(US4AIAnalysisInput.model_validate(self.data), result)

    def test_multiple_criteria_and_tasks_preserve_content_and_order(self):
        self.data["acceptance_criteria"] = ["Show the groups.", "Allow user corrections."]
        self.data["ai_tasks"].append({"category": "Another category", "task": "Another task"})
        self.data["system_purpose"] = "  Preserve surrounding whitespace.  "
        result = US4AIAnalysisInput.model_validate(self.data)
        self.assertEqual(result.model_dump(), self.data)

    def test_empty_purpose_and_story_are_rejected(self):
        for field in ("system_purpose", "user_story"):
            for value in ("", " \n\t", None):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValidationError):
                        US4AIAnalysisInput.model_validate({**self.data, field: value})

    def test_missing_required_fields_are_rejected(self):
        for field in ("system_purpose", "user_story", "ai_tasks"):
            with self.subTest(field=field):
                data = {key: value for key, value in self.data.items() if key != field}
                with self.assertRaises(ValidationError):
                    US4AIAnalysisInput.model_validate(data)

    def test_zero_ai_tasks_is_rejected(self):
        with self.assertRaises(ValidationError):
            US4AIAnalysisInput.model_validate({**self.data, "ai_tasks": []})

    def test_task_structure_and_list_types_are_validated(self):
        for value in ([{"task": "A task"}], [{"category": "A category"}],
                      [{"category": "", "task": "A task"}], ["A task"], "A task"):
            with self.subTest(ai_tasks=value):
                with self.assertRaises(ValidationError):
                    US4AIAnalysisInput.model_validate({**self.data, "ai_tasks": value})
        for value in ("A criterion", [123], None):
            with self.subTest(acceptance_criteria=value):
                with self.assertRaises(ValidationError):
                    US4AIAnalysisInput.model_validate({**self.data, "acceptance_criteria": value})


if __name__ == "__main__":
    unittest.main()
