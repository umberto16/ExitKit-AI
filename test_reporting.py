"""Checks for the short reader-facing report."""

import json
import unittest
from unittest.mock import patch

import requests

from core import EnglishOutputError, _english_prose, _validate, analyze
from reporting import build_markdown, build_pdf
from streamlit.testing.v1 import AppTest


def example_data():
    return {
        "summary": "Передача проекта по переносу клиентов.",
        "projects": [{"name": "CRM Migration", "status": "идёт тестирование", "background": "80% записей загружено", "next_steps": "завершить импорт", "decisions": "", "source": "example.txt"}],
        "tasks": [{"task": "Проверить дубликаты клиентов", "owner": "Игорь Петров", "deadline": "15.11.2026", "status": "в работе", "details": "проверить записи в тестовой среде", "source": "example.txt"}],
        "risks": [{"risk": "часть карточек без адреса электронной почты", "impact": "", "mitigation": "согласовать правило обработки с отделом продаж", "source": "example.txt"}],
        "contacts": [{"name": "Мария Смирнова", "role": "руководитель продаж", "context": "согласует запуск", "source": "example.txt"}],
        "instructions": [],
        "agreements": [],
        "open_questions": ["Кто будет поддерживать интеграцию?"],
    }


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.api_key = "private-api-key-for-test"
        self.source = "Конфиденциальный текст документа"
        self.filename = "private-document.txt"
        environment = patch.dict("os.environ", {"OPENROUTER_API_KEY": self.api_key})
        environment.start()
        self.addCleanup(environment.stop)
        request = patch("core.requests.post")
        self.post = request.start()
        self.addCleanup(request.stop)
        self.response = self.post.return_value
        self.response.status_code = 200
        self.envelope = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": json.dumps(example_data())},
            }],
        }
        self.response.json.return_value = self.envelope

    def assert_sanitized_error(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern) as raised:
            analyze(self.source, self.filename)
        for private_value in (self.api_key, self.source, self.filename):
            self.assertNotIn(private_value, str(raised.exception))
        self.assertLessEqual(self.post.call_count, 1)

    def test_explicit_source_question_survives_unknown_model_question(self):
        data = example_data()
        data["open_questions"] = ["неизвестно"]
        self.envelope["choices"][0]["message"]["content"] = json.dumps(data)
        result = analyze("Проект CRM: проверить дубликаты клиентов.\nОткрытый вопрос: кто отвечает за интеграцию?", "example.txt")
        self.assertEqual(result["open_questions"], ["кто отвечает за интеграцию?"])
        self.assertEqual(result["tasks"][0]["source"], "example.txt")

    def test_repeated_english_field_does_not_discard_russian_report(self):
        data = example_data()
        data["tasks"][0]["status"] = "In progress"
        self.envelope["choices"][0]["message"]["content"] = json.dumps(data)
        result = analyze("Задача: проверить дубликаты клиентов", "example.txt")
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(result["tasks"][0]["status"], "Неизвестно")
        self.assertTrue(result["open_questions"])
        self.assertEqual(result["projects"][0]["background"], data["projects"][0]["background"])
        self.assertNotIn("In progress", build_markdown(result, ["example.txt"]))

    def test_missing_or_blank_key_fails_before_sending_document(self):
        for value in ({}, {"OPENROUTER_API_KEY": "  "}):
            with self.subTest(value=value), patch.dict("os.environ", value, clear=True):
                self.assert_sanitized_error("OPENROUTER_API_KEY")
        self.post.assert_not_called()

    def test_http_failures_are_actionable_without_exposing_response(self):
        for status, category in ((401, "авторизац"), (403, "авторизац"), (402, "квот"), (429, "квот"), (404, "маршрут"), (502, "недоступна"), (503, "недоступна")):
            with self.subTest(status=status):
                self.post.reset_mock()
                self.response.status_code = status
                self.response.text = f"{self.api_key} {self.source} {self.filename}"
                self.assert_sanitized_error(category)

    def test_error_envelope_is_not_treated_as_a_report(self):
        for code, category in ((429, "квот"), (401, "авторизац"), (503, "недоступна")):
            with self.subTest(code=code):
                self.post.reset_mock()
                self.response.json.return_value = {
                    "error": {"code": code, "message": f"{self.api_key} {self.source} {self.filename}"},
                }
                self.assert_sanitized_error(category)

    def test_network_errors_do_not_expose_request_details(self):
        for error, category in ((requests.Timeout, "180 секунд"), (requests.ConnectionError, "связаться")):
            with self.subTest(error=error):
                self.post.reset_mock()
                self.post.side_effect = error(f"{self.api_key} {self.source} {self.filename}")
                self.assert_sanitized_error(category)

    def test_invalid_transport_json_does_not_expose_response(self):
        self.response.json.side_effect = ValueError(f"{self.api_key} {self.source} {self.filename}")
        self.assert_sanitized_error("JSON")

    def test_malformed_envelopes_fail_without_another_request(self):
        for envelope in (None, [], {}, {"choices": []}, {"choices": [None]}, {"choices": [{"message": None, "finish_reason": "stop"}]}):
            with self.subTest(envelope=envelope):
                self.post.reset_mock()
                self.response.json.return_value = envelope
                self.assert_sanitized_error("формат")

    def test_empty_content_is_not_a_report(self):
        for content in (None, "", "  ", []):
            with self.subTest(content=content):
                self.post.reset_mock()
                self.envelope["choices"][0]["message"]["content"] = content
                self.assert_sanitized_error("пустой ответ")

    def test_malformed_report_json_is_sanitized(self):
        self.envelope["choices"][0]["message"]["content"] = f"{self.api_key} {self.source} {self.filename}"
        self.assert_sanitized_error("JSON")

    def test_truncated_response_is_rejected_even_if_json_is_valid(self):
        self.envelope["choices"][0]["finish_reason"] = "length"
        self.assert_sanitized_error("обрезан")

    def test_refused_response_is_rejected_even_if_json_is_valid(self):
        for finish_reason, refusal in (("content_filter", None), ("stop", f"{self.api_key} {self.source} {self.filename}")):
            with self.subTest(finish_reason=finish_reason):
                self.post.reset_mock()
                self.envelope["choices"][0]["finish_reason"] = finish_reason
                self.envelope["choices"][0]["message"]["refusal"] = refusal
                self.assert_sanitized_error("отказалась")

    def test_incomplete_response_is_not_accepted_as_a_report(self):
        self.envelope["choices"][0]["finish_reason"] = "error"
        self.assert_sanitized_error("не завершила")

    def test_non_object_or_empty_report_is_rejected(self):
        for data, category in (([], "объекта"), ({}, "пустые разделы")):
            with self.subTest(data=data):
                self.post.reset_mock()
                self.envelope["choices"][0]["message"]["content"] = json.dumps(data)
                self.assert_sanitized_error(category)


class ReportingTests(unittest.TestCase):

    def test_short_russian_report_and_pdf(self):
        data = example_data()
        markdown = build_markdown(data, ["example.txt"])
        self.assertIn("## Текущее состояние", markdown)
        self.assertIn("## Что делать дальше", markdown)
        self.assertIn("## На что обратить внимание", markdown)
        self.assertIn("## Что нужно уточнить", markdown)
        self.assertIn("Игорь Петров", markdown)
        self.assertNotIn("Handover Package", markdown)
        self.assertNotIn("###", markdown)
        self.assertTrue(build_pdf(data, ["example.txt"]).startswith(b"%PDF"))

    def test_english_model_prose_is_rejected(self):
        data = example_data()
        data["tasks"][0]["status"] = "In progress"
        with self.assertRaises(EnglishOutputError):
            _validate(data, "example.txt")

    def test_source_filename_does_not_count_as_english_prose(self):
        data = example_data()
        data["summary"] = "Документ: employee_handover (1).txt"
        self.assertEqual(_validate(data, "employee_handover (1).txt")["summary"], data["summary"])
        self.assertFalse(_english_prose("Документ CRM Migration Guide"))


    def test_explicitly_unassigned_task_has_no_invented_owner(self):
        data = example_data()
        data["tasks"][0]["owner"] = "Руководитель продаж"
        data["tasks"][0]["status"] = "ответственный пока не назначен"
        self.assertEqual(_validate(data, "example.txt")["tasks"][0]["owner"], "Не назначен")

    def test_named_task_owner_takes_precedence(self):
        data = example_data()
        data["tasks"][0]["task"] = "Игорь Петров должен проверить дубликаты"
        data["tasks"][0]["owner"] = "Не назначен"
        self.assertEqual(_validate(data, "example.txt")["tasks"][0]["owner"], "Игорь Петров")

    def test_old_english_prose_is_hidden_from_report(self):
        data = example_data()
        data["projects"][0]["background"] = "The project is waiting for customer approval"
        markdown = build_markdown(data, ["example.txt"])
        self.assertNotIn("customer approval", markdown)

    def test_existing_session_report_is_rebuilt(self):
        app = AppTest.from_file("app.py")
        app.session_state["data"] = example_data()
        app.session_state["filenames"] = ["example.txt"]
        app.session_state["report"] = "Handover Package"
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("## Текущее состояние", app.session_state["report"])
        self.assertNotIn("Handover Package", app.session_state["report"])


if __name__ == "__main__":
    unittest.main()
