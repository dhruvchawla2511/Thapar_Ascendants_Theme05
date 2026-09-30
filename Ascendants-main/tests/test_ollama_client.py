import io
import json
import os
import unittest
import urllib.error
from unittest import mock

from agent.llm_client import (
    LLMModelNotFoundError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    OllamaClient,
    normalize_host,
)


def fake_response(body):
    if not isinstance(body, str):
        body = json.dumps(body)
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = body.encode()
    return resp


def http_error(code, body):
    return urllib.error.HTTPError(
        "http://x", code, "err", {}, io.BytesIO(body.encode())
    )


PATCH = "urllib.request.urlopen"


class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient(host="127.0.0.1:11434", model="qwen2.5:14b")

    def test_host_normalization(self):
        self.assertEqual(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434")
        self.assertEqual(normalize_host("http://h:1/"), "http://h:1")
        self.assertEqual(normalize_host(""), "http://127.0.0.1:11434")

    def test_from_env_defaults_and_overrides(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = OllamaClient.from_env()
            self.assertEqual(c.model, "qwen2.5:14b")
            self.assertEqual(c.host, "http://127.0.0.1:11434")
        env = {"OLLAMA_HOST": "box:9", "OLLAMA_MODEL": "m", "OLLAMA_TIMEOUT": "5", "OLLAMA_NUM_CTX": "abc"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = OllamaClient.from_env()
            self.assertEqual((c.host, c.model, c.timeout), ("http://box:9", "m", 5.0))
            self.assertEqual(c.num_ctx, 8192)  # invalid value falls back

    def test_chat_success_and_payload(self):
        with mock.patch(PATCH, return_value=fake_response({"message": {"content": "hi"}})) as m:
            out = self.client.chat([{"role": "user", "content": "x"}], json_mode=True)
        self.assertEqual(out, "hi")
        request = m.call_args[0][0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen2.5:14b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")
        self.assertTrue(request.full_url.endswith("/api/chat"))

    def test_no_format_when_not_json_mode(self):
        with mock.patch(PATCH, return_value=fake_response({"message": {"content": "hi"}})) as m:
            self.client.chat([{"role": "user", "content": "x"}])
        self.assertNotIn("format", json.loads(m.call_args[0][0].data))

    def test_connection_refused(self):
        err = urllib.error.URLError(ConnectionRefusedError("refused"))
        with mock.patch(PATCH, side_effect=err):
            with self.assertRaises(LLMUnavailableError):
                self.client.chat([])

    def test_timeout_variants(self):
        for exc in (TimeoutError(), urllib.error.URLError(TimeoutError())):
            with mock.patch(PATCH, side_effect=exc):
                with self.assertRaises(LLMTimeoutError):
                    self.client.chat([])

    def test_model_not_found_http_404(self):
        with mock.patch(PATCH, side_effect=http_error(404, '{"error":"model \'x\' not found"}')):
            with self.assertRaises(LLMModelNotFoundError) as ctx:
                self.client.chat([])
        self.assertIn("ollama pull", str(ctx.exception))

    def test_other_http_error(self):
        with mock.patch(PATCH, side_effect=http_error(500, '{"error":"boom"}')):
            with self.assertRaises(LLMResponseError):
                self.client.chat([])

    def test_malformed_json(self):
        with mock.patch(PATCH, return_value=fake_response("not json")):
            with self.assertRaises(LLMResponseError):
                self.client.chat([])

    def test_missing_content(self):
        with mock.patch(PATCH, return_value=fake_response({"message": {}})):
            with self.assertRaises(LLMResponseError):
                self.client.chat([])

    def test_error_field_in_body(self):
        with mock.patch(PATCH, return_value=fake_response({"error": "model 'q' not found"})):
            with self.assertRaises(LLMModelNotFoundError):
                self.client.chat([])

    def test_availability_helpers_never_raise(self):
        with mock.patch(PATCH, side_effect=urllib.error.URLError("down")):
            self.assertFalse(self.client.is_available())
            self.assertFalse(self.client.model_installed())
        tags = {"models": [{"name": "qwen2.5:14b"}]}
        with mock.patch(PATCH, return_value=fake_response(tags)):
            self.assertTrue(self.client.is_available())
            self.assertTrue(self.client.model_installed())
        with mock.patch(PATCH, return_value=fake_response({"models": [{"name": "other:1b"}]})):
            self.assertFalse(self.client.model_installed())


if __name__ == "__main__":
    unittest.main()
