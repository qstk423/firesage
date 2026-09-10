import io
import json
import os
import sys
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from rag.llm import LLMClient


class _StreamResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._chunks)


class LLMClientTest(unittest.TestCase):
    def _client(self, **extra):
        env = {
            "LLM_BASE_URL": "http://127.0.0.1:8320/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            **extra,
        }
        with patch.dict(os.environ, env, clear=True):
            return LLMClient()

    def test_fast_defaults(self):
        client = self._client()
        self.assertEqual(client.max_tokens, 480)
        self.assertEqual(client.temperature, 0)

    def test_stream_collects_final_metrics(self):
        client = self._client(LLM_STREAM="1")
        final = {
            "choices": [{"delta": {}}],
            "final_content": '{"conclusion":"ok"}',
            "firesage_protections": ["citation_guard"],
            "ttft_ms": 321,
            "latency_ms": 1234,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
        chunks = [
            b'data: {"choices":[{"delta":{"content":"raw"}}]}\n',
            ("data: " + json.dumps(final) + "\n").encode(),
            b"data: [DONE]\n",
        ]
        with patch("urllib.request.urlopen", return_value=_StreamResponse(chunks)):
            result = client.complete("system", "user")
        self.assertEqual(result, '{"conclusion":"ok"}')
        self.assertEqual(client.last_ttft_ms, 321)
        self.assertEqual(client.last_model_latency_ms, 1234)
        self.assertEqual(client.last_usage["completion_tokens"], 20)
        self.assertEqual(client.last_protections, ["citation_guard"])


if __name__ == "__main__":
    unittest.main()
