import unittest

from agent.llm import LLMClient


class FakeStreamingResponse:
    def __init__(self) -> None:
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode=True):
        del decode_unicode
        yield 'data: {"choices":[{"delta":{"content":"你"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"好"}}]}'
        yield "data: [DONE]"

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self) -> None:
        self.payload = None
        self.response = FakeStreamingResponse()

    def post(self, *args, **kwargs):
        del args
        self.payload = kwargs
        return self.response


class LLMStreamingTests(unittest.TestCase):
    def test_openai_compatible_sse_deltas_are_yielded(self) -> None:
        session = FakeSession()
        client = LLMClient("key", "https://example.test/v1", "model", session=session)
        tokens = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(tokens, ["你", "好"])
        self.assertTrue(session.payload["json"]["stream"])
        self.assertTrue(session.payload["stream"])
        self.assertTrue(session.response.closed)


if __name__ == "__main__":
    unittest.main()
