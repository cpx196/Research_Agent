import unittest

from agent.graph import LangGraphResearchAgent
from agent.llm import DemoLLM


class StreamingDemoLLM(DemoLLM):
    def __init__(self) -> None:
        self.streamed: list[str] = []

    def chat_stream(self, messages, tools=None):
        for token in super().chat_stream(messages, tools):
            self.streamed.append(token)
            yield token


class StreamingTests(unittest.TestCase):
    def test_writer_emits_incremental_tokens(self) -> None:
        llm = StreamingDemoLLM()
        tokens: list[str] = []
        agent = LangGraphResearchAgent(llm, verbose=False, token_callback=tokens.append)
        answer = agent.run("你好")
        self.assertTrue(tokens)
        self.assertEqual("".join(tokens).strip(), answer)
        self.assertEqual("".join(llm.streamed).strip(), answer)


if __name__ == "__main__":
    unittest.main()
