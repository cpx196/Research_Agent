"""Small OpenAI-compatible LLM wrapper.

The wrapper intentionally returns the provider's decoded JSON response. The
agent owns message history and tool-loop decisions; this module only handles
HTTP transport and configuration.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - requirements.txt installs it
    load_dotenv = None  # type: ignore[assignment]


class LLMClientError(RuntimeError):
    """Raised when the LLM cannot be configured or contacted."""


class LLMClient:
    """HTTP client for an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMClientError("LLM_API_KEY is empty.")
        if not base_url.strip():
            raise LLMClientError("LLM_BASE_URL is empty.")
        if not model.strip():
            raise LLMClientError("LLM_MODEL is empty.")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Build a client from ``.env`` and process environment variables."""

        if load_dotenv is not None:
            load_dotenv()

        # Prefer DashScope's documented variable, while retaining the generic
        # variable for backwards compatibility with the original V0 scaffold.
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY", "")
        base_url = os.getenv(
            "LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        model = os.getenv("LLM_MODEL", "qwen3.8-flash")
        timeout_raw = os.getenv("LLM_TIMEOUT", "60")
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise LLMClientError("LLM_TIMEOUT must be a number.") from exc
        return cls(api_key, base_url, model, timeout=timeout)

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Call the model and return its raw decoded JSON response."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
        }
        if tools:
            payload["tools"] = list(tools)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self.session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise LLMClientError("LLM request timed out.") from exc
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", "")
            detail = detail[:300].strip()
            suffix = f": {detail}" if detail else ""
            raise LLMClientError(f"LLM request failed{suffix}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError("LLM returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise LLMClientError("LLM returned an unexpected response shape.")
        return data

    def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ):
        """Yield assistant content deltas from an OpenAI-compatible SSE API."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
        }
        if tools:
            payload["tools"] = list(tools)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        try:
            response = self.session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise LLMClientError("LLM streaming request timed out.") from exc
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", "")[:300].strip()
            suffix = f": {detail}" if detail else ""
            raise LLMClientError(f"LLM streaming request failed{suffix}") from exc

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError as exc:
                    raise LLMClientError("LLM returned invalid streaming JSON.") from exc
                choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content") if isinstance(delta, Mapping) else None
                if content:
                    yield str(content)
        finally:
            response.close()


class DemoLLM:
    """Deterministic local model used to exercise the full loop without a key.

    This is deliberately a demo adapter, not a replacement for a real LLM.
    It makes the CLI and local RAG smoke tests runnable without an API key.
    """

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del tools
        user_query = ""
        called: set[str] = set()
        observations: list[str] = []
        for message in messages:
            role = message.get("role")
            if role == "user":
                user_query = str(message.get("content", ""))
            if role == "assistant":
                for tool_call in message.get("tool_calls", []) or []:
                    function = tool_call.get("function", {})
                    called.add(str(function.get("name", "")))
            if role == "tool":
                observations.append(str(message.get("content", "")))

        query_lower = user_query.lower()
        english_words = set(query_lower.replace("!", " ").replace("?", " ").split())
        if (
            any(token in query_lower for token in ("联网", "网络搜索", "web access", "internet access"))
            and any(token in query_lower for token in ("能", "不能", "可以", "无法", "能力", "can", "why"))
        ):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "可以使用 web_search 工具检索实时网络信息。",
                        }
                    }
                ]
            }
        if any(token in user_query for token in ("你好", "您好")) or english_words.intersection({"hello", "hi"}):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "你好！我是 Research Agent，可以帮你检索论文、查询本地资料或完成计算。",
                        }
                    }
                ]
            }
        if any(token in user_query for token in ("计算", "算一下")) or "3947" in query_lower:
            if "calculator" not in called:
                return self._tool_call(
                    "calculator",
                    {"expression": "3947 * 8123"},
                    "demo-calculation",
                )

        is_local_query = any(
            token in user_query.lower()
            for token in ("本地", "local", "ust", "本地资料", "本地论文")
        )
        if is_local_query and "local_search" not in called:
            return self._tool_call(
                "local_search",
                {"query": user_query, "top_k": 5},
                "demo-local-search",
            )

        is_paper_query = any(token in user_query for token in ("论文", "paper", "文献"))
        is_investigation = "调查" in user_query or "总结" in user_query
        if is_paper_query and not is_local_query and "paper_search" not in called:
            max_results = 3 if "3" in user_query else 5
            return self._tool_call(
                "paper_search",
                {"query": "V-JEPA 2", "max_results": max_results},
                "demo-paper-search",
            )

        needs_web = any(
            token in user_query
            for token in ("开源", "代码", "checkpoint", "官方", "最新", "实时", "搜索", "是谁", "谁是")
        ) or any(token in query_lower for token in ("latest", "official", "search", "who is", "current"))
        if (is_investigation or needs_web) and "web_search" not in called:
            return self._tool_call(
                "web_search",
                {"query": user_query, "max_results": 5},
                "demo-web-search",
            )

        if observations:
            if "计算" in user_query or "3947" in query_lower:
                content = f"计算结果是：{observations[-1]}"
            elif is_investigation:
                content = "已完成论文与网页两步检索；请依据上方 Observation 总结发布时间、代码和 checkpoint 信息。"
            elif is_local_query:
                if "数据" in user_query:
                    content = (
                        "根据 vjepa2.pdf 第 8–9 页，V-JEPA 2-AC 使用 Droid 数据集：约 62 小时的无标注、"
                        "通过远程操作采集的桌面 Franka Emika Panda 机械臂视频，并使用末端执行器状态信号。"
                    )
                elif "训练目标" in user_query:
                    content = (
                        "根据 vjepa2.pdf 第 4 页，V-JEPA 2 用 predictor 预测被遮挡 patch 的表征，"
                        "再以 EMA encoder 生成的目标表征计算 L1 loss；V-JEPA 2-AC 则在冻结的视频 encoder 上，"
                        "按过去视频、action 和末端状态自回归预测未来表征。"
                    )
                elif "action" in query_lower or "使用" in user_query:
                    content = (
                        "根据 vjepa2.pdf 第 4、8 页，V-JEPA 2-AC 将 action 与视频/末端状态按时间交错输入"
                        " frame-causal predictor，自回归预测未来视频表征，再放入 model-predictive control loop 规划动作。"
                    )
                else:
                    content = (
                        "已完成本地论文检索；回答依据见 Observation 中的 Source、Page 和 Text，"
                        "资料库没有证据的部分不会编造。"
                    )
            elif is_paper_query:
                content = "已完成论文检索，结果详见上方 paper_search Observation。"
            else:
                content = "已完成工具检索。"
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Transformer 是一种以自注意力机制为核心的神经网络架构，"
                            "能够并行处理序列中的各个位置，并建模长距离依赖。"
                        ),
                    }
                }
            ]
        }

    def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ):
        """Provide the same incremental interface for deterministic demos."""

        response = self.chat(messages=messages, tools=tools)
        message = response.get("choices", [{}])[0].get("message", {})
        content = str(message.get("content") or "")
        for offset in range(0, len(content), 8):
            yield content[offset:offset + 8]

    @staticmethod
    def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
        import json

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                }
            ]
        }
