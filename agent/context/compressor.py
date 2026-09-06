"""Rule-based Tool Result compression for V4."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .budget import estimate_tokens


@dataclass
class CompressionResult:
    text: str
    raw_tokens: int
    final_tokens: int
    compressed: bool
    dropped_fields: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        if self.raw_tokens <= 0:
            return 0.0
        return round(1.0 - self.final_tokens / self.raw_tokens, 4)


class ToolResultCompressor:
    """Preserve citation-bearing fields while shrinking raw tool output."""

    _local_result = re.compile(
        r"(?P<head>\[Result\s+\d+\].*?\nText:\s*)(?P<body>.*?)(?=\n\n\[Result\s+\d+\]|\Z)",
        re.DOTALL,
    )

    def __init__(
        self,
        threshold_tokens: int = 2500,
        max_chars: int = 12000,
        max_evidence_chars: int = 1800,
        extractor_llm: Any | None = None,
        llm_extractor_enabled: bool = False,
    ) -> None:
        self.threshold_tokens = max(1, threshold_tokens)
        self.max_chars = max(256, max_chars)
        self.max_evidence_chars = max(128, max_evidence_chars)
        self.extractor_llm = extractor_llm
        self.llm_extractor_enabled = llm_extractor_enabled

    def compress_tool_result(
        self,
        result: str,
        tool_name: str = "tool",
        query: str = "",
    ) -> CompressionResult:
        raw = str(result or "")
        raw_tokens = estimate_tokens(raw)
        if raw_tokens <= self.threshold_tokens and len(raw) <= self.max_chars:
            return CompressionResult(raw, raw_tokens, raw_tokens, False)

        if tool_name == "local_search" and self._local_result.search(raw):
            compressed = self._compress_local(raw)
        else:
            compressed = self._compress_generic(raw)
        compressed = compressed[: self.max_chars].rstrip()
        if self.llm_extractor_enabled and self.extractor_llm is not None:
            extracted = self._extract_with_llm(raw, tool_name, query)
            if extracted:
                compressed = extracted[: self.max_chars].rstrip()
        return CompressionResult(
            text=compressed,
            raw_tokens=raw_tokens,
            final_tokens=estimate_tokens(compressed),
            compressed=True,
            dropped_fields=["verbose_tool_status", "overlong_result_text"],
        )

    def _extract_with_llm(self, raw: str, tool_name: str, query: str) -> str:
        """Optionally run Level-2 evidence extraction for very large results."""

        prompt = (
            "Extract only evidence relevant to the current research step. Do not answer the user, "
            "add outside knowledge, or change source/page/URL. Preserve citation fields. "
            "Return concise evidence bullets.\n\n"
            f"Current research step: {query}\nTool: {tool_name}\nRaw Tool Result:\n{raw}"
        )
        try:
            response = self.extractor_llm.chat(
                messages=[
                    {"role": "system", "content": "You extract faithful evidence records."},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
            )
            message: Mapping[str, Any] | None = None
            if isinstance(response, Mapping):
                choices = response.get("choices", [])
                if choices and isinstance(choices[0], Mapping):
                    candidate = choices[0].get("message")
                    if isinstance(candidate, Mapping):
                        message = candidate
            content = str(message.get("content", "") if message else "").strip()
            if not content or content.startswith("ToolError:"):
                return ""
            # For local evidence, do not accept an extraction that drops both
            # citation anchors; the rule-based result remains the safe fallback.
            if tool_name == "local_search" and not ("Source:" in content and "Page:" in content):
                return ""
            return content
        except Exception:
            return ""

    def _compress_local(self, raw: str) -> str:
        headers = []
        for line in raw.splitlines():
            if line.startswith("[RAG]"):
                continue
            if line.startswith("[Result "):
                headers.append(line)
        parts: list[str] = []
        for match in self._local_result.finditer(raw):
            body = " ".join(match.group("body").split())
            parts.append(match.group("head") + body[: self.max_evidence_chars].rstrip())
        return "\n\n".join(parts) if parts else self._compress_generic(raw)

    def _compress_generic(self, raw: str) -> str:
        lines: list[str] = []
        for line in raw.splitlines():
            clean = " ".join(line.split())
            if not clean:
                continue
            # Keep citation-bearing key/value lines and ordinary evidence text;
            # omit repeated RAG progress/status lines.
            if clean.startswith("[RAG]"):
                continue
            lines.append(clean)
        return "\n".join(lines)[: self.max_chars].rstrip()
