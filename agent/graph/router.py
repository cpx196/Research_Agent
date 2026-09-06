"""Hybrid semantic query resolver and route classifier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Literal


Route = Literal["direct", "react", "research"]
Complexity = Literal["simple", "tool", "research"]


_KNOWN_TOPICS = {
    "v-jepa 2": "V-JEPA 2",
    "v-jepa": "V-JEPA",
    "jepa-wam": "JEPA-WAM",
    "jepa": "JEPA",
    "dinov3": "DINOv3",
    "dino v3": "DINOv3",
    "dino": "DINO",
    "patch policy": "Patch Policy",
    "transformer": "Transformer",
    "masked autoencoder": "MAE",
    "masked autoencoders": "MAE",
    "mae encoder": "MAE",
    "danking": "DANK1NG",
}
_TOPIC_EXPANSIONS = {
    "JEPA": "JEPA Joint Embedding Predictive Architecture",
    "V-JEPA": "V-JEPA Video Joint Embedding Predictive Architecture",
    "V-JEPA 2": "V-JEPA 2 video world model",
    "JEPA-WAM": "JEPA-WAM world action model",
    "DINO": "DINO self-supervised visual representation",
    "DINOv3": "DINOv3 self-supervised visual representation",
    "Patch Policy": "Patch Policy robot learning",
    "DANK1NG": "DANK1NG",
    "MAE": "Masked Autoencoders Are Scalable Vision Learners MAE encoder",
}
_LOCAL_LIBRARY_TOPIC_MARKERS = (
    "jepa", "v-jepa", "dino", "world model", "robot", "patch policy",
)
_LATIN_STOPWORDS = {
    "about", "answer", "are", "can", "could", "find", "for", "from", "have",
    "information", "latest", "learn", "look", "more", "paper", "papers", "please",
    "related", "research", "search", "show", "some", "tell", "that", "the", "these",
    "this", "those", "want", "what", "which", "who", "with", "would", "you",
}


@dataclass(frozen=True)
class RouterDecision:
    """Validated routing and query-resolution output used by the graph."""

    route: Route
    complexity: Complexity
    need_planning: bool
    need_web: bool
    need_local_rag: bool
    need_verification: bool
    suggested_workers: int
    confidence: float
    source: str = "rules"
    allowed_tools: tuple[str, ...] = ()
    original_query: str = ""
    standalone_query: str = ""
    topic: str = ""
    intent: str = "general"
    primary_tool: str = ""
    fallback_tools: tuple[str, ...] = ()
    search_queries: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["allowed_tools"] = list(self.allowed_tools)
        result["fallback_tools"] = list(self.fallback_tools)
        return result


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        content = value.get("content")
        if content is not None:
            return str(content)
    choices = value.get("choices", []) if isinstance(value, Mapping) else []
    if not choices or not isinstance(choices[0], Mapping):
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", "") or "") if isinstance(message, Mapping) else ""


def _json_object(content: str) -> Mapping[str, Any] | None:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, Mapping) else None


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "yes", "1"}:
            return True
        if value.strip().lower() in {"false", "no", "0"}:
            return False
    return default


def _number(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _workers(value: Any, default: int) -> int:
    try:
        return max(0, min(int(value), 8))
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    result: list[str] = []
    for item in value:
        name = str(item).strip()
        if name and name not in result:
            result.append(name)
    return tuple(result)


def _search_query_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key).strip(): str(item).strip()
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


def validate_router_output(payload: Mapping[str, Any], *, source: str = "llm") -> RouterDecision | None:
    """Validate and normalize an LLM-produced semantic routing object."""

    route = str(payload.get("route", "")).strip().lower()
    complexity = str(payload.get("complexity", "")).strip().lower()
    if route not in {"direct", "react", "research"}:
        return None
    if complexity not in {"simple", "tool", "research"}:
        complexity = {"direct": "simple", "react": "tool", "research": "research"}[route]

    need_planning = _bool(payload.get("need_planning"), route == "research")
    need_web = _bool(payload.get("need_web"))
    need_local = _bool(payload.get("need_local_rag"))
    need_verification = _bool(payload.get("need_verification"), route == "research")
    workers_default = 0 if route == "direct" else 1 if route == "react" else 3
    workers = _workers(payload.get("suggested_workers"), workers_default)

    if route == "direct" and (need_web or need_local or need_planning or need_verification):
        route = "research" if need_planning else "react"
        complexity = "research" if route == "research" else "tool"
        workers = max(workers, 3 if route == "research" else 1)
    if route == "react" and (need_planning or need_verification):
        route = "research"
        complexity = "research"
        need_planning = True
        need_verification = True
        workers = max(workers, 1)
    if route == "research":
        need_planning = True
        need_verification = True
        workers = max(workers, 1)
    if route == "direct":
        workers = 0
    elif workers == 0:
        workers = 1

    return RouterDecision(
        route=route,  # type: ignore[arg-type]
        complexity=complexity,  # type: ignore[arg-type]
        need_planning=need_planning,
        need_web=need_web,
        need_local_rag=need_local,
        need_verification=need_verification,
        suggested_workers=workers,
        confidence=_number(payload.get("confidence"), 0.6),
        source=source,
        allowed_tools=_string_list(payload.get("allowed_tools")),
        original_query=str(payload.get("original_query", "")).strip(),
        standalone_query=str(payload.get("standalone_query", "")).strip(),
        topic=str(payload.get("topic", "")).strip(),
        intent=str(payload.get("intent", "general")).strip().lower() or "general",
        primary_tool=str(payload.get("primary_tool", "")).strip(),
        fallback_tools=_string_list(payload.get("fallback_tools")),
        search_queries=_search_query_map(payload.get("search_queries")),
    )


def _canonical_topic(value: str) -> str:
    normalized = " ".join(value.strip().split())
    lowered = normalized.lower()
    dino = "DINOv3" if any(token in lowered for token in ("dinov3", "dino v3")) else (
        "DINO" if "dino" in lowered else ""
    )
    jepa = "V-JEPA 2" if "v-jepa 2" in lowered else (
        "V-JEPA" if "v-jepa" in lowered else "JEPA" if "jepa" in lowered else ""
    )
    if dino and jepa:
        return f"{dino} vs {jepa}"
    for marker, canonical in _KNOWN_TOPICS.items():
        if marker in lowered:
            return canonical
    return normalized


def _is_local_library_topic(topic: str) -> bool:
    lowered = topic.lower()
    return any(marker in lowered for marker in _LOCAL_LIBRARY_TOPIC_MARKERS)


def _topic_from_text(text: str) -> str:
    lowered = text.lower()
    combined = _canonical_topic(text)
    if " vs " in combined:
        return combined
    for marker, canonical in _KNOWN_TOPICS.items():
        if marker in lowered:
            return canonical

    person = re.search(r"([\u3400-\u9fff]{2,12})(?:是谁|是哪个人)", text)
    if person:
        return person.group(1)
    learning = re.search(
        r"(?:学习|了解|研究|关于)\s*([\u3400-\u9fff]{2,12}?)(?:相关|方面|的信息|的内容|$)",
        text,
    )
    if learning:
        return learning.group(1)

    candidates = re.findall(r"[A-Za-z][A-Za-z0-9._-]{2,}", text)
    for candidate in candidates:
        if candidate.lower() not in _LATIN_STOPWORDS:
            return _canonical_topic(candidate)
    return ""


def _history_topic(history: Sequence[Mapping[str, str]]) -> str:
    for preferred_role in ("user", "assistant"):
        for item in reversed(history[-12:]):
            if str(item.get("role", "user")) != preferred_role:
                continue
            topic = _topic_from_text(str(item.get("content", "")))
            if topic:
                return topic
    return ""


def _is_follow_up(query: str) -> bool:
    normalized = query.strip().lower()
    markers = (
        "这个", "这些", "那个", "它", "他们", "呢", "继续", "展开", "再找",
        "看看", "相关的", "more", "those", "them", "it", "continue",
    )
    return len(normalized) <= 40 and any(marker in normalized for marker in markers)


def _infer_intent(query: str, proposed: str = "general") -> str:
    normalized = query.lower()
    if (re.search(r"计算(?!机)", normalized) or any(token in normalized for token in ("算一下", "calculate"))) or re.fullmatch(
        r"\s*[\d\s()+\-*/%.]+\s*", query
    ):
        return "calculation"
    if any(token in normalized for token in ("比较", "对比", "分析", "报告", "综述", "compare", "review")):
        return "research"
    if any(token in normalized for token in ("本地论文", "本地资料", "ust论文", "local paper")):
        return "local_research"
    if any(token in normalized for token in ("论文", "文献", "paper", "papers")):
        return "paper_search"
    if any(token in normalized for token in ("是谁", "谁是", "who is", "最新", "实时", "新闻", "官方")):
        return "web_lookup"
    return proposed if proposed not in {"", "unknown"} else "general"


def _resolved_query(original: str, proposed: str, topic: str, intent: str) -> str:
    candidate = " ".join(proposed.strip().split())
    topic_present = bool(topic and topic.lower() in candidate.lower())
    if intent == "paper_search" and topic and (not candidate or not topic_present or _is_follow_up(original)):
        return f"查找并推荐 {topic} 相关核心论文"
    if topic and _is_follow_up(original) and (not candidate or not topic_present):
        return f"围绕 {topic} 回答：{original}"
    return candidate or original


def _tool_policy(
    decision: RouterDecision,
    *,
    query: str,
    topic: str,
    intent: str,
    available_tools: Sequence[str],
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    available = {str(name) for name in available_tools}
    if decision.route == "direct":
        return (), "", ()

    selected: list[str] = []

    def add(name: str) -> None:
        if name in available and name not in selected:
            selected.append(name)

    if intent == "calculation":
        add("calculator")
    elif intent == "paper_search":
        # Known local-library topics always collect local and external paper
        # evidence.  local_search is a first-class source, not a fallback.
        if _is_local_library_topic(topic):
            add("local_search")
        add("paper_search")
        if decision.need_web or not selected:
            add("web_search")
    elif intent == "local_research":
        add("local_search")
        if decision.need_web:
            add("web_search")
    else:
        if _is_local_library_topic(topic):
            add("local_search")
            add("paper_search")
        if decision.need_web:
            add("web_search")
        if decision.need_local_rag:
            add("local_search")
        for name in decision.allowed_tools:
            add(name)

    if not selected:
        for name in decision.allowed_tools:
            add(name)
    if not selected and len(available) == 1:
        add(next(iter(available)))

    preferred = decision.primary_tool if decision.primary_tool in selected else ""
    if not preferred and selected:
        preferred = selected[0]
    fallbacks = [name for name in decision.fallback_tools if name in selected and name != preferred]
    fallbacks.extend(name for name in selected if name != preferred and name not in fallbacks)
    return tuple(selected), preferred, tuple(fallbacks)


def _search_queries(
    decision: RouterDecision,
    *,
    standalone_query: str,
    topic: str,
    intent: str,
    allowed_tools: Sequence[str],
) -> dict[str, str]:
    expanded_topic = _TOPIC_EXPANSIONS.get(topic, topic).strip()
    result: dict[str, str] = {}
    for tool in allowed_tools:
        proposed = str(decision.search_queries.get(tool, "")).strip()
        if topic and proposed and topic.lower() not in proposed.lower():
            proposed = ""
        if proposed:
            result[tool] = proposed
        elif tool == "paper_search":
            result[tool] = expanded_topic or standalone_query
        elif tool == "local_search":
            result[tool] = f"{expanded_topic} {standalone_query}".strip()
        elif tool == "web_search" and intent == "paper_search":
            result[tool] = f"{expanded_topic} research paper official".strip()
        elif tool == "calculator":
            result[tool] = standalone_query
        else:
            result[tool] = standalone_query
    return result


def _finalize_decision(
    decision: RouterDecision,
    query: str,
    *,
    available_tools: Sequence[str],
    conversation_history: Sequence[Mapping[str, str]] | None = None,
) -> RouterDecision:
    history = list(conversation_history or [])
    topic = _canonical_topic(decision.topic) if decision.topic else _topic_from_text(query)
    if not topic and _is_follow_up(query):
        topic = _history_topic(history)
    intent = _infer_intent(query, decision.intent)
    standalone = _resolved_query(query, decision.standalone_query, topic, intent)

    route = decision.route
    complexity = decision.complexity
    need_planning = decision.need_planning
    need_verification = decision.need_verification
    workers = decision.suggested_workers
    if intent in {"paper_search", "local_research"}:
        route = "research"
        complexity = "research"
        need_planning = True
        need_verification = True
        workers = max(workers, 1)
    elif intent == "calculation":
        route = "react"
        complexity = "tool"
        workers = max(workers, 1)

    normalized = RouterDecision(
        route=route,
        complexity=complexity,
        need_planning=need_planning,
        need_web=decision.need_web,
        need_local_rag=decision.need_local_rag,
        need_verification=need_verification,
        suggested_workers=workers,
        confidence=decision.confidence,
        source=decision.source,
        allowed_tools=decision.allowed_tools,
        original_query=query,
        standalone_query=standalone,
        topic=topic,
        intent=intent,
        primary_tool=decision.primary_tool,
        fallback_tools=decision.fallback_tools,
        search_queries=decision.search_queries,
    )
    allowed, primary, fallbacks = _tool_policy(
        normalized,
        query=query,
        topic=topic,
        intent=intent,
        available_tools=available_tools,
    )
    return RouterDecision(
        route=normalized.route,
        complexity=normalized.complexity,
        need_planning=normalized.need_planning,
        need_web=normalized.need_web,
        need_local_rag=normalized.need_local_rag,
        need_verification=normalized.need_verification,
        suggested_workers=normalized.suggested_workers,
        confidence=normalized.confidence,
        source=normalized.source,
        allowed_tools=allowed,
        original_query=query,
        standalone_query=standalone,
        topic=topic,
        intent=intent,
        primary_tool=primary,
        fallback_tools=fallbacks,
        search_queries=_search_queries(
            normalized,
            standalone_query=standalone,
            topic=topic,
            intent=intent,
            allowed_tools=allowed,
        ),
    )


def allowed_tools_for_decision(
    decision: RouterDecision,
    query: str,
    *,
    available_tools: Sequence[str],
) -> tuple[str, ...]:
    """Compatibility helper retained for callers of the previous Router API."""

    return _finalize_decision(
        decision,
        query,
        available_tools=available_tools,
    ).allowed_tools


def _rule_decision(query: str, *, available_tools: Sequence[str]) -> RouterDecision | None:
    """Handle only high-confidence, unambiguous cases."""

    normalized = query.strip().lower()
    if not normalized:
        return RouterDecision("direct", "simple", False, False, False, False, 0, 1.0)

    greetings = ("你好", "您好", "谢谢", "感谢", "你是谁", "你是什么模型")
    if any(token in normalized for token in greetings) or re.fullmatch(r"(?:hello|hi)", normalized):
        return RouterDecision("direct", "simple", False, False, False, False, 0, 0.99)

    capability_question = (
        any(token in normalized for token in ("联网", "网络搜索", "web access", "internet access"))
        and any(token in normalized for token in ("能", "不能", "可以", "无法", "能力", "can", "why"))
    )
    if capability_question:
        return RouterDecision("direct", "simple", False, False, False, False, 0, 0.98)

    has_calculation_word = bool(re.search(r"计算(?!机)", normalized)) or any(
        token in normalized for token in ("算一下", "calculate")
    )
    arithmetic_expression = bool(re.fullmatch(r"\s*[\d\s()+\-*/%.]+\s*", query))
    if (has_calculation_word or arithmetic_expression) and "calculator" in available_tools:
        return RouterDecision("react", "tool", False, False, False, False, 1, 0.98)
    if re.search(r"比较.*\d+.*\d+.*(大|小|greater|less)", normalized) and not re.search(
        r"jepa|dino|论文|paper|机器人|robot", normalized
    ):
        return RouterDecision("direct", "simple", False, False, False, False, 0, 0.97)

    local_marker = any(
        token in normalized
        for token in ("本地论文", "本地资料", "papers/", "ust论文", "根据我本地", "local paper", "local papers")
    )
    paper_marker = any(token in normalized for token in ("论文", "文献", "paper", "papers"))
    web_marker = any(
        token in normalized
        for token in (
            "今天", "最新", "最近", "现在", "实时", "官方", "公开资料", "公开信息",
            "联网", "网页", "开源", "新闻", "是谁", "谁是", "who is",
        )
    )
    research_marker = any(
        token in normalized
        for token in ("比较", "对比", "分析", "系统", "优缺点", "五个方面", "报告", "综述", "compare", "systematic")
    )
    multi_method = sum(
        token in normalized
        for token in ("jepa", "dino", "v-jepa", "jepa-wam", "patch policy", "mae encoder", "masked autoencoder")
    ) >= 2
    known_local_topic = _topic_from_text(query)

    if local_marker and research_marker:
        return RouterDecision("research", "research", True, web_marker, True, True, 3, 0.96)
    if local_marker:
        return RouterDecision("research", "research", True, web_marker, True, True, 1, 0.94)
    # Explicit paper requests are deterministic; semantic query resolution in
    # _finalize_decision still uses conversation history to recover the topic.
    if paper_marker and any(name in available_tools for name in ("paper_search", "local_search", "web_search")):
        return RouterDecision("research", "research", True, web_marker, False, True, 1, 0.95)
    if research_marker and multi_method:
        return RouterDecision(
            "research", "research", True, True, "local_search" in available_tools, True, 3, 0.9
        )
    if _is_local_library_topic(known_local_topic):
        return RouterDecision("research", "research", True, web_marker, True, True, 2, 0.94)
    if web_marker and "web_search" in available_tools:
        return RouterDecision("react", "tool", False, True, False, False, 1, 0.92)
    return None


class HybridRouter:
    """Rules for obvious cases, one semantic resolver call, then safe fallback."""

    def __init__(self, llm: Any, available_tools: Sequence[str]) -> None:
        self.llm = llm
        self.available_tools = tuple(available_tools)

    def classify(
        self,
        query: str,
        *,
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        session_preferences: Mapping[str, str] | None = None,
    ) -> RouterDecision:
        history = list(conversation_history or [])
        ruled = _rule_decision(query, available_tools=self.available_tools)
        if ruled is not None:
            return _finalize_decision(
                ruled,
                query,
                available_tools=self.available_tools,
                conversation_history=history,
            )

        history_lines = [
            f"{str(item.get('role', 'user')).upper()}: {str(item.get('content', '')).strip()}"
            for item in history[-12:]
            if str(item.get("content", "")).strip()
        ]
        preference_text = json.dumps(dict(session_preferences or {}), ensure_ascii=False)
        prompt = (
            "You are the semantic query resolver and router for a research agent. Do not answer the user. "
            "Resolve follow-up references using conversation history and return only one JSON object. "
            "Required keys: original_query, standalone_query, topic, intent, route "
            "(direct|react|research), complexity (simple|tool|research), need_planning, need_web, "
            "need_local_rag, need_verification, suggested_workers (0-8), confidence (0-1), "
            "allowed_tools (array), primary_tool, fallback_tools (array), and search_queries "
            "(object keyed by tool name).\n\n"
            "standalone_query must be independently searchable and must include the topic inherited from history. "
            "Use intent=paper_search for requests to find papers or literature, and prefer paper_search. "
            "Use local_search only for the local paper library, web_search for current/public web facts, and "
            "calculator for arithmetic. direct means no tools; react means a short bounded tool loop; research "
            "means Planner/Researcher/Writer/Verifier. Only list available tools.\n\n"
            f"Available tools: {', '.join(self.available_tools) or '(none)'}\n"
            f"Session preferences: {preference_text}\n"
            f"Conversation history:\n{chr(10).join(history_lines) or '(none)'}\n\n"
            f"Current user query: {query}"
        )
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "Resolve and classify requests. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
            )
            parsed = _json_object(_text(response))
            if parsed is not None:
                decision = validate_router_output(parsed, source="llm")
                if decision is not None:
                    return _finalize_decision(
                        decision,
                        query,
                        available_tools=self.available_tools,
                        conversation_history=history,
                    )
        except Exception:
            pass

        fallback = RouterDecision(
            "react",
            "tool",
            False,
            "web_search" in self.available_tools,
            False,
            False,
            1,
            0.2,
            "fallback",
        )
        return _finalize_decision(
            fallback,
            query,
            available_tools=self.available_tools,
            conversation_history=history,
        )
