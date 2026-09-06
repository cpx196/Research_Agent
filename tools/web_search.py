"""Web search tool with several simple OpenAI-agent-friendly backends."""

from __future__ import annotations

import os
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests


def _timeout() -> float:
    try:
        return float(os.getenv("RESEARCH_AGENT_HTTP_TIMEOUT", "15"))
    except ValueError:
        return 15.0


def _limit(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_results must be an integer") from exc
    return max(1, min(result, 10))


def _clean_result(item: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(item.get("title") or "Untitled").strip(),
        "url": str(item.get("url") or item.get("link") or "").strip(),
        "snippet": str(item.get("snippet") or item.get("content") or item.get("description") or "").strip(),
    }


def _search_tavily(query: str, limit: int) -> list[dict[str, str]]:
    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": limit},
        timeout=_timeout(),
    )
    response.raise_for_status()
    data = response.json()
    return [_clean_result(item) for item in data.get("results", [])]


def _search_serper(query: str, limit: int) -> list[dict[str, str]]:
    api_key = os.getenv("SERPER_API_KEY") or os.getenv("SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not configured")
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": limit},
        timeout=_timeout(),
    )
    response.raise_for_status()
    data = response.json()
    return [_clean_result(item) for item in data.get("organic", [])[:limit]]


def _search_brave(query: str, limit: int) -> list[dict[str, str]]:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not configured")
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        params={"q": query, "count": limit},
        timeout=_timeout(),
    )
    response.raise_for_status()
    data = response.json()
    return [_clean_result(item) for item in data.get("web", {}).get("results", [])[:limit]]


def _search_bing(query: str, limit: int) -> list[dict[str, str]]:
    api_key = os.getenv("BING_SEARCH_API_KEY") or os.getenv("SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("BING_SEARCH_API_KEY is not configured")
    response = requests.get(
        "https://api.bing.microsoft.com/v7.0/search",
        headers={"Ocp-Apim-Subscription-Key": api_key},
        params={"q": query, "count": limit, "responseFilter": "Webpages"},
        timeout=_timeout(),
    )
    response.raise_for_status()
    data = response.json()
    return [_clean_result(item) for item in data.get("webPages", {}).get("value", [])[:limit]]


def _search_duckduckgo(query: str, limit: int) -> list[dict[str, str]]:
    """Use DuckDuckGo's no-key endpoints as a lightweight fallback."""

    response = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        headers={"User-Agent": "research-agent-v0/1.0"},
        timeout=_timeout(),
    )
    response.raise_for_status()
    data = response.json()
    results: list[dict[str, str]] = []
    if data.get("AbstractText") or data.get("AbstractURL"):
        results.append(
            _clean_result(
                {
                    "title": data.get("Heading") or query,
                    "url": data.get("AbstractURL"),
                    "snippet": data.get("AbstractText"),
                }
            )
        )

    def collect(topics: list[dict[str, Any]]) -> None:
        for topic in topics:
            if len(results) >= limit:
                return
            if topic.get("Topics"):
                collect(topic["Topics"])
                continue
            if topic.get("FirstURL") or topic.get("Text"):
                results.append(
                    _clean_result(
                        {
                            "title": topic.get("Text", "").split(" - ", 1)[0],
                            "url": topic.get("FirstURL"),
                            "snippet": topic.get("Text"),
                        }
                    )
                )

    collect(data.get("RelatedTopics", []))
    if results:
        return results[:limit]

    # The Instant Answer endpoint often has no entries for niche queries.
    # Its HTML endpoint is still keyless and provides ordinary result links.
    html_response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "research-agent-v0/1.0"},
        timeout=_timeout(),
    )
    html_response.raise_for_status()
    parser = _DuckDuckGoHTMLParser(limit)
    parser.feed(html_response.text)
    if parser.results:
        return parser.results[:limit]
    # DuckDuckGo sometimes returns an HTTP 202 bot-check page. Bing's public
    # HTML results keep the no-key local setup useful in that case.
    return _search_bing_html(query, limit)


def _search_bing_html(query: str, limit: int) -> list[dict[str, str]]:
    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "count": limit},
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        timeout=_timeout(),
    )
    response.raise_for_status()
    parser = _BingHTMLParser(limit)
    parser.feed(response.text)
    parser.close()
    return parser.results[:limit]


def _redirect_url(href: str) -> str:
    parsed = urlparse(href)
    target = parse_qs(parsed.query).get("uddg", [""])[0]
    return unquote(target) if target else href


class _DuckDuckGoHTMLParser(HTMLParser):
    """Parse only the stable result anchors from DuckDuckGo's HTML page."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.results: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None
        self.capture: str | None = None
        self.buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if "result__a" in classes:
            if self.current and self.current.get("title"):
                self.results.append(self.current)
            self.current = {
                "title": "",
                "url": _redirect_url(attributes.get("href") or ""),
                "snippet": "",
            }
            self.capture = "title"
            self.buffer = []
        elif "result__snippet" in classes and self.current is not None:
            self.capture = "snippet"
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self.capture or self.current is None:
            return
        self.current[self.capture] = " ".join("".join(self.buffer).split())
        self.capture = None
        self.buffer = []

    def close(self) -> None:
        super().close()
        if self.current and self.current.get("title"):
            self.results.append(self.current)
        self.current = None


class _BingHTMLParser(HTMLParser):
    """Extract ordinary result cards from Bing's server-rendered HTML."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.results: list[dict[str, str]] = []
        self.in_result = False
        self.result_depth = 0
        self.in_heading = False
        self.capture = ""
        self.buffer: list[str] = []
        self.current: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "li" and "b_algo" in classes and len(self.results) < self.limit:
            self.in_result = True
            self.result_depth = 1
            self.current = {"title": "", "url": "", "snippet": ""}
            return
        if not self.in_result or self.current is None:
            return
        self.result_depth += 1
        if tag == "h2":
            self.in_heading = True
        elif tag == "a" and self.in_heading and not self.current["url"]:
            self.current["url"] = attributes.get("href") or ""
            self.capture = "title"
            self.buffer = []
        elif tag == "p" and not self.current["snippet"]:
            self.capture = "snippet"
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_result or self.current is None:
            return
        if tag == "a" and self.capture == "title":
            self.current["title"] = " ".join("".join(self.buffer).split())
            self.capture = ""
            self.buffer = []
        elif tag == "p" and self.capture == "snippet":
            self.current["snippet"] = " ".join("".join(self.buffer).split())
            self.capture = ""
            self.buffer = []
        elif tag == "h2":
            self.in_heading = False
        self.result_depth -= 1
        if tag == "li" and self.result_depth <= 0:
            if self.current.get("title") and self.current.get("url"):
                self.results.append(self.current)
            self.current = None
            self.in_result = False
            self.capture = ""
            self.buffer = []


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return normalized title/URL/snippet blocks."""

    if not isinstance(query, str) or not query.strip():
        return "ToolError: query must be a non-empty string"
    try:
        limit = _limit(max_results)
    except ValueError as exc:
        return f"ToolError: {exc}"

    provider = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo").strip().lower()
    backends = {
        "tavily": _search_tavily,
        "serper": _search_serper,
        "brave": _search_brave,
        "bing": _search_bing,
        "duckduckgo": _search_duckduckgo,
        "ddg": _search_duckduckgo,
    }
    backend = backends.get(provider)
    if backend is None:
        return f"ToolError: Unknown web search provider {provider}"

    try:
        results = backend(query.strip(), limit)
    except requests.Timeout:
        return "ToolError: Search request timed out."
    except requests.RequestException as exc:
        return f"ToolError: Search request failed: {exc}"
    except (RuntimeError, ValueError) as exc:
        return f"ToolError: {exc}"
    except (KeyError, TypeError, AttributeError) as exc:
        return f"ToolError: Search response could not be parsed: {exc}"

    if not results:
        return "No web results found."
    blocks = []
    for result in results[:limit]:
        blocks.append(
            "\n".join(
                [
                    f"Title: {result['title']}",
                    f"URL: {result['url']}",
                    f"Snippet: {result['snippet']}",
                ]
            )
        )
    return "\n\n".join(blocks)


web_search_schema = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current or external information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results, capped at 10.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
