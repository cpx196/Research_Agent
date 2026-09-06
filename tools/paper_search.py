"""arXiv paper search tool."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote_plus

import requests


ATOM = "http://www.w3.org/2005/Atom"


def _limit(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_results must be an integer") from exc
    return max(1, min(result, 10))


def _text(parent: ET.Element, tag: str, default: str = "") -> str:
    element = parent.find(f"{{{ATOM}}}{tag}")
    return " ".join((element.text or "").split()) if element is not None else default


def _search_arxiv(query: str, limit: int) -> list[dict[str, str]]:
    search_query = f'all:"{query.replace(chr(34), "")}"'
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query={quote_plus(search_query)}&start=0&max_results={limit}&sortBy=relevance"
    )
    response = requests.get(
        url,
        headers={"User-Agent": "research-agent-v0/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    results: list[dict[str, str]] = []
    for entry in root.findall(f"{{{ATOM}}}entry")[:limit]:
        paper_id = _text(entry, "id")
        paper_id = re.sub(r"^http://", "https://", paper_id)
        authors = [
            " ".join((author.findtext(f"{{{ATOM}}}name") or "").split())
            for author in entry.findall(f"{{{ATOM}}}author")
        ]
        abstract = _text(entry, "summary")
        if len(abstract) > 800:
            abstract = abstract[:797].rstrip() + "..."
        results.append(
            {
                "title": _text(entry, "title"),
                "authors": ", ".join(author for author in authors if author),
                "published": _text(entry, "published")[:10],
                "arxiv": paper_id,
                "abstract": abstract,
            }
        )
    return results


def paper_search(query: str, max_results: int = 5) -> str:
    """Search arXiv by keywords and return normalized paper metadata."""

    if not isinstance(query, str) or not query.strip():
        return "ToolError: query must be a non-empty string"
    try:
        limit = _limit(max_results)
        results = _search_arxiv(query.strip(), limit)
    except requests.Timeout:
        return "ToolError: Paper search request timed out."
    except requests.RequestException as exc:
        return f"ToolError: Paper search request failed: {exc}"
    except ET.ParseError as exc:
        return f"ToolError: Paper search response could not be parsed: {exc}"
    except ValueError as exc:
        return f"ToolError: {exc}"

    if not results:
        return "No papers found."
    blocks = []
    for result in results[:limit]:
        blocks.append(
            "\n".join(
                [
                    f"Title: {result['title']}",
                    f"Authors: {result['authors']}",
                    f"Published: {result['published']}",
                    f"arXiv: {result['arxiv']}",
                    f"Abstract: {result['abstract']}",
                ]
            )
        )
    return "\n\n".join(blocks)


paper_search_schema = {
    "type": "function",
    "function": {
        "name": "paper_search",
        "description": "Search arXiv for academic papers by keywords.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Academic search query."},
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of papers, capped at 10.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
