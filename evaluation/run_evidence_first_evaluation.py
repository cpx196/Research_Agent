"""Run the flagship query against the three Evidence-First baselines."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from agent.graph import LangGraphResearchAgent
from agent.llm import DemoLLM, LLMClient, LLMClientError
from main import FLAGSHIP_QUERY


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "evidence_first_results.json"


def _agent(llm: Any, verifier: str) -> LangGraphResearchAgent:
    return LangGraphResearchAgent(
        llm,
        verifier_mode=verifier,
        max_iterations=2,
        max_verification_tool_calls=2,
        context_enabled=True,
        verbose=False,
    )


def _structural_metrics(state: dict[str, Any]) -> dict[str, Any]:
    claims = list(state.get("claims", []))
    total = len(claims)
    unsupported = sum(
        1 for item in claims if item.get("status") in {"UNSUPPORTED", "CONTRADICTED"}
    )
    covered = sum(1 for item in claims if item.get("supporting_evidence_ids"))
    repairs = list(state.get("claim_audits", []))
    return {
        "claim_count": total,
        "unsupported_claim_rate": round(unsupported / total, 4) if total else None,
        "claim_evidence_coverage": round(covered / total, 4) if total else None,
        "repair_attempts": len(repairs),
        "changed_claims": sum(bool(item.get("changed")) for item in repairs),
        "human_metrics_pending": [
            "citation_entailment",
            "error_detection_precision_recall",
            "repair_correctness",
            "correct_claim_preservation",
            "conclusion_usefulness",
        ],
    }


def run(
    output: Path,
    *,
    demo: bool = False,
    cases_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    llm: Any = DemoLLM() if demo else LLMClient.from_env()
    records: list[dict[str, Any]] = []
    cases = [{"id": "FLAGSHIP", "query": FLAGSHIP_QUERY}]
    if cases_path is not None:
        loaded = json.loads(cases_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("cases file must contain a JSON array")
        cases = loaded[:limit] if limit is not None else loaded
    for case in cases:
        query = str(case.get("query", "")).strip()
        if not query:
            continue
        for label, verifier in (
            ("writer_only", "none"),
            ("whole_answer_verifier", "simple"),
            ("claim_level_targeted_repair", "active"),
        ):
            agent = _agent(llm, verifier)
            answer = agent.run(query)
            state = dict(agent.last_state)
            records.append({
                "case_id": str(case.get("id", "")),
                "category": str(case.get("category", "flagship")),
                "query": query,
                "method": label,
                "verifier": verifier,
                "answer": answer,
                "claims": state.get("claims", []),
                "claim_evidence": state.get("claim_evidence", []),
                "claim_audits": state.get("claim_audits", []),
                "structural_metrics": _structural_metrics(state),
                "stats": agent.last_run_stats,
            })
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(str(record.get("case_id", "")), []).append(record)
    for case_records in by_case.values():
        baseline = next((item for item in case_records if item["method"] == "writer_only"), None)
        if baseline is None:
            continue
        base_tools = int(baseline.get("stats", {}).get("tool_calls", 0))
        base_latency = float(baseline.get("stats", {}).get("latency_seconds", 0.0))
        for record in case_records:
            record["structural_metrics"]["extra_tool_calls_vs_writer"] = (
                int(record.get("stats", {}).get("tool_calls", 0)) - base_tools
            )
            record["structural_metrics"]["latency_overhead_seconds_vs_writer"] = round(
                float(record.get("stats", {}).get("latency_seconds", 0.0)) - base_latency,
                3,
            )
    result = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "query": FLAGSHIP_QUERY if cases_path is None else None,
        "case_count": len(cases),
        "note": "Human labels are required for entailment, detection precision/recall, and repair correctness.",
        "runs": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use deterministic DemoLLM")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=Path, help="Optional labeled JSON case set")
    parser.add_argument("--limit", type=int, help="Limit the number of loaded cases")
    args = parser.parse_args()
    try:
        result = run(args.output, demo=args.demo, cases_path=args.cases, limit=args.limit)
    except LLMClientError as exc:
        print(f"Configuration error: {exc}. Use --demo or configure .env.")
        return 2
    print(f"Wrote {len(result['runs'])} baseline runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
