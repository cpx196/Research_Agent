"""Command-line entry point for Research Agent V5."""

from __future__ import annotations

import argparse

from agent.agent import ResearchAgent
from agent.llm import DemoLLM, LLMClient, LLMClientError
from agent.graph import LangGraphResearchAgent, MCPResearchAgent


FLAGSHIP_QUERY = (
    "根据本地论文与公开资料，比较 V-JEPA 2、DINOv3 和普通 MAE Encoder 所学习视觉表征的特点。"
    "分析这些表征在空间感知、时序建模、物体交互和机器人控制中的适用性，并判断哪一种更有利于"
    "机器人控制，说明判断成立的条件和证据。"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Research Agent V5 workflow")
    parser.add_argument(
        "--mode",
        choices=("legacy", "graph_baseline", "graph_context", "graph_mcp", "graph"),
        default="graph_context",
        help="legacy=V0/V2, graph_baseline=V3, graph_context=V4, graph_mcp=V5, graph=V3 alias",
    )
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum LLM/tool loop steps")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum V3/V4 Verifier -> Researcher iterations",
    )
    parser.add_argument(
        "--verifier",
        choices=("none", "simple", "active"),
        default="active",
        help="Verifier workflow: simple baseline or active claim verification (default: active)",
    )
    parser.add_argument(
        "--max-verification-tool-calls",
        type=int,
        choices=range(1, 5),
        default=2,
        metavar="1-4",
        help="Tool-call budget for Active Verifier (default: 2)",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable trace output")
    parser.add_argument("--demo", action="store_true", help="Use a deterministic local demo model")
    parser.add_argument("--trace-file", help="Append trace output to this file")
    parser.add_argument(
        "--flagship-demo",
        action="store_true",
        help="Run the V-JEPA 2 / DINOv3 / MAE Claim-verification demo once and exit",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mode_label = {
        "legacy": "V0/V2 Legacy",
        "graph_baseline": "V3 LangGraph Baseline",
        "graph_context": "V4 LangGraph Context",
        "graph_mcp": "V5 LangGraph MCP",
        "graph": "V3 LangGraph Baseline",
    }[args.mode]
    if args.demo:
        llm = DemoLLM()
        print(f"Research Agent {mode_label} (demo mode)")
    else:
        try:
            llm = LLMClient.from_env()
        except LLMClientError as exc:
            print(f"Configuration error: {exc}")
            print("Set DASHSCOPE_API_KEY (or LLM_API_KEY), or run with --demo.")
            return 2
        print(f"Research Agent {mode_label}")

    if args.mode == "graph_mcp":
        agent = MCPResearchAgent(
            llm=llm,
            max_iterations=args.max_iterations,
            verifier_mode=args.verifier,
            max_verification_tool_calls=args.max_verification_tool_calls,
            verbose=not args.quiet,
            trace_file=args.trace_file,
        )
    elif args.mode in {"graph", "graph_baseline", "graph_context"}:
        agent = LangGraphResearchAgent(
            llm=llm,
            max_iterations=args.max_iterations,
            verifier_mode=args.verifier,
            max_verification_tool_calls=args.max_verification_tool_calls,
            context_enabled=args.mode == "graph_context",
            verbose=not args.quiet,
            trace_file=args.trace_file,
        )
    else:
        agent = ResearchAgent(
            llm=llm,
            max_steps=args.max_steps,
            verbose=not args.quiet,
            trace_file=args.trace_file,
        )
    if args.flagship_demo:
        if args.mode == "legacy":
            print("Flagship demo requires a graph mode.")
            return 2
        answer = agent.run(FLAGSHIP_QUERY)
        if args.quiet:
            print(f"\n[Final Answer]\n{answer}")
        state = getattr(agent, "last_state", {})
        print("\n[Claim Audit]")
        for claim in state.get("claims", []):
            print(
                f"{claim.get('claim_id')} [{claim.get('claim_type')}/{claim.get('status')}]: "
                f"{claim.get('repaired_text') or claim.get('text')}"
            )
        for audit in state.get("claim_audits", []):
            if audit.get("changed"):
                print(f"REPAIR {audit.get('claim_id')}: - {audit.get('original_claim')}")
                print(f"                    + {audit.get('final_claim')}")
        return 0

    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            raw_query = input("\nQuery > ")
        except EOFError:
            print()
            break
        query = raw_query.strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        answer = agent.run(query)
        if args.quiet:
            print(f"\n[Final Answer]\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
