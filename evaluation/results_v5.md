# V5 Smoke Results

本次只做最小验收，不做大规模 benchmark。

```json
{
  "protocol": {
    "status": {
      "research-mcp": "connected"
    },
    "tools": [
      "calculator",
      "paper_search"
    ],
    "calculator": "42",
    "resources": 1,
    "prompts": 1
  },
  "skill": {
    "name": "paper_research",
    "status": "ok",
    "tools": [
      "local_search"
    ]
  },
  "graph_runs": [
    {
      "query": "计算 12 * 8",
      "answer": "基于当前研究证据：\n- [Tool Source] MCP\n[MCP Server] research-mcp\n[MCP Tool] calculator\n[Transport] stdio\n[Latency] 2211.6ms\n[Fallback] None\n[Observation]\n96（来源：calculator）",
      "mcp_tools": [
        "calculator",
        "paper_search"
      ],
      "tool_sources": [
        "MCP"
      ],
      "discovery_error": ""
    },
    {
      "query": "计算 7 * 6",
      "answer": "基于当前研究证据：\n- [Tool Source] MCP\n[MCP Server] research-mcp\n[MCP Tool] calculator\n[Transport] stdio\n[Latency] 2132.2ms\n[Fallback] None\n[Observation]\n42（来源：calculator）",
      "mcp_tools": [
        "calculator",
        "paper_search"
      ],
      "tool_sources": [
        "MCP"
      ],
      "discovery_error": ""
    }
  ]
}
```
