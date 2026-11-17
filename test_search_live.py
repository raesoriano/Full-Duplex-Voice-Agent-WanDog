"""Live end-to-end check: does WanDog's spawned OnIt 'net' MCP server
serve the `search` tool, and does it return real results?

Run: .venv/bin/python test_search_live.py
"""
import asyncio
import os

os.environ.pop("OLLAMA_API_KEY", None)  # force the free-tier path

import config
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=config.ONIT_PYTHON,
        args=["-m", "src.mcp.servers.run", "--stdio",
              "--name", "ToolsNetMCPServer",
              "--module", "tasks.tools",
              "--profile", "net",
              "--log-level", "INFO"],
        env={**os.environ,
             "ONIT_DATA_PATH": config.MCP_DATA_PATH,
             "ONIT_STDIO_SPAWNED": "1",
             "ONIT_APPROVAL_CHANNEL": "1",
             "ONIT_COMMAND_ALLOWLIST": "1"},
        cwd=config.ONIT_REPO,
    )
    cm = stdio_client(params)
    read, write = await cm.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()

    tools = await session.list_tools()
    names = [t.name for t in tools.tools]
    print("TOOLS:", names)
    assert "search" in names, "search tool missing from net profile"

    res = await session.call_tool(
        "search", {"query": "what is the capital of France",
                   "type": "web", "max_results": 3})
    text = str(res.content[0].text)
    print("WEB SEARCH:", text[:300])
    assert "error" not in text[:60], f"web search failed: {text[:200]}"

    res = await session.call_tool(
        "search", {"query": "AI regulation", "type": "news",
                   "max_results": 3})
    text = str(res.content[0].text)
    print("NEWS SEARCH:", text[:300])
    assert "error" not in text[:60], f"news search failed: {text[:200]}"

    await session.__aexit__(None, None, None)
    await cm.__aexit__(None, None, None)
    print("PASS: search tool is live and returning results over MCP")


if __name__ == "__main__":
    asyncio.run(main())
