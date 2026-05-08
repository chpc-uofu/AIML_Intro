"""
FastMCP quickstart example.

Run from the repository root:
    uv run examples/snippets/servers/fastmcp_quickstart.py
"""

# from mcp.server.fastmcp import FastMCP, http_app
from fastmcp import FastMCP
from fastapi import FastAPI
import uvicorn
from datetime import date
from ddgs import DDGS
# Create an MCP server
#mcp = FastMCP("Demo", json_response=True)
mcp = FastMCP("QuackPower")

# Add an addition tool
#@mcp.tool()
#def add(a: int, b: int) -> int:
#    """Add two numbers"""
#    return a + b + b + a #Yes this is wrong on purpose to see if it really calls the tool

@mcp.tool()
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    print("calling duck army", flush=True)
    """Search the web with DuckDuckGo and return the top results."""
    if not query.strip():
        return "Please provide a search query."

    max_results = max(1, min(max_results, 10))

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        lines = []
        for i, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            href = result.get("href", "")
            body = result.get("body", "")

            lines.append(
                f"{i}. {title}\n"
                f"   URL: {href}\n"
                f"   Summary: {body}"
            )

        return "\n\n".join(lines)

    except Exception as exc:
        return f"DuckDuckGo search failed: {exc}"

#@mcp.tool()
#def get_current_time() -> str:
#    print("Calling get_current_time", flush=True)
#    return datetime.now().strftime("%d/%m/%Y")

# Add a dynamic greeting resource
#@mcp.resource("greeting://{name}")
#def get_greeting(name: str) -> str:
#    """Get a personalized greeting"""
#    return f"Greetings, solicitations and felicitations, {name}!"

# Add a prompt
#@mcp.prompt()
#def greet_user(name: str, style: str = "friendly") -> str:
#    """Generate a greeting prompt"""
#    styles = {
#        "friendly": "Please write a warm, friendly greeting",
#        "formal": "Please write a formal, professional greeting",
#        "casual": "Please write a casual, relaxed greeting",
#    }

#    return f"{styles.get(style, styles['friendly'])} for someone named {name}."

#FastAPI for conversion if needed
#mcp_app = mcp.http_app(path='/mcp')
#app = FastAPI(title="Quack Powered Searches", lifespan=mcp_app.lifespan)
#app.mount('/mcp', mcp.streamable_http_app())
#app.mount("/ddgs", mcp_app)
#for route in app.routes:
#       print(route.path, route.name)
# Run with streamable HTTP transport
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    #uvicorn.run("mcp_server_quack:app", host="127.0.0.1", port=8000, reload=True)

