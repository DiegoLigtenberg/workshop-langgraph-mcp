from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel
from typing import Annotated, List
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph_mcp.configuration import get_llm

"""
LangGraph Agent with Remote HTTP MCP + External Package

This example shows:
- Remote MCP server via Streamable HTTP (Supabase on Railway)
- External MCP package via uv (Office Word)
- Mix of HTTP and stdio transports
- Perfect for production deployments and workshops

Flow: User types in web interface → Agent calls tools → Response streams back

Example:
  User: "Query the database for top listener, then create a Word doc with the results"
  Agent: *calls Supabase tools via HTTP, then Word tools via stdio*
  Result: "Diego listened to 340 songs. Created report.docx with the data."
"""

VERBOSE = False


# Define the state of the graph
class MessageState(BaseModel):
    messages: Annotated[List, add_messages]


def create_assistant(llm_with_tools):
    """Create an assistant function with access to the LLM"""

    async def assistant(state: MessageState):
        state.messages = await llm_with_tools.ainvoke(state.messages)
        return state

    return assistant


def build_graph(tools):
    """Build and return the LangGraph ReAct agent with MCP tools"""
    llm = get_llm("openai")
    llm_with_tools = llm.bind_tools(tools)

    builder = StateGraph(MessageState)
    builder.add_node("assistant", create_assistant(llm_with_tools))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


async def validate_servers(all_servers):
    """Validate and filter MCP servers, returning only successful ones"""
    successful_servers = {}
    for server_name, server_config in all_servers.items():
        try:
            test_client = MultiServerMCPClient({server_name: server_config})
            await test_client.get_tools()
            successful_servers[server_name] = server_config
            print(f"✓ Successfully loaded: {server_name}")
        except Exception as e:
            print(f"✗ Failed to load {server_name}: {e}")
    return successful_servers


async def setup_langgraph_app():
    """Setup the LangGraph app with MCP tools"""

    # Define MCP servers
    all_servers = {
        # Remote HTTP MCP server (Supabase via Railway)
        # Serverside access to supabase data (no credentials needed)
        "supabase": {
            "url": "https://mcp-workshop-server.up.railway.app",
            "transport": "streamable_http",
        },
        # External package (Office Word via uv)
        "office_word": {
            "command": "uv",
            "args": [
                "tool",
                "run",
                "--from",
                "office-word-mcp-server",
                "word_mcp_server",
            ],
            "transport": "stdio",
        },
    }

    # Validate server connection
    successful_servers = await validate_servers(all_servers)

    if successful_servers:
        client = MultiServerMCPClient(successful_servers)
        tools = await client.get_tools()

        print(f"\nLoaded {len(tools)} tools from {len(successful_servers)} server(s):")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        return build_graph(tools)
    else:
        print("No servers loaded! Terminating.")
        raise RuntimeError("No MCP servers available")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.langgraph_app = await setup_langgraph_app()
    yield


app = FastAPI(lifespan=lifespan)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return RedirectResponse(url="/static/chat.html")


@app.post("/chat")
async def chat_endpoint(
    request: Request, user_input: str = Form(...), thread_id: str = Form(None)
):
    print("Received user_input:", user_input)
    if not thread_id:
        thread_id = "demo-user-1"

    config = {"configurable": {"thread_id": thread_id}}
    langgraph_app = request.app.state.langgraph_app

    async def event_stream():
        final_message = None
        async for event in langgraph_app.astream_events(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        ):
            if VERBOSE:
                print("Event:", event)
            if event.get("event") == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content + " "
            elif event.get("event") == "on_tool_start":
                tool_name = event.get("name", "tool")
                tool_args = event.get("data", {}).get("input", {})
                yield f"\n__TOOL_CALL__:Calling tool '{tool_name}' with args {tool_args}\n"
            elif event.get("event") == "on_tool_end":
                tool_name = event.get("name", "tool")
                tool_output = event.get("data", {}).get("output", "")
                yield f"\n__TOOL_CALL_RESULT__:Tool '{tool_name}' returned: {tool_output}\n"
            elif event.get("event") in ("on_chain_stream", "on_chain_end"):
                messages = []
                if (
                    "data" in event
                    and "chunk" in event["data"]
                    and "messages" in event["data"]["chunk"]
                ):
                    messages = event["data"]["chunk"]["messages"]
                elif (
                    "data" in event
                    and "output" in event["data"]
                    and "messages" in event["data"]["output"]
                ):
                    messages = event["data"]["output"]["messages"]
                if messages:
                    final_message = messages[-1].content
        if final_message:
            yield f"\n__FINAL__:{final_message}"

    return StreamingResponse(event_stream(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
