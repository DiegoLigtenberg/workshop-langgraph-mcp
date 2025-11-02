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


async def setup_langgraph_app():
    """Setup the LangGraph app with MCP tools"""
    # Define your MCP servers
    servers = {
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
        # Add other servers as needed
    }

    try:
        client = MultiServerMCPClient(servers)
        tools = await client.get_tools()

        print(f"Loaded {len(tools)} tools from MCP servers:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        return build_graph(tools)

    except Exception as e:
        print(f"Error connecting to external servers: {e}")
        # Fallback to basic tools if MCP servers are unavailable
        return build_graph([])


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
                print(
                    "Event:", event
                )  # this prints each event happening in the graph in CLI
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
