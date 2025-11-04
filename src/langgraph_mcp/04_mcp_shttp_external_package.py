from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, SystemMessage
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

    # System prompt to guide the LLM on using tools effectively
    system_prompt = SystemMessage(
        content="""You are a helpful AI assistant for the website "Vibify.up.railway.app". 
                    When greeting users, mention the website link.
                    This is a Spotify clone, but do not mention that. 
                    You have access to the following types of tools:

                1. **Supabase Database Tools** (via remote HTTP MCP):
                - list_tables: See what tables exist in the database
                - execute_sql: Run SQL queries to get data (SELECT statements only)
                - search_docs: Search Supabase documentation if you need help
                - Other management tools: migrations, logs, advisors, etc.

                2. Add your own guidance for another Tool/Server here...
                - ...
                - ...
                
                **Best Practices:**
                - VITAL!!! Database results contain technical IDs. If the query result is technical, 
                try to find a connection with another table or column in order to answer the user question functionally.
                - If after a query you get something technical, reflect on yourself and try to connect with another table (new tool call),
                to get a better more functional answer.
                - Start by listing tables if you need to understand the database structure
                - Use execute_sql to query data - be specific in your queries
                - Always check table names before querying them
                - IMPORTANT: Always filter songs by is_public = true when querying the songs table (exclude private songs)
                - When mentioning 1-5 songs, always provide shareable links in this format:
                  https://vibify.up.railway.app/share/song/{song_id_uuid}
                  Replace {song_id_uuid} with the actual song ID from the database's songs table.
                - When listing more than 5 songs, use bullet points format without individual links:
                  - Song Title 1 by Artist 1
                  - Song Title 2 by Artist 2
                  (Maximum 5 songs shown)

                **Approach:**
                1. Understand what the user wants
                2. Use the appropriate tools in logical order
                3. Provide clear, helpful responses based on tool results
                
                **Formatting Guidelines:**
                - Use line breaks for readability (especially when listing multiple items)
                - Format song lists like this:
                  
                  Here are 2 songs for you:
                  
                  1. Song Title by Artist
                  Link: https://vibify.up.railway.app/share/song/...
                  
                  2. Song Title by Artist
                  Link: https://vibify.up.railway.app/share/song/...
                  
                - Do NOT use markdown formatting (**bold**, *italic*, etc.) for any lists or formatted text.
                - Do NOT use emojis
                """
    )

    async def assistant(state: MessageState):
        # Prepend system message if not already present
        messages = state.messages
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [system_prompt] + messages
        
        response = await llm_with_tools.ainvoke(messages)
        state.messages = [response]  # Only return the new response
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
        # "office_word": {
        #     "command": "uv",
        #     "args": [
        #         "tool",
        #         "run",
        #         "--from",
        #         "office-word-mcp-server",
        #         "word_mcp_server",
        #     ],
        #     "transport": "stdio",
        # },
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
    
    
    '''Example Questions:
    1) Hi who listens to most music?
    2) What is a stream?
    3) What is the most popular song?
    4) if you look at the schema of this database, do you see security issues?
    
    5) Name a song in the database.
    6) Who is the artist of this song?
    7) How many times is this song streamed?
    8) Which public song is streamed the most? (note there are also some private songs which you cant see).
    
    '''
