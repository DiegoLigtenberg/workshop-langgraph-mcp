from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
from contextlib import asynccontextmanager
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel
from typing import Annotated, List
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph_mcp.configuration import get_llm
from langgraph_mcp.streaming_utils import (
    chat_endpoint_handler,
    truncate_messages_safely,
)

"""
LangGraph Agent with Remote HTTP MCP + External Package

This example shows:
- Remote MCP server via Streamable HTTP (Supabase on Railway)
- Local MCP Server via Streamable HTTP (Code Explorer)

Example:
  User: "Query the database for top listener, then create a Word doc with the results"
  Agent: *calls Supabase tools via HTTP, or Code Explorer tools via HTTP*
  Result: "Diego listened to 340 songs."
"""

# put verbose to true to see chat and tool results in terminal
VERBOSE = True


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

                1. Supabase Database Tools Usage (via remote HTTP MCP):
                - list_tables: See what tables exist in the database
                - execute_sql: Run SQL queries to get data (SELECT statements only)
                - search_docs: Search Supabase documentation can not be used for helped in this project.
                - Other management tools: migrations, logs, advisors, etc.          
                
                Best Practices:
                - CRITICAL!!! Database results contain technical IDs. If the result of query/queries is/are technical, 
                try really hard to find a connection with another table or column in order to make the answer 
                more functionally sound and non-technical-user friendly.
                - CRITICAL: When querying songs table or joining with songs table, you MUST filter by is_public = true in the SAME query. 
                Never query other tables first to get song IDs and then check songs separately. Always JOIN songs table and filter by is_public = true in one query.
                - CRITICAL: When querying the genres table by genre name, you MUST use ILIKE instead of = due to capitalization differences in the table.
                For example: WHERE genres.name ILIKE '%jazz%' NOT WHERE genres.name = 'Jazz'. This ensures case-insensitive matching.
                - CRITICAL: If after a query you get something technical, reflect on yourself and try to connect with another table (new tool call),
                to get a better more functional answer.
                - CRITICAL: Start by listing tables if you need to understand the database structure
                - CRITICAL: When you do multiple tool calls, or multiple queries, make sure these CRITICAL steps are always applied to the final result.
                
                - Use execute_sql to query data - be specific in your queries
                - Always check table names before querying them
                - docs tool contains no relevant information regarding vibify project.
                - When mentioning 1-5 songs, always provide shareable links in this format:
                  https://vibify.up.railway.app/share/song/{song_id_uuid}
                  Replace {song_id_uuid} with the actual song ID from the database's songs table.
                - When listing more than 5 songs, use bullet points format without individual links:
                  - Song Title 1 by Artist 1
                  - Song Title 2 by Artist 2
                  (Maximum 5 songs shown)

                Approach:
                1. Understand what the user wants
                2. Use the appropriate tools in logical order
                3. Provide clear, functional and non-technical responses based on tool results
                
                Formatting Guidelines:
                - Use line breaks for readability (especially when listing multiple items)
                - Format song lists like this:
                  
                  Here are 2 songs for you:
                  
                  1. Song Title by Artist
                  Link: https://vibify.up.railway.app/share/song/...
                  
                  2. Song Title by Artist
                  Link: https://vibify.up.railway.app/share/song/...
                  
                - Do NOT use markdown formatting (**bold**, *italic*, [links](url), etc.)
                - Write URLs as plain text (e.g., "Link: https://example.com" not "[text](url)")
                - Do NOT use emojis
                """
    )

    async def assistant(state: MessageState):
        # Always ensure system message is first (remove any existing system messages first)
        messages = state.messages
        # Remove system messages and truncate msg history to preserve token usage
        messages = truncate_messages_safely(messages)

        # Always prepend system message
        messages = [system_prompt] + messages

        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

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
    import traceback

    successful_servers = {}
    for server_name, server_config in all_servers.items():
        try:
            print(
                f"Testing connection to {server_name} at {server_config.get('url', 'stdio')}..."
            )
            test_client = MultiServerMCPClient({server_name: server_config})
            await test_client.get_tools()
            successful_servers[server_name] = server_config
            print(f"Successfully loaded: {server_name}")
        except Exception as e:
            print(f"Failed to load {server_name}: {e}")
            print(f"   Full traceback:\n{traceback.format_exc()}")
    return successful_servers


async def setup_langgraph_app():
    """Setup the LangGraph app with MCP tools"""

    # Define MCP servers
    all_servers = {
        # Remote HTTP MCP server (Supabase via Railway)
        # Serverside access to supabase data (database  credentials are handled by the server)
        "supabase": {
            "url": "https://mcp-workshop-server.up.railway.app",
            "transport": "streamable_http",
        },
        # uncomment to use code explorer mcp server
        # Code Explorer MCP server (via Streamable HTTP)
        # "code-explorer": {
        #     "url": "http://127.0.0.1:8001",  # FastMCP streamable-http exposes at root, not /mcp
        #     "transport": "streamable_http",
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
    return await chat_endpoint_handler(request, user_input, thread_id, VERBOSE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
