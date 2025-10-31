import asyncio
from pathlib import Path
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel
from typing import Annotated, List
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from poetry_langgraph_mcp.configuration import get_llm

"""
LangGraph ReAct Agent with External MCP Servers

This example shows how to connect to external MCP servers like Composio
alongside your local servers.

Flow: Human Question → Assistant (calls tools from multiple sources) → Tool Execution → Assistant (final answer)
"""

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

async def run_external_mcp_agent(input_state):
    """Load MCP tools from local and external servers"""
    current_dir = Path(__file__).parent
    
    # Define servers - mix of local and external
    servers = {
        # Your local math server
        "local_math": {
            "command": "python",
            "args": [str(current_dir / "local_mcp_servers" / "math_server.py")],
            "transport": "stdio",
        },
        # Your local weather server  
        "local_weather": {
            "command": "python",
            "args": [str(current_dir / "local_mcp_servers" / "weather_server.py")],
            "transport": "stdio",
        },
        # Office Word MCP Server (using uv - no local installation needed)
        "office_word": {
            "command": "uv",
            "args": ["tool", "run", "--from", "office-word-mcp-server", "word_mcp_server"],
            "transport": "stdio",
        },
    }
    
    try:
        client = MultiServerMCPClient(servers)
        tools = await client.get_tools()
        
        print(f"Loaded {len(tools)} tools from MCP servers:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        graph = build_graph(tools)
        config = {"configurable": {"thread_id": "1"}}
        
        result = await graph.ainvoke(input_state, config)
        return result
        
    except Exception as e:
        print(f"Error connecting to external servers: {e}")
        print("Note: External servers may require API keys or may be unavailable")
        return None

if __name__ == "__main__":
    # Test with local servers only
    input_state = {"messages": [HumanMessage(content="What's 15 * 8? Then create a new Word document called 'full_report.docx' \
        with the title 'Math Report' and add a heading 'Calculation Results' \
        followed by a paragraph explaining that 15 * 8 = 120.\
        Finally write a new paragraph explaining the weather in tokyo.")]}
    result = asyncio.run(run_external_mcp_agent(input_state))
    
    if result:
        for m in result['messages']:
            m.pretty_print()
    else:
        print("Failed to run agent - check server connections")
