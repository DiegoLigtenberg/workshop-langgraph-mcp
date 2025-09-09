import asyncio
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel
from typing import Annotated, List
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from poetry_langgraph_mcp.configuration import get_llm

"""
LangGraph ReAct Agent with Multiple MCP Servers

Flow: Human Question → Assistant (calls MCP tools from multiple servers) → Tool Execution → Assistant (final answer)

Example:
  Human: "What's 3 + 4 and what's the weather in NYC?"
  Assistant: *calls math add(3, 4) and weather get_weather("nyc")*
  MCP Tools: returns 7 and "Sunny, 72°F"
  Assistant: "3 + 4 = 7. Weather in NYC is Sunny, 72°F"
"""

# Define the state of the graph.
class MessageState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]

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
    # Define nodes
    builder.add_node("assistant", create_assistant(llm_with_tools))
    builder.add_node("tools", ToolNode(tools))
    # Define edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    # note: The tool call output will be sent back to the assistant node (to 'summarize' the tool call)
    builder.add_edge("tools", "assistant") 

    memory = MemorySaver()
    react_graph_memory = builder.compile(checkpointer=memory)
    return react_graph_memory

async def run_mcp_agent(input_state):
    """Load MCP tools from multiple servers and run the LangGraph agent"""
    # Get absolute paths to server files
    current_dir = Path(__file__).parent
    math_server_path = current_dir / "math_server.py"
    weather_server_path = current_dir / "weather_server.py"
    
    # Initialize MultiServerMCPClient with both servers
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [str(math_server_path)],
                "transport": "stdio",
            },
            "weather": {
                "command": "python", 
                "args": [str(weather_server_path)],
                "transport": "stdio",
            }
        }
    )
    
    # Load tools from all servers
    tools = await client.get_tools()
    
    print(f"Loaded {len(tools)} MCP tools from multiple servers:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    graph = build_graph(tools)
    config = {"configurable": {"thread_id": "1"}}
    
    # Test with math question
    result = await graph.ainvoke(input_state, config)
    
    return result

if __name__ == "__main__":    
    input_state = {"messages": [HumanMessage(content="What's (3 + 5) * 12?")]}
    result = asyncio.run(run_mcp_agent(input_state))
    for m in result['messages']:
        m.pretty_print()
        
    input_state = {"messages": [HumanMessage(content="What's the weather forecast in london?")]}
    result = asyncio.run(run_mcp_agent(input_state))
    for m in result['messages']:
        m.pretty_print()