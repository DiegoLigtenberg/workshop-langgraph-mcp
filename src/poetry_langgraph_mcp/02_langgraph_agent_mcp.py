import asyncio
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel
from typing import Annotated, List
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from poetry_langgraph_mcp.configuration import get_llm

"""
LangGraph ReAct Agent with MCP Server Tools

Flow: Human Question → Assistant (calls MCP tool) → Tool Execution → Assistant (final answer)

Example:
  Human: "Add 3 and 4"
  Assistant: *calls MCP add(3, 4)*
  MCP Tool: returns 7
  Assistant: "The result of adding 3 and 4 is 7"
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

async def run_mcp_agent():
    """Load MCP tools and run the LangGraph agent"""
    server_path = Path(__file__).parent / "math_server.py"
    server_params = StdioServerParameters(command="python", args=[str(server_path)])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)            
            
            print(f"Loaded {len(tools)} MCP tools from the sever::")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            graph = build_graph(tools)
            config = {"configurable": {"thread_id": "1"}}
            
            input_state = {"messages": [HumanMessage(content="Add 3 and 4.")]}
            result = await graph.ainvoke(input_state, config)
            
            # Show only the final answer
            # final_answer = result['messages'][-1].contentac
            return result

if __name__ == "__main__":
    result = asyncio.run(run_mcp_agent())
    # print(f"Final Answer: {result}")
    for m in result['messages']:
        m.pretty_print()