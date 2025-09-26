from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel
from typing import Annotated, List
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph_mcp.configuration import get_llm
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod

"""
LangGraph ReAct Agent with Local Tools

Flow: Human Question → Assistant (calls tool) → Tool Execution → Assistant (final answer (summarizing the tool execution))

Example:
  Human: "Add 3 and 4"
  Assistant: *calls add(3, 4)*
  Tool: returns 7
  Assistant: "The result of adding 3 and 4 is 7"
"""


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


# Define the state of the graph.
class MessageState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    
def assistant(state: MessageState):    
    state.messages = llm_with_tools.invoke(state.messages)
    return state

def build_graph(tools):    
    builder = StateGraph(MessageState)
    # Define nodes
    builder.add_node("assistant", assistant)
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
    
    png_bytes = react_graph_memory.get_graph().draw_mermaid_png()
    with open("workshop-langgraph-mcp/model_graph.png", "wb") as f:
        f.write(png_bytes)        
        return react_graph_memory

if __name__ == "__main__":
    # setup llm and tools
    llm = get_llm("openai")
    tools = [add, multiply, divide] # this will come from mcp server
    llm_with_tools = llm.bind_tools(tools)
    
    # build graph
    react_graph_memory = build_graph(tools=tools)

    # setup config 
    config = {"configurable": {"thread_id": "1"}}
    
    # setup input state
    input_state = MessageState(messages=[HumanMessage(content="Please, make a word document where you add 3 and 4")])

    # Run the graph with the input state and the config from the langraph
    result = react_graph_memory.invoke(input_state, config)

    for m in result['messages']:
        m.pretty_print()
        # print(m)
  
    # Graph has memory with same thread id  
    '''
    input_state = MessageState(messages=[HumanMessage(content="what was my last question?")])
    result = react_graph_memory.invoke(input_state, config)
    for m in result['messages']:
        m.pretty_print()
    '''