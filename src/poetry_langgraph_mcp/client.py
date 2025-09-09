import asyncio
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from poetry_langgraph_mcp.configuration import get_llm

async def run_mcp_demo():
    server_path = Path(__file__).parent / "math_server.py"
    server_params = StdioServerParameters(command="python", args=[str(server_path)])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            llm = get_llm("openai")
            agent = create_react_agent(llm, tools)
            
            response = await agent.ainvoke({"messages": [{"role": "user", "content": "What's (3 + 5) * 12?"}]})
            return response["messages"][-1].content

if __name__ == "__main__":
    result = asyncio.run(run_mcp_demo())
    print(f"Agent: {result}")
