import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_individual_servers():
    """Test each MCP server individually to find the problematic one"""
    current_dir = Path(__file__).parent
    
    # Test servers one by one
    servers_to_test = [
        {
            "name": "local_math",
            "config": {
                "command": "python",
                "args": [str(current_dir / "math_server.py")],
                "transport": "stdio",
            }
        },
        {
            "name": "local_weather", 
            "config": {
                "command": "python",
                "args": [str(current_dir / "weather_server.py")],
                "transport": "stdio",
            }
        },
        {
            "name": "office_word",
            "config": {
                "command": "uv",
                "args": ["tool", "run", "--from", "office-word-mcp-server", "word_mcp_server"],
                "transport": "stdio",
            }
        }
    ]
    
    for server in servers_to_test:
        print(f"\n🔍 Testing {server['name']}...")
        try:
            client = MultiServerMCPClient({server['name']: server['config']})
            tools = await client.get_tools()
            print(f"✅ {server['name']}: Loaded {len(tools)} tools")
            for tool in tools:
                print(f"   - {tool.name}: {tool.description}")
        except Exception as e:
            print(f"❌ {server['name']}: Error - {e}")

if __name__ == "__main__":
    asyncio.run(test_individual_servers())
