# MCP (Model Context Protocol) with LangGraph

## MCP Components

### MCP Servers
Tools/functions that provide specific capabilities:
- **Math Server**: `add()`, `multiply()`, `divide()` functions
- **Weather Server**: `get_weather()`, `get_forecast()` functions  
- **File Server**: `read_file()`, `write_file()` functions

### MCP Clients
Programs that connect to MCP servers and expose their tools to LLM frameworks:
- **Claude Desktop**: Uses MCP to access local files, databases, and APIs
- **VS Code Extensions**: Connect to MCP servers for code assistance

### MCP Hosts  
Applications that combine MCP servers + clients to create full AI workflows:
- **Claude Desktop App**: Built-in MCP client for file access and web search
- **VS Code with MCP**: Code editor that can read files, run commands, search code

## Resources
- [MCP Servers Directory](https://mcpservers.org/)