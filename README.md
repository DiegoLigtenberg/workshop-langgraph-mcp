# MCP (Model Context Protocol) with LangGraph

This project demonstrates how to use **Model Context Protocol (MCP)** with **LangGraph** to create AI agents that can access external tools and services. We use `MultiServerMCPClient` from `langchain-mcp-adapters` to connect to multiple MCP servers simultaneously.

## Quick Start

### Setup this project with Poetry


1. Install Poetry:
    ```bash
    # Simplest method (works on Windows/Mac/Linux)
    pip install poetry
    ```
    
    ```bash
    # Official installer (recommended)
    # Windows PowerShell:
    powershell -c "(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"
    
    # Mac/Linux:
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. Configure Poetry to create venv in project:
    ```bash
    poetry config virtualenvs.in-project true
    ```

3. Update lock file and install:
    ```bash
    poetry lock
    poetry install
    ```

4. Select Python interpreter in VS Code:
    - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
    - Type "Python: Select Interpreter"
    - Choose the `.venv` one from your project
    - Done! You can now run Python files normally.

Alternative: Use poetry run python your_script.py if needed.

Adding new packages:
poetry add package-name
(No need for lock/install - poetry add does it automatically)

## MCP Architecture

![MCP Architecture](resources/readme_mcp_explanation.png)

### How MCP Works

**MCP (Model Context Protocol)** enables AI applications to securely connect to external tools and data sources. The architecture consists of three main components:

#### 1. **MCP Servers** 🔧
- **Purpose**: Provide specific tools and capabilities
- **Examples**: Math operations, weather data, file manipulation, database access
- **In this project**: 
  - `math_server.py` - Basic math operations (`add`, `multiply`, `divide`)
  - `weather_server.py` - Weather information (`get_weather`, `get_forecast`)
  - External Office Word server - Document creation and editing

#### 2. **MCP Client** 🌉
- **Purpose**: Bridge between AI applications and MCP servers
- **In this project**: `MultiServerMCPClient` from `langchain-mcp-adapters`
- **Other examples**: Claude Desktop (built-in MCP client for file access and web search)
- **Key Features**:
  - Connects to multiple MCP servers simultaneously
  - Aggregates tools from all connected servers
  - Provides unified interface to LangGraph agents

#### 3. **MCP Host** 🤖
- **Purpose**: The AI application that uses MCP tools
- **In this project**: LangGraph agents that can call MCP tools
- **Other examples**: 
  - **Cursor** - AI code editor with MCP integration for file operations
  - **Claude Code** - AI coding assistant with MCP tool access
  - **Lovable** - AI development platform using MCP for project management
- **Flow**: Human Question → LangGraph Agent → MCP Tools → Response

### Why MultiServerMCPClient?

Instead of connecting to one server at a time, `MultiServerMCPClient` allows you to:
- **Connect to multiple servers** (math, weather, office, etc.) simultaneously
- **Access all tools** from a single unified interface
- **Scale easily** by adding new servers without changing agent code

## Project Examples

This repository contains several examples demonstrating MCP integration:

- **`01_langgraph_agent_no_mcp.py`** - Basic LangGraph agent with local tools
- **`02_langgraph_agent_mcp.py`** - LangGraph agent with single MCP server
- **`03_langgraph_agent_mcp_multiply.py`** - LangGraph agent with multiple local MCP servers
- **`04_mcp_external_servers_local.py`** - LangGraph agent with local + external MCP servers

## Resources
- [MCP Servers Directory](https://mcpservers.org/) - Find more MCP servers
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters) - Official adapter library