from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
import os
import platform
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Platform detection
IS_WINDOWS = platform.system() == "Windows"

# Supabase MCP server configuration
SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN", "")
SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "")


def get_command_for_platform(command: str, args: list[str]) -> tuple[str, list[str]]:
    """
    Returns platform-appropriate command and args.
    On Windows, wraps npx/npm/node commands with 'cmd /c'.

    Args:
        command: The command to run (e.g., "npx", "python", "uv")
        args: List of arguments for the command

    Returns:
        Tuple of (command, args) adjusted for the platform
    """
    if IS_WINDOWS and command in ["npx", "npm", "node"]:
        return "cmd", ["/c", command] + args
    return command, args


def get_llm(llm_type="openai"):
    """
    Returns an LLM instance.
    llm_type: "qwen" (default) or "openai"
    """
    if llm_type == "openai":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        version = os.getenv("AZURE_OPENAI_MODEL_VERSION", "2024-08-01-preview")
        return AzureChatOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=version, model=model
        )
    else:
        return ChatOllama(model="qwen3:8b")


if __name__ == "__main__":
    llm = get_llm(llm_type="openai")
    print(llm.invoke("Hello, how are you?").content)
