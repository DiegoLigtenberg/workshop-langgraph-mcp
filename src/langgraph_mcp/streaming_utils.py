"""Shared streaming utilities for LangGraph chat endpoints"""

from fastapi import Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import uuid
import re
import json


async def create_event_stream(
    langgraph_app, user_input: str, thread_id: str, verbose: bool = False
):
    """Create an async generator that streams LangGraph events to the frontend"""
    config = {"configurable": {"thread_id": thread_id}}
    tool_results_shown = set()
    tools_were_used = False
    final_message = None

    async for event in langgraph_app.astream_events(
        {"messages": [HumanMessage(content=user_input)]}, config=config
    ):
        event_type = event.get("event")

        # Stream AI response chunks
        if event_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content + " "

        # Tool calls
        if event_type == "on_tool_start":
            tools_were_used = True
            tool_name = event.get("name", "tool")
            tool_args = event.get("data", {}).get("input", {})
            yield f"\n__TOOL_CALL__:Calling tool '{tool_name}' with args {tool_args}\n"

        if event_type == "on_tool_end":
            tool_name = event.get("name", "tool")
            # LangGraph on_tool_end events have run_id at the top level (unique UUID per tool call)
            tool_id = event.get("run_id")
            tool_output = event.get("data", {}).get("output", "")

            if isinstance(tool_output, ToolMessage):
                tool_output = tool_output.content

            tool_output = _clean_tool_output(str(tool_output))

            if verbose:
                print(f"\n{'='*60}")
                print(f"Tool Result: {tool_name} (id: {tool_id})")
                print(f"{'='*60}")
                print(tool_output)
                print(f"{'='*60}\n")

            if tool_id not in tool_results_shown:
                yield f"\n__TOOL_CALL_RESULT__:Tool '{tool_name}' returned: {tool_output}\n"
                tool_results_shown.add(tool_id)

        if event_type == "on_chain_end" and final_message is None:
            event_name = event.get("name", "")
            tags = event.get("tags", {})
            if event_name in ("LangGraph", "") and "node" not in tags:
                messages = event.get("data", {}).get("output", {}).get("messages", [])
                if messages:
                    final_message = _extract_final_message(messages)
                    if final_message and verbose:
                        print(
                            f"Captured final message (tools_used={tools_were_used}): {final_message[:100]}..."
                        )

    if final_message:
        if verbose:
            print(f"Sending final message after event loop: {final_message[:100]}...")
        yield f"\n__FINAL__:{final_message}"
        if verbose:
            print("Final message sent successfully")
    else:
        if verbose:
            print("WARNING: No final message captured")


async def chat_endpoint_handler(
    request: Request, user_input: str, thread_id: str = None, verbose: bool = False
):
    """Handle chat endpoint - streams LangGraph events to frontend"""
    # Ensure thread_id is valid and unique
    if not thread_id or (isinstance(thread_id, str) and not thread_id.strip()):
        thread_id = str(uuid.uuid4())
        print(f"Generated new thread_id: {thread_id}")
    else:
        print(f"Using thread_id: {thread_id}")

    langgraph_app = request.app.state.langgraph_app
    return StreamingResponse(
        create_event_stream(langgraph_app, user_input, thread_id, verbose),
        media_type="text/plain",
    )


def _clean_tool_output(tool_output: str) -> str:
    """
    Extract and pretty-print JSON content from Supabase MCP tool output.
    MCP server wraps tool results in JSON.stringify().
    """
    # Parse outer JSON (MCP server wraps all results in JSON.stringify)
    try:
        outer_parsed = json.loads(tool_output)
        if isinstance(outer_parsed, str):
            inner_output = outer_parsed
        else:
            return json.dumps(outer_parsed, indent=2)
    except (json.JSONDecodeError, ValueError):
        inner_output = tool_output

    # Extract JSON from <untrusted-data> tags if present
    uuid_match = re.search(r"<untrusted-data-([^>]+)>", inner_output)
    if uuid_match:
        uuid = uuid_match.group(1)
        pattern = rf"<untrusted-data-{re.escape(uuid)}>(.*?)</untrusted-data-{re.escape(uuid)}>"
        match = re.search(pattern, inner_output, re.DOTALL)
        if match:
            # Extract JSON data between tags, removing any verbose text before/after
            json_data = match.group(1).strip()
            json_data = re.sub(r"^[^[{]*", "", json_data)
            last_bracket = max(json_data.rfind("]"), json_data.rfind("}"))
            if last_bracket >= 0:
                json_data = json_data[: last_bracket + 1]
            json_data = json_data.strip()
            try:
                parsed_json = json.loads(json_data)
                return json.dumps(parsed_json, indent=2)
            except (json.JSONDecodeError, ValueError):
                return json_data

    # Try to parse as JSON
    try:
        parsed_json = json.loads(inner_output)
        return json.dumps(parsed_json, indent=2)
    except (json.JSONDecodeError, ValueError):
        return inner_output


def _extract_final_message(messages: list) -> str | None:
    """Extract final AIMessage with finish_reason='stop' from message list"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            finish_reason = getattr(msg, "response_metadata", {}).get(
                "finish_reason", ""
            )
            if finish_reason == "stop":
                msg_content = str(msg.content)
                if msg_content.strip():
                    return msg_content
