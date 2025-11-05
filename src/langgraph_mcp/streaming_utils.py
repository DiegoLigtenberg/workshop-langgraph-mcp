"""Shared streaming utilities for LangGraph chat endpoints"""

from fastapi import Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
import uuid


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
        if verbose:
            print("Event:", event_type, event.get("name"))

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
            tool_id = event.get("data", {}).get("run_id") or tool_name
            tool_output = event.get("data", {}).get("output", "")
            if tool_id not in tool_results_shown:
                yield f"\n__TOOL_CALL_RESULT__:Tool '{tool_name}' returned: {tool_output}\n"
                tool_results_shown.add(tool_id)

        # Capture final message when agent is done (finish_reason='stop')
        if event_type == "on_chain_end" and final_message is None:
            event_name = event.get("name", "")
            tags = event.get("tags", {})
            # Only process top-level graph end
            if event_name in ("LangGraph", "") and "node" not in tags:
                messages = event.get("data", {}).get("output", {}).get("messages", [])

                if messages:
                    if verbose:
                        print(
                            f"Checking {len(messages)} messages for final AIMessage with finish_reason='stop'"
                        )

                    # Find final AI message with finish_reason='stop'
                    from langchain_core.messages import AIMessage

                    for msg in reversed(messages):
                        if (
                            isinstance(msg, AIMessage)
                            and hasattr(msg, "content")
                            and msg.content
                        ):
                            finish_reason = getattr(msg, "response_metadata", {}).get(
                                "finish_reason", ""
                            )
                            if verbose:
                                print(
                                    f"Found AIMessage with finish_reason='{finish_reason}', content length: {len(str(msg.content))}"
                                )
                            if finish_reason == "stop":
                                msg_content = (
                                    msg.content
                                    if isinstance(msg.content, str)
                                    else str(msg.content)
                                )
                                if msg_content.strip():
                                    final_message = msg_content
                                    if verbose:
                                        print(
                                            f"Captured final message (tools_used={tools_were_used}): {final_message[:100]}..."
                                        )
                                    break

    # After all events are processed, send final message if we captured it
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
    # Enable verbose for debugging "name a song" issue
    debug_verbose = True  # Temporarily enable to debug

    # Ensure thread_id is valid and unique
    if not thread_id or not thread_id.strip():
        # Generate a unique thread_id for this session
        thread_id = str(uuid.uuid4())
        print(f"Generated new thread_id: {thread_id}")
    else:
        print(f"Using thread_id: {thread_id}")

    langgraph_app = request.app.state.langgraph_app

    async def event_stream():
        try:
            async for chunk in create_event_stream(
                langgraph_app, user_input, thread_id, debug_verbose
            ):
                yield chunk
                # Ensure each chunk is flushed
                if verbose:
                    print(f"Yielded chunk of length {len(chunk)}")
        except Exception as e:
            if verbose:
                print(f"Error in event stream: {e}")
            raise
        finally:
            if verbose:
                print("Event stream completed")

    return StreamingResponse(event_stream(), media_type="text/plain")
