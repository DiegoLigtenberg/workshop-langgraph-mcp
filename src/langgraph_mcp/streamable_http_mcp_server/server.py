"""
MCP Stdio to Streamable HTTP Bridge

Runs the official @supabase/mcp-server-supabase via stdio and exposes it via Streamable HTTP.
Provides HTTP access to 30+ Supabase MCP tools (made in typescript), converted to streamable http with bridge pattern.
"""

import os
import asyncio
import json
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Validate credentials
SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN")
SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF")

if not SUPABASE_ACCESS_TOKEN or not SUPABASE_PROJECT_REF:
    raise ValueError("Missing SUPABASE_ACCESS_TOKEN or SUPABASE_PROJECT_REF")

# Global subprocess for the MCP server
mcp_process: Optional[asyncio.subprocess.Process] = None
message_id_counter = 0
pending_responses = {}


async def start_mcp_subprocess():
    """Start the official Supabase MCP server as a subprocess"""
    global mcp_process

    print("Starting Supabase MCP server...")

    mcp_process = await asyncio.create_subprocess_exec(
        "npx",
        "-y",
        "@supabase/mcp-server-supabase",
        f"--access-token={SUPABASE_ACCESS_TOKEN}",
        f"--project-ref={SUPABASE_PROJECT_REF}",
        "--read-only",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    asyncio.create_task(read_mcp_responses())

    print("MCP server started successfully")
    return mcp_process


async def read_mcp_responses():
    """Continuously read JSON-RPC responses from MCP subprocess"""
    global mcp_process

    while mcp_process and mcp_process.stdout:
        try:
            line = await mcp_process.stdout.readline()
            if not line:
                print("MCP subprocess stdout closed - subprocess may have crashed")
                # Check if subprocess is still alive
                if mcp_process and mcp_process.returncode is not None:
                    print(f"MCP subprocess exited with code {mcp_process.returncode}")
                    # Clear all pending requests
                    for future in pending_responses.values():
                        if not future.done():
                            future.set_exception(RuntimeError("MCP subprocess crashed"))
                    pending_responses.clear()
                break

            response = json.loads(line.decode().strip())

            # Match response to pending request
            if "id" in response and response["id"] in pending_responses:
                future = pending_responses[response["id"]]
                if not future.done():
                    future.set_result(response)
                del pending_responses[response["id"]]

        except json.JSONDecodeError as e:
            print(f"Failed to parse MCP response: {e}")
            continue
        except Exception as e:
            print(f"Error reading MCP response: {e}")
            import traceback

            traceback.print_exc()
            break


async def send_mcp_request(request: dict, timeout: float = 30.0) -> dict:
    """Send a JSON-RPC request to MCP subprocess and wait for response"""
    global mcp_process, message_id_counter

    # Check if subprocess is still alive
    if not mcp_process:
        raise RuntimeError("MCP subprocess not running")

    if mcp_process.returncode is not None:
        raise RuntimeError(
            f"MCP subprocess crashed (exit code: {mcp_process.returncode})"
        )

    if not mcp_process.stdin:
        raise RuntimeError("MCP subprocess stdin not available")

    client_id = request.get("id")

    message_id_counter += 1
    internal_id = message_id_counter
    request["id"] = internal_id

    loop = asyncio.get_event_loop()
    future = loop.create_future()
    pending_responses[internal_id] = future

    request_json = json.dumps(request) + "\n"
    mcp_process.stdin.write(request_json.encode())
    await mcp_process.stdin.drain()

    try:
        response = await asyncio.wait_for(future, timeout=timeout)
        response["id"] = client_id
        return response
    except asyncio.TimeoutError:
        if internal_id in pending_responses:
            del pending_responses[internal_id]
        raise RuntimeError(f"MCP request timed out after {timeout}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    await start_mcp_subprocess()
    print("\n" + "=" * 60)
    print("MCP Stdio to Streamable HTTP Bridge")
    print("=" * 60)
    print("Running: @supabase/mcp-server-supabase (official)")
    print("Credentials: Server-side (30+ tools available)")
    print("Transport: Streamable HTTP")
    print("Ready for connections")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    global mcp_process
    if mcp_process:
        mcp_process.terminate()
        await mcp_process.wait()


# FastAPI app
app = FastAPI(title="Supabase MCP Workshop Server", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home(request: Request):
    """Server information endpoint"""
    base_url = str(request.base_url).rstrip("/")

    return {
        "name": "Supabase MCP Server",
        "version": "1.0.0",
        "transport": "streamable_http",
        "description": "Official Supabase MCP via HTTP bridge",
        "tools": "30+ tools available",
        "security": "Read-only mode",
        "client_config": {
            "supabase": {"url": base_url, "transport": "streamable_http"}
        },
    }


@app.post("/")
async def mcp_endpoint(request: Request):
    """
    Main MCP endpoint for Streamable HTTP transport.
    Per MCP spec, messages are sent to the base URL via POST.
    """
    message_id = 0
    try:
        message = await request.json()
        message_id = message.get("id", 0)
        method = message.get("method", "unknown")

        print(f"Received: {method} (id: {message_id})")

        if method.startswith("notifications/"):
            global mcp_process
            if mcp_process and mcp_process.stdin and mcp_process.returncode is None:
                request_json = json.dumps(message) + "\n"
                mcp_process.stdin.write(request_json.encode())
                await mcp_process.stdin.drain()
            print("Notification forwarded (no response expected)")
            return Response(status_code=204)

        response = await send_mcp_request(message)
        print(f"Responding to id: {response.get('id')}")
        return response

    except RuntimeError as e:
        # RuntimeError usually means subprocess issue
        error_msg = str(e)
        print(f"RuntimeError: {error_msg}")
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": f"Internal error: {error_msg}"},
            "id": message_id,
        }
    except Exception as e:
        import traceback

        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            "id": message_id,
        }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
