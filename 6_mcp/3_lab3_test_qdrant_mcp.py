import asyncio
import subprocess
import traceback
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    StdioConnectionParams,
    StdioServerParameters,
)


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

INSTRUCTIONS = """
You are testing a Qdrant MCP connection.

Use the available Qdrant tools to store information in the
knowledge base when appropriate.

If the Qdrant tools are available, say that the connection works.
"""


REQUEST = """
Test the Qdrant knowledge base.

Store this fact in the knowledge base:

"NVIDIA is a technology company known for its GPUs and AI computing hardware."

After storing it, confirm that the Qdrant MCP server is working.
"""


# ---------------------------------------------------------
# 2. Qdrant MCP configuration
# ---------------------------------------------------------

# Use an absolute path so there is no ambiguity about
# where Qdrant's local database is created.
vectordb_path = (Path(__file__).parent / "memory" / "qdrant").resolve()

print("=" * 60)
print("QDRANT MCP TEST")
print("=" * 60)

print(f"Python file : {Path(__file__).resolve()}")
print(f"Qdrant path : {vectordb_path}")
print()


vectorstore_params = StdioServerParameters(
    command="uvx",
    args=[
        "mcp-server-qdrant",
    ],
    env={
        "QDRANT_LOCAL_PATH": str(vectordb_path),
        "COLLECTION_NAME": "knowledge",
    },
)


# ---------------------------------------------------------
# 3. Create MCP toolset
# ---------------------------------------------------------

vectordb_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=vectorstore_params,
        timeout=120,
    )
)


# ---------------------------------------------------------
# 4. Create ADK agent
# ---------------------------------------------------------

agent = LlmAgent(
    name="VectorDB_test_agent",
    instruction=INSTRUCTIONS,

    # IMPORTANT:
    # Replace this with the model you are already using.
    model="gemini-3.1-flash-lite",

    tools=[
        vectordb_server,
    ],
)


# ---------------------------------------------------------
# 5. Run the test
# ---------------------------------------------------------

async def main():

    print("Creating ADK runner...")
    
    runner = InMemoryRunner(
        agent=agent
    )

    print("Running agent...")
    print()

    try:

        events = await runner.run_debug(REQUEST)

        print()
        print("=" * 60)
        print("FINAL RESPONSE")
        print("=" * 60)

        found_final = False

        for event in events:

            if (
                event.is_final_response()
                and event.content
                and event.content.parts
            ):

                found_final = True

                for part in event.content.parts:

                    if hasattr(part, "text") and part.text:
                        print(part.text)

        if not found_final:
            print("No final response was found.")

        print()
        print("=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)

    except Exception as e:

        print()
        print("=" * 60)
        print("TEST FAILED")
        print("=" * 60)

        print()
        print("Exception:")
        print(repr(e))

        print()
        print("Full traceback:")
        traceback.print_exc()

        # ExceptionGroup handling
        if hasattr(e, "exceptions"):

            print()
            print("=" * 60)
            print("SUB-EXCEPTIONS")
            print("=" * 60)

            def print_exception_group(exc, level=0):

                prefix = "  " * level

                print(
                    f"{prefix}{type(exc).__name__}: {exc}"
                )

                if hasattr(exc, "exceptions"):

                    for sub in exc.exceptions:
                        print_exception_group(
                            sub,
                            level + 1
                        )

            print_exception_group(e)


# ---------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())