import asyncio
import os
import traceback
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    vectordb_path = (
        Path(__file__).parent / "memory" / "qdrant"
    ).resolve()

    print("Qdrant path:")
    print(vectordb_path)
    print()

    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-qdrant"],
        env={
            **os.environ,
            "QDRANT_LOCAL_PATH": str(vectordb_path),
            "COLLECTION_NAME": "knowledge",
        },
    )

    print("Starting MCP server...")

    try:

        async with stdio_client(server_params) as (read, write):

            print("MCP process started.")
            print("Initializing MCP session...")

            async with ClientSession(read, write) as session:

                await session.initialize()

                print("MCP session initialized!")

                tools = await session.list_tools()

                print("\nAvailable tools:")
                print("=" * 60)

                for tool in tools.tools:
                    print(f"\nName: {tool.name}")
                    print(f"Description: {tool.description}")

                print("\n" + "=" * 60)
                print("SUCCESS")


    except BaseException as e:

        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)

        traceback.print_exception(e)


if __name__ == "__main__":
    asyncio.run(main())