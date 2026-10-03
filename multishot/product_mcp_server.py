"""Keep native/model logs separate from the product MCP stdio protocol."""

from __future__ import annotations

import os
import sys

import anyio
from mcp.server.stdio import stdio_server


async def run() -> None:
    # Keep a dedicated protocol descriptor before redirecting fd 1. This also
    # catches native library output that contextlib.redirect_stdout cannot.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1) as protocol:
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        from .mcp_asset_server import mcp

        async with stdio_server(stdout=anyio.wrap_file(protocol)) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream, write_stream, mcp._mcp_server.create_initialization_options()
            )


def main() -> None:
    anyio.run(run)


if __name__ == "__main__":
    main()
