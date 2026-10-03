import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ProductMCPTransportTest(unittest.TestCase):
    def test_worker_keeps_real_stdio_for_mcp_subprocesses(self):
        from app.workers.celery_app import celery_app

        self.assertFalse(celery_app.conf.worker_redirect_stdouts)

    def test_python_and_native_stdout_do_not_corrupt_tool_response(self):
        script = '''
import os
from multishot.mcp_asset_server import mcp
from multishot.product_mcp_server import main
@mcp.tool()
def noisy_probe() -> dict:
    print("model python log", flush=True)
    os.write(1, b"model native log\\n")
    return {"ok": True}
main()
'''
        async def check(log):
            params = StdioServerParameters(
                command=sys.executable, args=["-c", script], env=os.environ.copy(),
            )
            async with stdio_client(params, errlog=log) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("noisy_probe", {})
                    self.assertFalse(result.isError)
                    self.assertIn('"ok":true', result.content[0].text.replace(" ", ""))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stderr.log"
            with path.open("w") as log:
                asyncio.run(check(log))
            output = path.read_text()
        self.assertIn("model python log", output)
        self.assertIn("model native log", output)


if __name__ == "__main__":
    unittest.main()
