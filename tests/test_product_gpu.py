import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from multishot.product_gpu import asset_gpu_interceptor, set_llm_sleep
from multishot.product_pipeline import generate_shot_videos


class ProductGPULifecycleTest(unittest.TestCase):
    def test_sleep_and_wake_are_idempotent_and_confirmed(self):
        sleeping = False
        transitions = []
        def respond(request):
            nonlocal sleeping
            if request.url.path == "/is_sleeping":
                return httpx.Response(200, json={"is_sleeping": sleeping})
            transitions.append(request.url.path)
            sleeping = request.url.path == "/sleep"
            return httpx.Response(200)
        client_type = httpx.Client
        with patch.dict("os.environ", {"MULTISHOT_VLLM_SLEEP_ENABLED": "1", "DASHSCOPE_BASE_URL": "http://127.0.0.1:8001/v1"}), patch(
            "multishot.product_gpu.httpx.Client",
            side_effect=lambda **kwargs: client_type(transport=httpx.MockTransport(respond)),
        ):
            set_llm_sleep(True)
            set_llm_sleep(True)
            set_llm_sleep(False)
            set_llm_sleep(False)
        self.assertFalse(sleeping)
        self.assertEqual(transitions, ["/sleep", "/wake_up"])

    def test_sleep_control_rejects_remote_llm(self):
        with patch.dict("os.environ", {
            "MULTISHOT_VLLM_SLEEP_ENABLED": "1",
            "DASHSCOPE_BASE_URL": "https://remote.example/v1",
        }):
            with self.assertRaisesRegex(ValueError, "local DASHSCOPE_BASE_URL"):
                set_llm_sleep(True)

    def test_failed_asset_tool_still_wakes_llm(self):
        events = []
        async def run():
            intercept = asset_gpu_interceptor()
            async def work(request):
                raise RuntimeError("asset failed")
            await intercept(None, work)
        with patch("multishot.product_gpu.set_llm_sleep", side_effect=events.append):
            with self.assertRaisesRegex(RuntimeError, "asset failed"):
                asyncio.run(run())
        self.assertEqual(events, [True, False])

    def test_parallel_asset_tools_serialize_sleep_work_and_wake(self):
        events = []
        async def run():
            intercept = asset_gpu_interceptor()
            async def work(request):
                events.append("start")
                await asyncio.sleep(0.01)
                events.append("end")
            await asyncio.gather(intercept(None, work), intercept(None, work))
        with patch("multishot.product_gpu.set_llm_sleep", side_effect=lambda x: events.append("sleep" if x else "wake")):
            asyncio.run(run())
        self.assertEqual(events, ["sleep", "start", "end", "wake"] * 2)

    def test_wan_uses_dedicated_interpreter_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "wan_manifest_product.json").write_text('{"jobs":[]}')
            (project / "wan_video_report.json").write_text('{"completed_jobs":1}')
            with patch("multishot.product_pipeline.subprocess.run") as run:
                result = generate_shot_videos(
                    project, wan_root=project, checkpoint_dir=project,
                    frame_num=9, sample_steps=2,
                )
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/root/autodl-tmp/wan22-venv/bin/python")
        self.assertIn("pretest.run_wan22_i2v_manifest", command)
        self.assertEqual(result["completed_jobs"], 1)


if __name__ == "__main__":
    unittest.main()
