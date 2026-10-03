"""Serialize product GPU work and coordinate the local vLLM sleep lifecycle."""

from __future__ import annotations

import asyncio
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx


def _sleep_enabled() -> bool:
    return os.getenv("MULTISHOT_VLLM_SLEEP_ENABLED", "0").lower() in {"1", "true", "yes"}


def _control_url(path: str) -> str:
    base = os.getenv("DASHSCOPE_BASE_URL", "http://127.0.0.1:8001/v1")
    parsed = urlsplit(base)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("vLLM sleep control requires a local DASHSCOPE_BASE_URL")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def set_llm_sleep(sleeping: bool) -> None:
    if not _sleep_enabled():
        return
    with httpx.Client(timeout=180, trust_env=False) as client:
        status = client.get(_control_url("/is_sleeping"))
        status.raise_for_status()
        if bool(status.json()["is_sleeping"]) == sleeping:
            return
        response = client.post(
            _control_url("/sleep" if sleeping else "/wake_up"),
            params={"level": 1} if sleeping else None,
        )
        response.raise_for_status()
        status = client.get(_control_url("/is_sleeping"))
        status.raise_for_status()
        if bool(status.json()["is_sleeping"]) != sleeping:
            raise RuntimeError("vLLM did not reach the requested sleep state")


@contextmanager
def product_gpu_lock():
    path = Path(os.getenv(
        "MULTISHOT_PRODUCT_GPU_LOCK",
        str(Path(__file__).resolve().parents[1] / "outputs" / "product_gpu.lock"),
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def asset_gpu_interceptor():
    # Agent tool calls may be parallel; sleep/wake must encompass one complete
    # MCP subprocess lifetime before the next call changes the LLM state.
    lock = asyncio.Lock()

    async def intercept(request, handler):
        async with lock:
            await asyncio.to_thread(set_llm_sleep, True)
            try:
                return await handler(request)
            finally:
                await asyncio.to_thread(set_llm_sleep, False)

    return intercept
