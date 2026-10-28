#!/usr/bin/env python3
"""Submit once and poll Wan3 on the configured host. Errors stop the run."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from qwen140_integrated_episodes import sha, write_once
from qwen140_pilot_route_agent import config


def api_request(url: str, key: str, folder: Path, payload: dict | None = None) -> dict:
    headers = {"Authorization": "Bearer " + key}
    if payload is not None:
        headers.update({"Content-Type": "application/json", "X-DashScope-Async": "enable"})
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace").replace(key, "[REDACTED]")
        write_once(folder / "api_error.json", {"http_status": error.code, "response": detail})
        raise RuntimeError(f"Wan3 API HTTP {error.code}: {detail}") from error


def generate_video(first: Path, last: Path | None, prompt: str, seconds: int, folder: Path) -> Path:
    cfg = config()
    parts = urlsplit(cfg["DASHSCOPE_BASE_URL"])
    origin = parts.scheme + "://" + parts.netloc
    folder.mkdir(parents=True, exist_ok=True)
    files = [("first_frame", first)] + ([("last_frame", last)] if last else [])
    metadata = {"model": "wan3.0-video", "prompt": prompt, "duration": seconds,
                "resolution": "720P", "audio": False, "prompt_extend": False,
                "images": [{"type": kind, "path": str(path.resolve()), "sha256": sha(path)}
                           for kind, path in files]}
    write_once(folder / "request_metadata.json", metadata)
    if (folder / "video.mp4").exists():
        return folder / "video.mp4"
    create_file = folder / "create_response.json"
    if create_file.exists():
        response = json.loads(create_file.read_text())
    else:
        media = [{"type": kind, "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()}
                 for kind, path in files]
        payload = {"model": "wan3.0-video", "input": {"prompt": prompt, "media": media},
                   "parameters": {"resolution": "720P", "ratio": "adaptive", "duration": seconds,
                                  "audio": False, "prompt_extend": False, "seed": 42}}
        response = api_request(origin + "/api/v1/services/aigc/video-generation/video-synthesis",
                               cfg["DASHSCOPE_API_KEY"], folder, payload)
        write_once(create_file, response)
    if "task_id" not in response.get("output", {}):
        raise RuntimeError(f"Wan3 did not create a task: {response}")
    task_id = response["output"]["task_id"]
    for _ in range(180):
        result = api_request(origin + "/api/v1/tasks/" + task_id, cfg["DASHSCOPE_API_KEY"], folder)
        (folder / "latest_task.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        output = result.get("output", {})
        state = output.get("task_status")
        print("Wan3", state, flush=True)
        if state == "SUCCEEDED":
            write_once(folder / "completed_task.json", result)
            with urllib.request.urlopen(output["video_url"], timeout=120) as response:
                (folder / "video.mp4").write_bytes(response.read())
            return folder / "video.mp4"
        if state not in {"PENDING", "RUNNING"}:
            raise RuntimeError(f"Wan3 stopped: {result}")
        time.sleep(10)
    raise TimeoutError(f"Task {task_id} still pending; no replacement task submitted")
