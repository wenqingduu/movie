"""Yield the GPU at a saved shot, run a bounded trial, restore original pipeline."""
from __future__ import annotations

import json
import argparse
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from run_a14b_acceleration_trial import PROJECT, ROOT, sha, write

FORMAL = PROJECT / "outputs/entitybench_pulid_i2v_a14b_continuation_20ep"
PYTHON = "/root/autodl-tmp/envs/lightx2v-trial/bin/python"


def process_ids(fragment):
    matches = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory / "cmdline").read_bytes().split(b"\0")
            if fragment.encode() in args and b"python" in args[0]:
                matches.append(int(directory.name))
        except (FileNotFoundError, PermissionError, IndexError):
            pass
    return matches


def status(phase, **fields):
    write(ROOT / "status.json", dict(phase=phase, updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields))


def exists(pid):
    return Path(f"/proc/{pid}").exists()


def restore():
    shell = f"cd {shlex.quote(str(PROJECT))} && export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 && "
    shell += f"{PROJECT}/.venv/bin/python -u pretest/run_entitybench_a14b_continuation_20ep.py >> {FORMAL}/logs/pipeline.log 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", "entitybench_a14b20", shell], check=True)


def main(yield_now=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    logroot = ROOT / "logs"
    logroot.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", DTYPE="BF16", SENSITIVE_LAYER_DTYPE="None")
    for key in ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "GEMINI_API_KEYS"]:
        env.pop(key, None)
    previous_snapshot = None
    previous_path = ROOT / "formal_snapshot.json"
    if previous_path.exists():
        previous_snapshot = json.loads(previous_path.read_text())
    for route in ["lightx2v_distill_bf16_4step", "lightx2v_bf16_40step"]:
        with (logroot / f"{route}_preflight.log").open("a") as stream:
            subprocess.run([PYTHON, "pretest/run_a14b_acceleration_trial.py", "--route", route, "--preflight"],
                           cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    drivers = process_ids("pretest/run_entitybench_a14b_continuation_20ep.py")
    generators = process_ids("pretest/run_wan22_i2v_a14b_continuation.py")
    assert len(drivers) == len(generators) == 1, (drivers, generators)
    driver, generator = drivers[0], generators[0]
    formal_report = json.loads((FORMAL / "generation_report.json").read_text())
    if yield_now and formal_report.get("status") != "loading_model":
        raise ValueError("--yield-now is only for retrying during formal model initialization")
    initial = max(len(formal_report["jobs"]), len((previous_snapshot or {}).get("jobs", [])))
    status("waiting_for_saved_formal_shot", completed_shots=initial, target_completed_shots=initial + 1,
           formal_driver_pid=driver, formal_generator_pid=generator)
    print(f"Waiting for formal shot {initial + 1} before yielding GPU" if not yield_now else
          f"Retry: yielding during formal initialization, with {initial} shots already saved", flush=True)
    deadline = time.monotonic() + 2700
    while not yield_now and len(json.loads((FORMAL / "generation_report.json").read_text())["jobs"]) <= initial:
        assert exists(driver) and exists(generator), "Formal pipeline exited while waiting"
        if time.monotonic() > deadline:
            raise TimeoutError("No new saved shot in 45 minutes; formal pipeline left running")
        time.sleep(.2)
    yielded = False
    try:
        # Stop the waiting parent before interrupting its GPU child. No code/hash change.
        os.kill(driver, signal.SIGSTOP)
        yielded = True
        os.kill(generator, signal.SIGINT)
        deadline = time.monotonic() + 15
        while exists(generator) and time.monotonic() < deadline:
            time.sleep(.1)
        if exists(generator):
            os.kill(generator, signal.SIGTERM)
        os.kill(driver, signal.SIGTERM)
        os.kill(driver, signal.SIGCONT)
        deadline = time.monotonic() + 20
        while subprocess.run(["tmux", "has-session", "-t", "entitybench_a14b20"], capture_output=True).returncode == 0:
            if time.monotonic() > deadline:
                raise TimeoutError("Original tmux session did not exit")
            time.sleep(.2)
        snapshot = json.loads((FORMAL / "generation_report.json").read_text())
        # Model startup temporarily resets the progress report to zero. Keep the
        # saved checkpoint inventory until the normal resume scan repopulates it.
        if previous_snapshot and len(previous_snapshot["jobs"]) > len(snapshot["jobs"]):
            for row in previous_snapshot["jobs"]:
                assert sha(row["output_video"]) == row["video_sha256"]
                assert sha(row["last_frame"]) == row["last_frame_sha256"]
            snapshot = previous_snapshot
        assert all(row["status"] in {"generated", "resumed_verified"} for row in snapshot["jobs"])
        write(ROOT / "formal_snapshot.json", snapshot)
        frozen_hash = sha(FORMAL / "generation_manifest.json")
        write(FORMAL / "status.json", {"phase": "gpu_yielded_for_isolated_acceleration_trial",
              "completed_shots": len(snapshot["jobs"]), "trial_status": str(ROOT / "status.json"),
              "formal_config_unchanged": True, "automatic_resume_after_trial": True, "vlm_enabled": False})
        results = []
        for route in ["lightx2v_distill_bf16_4step", "lightx2v_bf16_40step"]:
            status("generating_trial", route=route, formal_completed_shots=len(snapshot["jobs"]))
            print(f"Starting isolated {route}", flush=True)
            try:
                with (logroot / f"{route}.log").open("a") as stream:
                    process = subprocess.run([PYTHON, "-u", "pretest/run_a14b_acceleration_trial.py", "--route", route],
                        cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
                results.append({"route": route, "exit_code": process.returncode})
            except subprocess.TimeoutExpired:
                results.append({"route": route, "status": "timed_out"})
        assert sha(FORMAL / "generation_manifest.json") == frozen_hash
        for row in snapshot["jobs"]:
            assert sha(row["output_video"]) == row["video_sha256"]
            assert sha(row["last_frame"]) == row["last_frame_sha256"]
        write(ROOT / "trial_process_results.json", results)
        write(ROOT / "formal_preservation.json", {"status": "passed", "preserved_shots": len(snapshot["jobs"]),
              "generation_manifest_sha256": frozen_hash, "video_and_tail_hashes_unchanged": True})
    finally:
        if yielded:
            if exists(driver):
                os.kill(driver, signal.SIGTERM)
                os.kill(driver, signal.SIGCONT)
            if exists(generator):
                os.kill(generator, signal.SIGTERM)
            deadline = time.monotonic() + 20
            while subprocess.run(["tmux", "has-session", "-t", "entitybench_a14b20"], capture_output=True).returncode == 0 and time.monotonic() < deadline:
                time.sleep(.2)
            restore()
            status("trial_generation_finished_formal_resumed", results=locals().get("results", []),
                   formal_resume_tmux="entitybench_a14b20", quality_evaluation="pending")
            print("Original formal pipeline resumed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yield-now", action="store_true", help="Retry during formal initialization; retain every saved shot")
    main(parser.parse_args().yield_now)
