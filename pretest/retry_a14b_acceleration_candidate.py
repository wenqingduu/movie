"""Retry the corrected 4-step request after the in-flight 40-step pilot saves.

The original supervisor is stopped while this helper owns the GPU slot; its
finally block restores the formal queue as soon as this helper resumes it.
"""
import json
import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from run_a14b_acceleration_trial import PROJECT, ROOT, sha, write

def still_running(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main(supervisor, pilot):
    try:
        deadline = time.monotonic() + 3500
        while still_running(pilot):
            if time.monotonic() > deadline:
                os.kill(pilot, signal.SIGTERM)
                raise TimeoutError("40-step pilot exceeded bounded wait")
            time.sleep(2)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", DTYPE="BF16", SENSITIVE_LAYER_DTYPE="None")
        for key in ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "GEMINI_API_KEYS"]:
            env.pop(key, None)
        print("Retry corrected 4-step request; formal supervisor will resume afterwards", flush=True)
        with (ROOT / "logs/lightx2v_distill_bf16_4step.log").open("a") as stream:
            process = subprocess.run(["/root/autodl-tmp/envs/lightx2v-trial/bin/python", "-u",
                    "pretest/run_a14b_acceleration_trial.py", "--route", "lightx2v_distill_bf16_4step"],
                    cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        write(ROOT / "corrected_4step_process.json", {"exit_code": process.returncode})
        frozen = json.loads((ROOT / "attempt1_pinned_host/formal_snapshot.json").read_text())
        for row in frozen["jobs"]:
            assert sha(row["output_video"]) == row["video_sha256"]
            assert sha(row["last_frame"]) == row["last_frame_sha256"]
        write(ROOT / "formal_preservation_six_shots.json", {"status": "passed", "preserved_shots": len(frozen["jobs"]),
              "video_and_tail_hashes_unchanged": True})
    finally:
        os.kill(supervisor, signal.SIGCONT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--pilot-pid", type=int, required=True)
    args = parser.parse_args()
    main(args.supervisor_pid, args.pilot_pid)
