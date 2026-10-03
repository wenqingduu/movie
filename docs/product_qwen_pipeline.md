# Product Qwen Pipeline

This is the application route, not the paired evaluation protocol. Evaluation
entrypoints and reports remain unchanged.

## Data Flow

1. The local planning LLM writes characters, scenes, and shots to
   `project_plan.json`.
2. The asset agent calls MCP tools to generate background and character
   reference images. Their project-local paths are stored in `asset_index.json`.
3. FaceLift builds one Gaussian face per character. The product adapter runs
   the existing pose calibration at 512px and records the role-specific path.
4. Each shot loads its scene reference first and its scheduled character
   references in shot order. Qwen-Image-2.1 consumes all of them together.
5. One 20-step denoising trajectory predicts `pred_x0` at step 12. Character
   identity mapping, continuous 3D rendering, pose checks, color harmonization,
   and the existing v7 mask prepare per-role residuals. There is no Control
   generation, saved fork latent, or second denoising trajectory.
6. Residual strength is 0.4 with the existing small-face scaling. Each role uses
   one active step when step-12 identity cosine is at least 0.40, otherwise six.
   Overlapping contributions are normalized and applied synchronously.
7. Unusable roles are skipped without restarting sampling. Closed-eye prompts
   and shots without named characters finish without 3D injection. CUDA OOM
   is an error, not a silent successful fallback.
8. Only final first frames enter the Wan manifest. Wan runs in its own Python
   environment, then ffmpeg assembles shot videos in manifest order.

## Models And Processes

| Stage | Default |
| --- | --- |
| Planning and asset tool calls | Local Qwen2.5-14B-Instruct-AWQ, vLLM port 8001 |
| Background and identity assets | SDXL Base 1.0, 1024px, 30 steps, guidance 5.0 |
| 3D assets | FaceLift, per-character Gaussian and pose calibration |
| First frames | Qwen-Image-2.1, 1024x576, sequential CPU offload |
| Videos | Wan2.2-TI2V-5B in `wan22-venv` |

SDXL matches the evaluation asset model. The product asset agent currently
generates one reference per character; it does not reproduce the benchmark's
three-candidate portrait selection. Do not describe those asset workflows as
identical.

The product MCP entrypoint is `multishot.product_mcp_server`. It reserves a
separate JSON-RPC descriptor and sends Python/native model logs to stderr.
Asset tools use short-lived processes; first frames share one MCP process for
all shots so Qwen weights load once. That process closes before Wan starts.
Celery disables `worker_redirect_stdouts`: its `LoggingProxy` lacks the real
stderr descriptor required by the MCP subprocess transport. Model logs remain
on stderr and do not enter the JSON-RPC stream.

## Local GPU Lifecycle

`scripts/start_local_llm.sh` enables vLLM sleep and Hermes automatic tool calls.
Sleep-control endpoints are development endpoints: keep this service bound to
localhost.

`scripts/start_local_worker.sh` configures the local LLM, enables sleep control,
and uses the installed ONNX CUDA libraries when available. Planning wakes the
LLM; each asset tool sleeps it and wakes it after the tool process exits.
FaceLift, calibration, first frames, and video generation leave it asleep.
The next planning request wakes it again.

Celery project jobs acquire `product_gpu_lock()` across the complete project.
This protects both GPU scheduling and the shared sleep state even if worker
concurrency increases. Direct Python/CLI callers must acquire the same lock
themselves when running alongside workers.

Overrides include `MULTISHOT_ASSET_GENERATION_MODEL`,
`MULTISHOT_DIFFUSION_MODEL_PATH`, `MULTISHOT_FACE_CALIBRATION_SIZE`,
`QWEN_IMAGE21_PYTHON`, `QWEN_IMAGE21_OFFLOAD_MODE`, and `WAN_PYTHON`.
Cloud LLM callers should set `MULTISHOT_VLLM_SLEEP_ENABLED=0`; sleep control
rejects remote LLM URLs.

Start the local LLM, MariaDB, Redis, backend, and worker before submitting work.
The backend serves `frontend/dist` when built; the Vite development server is
available separately on port 3000. The application accepts only `qwen_image21`
as its first-frame backend.

## Verification

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
bash -n scripts/start_local_llm.sh scripts/start_local_worker.sh
git diff --check
```

Earlier local integration testing generated a new background and character reference,
built FaceLift, calibrated all 273 camera-grid samples, produced an injected
1024x576 first frame, and passed it through Wan and ffmpeg. Wan testing used
9 frames and 2 sampling steps for wiring checks, not production quality.
The chain was resumed after aligning calibration settings.

Measured first-frame PyTorch peak allocation was approximately 8 GiB with
sequential CPU offload. Local vLLM sleep reduced whole-card occupancy from
36,910 MiB to 628 MiB and woke successfully afterward. This does not
establish safe or faster two-Qwen concurrency; that has not been benchmarked.
No evaluation metrics or claims of identity improvement are implied.

### Full Three-Shot Application Run

Project `proj_30_a45ac642` completed through the actual HTTP API and Celery
queue, starting from one Chinese story instruction about a mechanic in a
garage. The local LLM planned one character, one scene, and three shots; the
asset agent generated new SDXL references inside this project. No benchmark
assets or manually supplied plan were used, and the successful run did not
resume from an intermediate stage.
The test input explicitly requested one character, one scene, and exactly
three shots. The LLM produced the structured plan and prompts under those
constraints; the shot count was not chosen autonomously.

- FaceLift and all 273 pose-calibration samples completed for the character.
- All three Qwen first frames used the same scene and character references,
  ran 20 steps, and applied the step-12 3D residual without skips or role
  preparation failures. No Control image or denoising fork was generated.
- Wan used the product defaults: 49 frames per shot, 50 sampling steps,
  24 fps, and requested size `1280*704`. Actual video size was 1248x704.
- All three videos contain changing frames. Ordered ffmpeg assembly produced
  147 frames and 6.125 seconds; the project and all shot database records
  reached `succeeded`, and the final-video HTTP range request returned 206.
- The queue run completed in approximately 15.5 minutes without CUDA OOM.
  After inference ended, whole-card usage returned to the sleeping LLM's
  approximately 628 MiB.

The initial API attempt (`proj_30_b0344935`) failed at asset MCP startup with
`LoggingProxy.fileno`; it is retained as a failure record. The worker setting
above fixes that failure, with a regression test for real stdio preservation.

Evidence is stored under
`outputs/projects/proj_30_a45ac642/verification/`: `execution_monitor.json`,
`chain_verification.json`, and extracted middle frames for all three videos.
The chain audit checks reference paths/hashes, Gaussian calibration provenance,
first-frame/video input hashes, shot order, frame counts, non-frozen videos,
database status, and final-video HTTP delivery. It is not an identity or
visual-quality benchmark.

Known quality issue: `shot_003` duplicates the red background car even though
its scene reference has one car; the duplication persists in the generated
video. At the user's request, this shot was retained without regeneration and
the observation is recorded in `chain_verification.json`. Successful workflow
execution does not mean all visual-quality checks passed.

Browser evidence includes `frontend_completed_desktop.png` and
`frontend_completed_mobile.png`. The desktop page renders completed statuses
and the final-video preview. The existing layout overflows horizontally at a
390px viewport; this UI issue remains open and is separate from chain success.
