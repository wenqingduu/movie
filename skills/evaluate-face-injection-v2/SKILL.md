---
name: evaluate-face-injection-v2
description: Evaluate or audit movie videos using official EntityBench metrics, with bounded episode concurrency, input-verified resume and canonical aggregation. Use for full-story or benchmark video evaluation; use the original evaluate-face-injection skill for paired InsightFace/reference-DINO plugin diagnostics.
---

# Face Injection Evaluation v2

Use `/root/autodl-tmp/movie`. Evaluate existing videos using official EntityBench algorithms; a Control/Treatment pair or a particular video model is not required. Do not generate media when only evaluation is requested.

Read `LLM_HANDOFF.md`, `docs/entitybench_20ep_evaluation.md`, the requested run's manifest and current status. Resolve the cohort from its split, not a remembered episode count. Preserve unrelated worktree changes. Reports are the source of truth; do not embed experiment scores in this skill.

## Metric scope

- Freeze evaluator revision, pristine upstream file, any scope adapter, SHA256 and diff. Preserve official sampling, detector/CLIP thresholds, representative-crop choice, DINO features/centroid, boundaries and aggregation.
- The current pinned evaluator has **six** active VBench dimensions: subject_consistency, temporal_flickering, motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality. Some upstream docstrings still say seven; follow the version's actual constant.
- Official `cs_face` compares grounded **character regions** against a cross-shot appearance centroid. It is not InsightFace, similarity to an identity reference, or the project's face-box centroid. Do not copy historical project scores into the official table.
- The user's current scope is **official non-VLM only**: audited `--skip_vlm`, strict mode, no config overrides, no VLM credentials passed, final calls = 0, and missing metrics retained as null.
- Appearance fidelity, action alignment and LLM consistency are absent without VLM; the fidelity gate is unavailable. Label `cs_face/cs_object` provisional and scope `partial_without_vlm`. Do not claim equivalence to full paper tables. Later authorized VLM scoring must reapply the gate and recompute affected consistency, not just append fields.
- Add legacy identity diagnostics only when requested and keep them separate. Use the original skill's mapping/tracking rules for those diagnostics.

## Prefer bounded parallel evaluation

Schedule **complete episodes**. Centroids and continuation boundaries require the whole episode; do not split its shots across independent scoring jobs or average shard summaries.

On the current single 48GB GPU, start with **two evaluation processes on GPU 0**, normally four CPU threads each. Inspect free GPU memory, container CPU quota and host memory. Before widening a new expensive evaluation, measure a representative serial-versus-two-worker sample in a separate validation directory: wall time, peak GPU/host memory and score agreement. Increase to three/four only when measurement shows better total throughput and enough headroom. Low allocated memory does not prove spare compute; do not promise linear speedup. Avoid sharing the GPU with heavy generation unless co-running has been profiled and chosen.

Give workers disjoint splits and independent output/work/log directories, with the same frozen evaluator/model/config. Share cached weights read-only; finish first-time downloads before concurrent inference. Bound CPU/BLAS threads per worker.

Use [scripts/evaluate_parallel.py](scripts/evaluate_parallel.py) for the current manifest/generation-report/split/frozen-runtime contract. Read [execution details](references/official_parallel.md) for commands, paths and adaptation. The helper supports two workers on one GPU, weighted whole-episode partitioning, input-verified resume, independent caches and official aggregation.

The upstream `benchmarks/entitybench/eval/run_eval_distributed.py` assumes distinct GPUs and has no switch for this run's non-VLM adapter. Do not use `--n_gpus 2` on a one-GPU machine as a substitute. For another evaluator version, adapt orchestration while preserving score logic and test the changed contract.

## Resume and validate

- Verify canonical `<episode_id>/<scene:03d>_<shot:03d>.mp4` mappings against generation SHA, and hash scripts, full split and evaluator. Upstream episode caches do not themselves hash videos. Reject changed inputs/plans or use a fresh score directory; do not delete historic results.
- For continuation audits, verify actual child input = decoded own-route previous tail, parent video SHA, original order and cut flags. Run numbers may cover several windows; use full episode_id in output names.
- Diagnose failed workers, preserve logs and resume valid caches without changing metrics or borrowing another route's media. Do not declare complete while a worker or episode is missing.
- After successful worker exits, verify every expected episode exactly once, full shot counts, six applicable VBench values/coverage/failures, compatible manifests and zero VLM calls. Preserve official entity/boundary skipped/failed/missing counts. Call the **official deserializer and aggregator** once over the original full split. Save manifests, input fingerprints and execution provenance with JSON/Markdown.
- Compare serial and parallel per-episode and aggregate results in the validation sample. Explain drift before widening concurrency; never change crop selection or denominators to force agreement.
- Inspect representative single-character, multi-character, fallback, continuation and drift videos/contact sheets. Reuse valid sheets. Distinguish metric completion from pending visual review; process success is not an effect conclusion.

## Report

Lead with stage/coverage, then official results and paths. State cohort, aggregation unit, evaluator, VLM/gate status, missing items, concurrency and measured cost. A single route yields absolute system performance, not invented C/T deltas or plugin causality. Update requested run documentation when completing its handoff work. This skill alone does not authorize GitHub pushes or new paid API calls.
