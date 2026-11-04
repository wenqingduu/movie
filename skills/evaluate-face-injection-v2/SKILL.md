---
name: evaluate-face-injection-v2
description: Evaluate movie videos with EntityBench metrics and the project's default visible-single-face, localization-reviewed character-region DINO, using input-verified resume and episode aggregation. Use for full-story or benchmark evaluation; use the original evaluate-face-injection skill for paired InsightFace/reference-face DINO ablations.
---

# Face Injection Evaluation v2

Use `/root/autodl-tmp/movie`. Evaluate existing videos; a Control/Treatment pair or a particular video model is not required. By the user's standing choice, the default character-consistency result is **visible-single-face filtering plus confirmed localization correction on original character-region DINO**. Preserve official raw metrics alongside it. Do not generate media when only evaluation is requested.

Read `LLM_HANDOFF.md`, the requested run's manifest/status and [the default DINO protocol](references/visible_single_face_region_dino.md) before scoring or comparing character consistency. Resolve the cohort from its split, not a remembered episode count. Preserve unrelated worktree changes. Reports are the source of truth; do not embed experiment scores in this skill.

## Metric scope

- Freeze evaluator revision, pristine upstream file, any scope adapter, SHA256 and diff. Preserve official sampling, detector/CLIP thresholds, representative-crop choice, DINO features/centroid, boundaries and aggregation.
- The current pinned evaluator has **six** active VBench dimensions: subject_consistency, temporal_flickering, motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality. Some upstream docstrings still say seven; follow the version's actual constant.
- Official `cs_face` compares grounded **character regions** against a cross-shot appearance centroid. It is not InsightFace, similarity to an identity reference, or the project's face-box centroid. Do not copy historical project scores into the official table.
- Default project metric: `cs_face_visible_single_face_reviewed`, displayed as **人物一致性DINO（可见单脸＋定位纠错）**. Keep each official representative region's original pixels/sample, require exactly one qualifying RetinaFace face inside it, reject conflicting role claims, then remove only confirmed localization mistakes and **recompute** character centroids. This remains a conditional diagnostic, not the official EntityBench algorithm. Never overwrite raw `cs_face` or silently relabel it.
- The historical three exclusions belong to exact original crop SHA values. Reuse a decision only for the identical input crop and episode/role/shot; reruns require their own decisions. A regenerated same-named role is not automatically excluded. The linked protocol defines gates, review, aggregation and coverage.
- The user's current scope is **official non-VLM only**: audited `--skip_vlm`, strict mode, no config overrides, no VLM credentials passed, final calls = 0, and missing metrics retained as null.
- Appearance fidelity, action alignment and LLM consistency are absent without VLM; the fidelity gate is unavailable. Label `cs_face/cs_object` provisional and scope `partial_without_vlm`. Do not claim equivalence to full paper tables. Later authorized VLM scoring must reapply the gate and recompute affected consistency, not just append fields.
- The visible-single-face reviewed metric is now a default part of v2, not an opt-in legacy diagnostic. Paired first-frame/video InsightFace, reference-face DINO, and a face-only centroid remain separate optional diagnostics; use the original skill for those ablations.

## Default character-consistency workflow

1. Obtain complete, input-validated official per-episode scores and original representative crops, reusing valid caches. All other metrics retain their official definitions.
2. Apply the [visible-single-face protocol](references/visible_single_face_region_dino.md) to the frozen cohort. Record original raw and automatic-gated results; freeze localization-review decisions before computing the final reviewed result. Do not report the automatic intermediate as final.
3. Report the reviewed DINO as the default project character-consistency row, with official raw `cs_face` explicitly labeled as a comparison. Give eligible/scored coverage and valid episode counts for both. A mixed full-cohort replacement must use the same definition on both sides.

The parallel helper below performs **only step 1**. A successful official helper exit does not complete this default workflow; the diagnostic remains pending until step 2 is complete. Existing full140/rerun adapters have fixed cohort paths: use them only with their matching manifest, or make an isolated adapter for a different cohort while preserving the protocol.

## Prefer bounded parallel evaluation

Schedule **complete episodes**. Centroids and continuation boundaries require the whole episode; do not split its shots across independent scoring jobs or average shard summaries.

On a single 48GB GPU, two official-evaluation processes on GPU 0, normally four CPU threads each, are a starting configuration. Inspect actual GPU availability/free memory, container CPU quota and host memory first; do not assume the machine still has that card. Before widening a new expensive evaluation, measure a representative serial-versus-two-worker sample in a separate validation directory: wall time, peak GPU/host memory and score agreement. Increase to three/four only when measurement shows better total throughput and enough headroom. Low allocated memory does not prove spare compute; do not promise linear speedup. Avoid sharing the GPU with heavy generation unless co-running has been profiled and chosen. Stage diagnostic detection separately when memory cannot support co-running.

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

Lead with stage/coverage, then the default reviewed character-region DINO, official metrics and paths. Clearly separate `cs_face_visible_single_face_reviewed` from official raw `cs_face`; keep their own before/after baselines. If review/detection is pending, mark that row pending rather than substituting raw, reference-face or automatic-gated DINO. State cohort, aggregation unit, evaluator, VLM/gate status, exclusions, missing items, concurrency and measured cost. A single route yields absolute system performance, not invented C/T deltas or plugin causality. Update requested run documentation when completing its handoff work. This skill alone does not authorize GitHub pushes or new paid API calls.
