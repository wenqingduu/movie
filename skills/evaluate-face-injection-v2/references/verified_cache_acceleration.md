# Verified cache reuse and measured parallel evaluation

Use for an authorized evaluation after shot replacement or when resuming a
partially scored benchmark. Keep the existing scorer, metric definitions,
sampling, canonical pixels, missing-value handling and population unchanged.
This workflow does not authorize generation, API calls or a change in queue order.

## Choose the reuse level

| Input state | Reuse | Work still required |
|---|---|---|
| Complete episode unchanged, compatible scoring provenance | Its validated full episode result and diagnostic artifacts | Official aggregation over the requested full split |
| Some shots changed in an episode | Exact-video six-dimension per-shot VBench scores for unchanged shots | Original VBench inference for new videos; complete affected-episode grounding/presence, character/object centroids and continuation boundaries |
| New episode, missing provenance, incompatible inputs or settings | No unverified score reuse | Original scoring; bounded concurrency if it helps measured throughput |

Do not carry an old episode's cross-shot values into a changed sequence, even
when only one shot changed. Do not average changed-shot deltas or shard summaries
to estimate the other metrics. Include successful replacement videos regardless
of whether their scores improved; retain missing/failed items explicitly.

## Cache contract

Freeze a new manifest keyed by full `(episode_id, shot_key)` with:

- Canonical input video path and content SHA256, new/reused classification and
  the six original per-shot VBench values when eligible. Matching names, frame
  counts or file sizes alone are insufficient. Never reconstruct per-shot
  values from an episode average or treat an unavailable score as zero.
- Source score files and their SHA256, source video SHA binding, and compatible
  evaluator revision/adapter SHA, config, dimensions, sampling, model checkpoint
  fingerprints, numeric precision and relevant library/runtime versions.
- Current split, replacement/media manifests and SHA256; adapter/helper source
  hashes; worker plan and output directories. Validate recorded compatibility
  against the actual current workers' run manifests before accepting results.

Reuse only a successfully evaluated, finite value for every required dimension.
An incompatible video or provenance is a cache miss, scored normally in a new
output directory. A changed frozen manifest is an integrity error: preserve it
and create a versioned plan instead of rebinding it silently.

Wrap only the pinned scorer's per-shot VBench method in memory. Submit cache
misses to its original implementation, merge original and cached values in the
current shot order, and preserve original means/counts/failure semantics.
Do not patch grounding, crop selection or the official scoring source files.
Record per-episode reuse receipts with source hashes, reused/computed counts and
adapter/cache-manifest SHA. Preserve native newly computed outputs alongside the
merged per-shot diagnostic files.

The default reviewed character-region DINO has its own
[visibility and localization protocol](visible_single_face_region_dino.md).
Identical-input detections and localization decisions may be reused only under
that protocol's SHA/role/shot bindings. Changed membership requires recomputing
centroids after gating/review. A VBench cache hit does not validate a new DINO
crop, and automatic detection does not complete manual localization review.

## Validate and schedule

1. Preserve already complete compatible episode outputs and hashes. If a scorer
   is running, switch at a persisted checkpoint or let its current worker finish;
   prevent overlapping heavy generation/evaluation and duplicate writers.
   Archive incomplete conflicting artifacts rather than deleting results.
2. Validate a representative **complete-episode** sample in isolated directories.
   On identical candidate media, compare original scoring with cached serial
   scoring, then cached serial with cached two-worker scoring. Include changed/unchanged videos, multi-character
   crops and a B transition where the cohort has them. Old same-runtime evidence
   can be reused when its full contract still applies; remeasure on changed
   hardware, configuration, input profile or worker count.
3. Require matching metric applicability/nulls, evaluated/failed/skipped counts,
   presence states, representative sample/frame choices and canonical crop SHA.
   Numeric metric differences must meet the pinned validation tolerance
   (the existing project validation uses absolute tolerance `1e-5`). Preserve
   per-episode comparisons; investigate drift rather than loosening thresholds.
4. Measure wall time and peak GPU/host memory. Start with two workers on one GPU,
   four CPU/BLAS threads each, when hardware permits. Use whichever validated
   configuration gives better total throughput; do not infer speedup from spare
   memory or assume a fixed multiplier. Use disjoint whole-episode splits,
   balanced by actual shot count, and independent worker logs/work/artifact dirs.
5. After successful workers, verify each expected episode exactly once, complete
   shot coverage, compatible strict/skip_vlm manifests and zero VLM calls. Merge
   results and artifacts by episode; keep source directories alive when linking.
   Run the official resume validation and deserializer/aggregator over the
   complete requested split. Keep the default DINO pending until its review is
   actually complete.

Report total shots, reused/new VBench counts, preserved/recomputed episodes,
worker configuration, measured validation cost, missing items and actual elapsed
time. Parallel worker outputs may exist before the final merged directory grows;
check worker progress and exits before calling a job stalled. Estimate runtime
from remaining cache misses and measured throughput, with asset generation and
manual review accounted for separately.

## Existing implementation and evidence

These paths are relative to `/root/autodl-tmp/movie`:

- `pretest/qwen140_vbench_verified_cache_20261105.py`: validated cache-adapter
  pattern. Its `OUT`, `CACHE`, evaluator and cohort bindings are specific to the
  completed snapshot; use a separate versioned adapter/manifest for new inputs.
- `outputs/entitybench_v2_completed_snapshot_20261105/acceleration/`: original
  `validation.json`, `serial_cache_validation.json`, cache manifest/contracts,
  reuse receipts, worker execution records and README. These are the evidence
  for the measured speedup, metric/count agreement and canonical-pixel equality;
  read them for values rather than treating historical timings as guarantees.
- `pretest/qwen140_snapshot_accelerated_20261105.py`: checkpoint handoff and
  whole-episode sharding example. **Do not execute it for a new run**: its handoff
  is bound to old PIDs and a particular snapshot. Reuse the orchestration pattern
  in a new wrapper with current input/process contracts.
- [The generic parallel helper](official_parallel.md) handles whole-episode
  parallel scoring and input-verified resume. It does not yet automatically
  discover or inject the per-video VBench cache described here.
