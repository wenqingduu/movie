# Official non-VLM execution

The current story run completed generation and serial official non-VLM scoring on 2026-10-12:
`/root/autodl-tmp/movie/outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep`.
Read status/report files for live values. Do not rerun its complete evaluation merely because this skill was created.

## Frozen scorer and environment

- Official revision: `3776439e7037b2128ac3b5d2d1ce63e424cca166`.
- Pristine SHA: `2334aad4b61cae2a0ea38cc2ee9ae85125b6df4e128610903654fd2b49607bb4`.
- Adapter SHA: `79caf1327e7f1a63527c0e18bc9c936b8618b59080197d2f50677ee2ba5cb5a0`.
- Frozen `<run>/runtime/` includes evaluate_benchmark_official.py, evaluate_benchmark_partial.py, skip_vlm.patch and evaluation_provenance.json.
- Python: `/root/autodl-tmp/envs/entitybench-eval/bin/python`. The helper sets local model/cache paths and bounded CPU threads; no InsightFace or generation environment is needed.

For another scorer revision, audit and adapt the helper's check first. Do not overwrite scoring sources to pass it.

## Commands

Plan without inference or output mutation:

```bash
eval_python=/root/autodl-tmp/envs/entitybench-eval/bin/python
eval_helper=/root/.codex/skills/evaluate-face-injection-v2/scripts/evaluate_parallel.py
run_root=/root/autodl-tmp/movie/outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep
"$eval_python" "$eval_helper" --output-root "$run_root" --workers 2 --gpu-id 0 --dry-run
```

For a **future authorized evaluation**, omit `--dry-run` and execute inside tmux. Default output is `<run>/evaluation/official_parallel_v2`, preserving serial scores. Use `--out-dir` for a separate pilot or worker count; cached worker plans are immutable.

Workers receive whole episodes greedily balanced by actual shot count. Both run on the selected GPU, with four CPU/BLAS threads each. `.shards/worker_NN/results/` contains independent work and evaluator manifest; splits below `.shards/` suppress competing shard-level global reports. Logs/resource snapshots are separate. Aggregation waits for successful workers and validated episode coverage.

The default admission budget is 8GiB per worker plus 2GiB spare GPU memory, and workers × threads must fit the container CPU quota. This is a starting budget, not a measured peak guarantee; tune `--vram-per-worker-mib` from a representative pilot. Tune concurrency from total throughput, not idle memory. Two-worker GPU speedup was measured on the 22-video validation below; do not extrapolate it as a guaranteed full-cohort factor.

Read-only aggregation validation from complete serial scores, with no neural inference:

```bash
"$eval_python" "$eval_helper" --output-root "$run_root" \
  --out-dir "$run_root/evaluation/official_parallel_v2_reaggregation_validation" \
  --aggregate-only --source-evaluation-dir "$run_root/evaluation/official"
```

This validates frozen inputs, per-episode metrics and zero VLM calls, and applies the same official aggregation in a **new** directory. It tests aggregation mechanics, not parallel inference speed or independently rerun model agreement.

## Resume and failure

`parallel_plan.json` freezes media/script/split/evaluator fingerprints, helper SHA, method and worker/GPU/thread plan. `parallel_status.json` records phase, failure and timing. Official worker manifests must agree on configuration, scope, checkpoint fingerprints and exact library versions.

On worker failure, stop sibling process groups, preserve saved results/logs, record failed state and raise. Diagnose, then resume the valid plan; use a new directory if changing inputs/configuration. Do not reduce samples, disable dimensions, relax strictness or label a partial aggregate complete.

Final run_manifest.json, input_fingerprint.json, parallel_execution.json and report JSON/Markdown accompany canonical episode JSONs. Serial scores and videos stay read-only. Scheduling changes; official thresholds, five grounding samples, representative crops and aggregation do not.

## Validation of this skill

2026-10-12: skill metadata/schema and Python compilation passed. A no-inference dry run
assigned the 20 complete episodes to two disjoint shards of 182/181 shots on GPU 0.
Nine workflow checks passed: complete/disjoint balancing, two-shard aggregation equality,
missing/duplicate episode rejection, changed config/media rejection, zero-VLM enforcement,
bounded credential-free environment and unchanged scorer flags. Actual cached reaggregation
matched the original official all/easy/medium/hard JSON sections exactly. These checks do
not measure parallel neural-inference throughput.

For future helper changes, rerun the read-only checks:

```bash
/root/autodl-tmp/envs/entitybench-eval/bin/python /root/.codex/skills/evaluate-face-injection-v2/scripts/test_parallel.py
```

## Actual two-worker GPU probe

2026-10-12: the authorized full140 expansion used the first two previously completed
episodes (22 existing videos) to compare one versus two workers on the same GPU, each
with four CPU threads. Measured helper wall times: 228.28 seconds versus 157.35 seconds
(1.451× throughput). Every per-episode metric value matched within 1e-5 and all
evaluated/failed/skipped counts matched. Zero VLM calls. Artifacts: movie outputs/
entitybench_pulid_i2v_a14b_4step_full140/parallel_probe/validation.json and the two worker
evaluation directories. No video regeneration or score-based output selection occurred.
A changed machine, model/version, input size or worker count needs its own representative
probe; this small sample is not a measured full140 speedup.
