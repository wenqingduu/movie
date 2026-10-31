# Paired example screening

Use the project `.venv/bin/python` for InsightFace/transformers. Existing first-frame pairs are inexpensive to screen on CPU while the GPU experiment runs. The helper uses two ONNX/torch threads and CPUExecutionProvider only; do not silently add a GPU evaluation job.

```bash
cd /root/autodl-tmp/movie
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python /root/.codex/skills/movie-firstframe-figure-assets/scripts/select_pairs.py \
  --roots outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140 \
          outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep \
  --out outputs/firstframe_identity_figure_candidates \
  --min-face-height 128 --dino-top 24 --top 8
```

Accepted existing layouts are `firstframes/<episode>/shot_*/result.json`, `experiments/<run>/episodes/<episode>/shot_*/frame/injection/result.json`, `episodes/<episode>/shot_*/frame/injection/result.json`, and `<run>/shots/shot_*/result.json`. A missing layout should be adapted explicitly rather than interpreted as no improvements. Use only relevant roots, avoid scanning unrelated models or historical different Control implementations as interchangeable evidence.

The default `--target-mode single` requires one prepared target; `--target-mode multi` screens each applied role in frames with multiple prepared targets, and `--target-mode any` covers both. Only actually injected roles enter screening. All modes require matching Control/Treatment SHA and matching 3D-source reference SHA where recorded, and deduplicate identical C/T/reference triples. The 128px height comes from the step-k preparation and is a **candidate prefilter**; final detection boxes are saved separately. It is not an injection threshold and does not guarantee a large final face.

For multi-target records, matching uses the original whole prepared target roster and the existing evaluator's spatial one-to-one box assignment. A frame where a required face cannot be reliably matched is recorded as an error rather than forced to another face. Keep per-character rows and distinguish role-pair counts from frame counts. A six-step low-preview-identity role may show a larger change; check whether identity improved without changing its expression, age, gender, occlusion or intended action.

Diagnostics reuse the project's paired evaluator: confidence 0.5, final face selected spatially by IoU with the saved preparation box (minimum IoU 0.25), never by maximizing final identity. Reference DINO uses those final boxes expanded by 0.35, DINOv2 CLS and the original portrait. Cache files live only in the selection output; generated result files and official benchmark reports remain untouched.

`input_candidates.json` retains exclusions, `ranking_insightface.json` includes all valid IF diagnostics, and `selection.json` adds DINO/shortlist details. DINO is computed only on the requested IF top-N, not on the entire population. Re-run with a larger top-N only if the question warrants it. This tool does not produce a new whole-dataset aggregate or measure a first-frame-to-video effect.

Open shortlisted `comparison.jpg` and raw full frames. Check actual identity cues, face shape/eyes/nose, role mapping, expression, seams, lighting and loss of detail. A positive numerical delta can be imperceptible or accompany an unwanted expression change. A negative DINO delta does not automatically invalidate IF but must be disclosed. Prefer two visibly improved shots of the same episode-local person when explaining cross-shot identity; do not mix same-named people across episodes.

Keep full negative/near-zero results even if only positives enter an explanatory figure. Describe selection criteria and avoid presenting the figure as representative benchmark evidence.
