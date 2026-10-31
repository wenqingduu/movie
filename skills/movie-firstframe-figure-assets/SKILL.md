---
name: movie-firstframe-figure-assets
description: "Select paired Qwen/v7 identity-injection examples and export faithful first-frame denoising images, 3D guides, masks and plain image folders for paper figures. Use for the movie project's method figures and trajectory replays, not whole-video benchmark scoring."
---

# Movie First-frame Figure Assets

Work in `/root/autodl-tmp/movie`. Read `LLM_HANDOFF.md` and the requested run's records. Preserve ongoing jobs and unrelated changes. The deliverable is a folder of original PNG/JPG assets plus source/SHA metadata; an HTML preview alone is insufficient.

## Choose useful evidence

Use existing Control/Treatment pairs with identical prompt, seed/noise, model, references and sampler; only the local v7 injection differs. Check result hashes, applied roles, and the paired reference/Gaussian identity. A new pipeline episode versus the original benchmark is not a paired injection ablation.

For readable illustrations, favor a clearly visible, sufficiently large face, correct role mapping, visible improvement and preserved expression/scene. Face size is a presentation preference, not a new generation gate or evidence of improvement. Strong Control likeness can leave little room for improvement. The step-k identity cosine chooses the injection schedule; it is not a final Control-versus-Treatment improvement score.

Use final-frame reference-anchored InsightFace and reference-face DINO together, plus visual inspection. They do not measure the official character-region `cs_face`. Keep disagreements, negatives, missing detections and the screened population in the selection record; label selected positives as illustrative. If claiming cross-shot improvement, inspect at least two shots of the same episode-local person; improvement to a reference in one frame alone does not establish cross-shot consistency.

For CPU-only screening without interrupting a GPU queue, use [scripts/select_pairs.py](scripts/select_pairs.py). It validates existing image hashes, filters by optional preview face height (default 128px, **selection only**), evaluates final-face identity at spatially matched boxes, then computes reference DINO for the top candidates. `--target-mode single` is the default; use `multi` for roles in multi-target frames or `any` for both. It writes every ranked result, failures and a plain-folder shortlist with originals and equal-scale crops. Inspect the shortlist yourself before recommending it. Read [references/selection.md](references/selection.md) for commands, scope and interpretation.

## Replay when requested

If the user asks only to inspect or package existing images, do not regenerate them. If denoising trajectories are requested, freeze the exact original prompts, image ordering, seed, model revision, sampler settings, saved step-k preparation, aligned 3D guide and mask in a separate output root. Do not regenerate identity assets to obtain a more favorable result.

Read [references/replay.md](references/replay.md) before GPU work. The existing `pretest/qwen140_export_clara_trajectory.py` is a **two-shot Clara-specific implementation**, not a generic exporter. Its `TrajectoryRecorder` observes solver outputs without changing updates. Other selected shots need a separate adapter that preserves their actual conditioning and settings; do not edit a frozen/running script or pretend the Clara CLI accepts arbitrary shots. Queue GPU replay after active generation unless co-running was explicitly selected and measured. No repair Agent is required or implied.

Export actual `z_0..z_N`, clean `pred_x0_0..pred_x0_(N-1)`, Euler output before injection, actual state after injection, and CPU latent/velocity/sigma tensors. Observe both branches; before the fork, the prefix is shared. At zero-based step k, the injected next state is `z_(k+1)`. Never replace raw-state images with cleaner predictions without changing their label.

Verify the same fork state, recorded update counts, final decoded state, saved guide/mask and original-versus-replay final SHA/pixel difference. If reproduction drifts, disclose it and use intermediate states and final output from the same replay. Never claim a replay was saved during the original run.

## Deliver image files

Use [scripts/pack_trajectory.py](scripts/pack_trajectory.py) for completed replay roots. It copies PNG/JPG and prompts without changing pixels, checks hashes, keeps the two shot folders separate and writes an image manifest. It leaves large tensors/logs at the original source. Example:

```bash
python /root/.codex/skills/movie-firstframe-figure-assets/scripts/pack_trajectory.py \
  --source outputs/model_figure_clara_denoising_replay_20261029 \
  --out outputs/clara_figure_images_delivery
```

Preserve original full frames alongside crops and contact sheets. Use the same crop coordinates and magnification for paired C/T display. Do not retouch faces, recolor, sharpen, synthesize missing intermediate states or use image generation to improve comparison panels. Report the skill location, material folder, selected shots, exact metric definition and remaining limitations; representative positives do not replace aggregate experiment results.
