# Faithful trajectory replay and image delivery

The verified local example is `pretest/qwen140_export_clara_trajectory.py`, original run354 / i90_j105 shots 4:1 and 5:1 from `outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep/`. Read its actual `prepare`, `TrajectoryRecorder`, `worker`, and `queue` before adapting it. Completed images are under `outputs/model_figure_clara_denoising_replay_20261029/`; the plain material folder is `outputs/clara_figure_images_20261030/`. Do not rerun these already completed tasks just to package images.

## Freeze the original conditions

Collect actual `frame_record.json` or the image worker's `task.json`, original `result.json`, prepared metadata/preview, identity reference, FaceLift Gaussian provenance, pose-aligned harmonized reference and final mask. Follow paths and hashes from the record, not filenames guessed from character names.

Record image order, including the actual previous decoded tail when conditioning used it. Current multi-face frame generation can use selected conditioning images different from the complete 3D identity roster; preserve `separate_conditioning_images` behavior from `qwen140_v7_frame.py`. Do not append a missing-looking reference or replan a shot during replay.

For the Clara implementation, settings come from the original preparation plan: Qwen-Image-2.1 revision `790c92633540aa0cb11d9abf19eb46d861714758`, BF16, sequential CPU offload, original KV cache, 1024x576, 20 updates, zero-based step12, original size-scaled strength and 1/6-step schedule. These are example values, not universal defaults for other records. Copy the exact settings of a new selected shot.

Reuse the saved prepared target, 3D guide and mask to keep pose and identity fixed. Regenerate the original Control and capture its shared fork, replay Treatment through the original v7 implementation. Do not re-detect/re-render/adjust thresholds until an image looks preferable. If a preparation record is unavailable, record that limitation; a new preparation is a new experiment rather than exact saved-condition replay.

## Observe, do not change, updates

The local recorder wraps the official scheduler step, v7 `_predict`, `_euler_step` and `_decode`. Store detached CPU clones of the actual inputs/outputs. Call the original functions exactly once and restore hooks in `finally`. KV prefill calls must not be miscounted as main solver states.

- `z_0`: initial noisy latent.
- `z_n`: actual latent after n updates, including injection where applicable.
- `pred_x0_i = z_i - sigma_i * velocity_i`: clean prediction at input state i, before that Euler update.
- At step k, `euler_before` is the solver output before the residual; `actual_after` is the state sent to the next prediction, `z_(k+1)`.

For N=20, each branch should have 21 z states and 20 clean predictions, or 82 main trajectory images across Control/Treatment. Injection-before/after files, guides, masks and final frames are extra. Verify fork equality and final `treatment/z_20.png == treatment.png`. Save CPU tensors so figures can be rearranged without rerunning inference.

Compare both final replay branches against originals by SHA and pixel MAE. Byte-level PNG differences can be separated from decoded pixel equality; report both rather than claiming a false mismatch or false exactness. If different, explain the practical effect and keep replay states/final paired; preserve the original files.

## Queue and delivery

Use a separate output root and input/code snapshots. Preserve all valid completed states on resume. The existing Clara worker refuses a partial finished-result-without-export case so it can be inspected; do not remove this check or delete results automatically. For other shots, create a separate configurable adapter after checking the conditioning contract; the Clara CLI only accepts its two fixed shot keys.

On a busy single GPU, enqueue requested replay after the experiment queue, or report that it is pending. Do not start another heavy model simply because a momentary nvidia-smi sample shows idle memory. User requests for a skill or selection alone do not ask for new image/video generation.

Package with `scripts/pack_trajectory.py`. Link the actual material folder, not just HTML. Include reference, final Control/Treatment, separate z/pred_x0 sequences, injection-before/after, aligned guide/mask, actual prompts and a SHA manifest. Copy original pixels; contact sheets are previews, and face crops must be paired at identical coordinates/scale. Keep tensors at their source unless requested in the delivery.
