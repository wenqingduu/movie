# Default project character-consistency protocol

The user selected the original full140 **visible-single-face gate plus localization-reviewed character-region DINO** as the standing default for subsequent v2 evaluation. This changes the default project reporting row; it does not change official EntityBench's `cs_face` implementation or the other official metrics.

## Names and comparable quantities

| Quantity | Meaning |
|---|---|
| `cs_face_visible_single_face_reviewed` | Default project row: 人物一致性DINO（可见单脸＋定位纠错）. Same definition as the historical reviewed original diagnostic. |
| `cs_face_visible_single_face_auto` | Intermediate after automatic visibility/role-conflict gates, before localization review. Never substitute for the reviewed row. |
| Official raw `cs_face` | Unmodified EntityBench representative character-region centroid score; preserve original JSON/report and show separately. |
| Reference-face DINO / InsightFace | Separate identity-reference diagnostics, usually for paired injection ablation; not this cross-shot metric. |
| Face-only centroid DINO | Embeds a new face crop; also not this metric. |

Label the reviewed row as a project conditional diagnostic, not a paper-comparable official score. Use the same row definition on both sides of a comparison. Do not compare the reviewed original baseline against a rerun's raw official score.

## Immutable inputs and face gate

Freeze complete episode IDs/windows, original shot order, videos and SHA, official evaluator/config/model/environment, the original per-shot `_eval.json`/canonical crops, detection weights, gate source, review manifest and decisions. A score is keyed by `(episode_id, character, shot)`, not run number alone.

Start only from official character items with `status=present`. At that item's **original representative sample**, use the existing padded canonical **person-region** crop and exact original pixels. Preserve official grounding/CLIP thresholds, representative selection, padding and DINO features. The detector decides eligibility; it does not supply replacement pixels for the DINO input.

| Gate | Fixed rule |
|---|---|
| Detector | facexlib RetinaFace ResNet50, original weights, FP32; confidence >=0.9, NMS 0.4, native-size detection |
| Minimum face size | Native width **and** height >=24px, before any display resize |
| Region containment | `intersection(face_box, canonical_region) / area(face_box) >=0.80` |
| Uniqueness | Exactly one qualifying detected face inside the original region at its original sample |
| Role conflict | Reject **all** role items claiming the same face track within the shot, including claims at different sampled times |
| Tracking | Original spatial `build_tracks`: max gap1 sample, minimum similarity0.3, relative ambiguity0.15; no identity-cosine assignment |
| Missing | Reject ineligible/ambiguous crops, retain reasons and coverage; never assign missing values zero |

These are **evaluation** gates, unrelated to generation's 3D injection schedule, face-size policy or figure-selection preferences. Do not import the figure screen's 128px/224px/six-step priority or keep only injected/positive cases. Non-face/nonhuman/back-view and detector failures remain in the scheduled-population coverage, even when not eligible for this conditional score. Do not relax the gate for a style or method.

Use the pinned official five-frame sampling function with the video's **actual decoded frame count**. Do not hard-code 81 frames or a particular set of frame indices: stitched B videos can be longer. If reusing the optimized detector, detect every sample from the first canonical-claim sample through the last, preserving all intermediate tracking steps; reuse only with its saved full-baseline equivalence validation. Reuse original detections where input hashes match.

## Localization review

The historical reviewed baseline removed **three particular original crops**, not three role names for all future runs. Its authoritative decisions live at:

```text
outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/evaluation/
  character_region_visible_face_gate_20261025/remaining_low_score_review/
    review_manifest.json
    review_decisions.json
    recomputed/report.json
```

The three original exclusions are Walter at run227/i25_j77, shots3:6 and3:1, and Ji-hoon at run4/i35_j83, shot2:1. Apply an old exclusion only when the full episode/role/shot key **and original crop SHA** match, and the input/review protocol is compatible. Do not transfer it to a regenerated crop, another window, or the same role name elsewhere. Original retained decisions may be reused for identical unchanged inputs. The historical review was a sample and includes a later-known unresolved R051 association; do not describe it as full-population localization correction.

For new evaluated crops, use the original review design: provisional centroid scores select the lowest60 plus a deterministic random20 from the remaining scored crops, without replacement; if fewer than80 are available, review all. Freeze the selection, seed, ordering, crop SHA and full-frame context. For the current rerun adapter the random seed is20261025 and presentation shuffle seed20261101. Display scores hidden during the localization review.

Review original descriptions/schedule, the exact representative full frame and selected region. Exclude only confirmed non-face false localization or a wrong-person region **when the intended target is identifiable in the same frame**. Keep generated identity/style/clothing drift, valid profile/lighting/pose changes, missing-target cases that cannot be separated from generation failure, and unresolved identity associations. A low DINO score, resemblance to another identity, duplicate generated person or mismatch to the portrait alone is not an exclusion rule. Never tune thresholds or keep only cases whose exclusion improves the score.

Save every keep/exclude decision with key, crop SHA, evidence and reason; bind it to the review-manifest SHA **before** final recomputation. For mixed full-cohort updates, carry unchanged decisions by compatible exact inputs and review newly replaced crops; do not newly correct shared old inputs on only one comparison side. Report the actual review size/scope. Do not call a diagnostic final while its required review is incomplete.

## Recompute, rather than average stored scores

For each episode-local role with at least two retained appearances, normalize its existing region-DINO features `x_i`, compute `c = normalize(mean_i(x_i))`, then each score `dot(x_i, c)`. The scored appearance is included in its centroid, matching the frozen official function. Recompute the centroid **after** automatic/manual exclusions; deleting rows and averaging old `sim_to_centroid` values is incorrect.

Pool scored appearances within each episode using the frozen official `compute_cross_shot_consistency`; do not give each character equal weight. Aggregate valid episode scores with the frozen official deserializer/aggregator on the original split; do not average shard averages or replace null episodes with zero. Report all/easy/medium/hard where the split defines them. A role left with one appearance is unscored, so dropping one crop can remove additional scored appearances.

Saved unit-feature pairwise dot products may recover a Gram-equivalent feature representation and avoid new DINO inference. Verify feature order, pair count and matrix reconstruction, reproduce every original full per-role/per-episode raw centroid score and original full aggregation, then filter and recompute. Freeze tolerances and source functions. For the original reviewed baseline also reproduce its per-episode scores and final aggregate from the authoritative report. If pairwise data are absent, extract embeddings on the **unchanged canonical person-region pixels** with the pinned official DINO; do not use a newly cropped/aligned face or a reference portrait.

## Existing implementations and current replay

Project root is `/root/autodl-tmp/movie`:

- `pretest/evaluate_entitybench_qwen140_region_face_gate.py`: original gate config, Gram reconstruction and frozen official arithmetic loader.
- `pretest/evaluate_entitybench_face_only_dino_qwen140.py`: original RetinaFace cache/tracking and exact canonical-region recovery. Use these helpers; its separate face-only scoring mode is not this metric.
- `pretest/recompute_entitybench_qwen140_localization_review.py`: original three exact-crop exclusions and post-review centroid recomputation. It is an original-baseline reproducer, not a generic new-run scorer.
- `pretest/qwen140_rerun_region_diagnostic.py`: fixed current completed-rerun manifest merged into full140, baseline reproduction, input-validated extraction, changed-crop review and final aggregation. It does not discover an arbitrary future cohort automatically.

For **that exact saved rerun manifest**, after inspecting actual resources and processes:

```bash
cd /root/autodl-tmp/movie
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -u \
  pretest/qwen140_rerun_region_diagnostic.py extract --device cuda
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  pretest/qwen140_rerun_region_diagnostic.py score
# Review every frozen review/manifest.json entry and save review/decisions.json.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  pretest/qwen140_rerun_region_diagnostic.py final
```

Outputs: `outputs/entitybench_rerun_visible_region_identity_20261101/`. `automatic_report.json` is intermediate; `report.json` is final only after validated manual review. For different inputs create a new manifest/output and an isolated adapter preserving the above rules; never edit a frozen/running adapter in place. CPU is valid but can be slow or memory-limited; do not concurrently load large detector/recognition/DINO models when memory is insufficient. No VLM or paid API is needed for this protocol.

## Required report

Publish the default reviewed metric and raw official metric under distinct names. Include scheduled role-shot count, canonical present count, automatic eligible/retained count, post-review retained count, scored appearances, valid/missing episodes, gate rejection reasons, review coverage, exclusions and evidence. Preserve JSON plus readable tables and input/protocol hashes. If a mixed replacement changes the gate population, include coverage changes; a matched role-shot intersection is a supplementary comparison, never a silent replacement for the primary episode-macro score.

The official parallel helper only produces prerequisites. Its success is not proof that detection or localization review has completed. Never fill a pending default row using a face-only/reference DINO, raw `cs_face`, the old cohort's score or an automatic intermediate.
