"""24 additional frozen dynamic-zero shots; keep selected20 and all negatives.

Original40 generation -> unfiltered six VBench -> dynamic-only cumulative overlay
-> whole-episode official scoring -> score-hidden localization review materials.
No historical scheduler, generation loop or quality gate is resumed.
"""
from __future__ import annotations

import copy
import fcntl
import math
from pathlib import Path
import subprocess
import sys

import qwen140_original40_scope32_20261108 as prior

core = prior.core
ROOT, BASE = core.ROOT, core.BASE
SCOPE32 = prior.OUT
OUT = ROOT / 'outputs/entitybench_original40_next24_20261109'
SELECTED20 = ROOT / 'outputs/entitybench_original40_dynamic_gain_selected20_20261109'
CONTEXT_SCRIPT = ROOT / 'pretest/qwen140_original40_next24_context_20261109.py'
IDS = ['M0014', 'M0015', 'M0026', 'M0029', 'M0030', 'M0032', 'M0040', 'M0051',
       'M0052', 'M0053', 'M0057', 'M0058', 'M0059', 'M0061', 'M0064', 'M0067',
       'M0070', 'M0072', 'M0073', 'M0075', 'M0077', 'M0082', 'M0094', 'M0095']
read, write, sha, immutable = core.read, core.write, core.sha, core.immutable


def status(stage, **kw):
    write(OUT / 'status.json', dict(stage=stage, utc=core.now(), expected=24,
          generated=len(list((OUT / 'cases').glob('*/generation.json'))),
          scored=len(list((OUT / 'cases').glob('*/scores.json'))),
          quality_gate_enabled=False, large_queue_resumed=False, api_calls=0,
          old_media_overwritten=False, **kw))


def prepare():
    if (OUT / 'plan.json').exists():
        p = read(OUT / 'plan.json')
        for b in p['bindings']:
            assert sha(b['path']) == b['sha256'], b['path']
        return p
    assert read(SELECTED20 / 'status.json')['stage'] == 'complete_all_available_metrics_with_reviewed_dino'
    earlier = read(SCOPE32 / 'plan.json')
    attempted = {c['source_case'] for c in earlier['cases']}
    source = read(core.SOURCE / 'plan.json')
    lookup = {c['case_id']: c for c in source['cases']}
    original = ROOT / 'outputs/entitybench_full140_latest57_v2_141_20261106/snapshot.json'
    snapshot = copy.deepcopy(read(original))
    media = read(SELECTED20 / 'replacement_manifest.json')['videos']
    bindings = {}

    def bind(path):
        path = str(Path(path).resolve())
        bindings[path] = sha(path)

    for item in snapshot['episodes']:
        result = SELECTED20 / 'affected/results' / (item['episode_id'] + '.json')
        if result.exists():
            item.update(result=str(result), result_sha256=sha(result),
                        artifacts=str(result.parent / '.work/eval_artifacts' / item['episode_id']),
                        output_root=str(SELECTED20 / 'episode_media'))
        assert sha(item['result']) == item['result_sha256']
        bind(item['result'])
    snapshot['media'] = {k: dict(source=r['source'], sha256=r['sha256']) for k, r in media.items()}
    snapshot.update(scope='Full140 selected20 completed version, immutable baseline for next24 additional attempts.',
                    parent_snapshot_sha256=sha(original), parent_selected20_manifest_sha256=sha(SELECTED20 / 'replacement_manifest.json'))
    immutable(OUT / 'baseline_snapshot.json', snapshot)
    cases = []
    for i, case_id in enumerate(IDS, 33):
        assert case_id not in attempted
        c = lookup[case_id]
        key = c['episode_id'] + '/' + c['shot']
        assert not media[key]['replaced'] and not c.get('background_cue') and c['route'] != 'B'
        assert c['baseline_scores']['dynamic_degree'] == 0
        assert media[key]['sha256'] == c['baseline_video_sha256']
        directory = Path(c['output']).parent
        record = read(directory / 'result.json')
        score = read(directory / 'vbench.json')
        task = read(c['task_path'])
        generation_path = Path(record['video']).parent / 'generation.json'
        generation = read(generation_path)
        assert task == c['task'] and task['prompt'] == c['new_prompt']
        for field in ['prompt', 'seed', 'frames', 'fps', 'size', 'input_sha256']:
            assert generation['inputs'][field] == task[field], (case_id, field)
        assert task['frames'] == 81 and task['fps'] == 16 and task['steps'] == 4 and task['last_frame_path'] is None
        assert score['after']['dynamic_degree'] == 0
        assert generation['video_sha256'] == record['video_sha256'] == sha(record['video'])
        assert task['input_sha256'] == sha(task['input_path'])
        receipt = read(score['receipt'])
        slot = f"{receipt['case_ids'].index(case_id) + 1}:1"
        assert receipt['video_sha256'][slot] == record['video_sha256']
        assert receipt['scores'][slot] == score['after']
        assert receipt['official_sha256'] == sha(BASE / 'runtime/evaluate_benchmark_official.py')
        assert all(math.isfinite(score['after'][d]) for d in core.DIMS)
        cases.append(dict(case_id=f'N{i:02d}', source_case=case_id, episode_id=c['episode_id'],
              run=c['run'], shot=c['shot'], category=c['category'], original_description=c['official_description'],
              prompt=task['prompt'], input_path=task['input_path'], input_sha256=task['input_sha256'],
              seed=task['seed'], frames=81, fps=16, size=task['size'], distilled_video=record['video'],
              distilled_video_sha256=record['video_sha256'], distilled_scores=score['after'],
              distilled_score_source=str(directory / 'vbench.json'), distilled_score_source_sha256=sha(directory / 'vbench.json'),
              source_task=c['task_path'], source_task_sha256=sha(c['task_path']), baseline_is_dynamic_zero=True,
              frozen_benchmark_scores=c['baseline_scores']))
        for path in [directory / 'result.json', directory / 'vbench.json', c['task_path'],
                     generation_path, score['receipt']]:
            bind(path)
    assert len(cases) == len({c['episode_id'] + '/' + c['shot'] for c in cases}) == 24
    cfg = read(earlier['config'])
    assert cfg['infer_steps'] == 40 and cfg['sample_guide_scale'] == [3.5, 3.5] and cfg['enable_cfg']
    assert cfg['sample_shift'] == 5 and cfg['boundary'] == .9
    assert cfg['rope_type'] == 'torch_complex_rope' and cfg['double_precision_rope']
    for path in [Path(__file__), CONTEXT_SCRIPT, prior.__file__, core.__file__, earlier['config'],
                 OUT / 'baseline_snapshot.json', original, core.SOURCE / 'plan.json', SCOPE32 / 'plan.json',
                 SELECTED20 / 'replacement_manifest.json', SELECTED20 / 'official_aggregates.json',
                 SELECTED20 / 'visible_region/report.json', SELECTED20 / 'visible_region/review/decisions.json',
                 ROOT / 'pretest/qwen140_v2_shots_only_eval_20261105.py',
                 ROOT / 'pretest/qwen140_vbench_verified_cache_20261105.py',
                 BASE / 'runtime/evaluate_benchmark_partial.py', BASE / 'runtime/evaluate_benchmark_official.py']:
        bind(path)
    for b in earlier['bindings']:
        assert sha(b['path']) == b['sha256']
        bind(b['path'])
    p = dict(created_utc=core.now(), cases=cases, total=24, new_cases=24, zero_controls=24,
             dynamic_positive_controls=0, model_path=earlier['model_path'], config=earlier['config'],
             negative_prompt='', baseline_snapshot=str(OUT / 'baseline_snapshot.json'),
             parent_selected20=str(SELECTED20), prior_attempts_excluded=32, batch_size=8,
             selection='24 fresh original40 attempts; current selected20 baseline dynamic0, existing byte-identical enhanced4 control dynamic0, no background append or B route. Descriptions reviewed for explicit whole-body travel; no new action/camera text.',
             scope='Purposive24 same-prompt backend-recipe pairs; retain every result and decline. Cumulative benchmark replacements only for dynamic0_to1 vs completed selected20 baseline; other scores never filter media.',
             protocol='WanScheduler40, CFG3.5, shift5, boundary.9, noLoRA, BF16/SDPA/TorchComplexRoPE; exact existing PNG/prompt/seed/81frames16fps/size; no new3D, images, following videos, paidAPI or VLM.',
             quality_gate_enabled=False, old_media_overwritten=False, large_queue_auto_resume=False,
             bindings=[dict(path=k, sha256=v) for k, v in bindings.items()])
    immutable(OUT / 'plan.json', p)
    write(OUT / 'validation.json', dict(cases=24, fresh_original40_cases=24, current_dynamic_zero=24,
          same_prompt4_dynamic_zero=24, control_video_score_sha_verified=24, exact_png_prompt_seed_config=True,
          baseline_selected20_preserved=True, no_background_append=True, no_new_camera_or_actions=True,
          no_quality_gate=True, no_old_queue_resume=True, plan_sha256=sha(OUT / 'plan.json')))
    status('prepared_frozen_next24')
    return p


def configure():
    core.OUT, core.prepare, core.status = OUT, prepare, status
    prior.OUT, prior.prepare, prior.status = OUT, prepare, status


def run():
    p = prepare()
    with (OUT / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'Existing GPU jobs: do not duplicate'
        try:
            for start in range(0, 24, 8):
                ids = [c['case_id'] for c in p['cases'][start:start + 8]]
                tag = f'next24_batch_{start // 8 + 1:02d}'
                if any(not (OUT / 'cases' / cid / 'generation.json').exists() for cid in ids):
                    status('generating_original40_batch', case_ids=ids)
                    with (OUT / (tag + '_generation.log')).open('a') as log:
                        subprocess.run([core.PY, '-u', str(Path(__file__)), 'worker', ','.join(ids)],
                                       env=core.env(), stdout=log, stderr=subprocess.STDOUT, check=True)
                if any(not (OUT / 'cases' / cid / 'scores.json').exists() for cid in ids):
                    status('evaluating_six_vbench_batch', case_ids=ids)
                    with (OUT / (tag + '_evaluation.log')).open('a') as log:
                        subprocess.run([core.EVAL, '-u', str(Path(__file__)), 'evaluate', ','.join(ids), tag],
                                       env=core.env(), stdout=log, stderr=subprocess.STDOUT, check=True)
                status('batch_complete', batch=tag)
            status('generation_and_six_vbench_complete_starting_cumulative_context')
            with (OUT / 'context.log').open('a') as log:
                subprocess.run([core.EVAL, '-u', str(CONTEXT_SCRIPT), 'run'], env=core.env(),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            status('official_complete_default_dino_localization_review_pending',
                   context_status=str(OUT / 'replacement/status.json'), report=str(OUT / 'replacement/official_report.md'))
        except Exception as error:
            status('stopped_error_preserve_all_outputs', error=repr(error))
            raise


if __name__ == '__main__':
    configure()
    action = sys.argv[1]
    if action == 'prepare':
        p = prepare()
        print(dict(output=str(OUT), total=p['total'], plan_sha256=sha(OUT / 'plan.json')))
    elif action == 'run':
        run()
    elif action == 'worker':
        p = prepare()
        ids = set(sys.argv[2].split(','))
        core.prepare = lambda: {**p, 'cases': [c for c in p['cases'] if c['case_id'] in ids]}
        core.generate()
    elif action == 'evaluate':
        prior.evaluate(sys.argv[2].split(','), sys.argv[3])
    else:
        raise ValueError(action)
