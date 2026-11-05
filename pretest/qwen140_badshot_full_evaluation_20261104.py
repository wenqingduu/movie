"""Shot comparisons, versioned replacement, then complete full140 scoring."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import qwen140_reaggregate_partial_reruns as agg

ROOT, BASE = agg.ROOT, agg.BASE
BATCH = ROOT / 'outputs/entitybench_prompt_v2_badshots_20261104'
SNAPSHOT = ROOT / 'outputs/entitybench_full140_completed51_snapshot_20261104'
OUT = BATCH / 'full_evaluation'
EVAL = '/root/autodl-tmp/envs/entitybench-eval/bin/python'
EVALUATOR = BASE / 'runtime/evaluate_benchmark_partial.py'
read, write, sha = agg.read, agg.write, agg.sha
DIMENSIONS = ['subject_consistency', 'temporal_flickering', 'motion_smoothness',
              'dynamic_degree', 'aesthetic_quality', 'imaging_quality']


def immutable(path, value):
    if path.exists():
        assert read(path) == value, f'Frozen input changed: {path}'
    else:
        write(path, value)


def status(stage, **extra):
    write(OUT / 'status.json', {'updated_utc': datetime.now(timezone.utc).isoformat(),
          'stage': stage, 'vlm_calls': 0, **extra})


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'extension_plan.json'
    if path.exists():
        plan = read(path)
        for item in plan['sources']:
            assert sha(item['path']) == item['sha256'], item['path']
        return plan
    selection = read(SNAPSHOT / 'official/selection_manifest.json')
    assert selection['num_episodes'] == 140 and len(selection['videos']) == 2491
    cases = read(BATCH / 'cases.json') + read(BATCH / 'plan.json')['reused']
    assert len({(c['episode_id'], c['shot']) for c in cases}) == 218
    assert all(c['episode_id'] + '/' + c['shot'] in selection['videos'] for c in cases)
    sources = [Path(__file__).resolve(), ROOT / 'pretest/qwen140_prompt_v2_badshot_batch_extended_20261104.py',
               ROOT / 'pretest/qwen140_badshot_reviewed_dino_20261104.py',
               BATCH / 'plan.json', BATCH / 'cases.json', SNAPSHOT / 'official/selection_manifest.json',
               SNAPSHOT / 'reviewed_dino_full140.json', SNAPSHOT / 'cohort_snapshot.json',
               BASE / 'manifest.json', BASE / 'split_full140.json', EVALUATOR,
               ROOT / 'pretest/qwen140_rerun_region_diagnostic.py']
    plan = {'episodes': 140, 'shots': 2491, 'target_shots': 218,
            'affected_episodes': len({c['episode_id'] for c in cases}),
            'order': ['finish_v2_generation', 'evaluate_affected_complete_episodes',
                      'compare_target_shots_all_non_vlm_metrics', 'publish_versioned_replacements',
                      'evaluate_full140_all_non_vlm_metrics', 'visible_face_gate_and_localization_review'],
            'disk_reserve_gib': 0, 'disk_quota_gib': None, 'automatic_deletion': False,
            'replacement_policy': 'Use every completed v2 target, including declines. Retain baseline only for missing/failed outputs and report them. Never overwrite historic media or scores.',
            'scoring': 'Pinned official strict/skip_vlm;12 available metrics plus default reviewed region-DINO. VLM-dependent fields remain null.',
            'chain_policy': 'Spliced independent shots, not a regenerated chain. Save actual generation parent hashes and report joins that differ; score their boundaries without masking.',
            'manual_review': 'Default reviewed-DINO remains pending until score-hidden localization decisions are completed; never relabel automatic/raw as reviewed.',
            'sources': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in sources]}
    immutable(path, plan)
    immutable(OUT / 'baseline_selection.json', selection)
    status('queued_after_generation', affected_episodes=plan['affected_episodes'])
    return plan


def scorer():
    spec = importlib.util.spec_from_file_location('_badshot_official_scorer', EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert list(module.VBENCH_DIMENSIONS) == DIMENSIONS
    return module


def environment():
    env = {k: v for k, v in os.environ.items() if not any(t in k.upper() for t in
           ['API_KEY', 'SECRET', 'CREDENTIAL', 'ACCESS_TOKEN', 'DASHSCOPE', 'OPENAI', 'AZURE', 'GEMINI'])}
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(ROOT),
               DINO_MODEL_PATH='/root/autodl-tmp/models/dinov2-base',
               CLIP_MODEL_PATH='/root/autodl-tmp/models/clip-vit-base-patch32',
               GDINO_CONFIG='/root/autodl-tmp/models/entitybench_groundingdino/GroundingDINO_SwinT_OGC.py',
               GDINO_WEIGHTS='/root/autodl-tmp/models/entitybench_groundingdino/groundingdino_swint_ogc.pth',
               VBENCH_CACHE_DIR='/root/autodl-tmp/models/entitybench_vbench',
               TORCH_HOME='/root/autodl-tmp/models/entitybench_torch_cache', HF_HOME='/root/autodl-tmp/models/entitybench_hf_cache')
    return env


def mapping():
    if (OUT / 'replacement_manifest.json').exists():
        data = read(OUT / 'replacement_manifest.json')
        for row in data['videos'].values():
            assert sha(row['source']) == row['sha256']
        return data
    before = read(OUT / 'baseline_selection.json')
    for episode in before['episodes']:
        assert sha(episode['selected_result']) == episode['selected_result_sha256']
        assert sha(episode['baseline_result']) == episode['baseline_result_sha256']
    cases = read(BATCH / 'cases.json') + read(BATCH / 'plan.json')['reused']
    targets, missing = {}, []
    for case in cases:
        path = Path(case['output_root']) / 'episodes' / case['episode_id'] / ('shot_' + case['shot'].replace(':', '_')) / 'result.json'
        if path.exists():
            rec = read(path)
            assert sha(rec['video']) == rec['video_sha256']
            assert sha(rec['last_frame']) == rec['last_sha256']
            targets[case['episode_id'] + '/' + case['shot']] = {'record': rec, 'result_path': str(path), 'result_sha256': sha(path)}
        else:
            missing.append({'episode_id': case['episode_id'], 'shot': case['shot'], 'missing_result': str(path)})
    data = {'scope': 'Versioned full140 replacement of completed target shots; all previous media/scores retained.',
            'videos': {}, 'targets': targets, 'missing_targets_retained_baseline': missing,
            'continuation_generation_parent_differences': []}
    for ep in read(BASE / 'manifest.json')['episodes']:
        records = []
        previous_sha = None
        for shot in ep['shots']:
            key = ep['episode_id'] + '/' + shot['shot_key']
            old = before['videos'][key]
            assert sha(old['source']) == old['sha256']
            replacement = targets.get(key)
            source = replacement['record']['video'] if replacement else old['source']
            digest = replacement['record']['video_sha256'] if replacement else old['sha256']
            scene, number = map(int, shot['shot_key'].split(':'))
            link = OUT / 'candidate_media' / ep['episode_id'] / f'{scene:03d}_{number:03d}.mp4'
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists():
                assert link.resolve() == Path(source).resolve()
            else:
                link.symlink_to(Path(source).resolve())
            data['videos'][key] = {'source': source, 'sha256': digest, 'canonical': str(link),
                                   'before_source': old['source'], 'before_sha256': old['sha256'], 'replaced': bool(replacement)}
            records.append({'shot_key': shot['shot_key'], 'cut': shot['cut'], 'video': source, 'video_sha256': digest})
            if replacement and not shot['cut'] and previous_sha != replacement['record']['parent_video_sha256']:
                data['continuation_generation_parent_differences'].append({'episode_id': ep['episode_id'],
                    'shot': shot['shot_key'], 'generation_parent_sha256': replacement['record']['parent_video_sha256'],
                    'spliced_preceding_video_sha256': previous_sha})
            if not replacement and not shot['cut'] and previous_sha != before['videos'][ep['episode_id'] + '/' + ep['shots'][len(records) - 2]['shot_key']]['sha256']:
                data['continuation_generation_parent_differences'].append({'episode_id': ep['episode_id'],
                    'shot': shot['shot_key'], 'kind': 'unchanged_child_after_replaced_parent',
                    'previous_sequence_parent_sha256': before['videos'][ep['episode_id'] + '/' + ep['shots'][len(records) - 2]['shot_key']]['sha256'],
                    'spliced_preceding_video_sha256': previous_sha})
            previous_sha = digest
        immutable(OUT / 'episode_media' / 'episodes' / ep['episode_id'] / 'episode_result.json', {'episode_id': ep['episode_id'], 'shots': records})
    assert len(data['videos']) == 2491
    immutable(OUT / 'replacement_manifest.json', data)
    return data


def evaluate(which, data):
    prepare()
    for row in data['videos'].values():
        assert sha(row['source']) == row['sha256']
    full = read(BASE / 'split_full140.json')
    ids = {k.split('/')[0] for k in data['targets']}
    split = {t: [eid for eid in full[t] if which == 'full140' or eid in ids] for t in ['easy', 'medium', 'hard']}
    dst = OUT / which
    immutable(dst / 'split.json', split)
    (dst / 'results').mkdir(parents=True, exist_ok=True)
    if which == 'full140':
        for eid in ids:
            old = OUT / 'affected/results' / (eid + '.json')
            dest = dst / 'results' / old.name
            if not dest.exists():
                shutil.copyfile(old, dest)
    status('evaluating_' + which, episodes=sum(map(len, split.values())))
    cmd = [EVAL, '-u', str(EVALUATOR), '--results_dir', str(OUT / 'candidate_media'),
           '--scripts_dir', str(ROOT / 'benchmarks/entitybench/data/scripts'), '--split_json', str(dst / 'split.json'),
           '--out_dir', str(dst / 'results'), '--method_name', 'qwen140_prompt_v2_badshot_replacement',
           '--pillars', '1,2,3', '--skip_vlm', '--resume', '--device', 'cuda:0',
           '--vbench_cache_dir', '/root/autodl-tmp/models/entitybench_vbench']
    with (dst / 'evaluation.log').open('a') as log:
        subprocess.run(cmd, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    m = read(dst / 'results/run_manifest.json')
    assert m['strict'] and m['skip_vlm'] and not m['allow_config_override'] and m['vlm_call_stats']['total_calls'] == 0
    original = read(BASE / 'evaluation/official_parallel_v2/run_manifest.json')
    for field in ['config', 'python_version', 'libraries', 'model_checkpoints', 'pillars', 'evaluation_scope']:
        assert m[field] == original[field], (which, field, 'Scoring provenance changed')
    found = [eid for tier in split.values() for eid in tier]
    for eid in found:
        raw = read(dst / 'results' / (eid + '.json'))
        assert raw['episode_id'] == eid
        for name in agg.LABELS:
            metric = agg.metric(raw, name)
            assert metric['n_failed'] == 0, (eid, name, metric)
    write(dst / 'validation.json', {'complete_episodes': len(found), 'skip_vlm': True, 'vlm_calls': 0, 'input_manifest_sha256': sha(OUT / 'replacement_manifest.json')})


def per_shot(result, work, eid, key):
    scene, shot = map(int, key.split(':'))
    values = {}
    for dimension in DIMENSIONS:
        path = work / f'{eid}_{dimension}_eval_results.json'
        matches = [r['video_results'] for r in read(path)[dimension][1] if Path(r['video_path']).stem == f'{scene:03d}_{shot:03d}']
        assert len(matches) == 1, (eid, key, dimension)
        values[dimension] = float(matches[0])
    detail = read(work / 'eval_artifacts' / eid / 'intra_shot' / key.replace(':', '_') / '_eval.json')
    for kind, label in [('character', 'intra_character_presence'), ('object', 'intra_object_presence'), ('location', 'intra_location_presence')]:
        p = detail['presence'][kind]
        values[label] = p['n_present'] / p['n_total'] if p['n_total'] else None
    for prefix, label in [('face_', 'cs_face'), ('object_', 'cs_object')]:
        scores = [app['sim_to_centroid'] for name, item in result['per_entity_scores'].items() if name.startswith(prefix)
                  for app in item['per_shot_sim_to_centroid'] if app['shot'] == key]
        values[label] = math.fsum(scores) / len(scores) if scores else None
    values['intra_action_overall'] = None
    values['cs_face_visible_single_face_reviewed'] = None
    return values


def compare_shots(data):
    os.environ.update(environment())
    module = scorer()
    selected = {e['episode_id']: e for e in read(OUT / 'baseline_selection.json')['episodes']}
    episodes = {e['episode_id']: e for e in read(BASE / 'manifest.json')['episodes']}
    rows, pairs = [], []
    boundary = {}
    for eid, ep in episodes.items():
        keys = {k.split('/')[1] for k in data['targets'] if k.startswith(eid + '/')}
        if not keys:
            continue
        oldpath = Path(selected[eid]['selected_result'])
        old, new = read(oldpath), read(OUT / 'affected/results' / (eid + '.json'))
        for i, shot in enumerate(ep['shots']):
            key = shot['shot_key']
            if i and not shot['cut'] and (key in keys or ep['shots'][i - 1]['shot_key'] in keys):
                previous = ep['shots'][i - 1]['shot_key']
                a, b = data['videos'][eid + '/' + previous], data['videos'][eid + '/' + key]
                vals = {label: float(module.compute_transition_score(a[field], b[field], 1, 'cuda:0'))
                        for label, field in [('before', 'before_source'), ('after', 'source')]}
                boundary[eid, key] = vals
                pairs.append({'episode_id': eid, 'previous': previous, 'next': key, **vals})
            if key not in keys:
                continue
            a = per_shot(old, oldpath.parent / '.work', eid, key)
            b = per_shot(new, OUT / 'affected/results/.work', eid, key)
            edge = boundary.get((eid, key), {})
            a['cs_transition_boundary'], b['cs_transition_boundary'] = edge.get('before'), edge.get('after')
            rows.append({'episode_id': eid, 'shot': key, 'before': a, 'after': b})
    write(OUT / 'target_shot_comparison.json', {'scope': 'Shot-level diagnostics from complete episode context; cross-shot values use each full episode centroid; boundaries are actual adjacent pairs. Reviewed DINO pending localization review.', 'rows': rows})
    write(OUT / 'target_boundary_pairs.json', pairs)
    with (OUT / 'target_shot_comparison.csv').open('w', newline='') as stream:
        columns = ['episode_id', 'shot'] + [f'{m}_{label}' for m in [*agg.LABELS, 'cs_face_visible_single_face_reviewed'] for label in ['before', 'after', 'delta']]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in ['episode_id', 'shot']}
            for name in [*agg.LABELS, 'cs_face_visible_single_face_reviewed']:
                a, b = row['before'].get(name), row['after'].get(name)
                flat.update({name + '_before': a, name + '_after': b, name + '_delta': b - a if a is not None and b is not None else None})
            writer.writerow(flat)


def aggregate_full(data):
    module = scorer()
    selected = {r['episode_id']: r for r in read(OUT / 'baseline_selection.json')['episodes']}
    groups = {name: [] for name in ['original', 'before_v2', 'after_v2']}
    comparisons = []
    for ep in read(BASE / 'manifest.json')['episodes']:
        eid, tier = ep['episode_id'], ep['difficulty']
        paths = {'original': selected[eid]['baseline_result'], 'before_v2': selected[eid]['selected_result'],
                 'after_v2': str(OUT / 'full140/results' / (eid + '.json'))}
        for label, path in paths.items():
            raw = read(path)
            assert raw['num_shots'] == len(ep['shots'])
            groups[label].append(module._episode_result_from_json(eid, tier, raw))
        comparisons.append({'episode_id': eid, 'tier': tier, 'paths': paths})
    split = read(BASE / 'split_full140.json')
    reports = {label: module.aggregate_results(rows, split) for label, rows in groups.items()}
    write(OUT / 'full140_aggregates.json', reports)
    table = [{'metric': name, 'label': label, **{k: agg.agg_metric(v, name)['mean'] for k, v in reports.items()}}
             for name, label in agg.LABELS.items()]
    write(OUT / 'full140_official_comparison.json', {'episodes': 140, 'shots': 2491, 'replaced_shots': len(data['targets']),
          'missing_replacements': data['missing_targets_retained_baseline'], 'metrics': table, 'per_episode': comparisons,
          'default_reviewed_dino': 'pending_localization_review', 'vlm_calls': 0, 'scope': 'partial_without_vlm; raw cs_face/cs_object provisional without fidelity gate'})
    text = '# 全140集：v2镜头替换后官方非VLM指标\n\n默认定位复查DINO待完成复查；raw cs_face单独保留，不能替代默认指标。\n\n| 指标 | 原始benchmark | 替换前51集快照 | v2替换后 |\n|---|---:|---:|---:|\n'
    for row in table:
        text += '| ' + row['label'] + ' | ' + ' | '.join('—' if row[k] is None else f'{row[k]:.4f}' for k in groups) + ' |\n'
    (OUT / 'report.md').write_text(text)


def run():
    prepare()
    data = mapping()
    evaluate('affected', data)
    status('comparing_target_shots_all_available_metrics')
    compare_shots(data)
    immutable(OUT / 'replacement_applied.json', {'replacement_manifest_sha256': sha(OUT / 'replacement_manifest.json'),
              'shot_comparison_sha256': sha(OUT / 'target_shot_comparison.json'), 'historic_inputs_overwritten': False})
    evaluate('full140', data)
    aggregate_full(data)
    subprocess.run([EVAL, '-u', str(ROOT / 'pretest/qwen140_badshot_reviewed_dino_20261104.py'), 'auto'], env=environment(), check=True)
    status('official_full140_complete_reviewed_dino_pending_localization_review', official_episodes=140, replaced_shots=len(data['targets']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    else:
        try:
            run()
        except Exception as exc:
            status('stopped_with_recorded_error', error=repr(exc))
            raise
