"""Audit completed Qwen scores using the pinned official empty-summary schema.

No inference, report rewriting or modification of the frozen original runner.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

from audit_entitybench_qwen21_full140 import audit_media, verify_adapters
from prepare_entitybench_qwen21_full140 import OUTPUT, read
from prepare_entitybench_qwen21_20ep import sha256, write_json
from run_entitybench_qwen21_full140 import HELPER, METHOD

TIERS = ('all', 'easy', 'medium', 'hard')


def validate_missing_vlm(report, split, names):
    """Empty official summaries are {value: null, coverage: 0}, not mean summaries."""
    assert len(names) == len(set(names)) == 39
    for tier in TIERS:
        count = sum(len(split[t]) for t in TIERS[1:]) if tier == 'all' else len(split[tier])
        assert report[tier]['num_episodes'] == count
        for name in names:
            expected = dict(value=None, n_episodes_evaluated=0, n_episodes_total=count)
            assert report[tier][name] == expected, (tier, name, report[tier][name])


def validate_episode_absent(metrics, names):
    # The skip_vlm adapter omits most VLM metrics from episode JSON altogether.
    # Official aggregation treats omitted values as unavailable, never as zero.
    for name in names:
        if name in metrics:
            metric = metrics[name]
            assert metric['value'] is None and metric['n_evaluated'] == 0, (name, metric)


def audit(root):
    root = Path(root).resolve()
    verify_adapters(root)
    spec = importlib.util.spec_from_file_location('qwen140_posteval_helper', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    out = root / 'evaluation/official_parallel_v2'
    manifest = read(root / 'manifest.json')
    split = read(root / 'split_full140.json')
    assert split == read(root / 'split_20ep.json')
    ids = [eid for tier in TIERS[1:] for eid in split[tier]]
    assert len(ids) == len(set(ids)) == len(manifest['episodes']) == 140
    episodes, _, inputs = helper.freeze_inputs(root)
    assert set(episodes) == set(ids)
    assert sum(len(e['shots']) for e in episodes.values()) == 2491
    fingerprint = read(out / 'input_fingerprint.json')
    assert fingerprint['inputs'] == inputs and fingerprint['fingerprint'] == helper.fingerprint(inputs)
    execution = read(out / 'parallel_execution.json')
    status = read(out / 'parallel_status.json')
    assert execution['mode'] == 'parallel_inference' and execution['workers'] == 2
    assert execution['zero_vlm_calls'] and execution['helper_sha256'] == sha256(HELPER)
    assert status['phase'] == 'official_non_vlm_complete_pending_visual_review'
    assert status['complete_episodes'] == 140 and status['complete_shots'] == 2491
    plan = read(out / 'parallel_plan.json')
    assert plan['inputs'] == inputs and plan['helper_sha256'] == sha256(HELPER)
    assert plan['workers'] == 2 and plan['method'] == METHOD and not plan['aggregate_only']
    run_manifest = read(out / 'run_manifest.json')
    helper.validate_manifest(run_manifest)
    assert run_manifest['n_vlm_keys'] == 0
    scorer = helper.import_scorer(root)
    absent = [name for name in scorer.ALL_METRICS
              if name.startswith(('intra_face_', 'intra_object_', 'intra_location_', 'intra_action_', 'llm_'))
              and not name.endswith('_presence')]
    actual = read(out / f'report_{METHOD}.json')
    report_sha = sha256(out / f'report_{METHOD}.json')
    validate_missing_vlm(actual, split, absent)
    assert {p.stem for p in out.glob('*__run*.json')} == set(ids)
    worker_files = {}
    for index, assignments in enumerate(plan['shards']):
        worker = out / '.shards' / f'worker_{index:02d}'
        worker_manifest = read(worker / 'results/run_manifest.json')
        helper.validate_manifest(worker_manifest)
        assert worker_manifest['n_vlm_keys'] == 0
        compatible, differences = scorer.manifest_comparable(run_manifest, worker_manifest)
        assert compatible and run_manifest['libraries'] == worker_manifest['libraries'], differences
        expected_split = {tier: [eid for eid, assigned_tier in assignments if assigned_tier == tier]
                          for tier in TIERS[1:]}
        assert read(worker / 'split.json') == expected_split
        assert read(worker / 'input_fingerprint.json') == dict(
            full_inputs_fingerprint=helper.fingerprint(inputs), episodes=[eid for eid, _ in assignments])
        for eid, tier in assignments:
            assert eid not in worker_files and eid in split[tier]
            worker_files[eid] = worker / 'results' / f'{eid}.json'
    assert set(worker_files) == set(ids)
    results, schedules, metric_counts = [], 0, {}
    for episode in manifest['episodes']:
        eid, tier = episode['episode_id'], episode['difficulty']
        saved = read(out / f'{eid}.json')
        assert saved == read(worker_files[eid])
        assert saved['episode_id'] == eid and saved['tier'] == tier
        assert saved['num_shots'] == len(episode['shots'])
        for dimension in helper.DIMENSIONS:
            metric = saved['vbench_means'][dimension]
            assert metric['value'] is not None and math.isfinite(metric['value'])
            assert metric['n_evaluated'] == len(episode['shots']) and metric['n_failed'] == 0
        validate_episode_absent(saved['metrics'], absent)
        metric_counts[eid] = dict(metrics=saved['metrics'], vbench_means=saved['vbench_means'])
        results.append(scorer._episode_result_from_json(eid, tier, saved))
        artifacts = list(out.glob('**/eval_artifacts/' + eid))
        assert len(artifacts) == 1
        for shot in episode['shots']:
            ev = read(artifacts[0] / 'intra_shot' / shot['shot_key'].replace(':', '_') / '_eval.json')
            for kind, key in [('character', 'characters'), ('object', 'objects'), ('location', 'places')]:
                items = ev['presence'].get(kind, {}).get('items', [])
                assert {item['name'] for item in items} == set(shot['entity_schedule'].get(key, [])), (eid, shot['shot_key'], kind)
            schedules += 1
    recomputed = scorer.aggregate_results(results, split)
    assert all(recomputed[tier] == actual[tier] for tier in TIERS)
    # Revalidate current media against the already completed independent tail decode.
    media = read(root / 'audit/final_media.json')
    assert media['status'] == 'passed' and media['independent_tail_decode']
    assert media['manifest_sha256'] == sha256(root / 'manifest.json')
    assert media['generation_report_sha256'] == sha256(root / 'generation_report.json')
    quick = audit_media(root, 140, 2491, 1355, decode=False)
    assert sha256(out / f'report_{METHOD}.json') == report_sha
    result = dict(status='passed', episodes=140, shots=2491, verified_continuations=quick['continuations'],
        vbench_dimensions=helper.DIMENSIONS, vlm_calls=0, scope='partial_without_vlm',
        report_sha256=report_sha, schedule_verified=schedules, official_aggregation_verified=True,
        missing_vlm_metric_names=absent, missing_vlm_schema='value:null; n_episodes_evaluated:0',
        metrics_counts={tier: actual[tier] for tier in TIERS}, per_episode_metric_counts=metric_counts,
        source_script_sha256=sha256(__file__), helper_sha256=sha256(HELPER),
        media_audit_sha256=sha256(root / 'audit/final_media.json'))
    write_json(root / 'audit/final_official.json', result)
    print('Final official audit passed: 140 episodes / 2491 shots; 39 VLM metrics null; '
          'zero VLM calls; original official aggregation and report unchanged', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    audit(parser.parse_args().output_root)
