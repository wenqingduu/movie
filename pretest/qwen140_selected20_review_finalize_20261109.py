"""Finalize selected20 evaluation after score-hidden canonical localization review.

This adapter never starts scoring or generation and never changes frozen workers.
The comparison baseline is the exact-input expanded review from 2026-11-07.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import qwen140_original40_selected_gain_20261109 as job
import qwen140_localization_repair_gpu_20261108 as cached

OUT, REGION = job.OUT, job.OUT / 'visible_region'
read, write, sha = job.read, job.write, job.sha
diag = job.frozen.diag


def state():
    plan = read(OUT / 'plan.json')
    snapshot = read(job.SNAP)
    audit = read(cached.AUDIT / 'report.json')
    assert audit['snapshot_sha256'] == sha(job.SNAP)
    assert audit['decision_sha256'] == sha(cached.AUDIT / 'decisions.json')
    for path, digest in audit['known_review_sources'].items():
        assert sha(path) == digest, path
    for binding in plan['bindings']:
        assert sha(binding['path']) == binding['sha256'], binding['path']
    rows = cached.gate_maps()['current']
    rejected = {diag.key(r): r for r in audit['applied_exclusions']['current']}
    assert len(rows) == 4991
    for row in rows.values():
        if row['original_crop']:
            assert sha(row['original_crop']) == row['original_crop_sha256']
    for key, row in rejected.items():
        assert rows[key]['original_crop_sha256'] == row['original_crop_sha256']
    return plan, snapshot, audit, rows, rejected


def metric_equal(a, b):
    assert a['n_evaluated'] == b['n_evaluated'], (a, b)
    assert (a['value'] is None and b['value'] is None) or (
        a['value'] is not None and b['value'] is not None and abs(a['value'] - b['value']) < 1e-10
    ), (a, b)


def baseline():
    _, snapshot, audit, rows, rejected = state()
    mod = diag.gate.load_official_pure()
    expected = {r['episode_id']: r['current']['extended'] for r in audit['per_episode']}
    results = []
    for item in snapshot['episodes']:
        assert sha(item['result']) == item['result_sha256']
        value, _ = diag.compute(item['episode'], read(item['result']), rows, rejected, mod)
        metric_equal(value.metrics['cs_face'].to_json(), expected[item['episode_id']])
        results.append(value)
    aggregate = mod.aggregate_results(results, read(job.BASE / 'split_full140.json'))
    for tier in ['all', 'easy', 'medium', 'hard']:
        assert aggregate[tier]['cs_face'] == audit['current'][tier]['cs_face']
    result = dict(episodes=140, scheduled_role_shots=len(rows),
                  source=str(cached.AUDIT / 'report.json'), source_sha256=sha(cached.AUDIT / 'report.json'),
                  snapshot_sha256=sha(job.SNAP), baseline=aggregate,
                  coverage=audit['coverage']['current'], exact_crop_exclusions=len(rejected),
                  all140_per_episode_reproduced=True, raw_centroids_reproduced=True,
                  no_new_inference=True, vlm_calls=0)
    write(OUT / 'reviewed_dino_baseline_validation.json', result)
    print('Baseline reproduced: 140 episodes; exact-input exclusions:', len(rejected), flush=True)
    return result


def official_validation(plan, snapshot):
    changed = set(plan['recompute_episode_ids'])
    assert len(changed) == 18
    expected = read(job.BASE / 'evaluation/official_parallel_v2/run_manifest.json')
    dimensions = job.trial.core.DIMS
    receipts, shots, failures, paths, provenance = [], 0, {}, {}, {}
    exact_dynamic = {'before': [], 'after': []}
    for item in snapshot['episodes']:
        eid, ep = item['episode_id'], item['episode']
        path = OUT / 'affected/results' / (eid + '.json') if eid in changed else Path(item['result'])
        raw = read(path)
        exact_dynamic['before'].append(read(item['result'])['vbench_means']['dynamic_degree']['value'])
        exact_dynamic['after'].append(raw['vbench_means']['dynamic_degree']['value'])
        assert raw['num_shots'] == len(ep['shots'])
        manifest_path = path.resolve().parent / 'run_manifest.json'
        m = read(manifest_path)
        for field in ['config', 'python_version', 'libraries', 'model_checkpoints', 'pillars', 'evaluation_scope']:
            assert m[field] == expected[field], (eid, field)
        assert m['strict'] and m['skip_vlm'] and not m['allow_config_override']
        assert m['vlm_call_stats']['total_calls'] == 0
        provenance[str(manifest_path)] = sha(manifest_path)
        paths[eid] = dict(path=str(path), sha256=sha(path))
        for name in job.frozen.J.full.agg.LABELS:
            mv = job.frozen.J.full.agg.metric(raw, name)
            assert mv['n_failed'] == 0, (eid, name, mv)
            failures[name] = failures.get(name, 0) + mv['n_failed']
        if eid not in changed:
            continue
        receipt_path = path.resolve().parent / '.work' / (eid + '_vbench_reuse_receipt.json')
        receipt = read(receipt_path)
        assert receipt['cache_manifest_sha256'] == sha(OUT / 'vbench_cache_manifest.json')
        assert receipt['adapter_sha256'] == sha(job.ROOT / 'pretest/qwen140_vbench_verified_cache_20261105.py')
        assert receipt['unchanged_shots_reused'] == len(ep['shots'])
        assert receipt['changed_shots_computed'] == 0
        for dimension in dimensions:
            data = read(path.resolve().parent / '.work' / (eid + '_' + dimension + '_eval_results.json'))
            assert len(data[dimension][1]) == len(ep['shots'])
        receipts.append(dict(path=str(receipt_path), sha256=sha(receipt_path)))
        shots += len(ep['shots'])
    assert shots == 539 and len(receipts) == 18 and len(paths) == 140
    aggregates = read(OUT / 'official_aggregates.json')
    dynamic = read(OUT / 'dynamic_exact.json')
    for edition in ['before', 'after']:
        value = job.frozen.J.full.agg.agg_metric(aggregates[edition], 'dynamic_degree')['mean']
        exact = sum(exact_dynamic[edition]) / len(exact_dynamic[edition])
        assert abs(exact - dynamic[edition]) < 1e-12
        # The pinned official aggregator explicitly rounds means to four decimals.
        assert value == round(exact, 4)
    result = dict(episodes=140, shots=2491, recomputed_episodes=18, reused_episodes=122,
                  vbench_reused_shots=shots, new_vbench_inference_calls=0,
                  scoring_manifests=provenance, result_files=paths,
                  vbench_receipts=receipts, metric_failures=failures,
                  exact_dynamic_matches_episode_macro=True, official_mean_rounding_decimals=4,
                  strict=True, skip_vlm=True, vlm_calls=0,
                  scope='partial_without_vlm', fidelity_gate='unavailable')
    write(OUT / 'official_completion_validation.json', result)
    return result


def coverage(rows, rejected, results):
    return dict(scheduled_role_shots=len(rows), canonical_present=sum(r['original_status'] == 'present' for r in rows.values()),
                auto_eligible=sum(r['kept'] for r in rows.values()),
                retained=sum(r['kept'] and k not in rejected for k, r in rows.items()),
                exact_input_exclusions=sum(r['kept'] and k in rejected for k, r in rows.items()),
                scored_appearances=sum(r.metrics['cs_face'].n_evaluated for r in results),
                valid_episodes=sum(r.metrics['cs_face'].value is not None for r in results),
                missing_episodes=sum(r.metrics['cs_face'].value is None for r in results),
                gate_reasons=dict(Counter(r['reason'] for r in rows.values() if not r['kept'])))


def final():
    plan, snapshot, audit, oldrows, oldreject = state()
    # Completion prerequisites: never relabel an automatic score as reviewed.
    automatic = read(REGION / 'automatic_results.json')
    assert len(automatic) == 18
    review = read(REGION / 'review/manifest.json')
    decisions = read(REGION / 'review/decisions.json')
    assert decisions['manifest_sha256'] == sha(REGION / 'review/manifest.json')
    expected = {r['review_id']: r for r in review['rows']}
    assert len(decisions['decisions']) == len(expected)
    assert {r['review_id'] for r in decisions['decisions']} == set(expected)
    newrows = dict(oldrows)
    media = read(OUT / 'replacement_manifest.json')['videos']
    count = 0
    for ep in plan['episodes']:
        for shot in ep['shots']:
            data = read(REGION / 'extraction' / ep['episode_id'] / (shot['shot_key'].replace(':', '_') + '.json'))
            video = media[ep['episode_id'] + '/' + shot['shot_key']]
            artifact = OUT / 'affected/results/.work/eval_artifacts' / ep['episode_id'] / 'intra_shot' / shot['shot_key'].replace(':', '_') / '_eval.json'
            assert data['evaluation_sha256'] == sha(artifact)
            assert data['video_sha256'] == video['sha256'] == sha(video['canonical'])
            assert data['protocol_sha256'] == sha(REGION / 'protocol_v3_cuda.json')
            assert {r['character'] for r in data['rows']} == set(shot['entity_schedule'].get('characters', []))
            for row in data['rows']:
                if row['original_crop']:
                    assert sha(row['original_crop']) == row['original_crop_sha256']
                newrows[diag.key(row)] = row
            count += 1
    assert count == 539 and set(newrows) == set(oldrows)
    rejected = {k: r for k, r in oldreject.items() if newrows[k]['original_crop_sha256'] == r['original_crop_sha256']}
    for row in decisions['decisions']:
        ref = expected[row['review_id']]
        assert diag.key(row) == diag.key(ref)
        assert row['original_crop_sha256'] == ref['original_crop_sha256'] == newrows[diag.key(row)]['original_crop_sha256']
        assert row['decision'] in ['keep', 'exclude'] and row['reason'] and row['evidence']
        if row['decision'] == 'exclude':
            assert oldrows[diag.key(row)]['original_crop_sha256'] != row['original_crop_sha256'], 'Shared old input must be symmetric'
            rejected[diag.key(row)] = row
    official_validation(plan, snapshot)
    baseline()
    mod = diag.gate.load_official_pure()
    oldresults, newresults, per_episode = [], [], []
    check = {r['episode_id']: r['current']['extended'] for r in audit['per_episode']}
    changed = set(plan['recompute_episode_ids'])
    for item in snapshot['episodes']:
        ep, eid = item['episode'], item['episode_id']
        old = read(item['result'])
        new = read(OUT / 'affected/results' / (eid + '.json')) if eid in changed else old
        a, _ = diag.compute(ep, old, oldrows, oldreject, mod)
        b, _ = diag.compute(ep, new, newrows, rejected, mod)
        metric_equal(a.metrics['cs_face'].to_json(), check[eid])
        oldresults.append(a)
        newresults.append(b)
        per_episode.append(dict(episode_id=eid, before=a.metrics['cs_face'].to_json(), after=b.metrics['cs_face'].to_json()))
    split = read(job.BASE / 'split_full140.json')
    a, b = mod.aggregate_results(oldresults, split), mod.aggregate_results(newresults, split)
    report = dict(metric='cs_face_visible_single_face_reviewed', status='complete_with_localization_review',
                  scope='Full140 episode-macro, original canonical person-region DINO, visible-single-face gate and exact-input localization correction.',
                  baseline_source=str(cached.AUDIT / 'report.json'), baseline_sha256=sha(cached.AUDIT / 'report.json'),
                  before=a, after=b, per_episode=per_episode,
                  coverage=dict(before=coverage(oldrows, oldreject, oldresults), after=coverage(newrows, rejected, newresults)),
                  new_review_items=len(expected), new_exclusions=[r for r in decisions['decisions'] if r['decision'] == 'exclude'],
                  carried_exclusions=[r for k, r in oldreject.items() if k in rejected and rejected[k] == r],
                  review_manifest_sha256=sha(REGION / 'review/manifest.json'), decisions_sha256=sha(REGION / 'review/decisions.json'),
                  finalizer_sha256=sha(__file__), vlm_calls=0, fidelity_gate='unavailable', official_scope='partial_without_vlm')
    write(REGION / 'report.json', report)
    official = read(OUT / 'official_comparison.json')
    table = [dict(metric=report['metric'], label='人物一致性DINO（可见单脸＋定位纠错）',
                  before=a['all']['cs_face']['mean'], after=b['all']['cs_face']['mean'])] + official['metrics']
    write(OUT / 'complete_comparison.json', dict(metrics=table, episodes=140, shots=2491, attempted=32, selected=20,
          default_reviewed_dino=report['status'], coverage=report['coverage'], vlm_calls=0, scope='partial_without_vlm'))
    lines = ['# 20条动态增长替换后的完整140集评测', '',
             '140集／2491镜头，按episode等权；32条尝试，20条动态0→1被选中，12条保留原视频。媒体不覆盖，不以其他指标下降过滤。', '',
             '| 指标↑ | 替换前 | 替换后 | 差值 |', '|---|---:|---:|---:|']
    for row in table:
        x, y = row['before'], row['after']
        fmt = lambda v: '缺失' if v is None else f'{v:.6f}'
        delta = '缺失' if x is None or y is None else f'{y-x:+.6f}'
        lines.append(f"| {row['label']} | {fmt(x)} | {fmt(y)} | {delta} |")
    lines += ['', '默认人物DINO是项目条件诊断，使用11月7日扩展定位复核的同口径当前基线；raw cs_face另列，属于人物区域DINO，不是人脸识别。', '',
              f"新裁剪隐藏分数复核{len(expected)}项，确认错定位剔除{len(report['new_exclusions'])}项；真实漂移、重复和未决保留。剔除后重算完整episode人物中心。",
              f"DINO有效episode {report['coverage']['before']['valid_episodes']}→{report['coverage']['after']['valid_episodes']}／140；计分角色次 {report['coverage']['before']['scored_appearances']}→{report['coverage']['after']['scored_appearances']}，完整覆盖见visible_region/report.json。", '',
              'strict/skip_vlm，VLM调用0；外观保真、动作对齐、LLM一致性仍缺失，保真门控不可用，raw cs_face/cs_object为partial_without_vlm。不是完整论文全指标。', '',
              '18个受影响episode／539个完整上下文重新定位、评presence、跨镜头中心和真实接缝；六项VBench按视频SHA和评分来源复用。另122集复用后由官方函数聚合140集。', '',
              '这是按动态筛选的替换版本；相对冻结benchmark，提示词和后端配方均可能变化，不能归因为独立步数或LoRA消融。所有媒体、尝试和下降保留。']
    exact = read(OUT / 'dynamic_exact.json')
    lines += ['', '官方聚合表保留原函数的四位小数精度。按未舍入逐集动态均值独立核验：'
              f"{exact['before']:.12f}→{exact['after']:.12f}，增加{exact['delta_percentage_points']:.6f}个百分点；不是用表内舍入值作差。"]
    (OUT / 'complete_report.md').write_text('\n'.join(lines) + '\n')
    write(OUT / 'completion_validation.json', dict(official_validation_sha256=sha(OUT / 'official_completion_validation.json'),
          reviewed_report_sha256=sha(REGION / 'report.json'), complete_report_sha256=sha(OUT / 'complete_report.md'),
          review_decisions_sha256=sha(REGION / 'review/decisions.json'), media_selection_unchanged=True,
          episodes=140, shots=2491, selected=20, new_generation=False, vlm_calls=0))
    job.status('complete_all_available_metrics_with_reviewed_dino', selected_shots=20, episodes=140,
               reviewed_items=len(expected), default_dino=report['status'], report=str(OUT / 'complete_report.md'))
    print('Complete:', OUT / 'complete_report.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['baseline', 'final'])
    args = parser.parse_args()
    baseline() if args.action == 'baseline' else final()
