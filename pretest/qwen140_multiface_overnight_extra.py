#!/usr/bin/env python3
"""Deadline-aware additional low-presence/cs_face episodes for the overnight queue."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import time

import qwen140_multiface_focus_episode_batch as focused
import qwen140_multiface_weak_episode_batch as batch
import qwen140_weak4_aggregate as aggregate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/entitybench_qwen140_multiface_overnight_extra_20261029'
SELECTION = {'run974': 'i35_j53', 'run227': 'i10_j20', 'run470': 'i75_j84',
             'run1537': 'i25_j48', 'run893': 'i30_j43'}
DEADLINE = '2026-10-30T04:00:00+00:00'


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def estimate_seconds(item):
    d = Path(item['output_root'])
    n = len(list((d / 'episodes').glob('*/shot_*/result.json')))
    remaining = len(item['episode']['shots']) - n
    if (d / 'complete.json').exists():
        return 0
    roles = {name for s in item['episode']['shots'] for name in s['entity_schedule'].get('characters', [])}
    asset_seconds = 0 if (d / 'face_injection_policy.json').exists() else len(roles) * 150
    # Five minutes/shot from the measured single-4090 pipeline, plus evaluation and buffer.
    return remaining * 300 + asset_seconds + 480


def can_start(item, deadline, now=None):
    now = now or datetime.now(timezone.utc)
    return now.timestamp() + estimate_seconds(item) <= deadline.timestamp()


def index(plan):
    rows = []
    for i in plan['episodes']:
        d = Path(i['output_root']);rel = d.relative_to(OUT)
        n = len(list((d / 'episodes').glob('*/shot_*/result.json')))
        state = '完整生成/评测完成，视频待审' if (d / 'complete.json').exists() else '排队/运行/时间预算延期，见status.json'
        rows.append(f'<tr><td>{html.escape(i["episode_id"])}</td><td>{i["scores"]["intra_character_presence"]:.4f}</td><td>{i["scores"]["cs_face"]:.4f}</td><td>{n}/{len(i["episode"]["shots"])}</td><td>{state}</td><td><a href="{rel}/review/index.html">前后视频</a> · <a href="{rel}/report.md">指标</a> · <a href="{rel}/review/cs_face_crops/index.html">人物裁剪</a></td></tr>')
    (OUT / 'index.html').write_text('<meta charset="utf-8"><h1>夜间扩展：优先人物出现率与cs_face低分</h1><p>5集候选／62shot；按时间预算逐集领取。北京时间10月30日12:00前停止领取预计无法完成的新episode，已开始episode自然完成。延期集不冒充已完成。</p><p><a href="status.json">状态</a> · <a href="report.md">同一已完成子集正式汇总</a></p><table border="1" cellpadding="6"><tr><th>episode</th><th>原人物出现</th><th>原cs_face</th><th>shot进度</th><th>状态</th><th>结果</th></tr>' + ''.join(rows) + '</table>')


def configure():
    focused.OUT = OUT
    focused.SELECTION = SELECTION
    focused.configure()
    batch.CODE = list(dict.fromkeys(batch.CODE + ['pretest/qwen140_multiface_overnight_extra.py']))
    batch.index = index
    aggregate.OUT = OUT


def prepare():
    existed = (OUT / 'cohort.json').exists()
    plan = batch.prepare()
    if existed:
        return plan
    assert plan['num_episodes'] == 5 and plan['num_shots'] == 62
    total_roles = 0
    for item in plan['episodes']:
        ep = item['episode']
        desc = {v['name']: v['description'] for v in ep['entity_descriptions'].values()}
        names = {n for s in ep['shots'] for n in s['entity_schedule'].get('characters', [])}
        assert names <= set(desc) and all(desc[n] for n in names)
        total_roles += len(names)
        assert item['scores']['cs_face'] < plan['thresholds_exact_episode_means']['cs_face']
        assert item['scores']['intra_character_presence'] < plan['thresholds_exact_episode_means']['intra_character_presence']
        item['focus_group'] = 'both_low'
    plan.update(experiment_revision='unified_focus_pipeline_20261029_overnight_extension',
                unified_version_from_first_shot=True, total_role_identities=total_roles,
                deadline_utc=DEADLINE, primary_metrics=['cs_face', 'intra_character_presence'],
                selection_policy='Five new human-character complete episodes low in both official cs_face and character presence; exclude first weak4, focus4 and run700. First three are priority, last two reserve if time budget permits. Not all low episodes or a representative full benchmark.',
                excluded_runs=['run771', 'run455', 'run484', 'run1404', 'run700', 'run317', 'run15', 'run354', 'run348'],
                style_evidence={'path': str(OUT / 'preflight/baseline_style_samples.jpg'), 'sha256': batch.sha(OUT / 'preflight/baseline_style_samples.jpg'),
                                'inspection': '15 first/middle/last samples inspected: photographic human scenes; run893 has mixed anime/photo appearances in original, unify episode to photographic live-action'},
                deadline_policy='Before each new episode, require estimated remaining generation/assets/evaluation to fit before noon; preserve deferred episodes, never kill an active shot/episode or shut down instance',
                crop_review_policy='Saved official original/rerun canonical character crop sheets, without modifying metrics; manual identity inspection pending')
    assert total_roles == 31
    batch.write(OUT / 'cohort.json', plan)
    for item in plan['episodes']:
        f = Path(item['output_root']) / 'style_specs.json'
        styles = batch.read(f);styles['scope'] = 'Five queued overnight complete episodes; deadline admission'
        batch.write(f, styles)
    batch.write(OUT / 'eta.json', {'deadline_utc': DEADLINE, 'deadline_beijing': '2026-10-30T12:00:00+08:00',
        'queued_episodes': 5, 'queued_shots': 62, 'execution_order': 'after focus4 repairs and the two Clara figure trajectories',
        'scope': 'Budgeted candidate queue, no guarantee all reserve episodes fit; actual durations/API/repairs determine completed subset'})
    return plan


def run(plan, deadline):
    for f in plan['frozen_code']:
        assert batch.sha(f['path']) == f['sha256'], f['path']
    assert batch.sha(batch.BASE / 'manifest.json') == plan['baseline_manifest_sha256']
    assert batch.sha(batch.TEMPLATE) == plan['v7_template_sha256']
    index(plan)
    while True:
        if datetime.now(timezone.utc) >= deadline:
            batch.status('deadline_reached_pending_episodes', expected_episodes=5, expected_shots=62)
            return
        probe = subprocess.run([batch.QWEN, '-c', 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)'],
                               env={**__import__('os').environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}, capture_output=True, timeout=60)
        if probe.returncode == 0:
            break
        batch.status('waiting_for_gpu', deadline_utc=deadline.isoformat())
        time.sleep(30)
    errors, deferred = [], []
    for item in plan['episodes']:
        d = Path(item['output_root'])
        if (d / 'complete.json').exists():
            continue
        expected = estimate_seconds(item)
        if not can_start(item, deadline):
            deferred.append({'episode_id': item['episode_id'], 'estimated_seconds': expected, 'reason': 'Estimated complete episode exceeds remaining noon budget'})
            batch.write(OUT / 'deferred_episodes.json', deferred)
            continue
        batch.write(OUT / 'admission_latest.json', {'episode_id': item['episode_id'], 'admitted_utc': datetime.now(timezone.utc).isoformat(), 'estimated_seconds': expected, 'deadline_utc': deadline.isoformat()})
        try:
            env = batch.assets(item, plan)
            if not (d / 'rerun_manifest.json').exists():
                for action in ['prepare', 'refs']:
                    batch.command(item, action, [batch.WAN, '-u', 'pretest/qwen140_episode_single_pass.py', action, '--run', item['episode']['run']], env)
            if not (d / 'episodes' / item['episode_id'] / 'episode_result.json').exists():
                batch.command(item, 'generation', [batch.WAN, '-u', 'pretest/qwen140_episode_single_pass.py', 'run-remaining', '--run', item['episode']['run']], env)
            for action in ['prepare', 'run', 'baseline-aggregate', 'compare']:
                batch.command(item, 'evaluation_' + action, [batch.EVAL, '-u', 'pretest/qwen140_integrated_episode_eval.py', action, '--name', 'weak4_v1', '--runs', item['episode']['run']], env)
            batch.summarize(item)
        except Exception as exc:
            info = {'episode_id': item['episode_id'], 'recorded_utc': batch.now(), 'error': repr(exc), 'stage_status': batch.read(OUT / 'status.json'), 'action': 'Preserve valid prefix; record for repair supervisor; continue next episode if within deadline'}
            errors.append(info)
            batch.write(d / 'technical_exception.json', info)
            batch.write(OUT / 'technical_exceptions.json', errors)
        index(plan)
    completed = [i for i in plan['episodes'] if (Path(i['output_root']) / 'complete.json').exists()]
    state = ('complete_non_vlm_pending_video_visual_review' if len(completed) == len(plan['episodes']) else
             'batch_finished_with_recorded_exceptions' if errors else 'finished_with_deferred_episodes')
    batch.status(state, completed_episodes=len(completed), completed_shots=sum(len(i['episode']['shots']) for i in completed),
                 expected_episodes=5, expected_shots=62, technical_exceptions=len(errors), deferred_episodes=deferred)


def main():
    p = argparse.ArgumentParser();p.add_argument('action', choices=['prepare', 'run', 'aggregate']);p.add_argument('--deadline', default=DEADLINE)
    args = p.parse_args();configure();plan = prepare()
    if args.action == 'prepare':
        index(plan);batch.status('prepared', expected_episodes=5, expected_shots=62)
    elif args.action == 'aggregate':
        aggregate.aggregate()
    else:
        try:
            run(plan, utc(args.deadline))
        except Exception as exc:
            batch.status('batch_setup_exception_recorded', error=repr(exc));raise
        finally:
            try:
                aggregate.aggregate()
            except Exception as exc:
                batch.write(OUT / 'aggregation_exception.json', {'error': repr(exc), 'action': 'Preserve result and log; no score modification'})


if __name__ == '__main__':
    main()
