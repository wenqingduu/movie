#!/usr/bin/env python3
"""Second bounded cohort: weak official cs_face and character presence episodes."""
import argparse
from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path

import qwen140_multiface_weak_episode_batch as batch
import qwen140_weak4_aggregate as aggregate
from qwen140_cs_face_crop_review import build as build_crop_review

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/entitybench_qwen140_multiface_focus4_20261029'
SELECTION = {'run317': 'i25_j60', 'run15': 'i0_j19', 'run354': 'i90_j105', 'run348': 'i30_j49'}
GROUPS = {'run317': 'both_low', 'run15': 'presence_low', 'run354': 'cs_face_low', 'run348': 'both_low'}


def index(plan):
    rows = []
    for item in plan['episodes']:
        d = Path(item['output_root'])
        completed = len(list((d / 'episodes').glob('*/shot_*/result.json')))
        ready = (d / 'complete.json').exists()
        relative = d.relative_to(OUT)
        scores = item['scores']
        rows.append(f'<tr><td>{html.escape(item["episode_id"])}</td><td>{scores["intra_character_presence"]:.4f}</td><td>{scores["cs_face"]:.4f}</td><td>{completed}/{len(item["episode"]["shots"])}</td><td>{"生成/评测完成，视频待审" if ready else "排队或运行，详见status.json"}</td><td><a href="{relative}/review/index.html">前后视频</a> · <a href="{relative}/report.md">指标</a> · <a href="{relative}/review/cs_face_crops/index.html">裁剪对照</a></td></tr>')
    (OUT / 'index.html').write_text('<meta charset="utf-8"><h1>第二批：cs_face与人物出现率低分集</h1>'
        f'<p>{plan["num_episodes"]}个完整episode，{plan["num_shots"]}shot；统一冻结版本、多人3D、strict/skip_vlm。</p>'
        '<p>链接随阶段生成。官方cs_face为人物区域DINO；自动裁剪页不代表完成身份审阅。</p>'
        '<p><a href="report.md">最终/已完成配对子集汇总</a> · <a href="status.json">实时状态</a></p>'
        '<table border="1" cellpadding="7"><tr><th>episode</th><th>原出现率</th><th>原cs_face</th><th>shot完成数</th><th>状态</th><th>结果</th></tr>' + ''.join(rows) + '</table>')


def configure():
    original_summary = batch.summarize
    def summarize(item):
        original_summary(item)
        try:
            build_crop_review(item)
        except Exception as exc:
            batch.write(Path(item['output_root']) / 'crop_review_exception.json', {'error': repr(exc), 'official_scores_unchanged': True})
    batch.OUT = OUT
    batch.SELECTION = SELECTION
    batch.CODE = list(dict.fromkeys(batch.CODE + [
        'pretest/qwen140_multiface_focus_episode_batch.py', 'pretest/qwen140_weak4_aggregate.py',
        'pretest/qwen140_cs_face_crop_review.py', 'pretest/qwen140_single_pass_shot.py',
        'pretest/prepare_entitybench_character_assets.py']))
    batch.index = index
    batch.summarize = summarize
    aggregate.OUT = OUT


def prepare():
    existed = (OUT / 'cohort.json').exists()
    plan = batch.prepare()
    if existed:
        return plan
    assert plan['num_episodes'] == 4 and plan['num_shots'] == 55
    for item in plan['episodes']:
        episode = item['episode']
        desc = {v['name']: v['description'] for v in episode['entity_descriptions'].values()}
        for shot in episode['shots']:
            for name in shot['entity_schedule'].get('characters', []):
                assert name in desc and desc[name], (episode['episode_id'], shot['shot_key'], name)
        item['focus_group'] = GROUPS[episode['run']]
    plan.update(selection_policy='Exclude first weak4 and run700; bound new cohort to four complete <=15-shot episodes: two low in both cs_face/presence, one lowest remaining presence, one lowest remaining cs_face within <=15-shot episodes. Not a claim of all weak episodes.',
        experiment_revision='final_hardcut_cast_and_inherited_tail_roles_20261029',
        unified_version_from_first_shot=True,
        primary_metrics=['cs_face', 'intra_character_presence'],
        previous_cohort_reused=False,
        crop_review_policy='Automatic saved official canonical crop sheets for both original/rerun; no crop replacement, identity score claims or VLM calls',
        style_evidence={'path': str(OUT / 'preflight/baseline_style_samples.jpg'), 'sha256': batch.sha(OUT / 'preflight/baseline_style_samples.jpg'), 'inspection': 'First/middle/last sampled frames of all four episodes inspected: photographic rendering; monochrome photo in run317 allowed as shot-specific style within live-action episode'},
        excluded_runs=['run771', 'run455', 'run484', 'run1404', 'run700'])
    batch.write(OUT / 'cohort.json', plan)
    start = datetime.now(timezone.utc)
    batch.write(OUT / 'eta.json', {'created_utc': start.isoformat(), 'estimated_finish_utc': (start + timedelta(hours=7)).isoformat(),
        'finish_window_beijing': [(start + timedelta(hours=h)).astimezone(timezone(timedelta(hours=8))).isoformat() for h in [6, 8]],
        'range_hours': [6, 8], 'basis': '55 full shots x measured ~5 min plus 24 clean portraits/Gaussians/calibrations and official 12-metric evaluation; GPU/API must remain available',
        'condition': 'Record technical errors and continue other episodes; gate skips reuse same Control; API delays may change time or leave explicitly incomplete episodes'})
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'aggregate'])
    args = parser.parse_args()
    configure()
    plan = prepare()
    if args.action == 'prepare':
        index(plan)
        batch.status('prepared', expected_episodes=4, expected_shots=55)
    elif args.action == 'aggregate':
        aggregate.aggregate()
    else:
        try:
            batch.run(plan)
        except Exception as exc:
            batch.status('batch_setup_exception_recorded', error=repr(exc))
            raise
        finally:
            try:
                aggregate.aggregate()
            except Exception as exc:
                batch.write(OUT / 'aggregation_exception.json', {'error': repr(exc), 'action': 'Record and preserve per-episode results; no official score alteration'})


if __name__ == '__main__':
    main()
