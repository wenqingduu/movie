"""Prioritize unfinished original-below-average reviewed-DINO episodes.

Scheduling adapter only: immutable remaining126 generation/scoring functions,
episode output roots and cache contracts are reused unchanged. No repair Agent.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess

import qwen140_multiface_remaining_episode_batch as original

ROOT=Path('/root/autodl-tmp/movie')
OUT=ROOT/'outputs/entitybench_csface_priority_20261101'
SOURCE=ROOT/'outputs/entitybench_qwen140_multiface_remaining126_20261030'
BASE=ROOT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140'
REVIEW=BASE/'evaluation/character_region_visible_face_gate_20261025/remaining_low_score_review/recomputed'
B=original.batch


def status(state,episode=None,**kw):
    p=B.read(OUT/'queue.json') if (OUT/'queue.json').exists() else {}
    B.write(OUT/'status.json',{'updated_utc':B.now(),'status':state,'active_episode':episode,
        'selected_episodes':p.get('num_episodes'),'selected_shots':p.get('num_shots'),
        'repair_agent_enabled':False,**kw})


def configure():
    original.configure()
    # Route stage status to this queue; retain original data/output/runtime roots.
    B.status=status


def validate(plan):
    source=B.read(SOURCE/'cohort.json')
    assert B.sha(SOURCE/'cohort.json')==plan['source_cohort_sha256']
    assert B.sha(Path(__file__).resolve())==plan['queue_script_sha256']
    for row in source['frozen_code']:assert B.sha(row['path'])==row['sha256'],row['path']
    assert B.sha(BASE/'manifest.json')==source['baseline_manifest_sha256']
    assert B.sha(BASE/'generation_report.json')==source['baseline_generation_sha256']
    for row in plan['selection_inputs']:assert B.sha(row['path'])==row['sha256'],row['path']
    return source


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'queue.json').exists():
        plan=B.read(OUT/'queue.json');validate(plan);return plan
    source=B.read(SOURCE/'cohort.json')
    scores={f.stem:B.read(f)['metrics']['cs_face']['value'] for f in (REVIEW/'results').glob('*.json')}
    values=[v for v in scores.values() if v is not None];average=math.fsum(values)/len(values)
    assert len(scores)==140 and len(values)==131
    assert round(average,4)==B.read(REVIEW/'report.json')['after']['all']['cs_face']['mean']
    eligible=[];excluded=[]
    for row in source['episodes']:
        eid=row['episode_id'];d=Path(row['output_root']);value=scores[eid]
        if (d/'complete.json').exists():reason='already_completed'
        elif value is None:reason='original_reviewed_dino_not_applicable'
        elif value>=average:reason='at_or_above_original_reviewed_mean'
        else:
            eligible.append({'episode_id':eid,'original_reviewed_dino':value,'shots':len(row['episode']['shots'])});continue
        excluded.append({'episode_id':eid,'reason':reason,'original_reviewed_dino':value})
    eligible.sort(key=lambda r:(r['original_reviewed_dino'],r['episode_id']))
    by_id={r['episode_id']:r for r in source['episodes']}
    roles=sum(len(original.roles(by_id[r['episode_id']])) for r in eligible)
    shots=sum(r['shots'] for r in eligible)
    estimated_bytes=roles*original.ASSET_BYTES_PER_ROLE+shots*original.OTHER_BYTES_PER_SHOT
    hours=(shots*(original.GEN_SECONDS_PER_SHOT+original.EVAL_SECONDS_PER_SHOT)+roles*original.ASSET_SECONDS_PER_ROLE)/3600
    selection_files=[REVIEW/'report.json']+sorted((REVIEW/'results').glob('*.json'))
    plan={'created_utc':B.now(),'selection_metric':'cs_face_visible_single_face_reviewed',
        'threshold_exact_original_episode_macro_mean':average,'threshold_rounded':round(average,4),
        'baseline_valid_episodes':len(values),'num_episodes':len(eligible),'num_shots':shots,
        'total_scheduled_role_identities':roles,'episodes':eligible,'excluded':excluded,
        'source_cohort':str(SOURCE/'cohort.json'),'source_cohort_sha256':B.sha(SOURCE/'cohort.json'),
        'queue_script_sha256':B.sha(Path(__file__).resolve()),
        'selection_inputs':[{'path':str(p),'sha256':B.sha(p)} for p in selection_files],
        'agent_model':source['agent_model'],'generation_revision':source['experiment_revision'],
        'scope':'Only incomplete below-original-reviewed-mean episodes; ascending original score. Existing30 complete reruns including latest run594 are untouched.',
        'generation_changes':False,'repair_agent_enabled':False,
        'failure_policy':source['failure_policy'],'disk_policy':source['disk_policy'],
        'evaluation':'Whole-episode official12 non-VLM metrics after each complete episode. Default reviewed-DINO requires later automatic gate/localization review, explicitly pending, never mislabeled raw cs_face.',
        'estimated_remaining_output_gib_before_reserve':estimated_bytes/1024**3,
        'historical_estimate_continuous_hours':hours,'free_gib_at_prepare':shutil.disk_usage(OUT).free/1024**3,
        'estimate_note':'Prior single-card phase timings only; assets/API/disk waits and exception handling change completion time.'}
    B.write(OUT/'queue.json',plan,frozen=True)
    B.write(SOURCE/'active_queue.json',{'queue_root':str(OUT),'status':str(OUT/'status.json'),
        'selection':'Below original reviewed-DINO mean, ascending score; original cohort/source unchanged'})
    refresh(plan);status('prepared')
    return plan


def refresh(plan):
    source=B.read(SOURCE/'cohort.json');by_id={r['episode_id']:r for r in source['episodes']}
    rows=[];done=0
    for position,row in enumerate(plan['episodes'],1):
        item=by_id[row['episode_id']];d=Path(item['output_root'])
        complete=(d/'complete.json').exists();done+=int(complete)
        generated=len(list(d.glob('episodes/*/shot_*/result.json')))
        rows.append({'rank':position,**row,'output_root':str(d),'generated_shots':generated,
                     'complete_official_non_vlm':complete,'default_reviewed_dino':'pending_new_crop_review'})
    B.write(OUT/'progress.json',{'episodes':rows,'completed_episodes':done,'expected_episodes':len(rows)})
    lines=['# 人物一致性低分episode优先重跑','',
        f"原始默认DINO均值{plan['threshold_rounded']:.4f}（131有效集）；本队列{plan['num_episodes']}集/{plan['num_shots']}shot，按原分数从低到高。已完成30集不重跑。",
        '', '沿用冻结的多人3D集成：Agent A/B/C路由、人物描述绑定、episode画风、原v7门控；守护修复Agent关闭。每集自动官方12项非VLM，默认可见单脸＋定位纠错DINO另需新裁剪复查，不能以raw cs_face替代。',
        '', '[实时状态](status.json) · [进度](progress.json) · [冻结队列](queue.json) · [技术异常](technical_exceptions.json)',
        '', f"历史阶段计时估算连续运行约{plan['historical_estimate_continuous_hours']/24:.1f}天；预计新增约{plan['estimated_remaining_output_gib_before_reserve']:.1f}GiB。启动可用{plan['free_gib_at_prepare']:.1f}GiB，每集前估算并保留8GiB，不足时等待扩容，不自动删除。",
        '', '| 顺序 | episode | 原始默认DINO | shot进度 | 官方非VLM完成 |', '|---|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['rank']} | {row['episode_id']} | {row['original_reviewed_dino']:.4f} | {row['generated_shots']}/{row['shots']} | {row['complete_official_non_vlm']} |")
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


def run(plan):
    # Same lock as the original queue prevents two writers to shared episodes.
    with (SOURCE/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=validate(plan);by_id={r['episode_id']:r for r in source['episodes']}
        probe=subprocess.run([B.QWEN,'-c','import torch;assert torch.cuda.is_available()'],capture_output=True,timeout=60)
        assert probe.returncode==0,'CUDA unavailable'
        errors=B.read(OUT/'technical_exceptions.json') if (OUT/'technical_exceptions.json').exists() else []
        for entry in plan['episodes']:
            item=by_id[entry['episode_id']];d=Path(item['output_root'])
            if (d/'complete.json').exists():continue
            validate(plan)
            assert B.sha(d/'cohort_config.json')==item['cohort_config_sha256']
            assert B.sha(d/'style_specs.json')==item['style_specs_sha256']
            needed=original.disk_required(item)
            while shutil.disk_usage(OUT).free<needed:
                status('waiting_for_disk_space',item['episode_id'],needed_gib=needed/1024**3,
                    free_gib=shutil.disk_usage(OUT).free/1024**3);__import__('time').sleep(60)
            cfg=original.api_config();assert cfg.get('SHOT_AGENT_MODEL',cfg['PORTRAIT_QWEN_MODEL'])==plan['agent_model']
            try:
                env=B.assets(item,source)
                if not (d/'rerun_manifest.json').exists():
                    for action in ['prepare','refs']:
                        B.command(item,action,[B.WAN,'-u','pretest/qwen140_episode_single_pass.py',action,'--run',item['episode']['run']],env)
                if not (d/'episodes'/item['episode_id']/'episode_result.json').exists():
                    B.command(item,'generation',[B.WAN,'-u','pretest/qwen140_episode_single_pass.py','run-remaining','--run',item['episode']['run']],env)
                for action in ['prepare','run','baseline-aggregate','compare']:
                    B.command(item,'evaluation_'+action,[B.EVAL,'-u','pretest/qwen140_integrated_episode_eval.py',action,'--name','weak4_v1','--runs',item['episode']['run']],env)
                original.summarize(item)
                print(B.now(),'Completed priority episode',item['episode_id'],flush=True)
            except Exception as exc:
                error={'recorded_utc':B.now(),'episode_id':item['episode_id'],'error':repr(exc),
                    'action':'Keep valid prefix and continue next episode; no repair Agent or model/gate change'}
                errors.append(error);B.write(OUT/'technical_exceptions.json',errors);B.write(d/'technical_exception.json',error)
                logs='\n'.join(p.read_text(errors='replace')[-6000:] for p in (d/'logs').glob('*.log'))
                if any(t in logs for t in ['insufficient_quota','invalid_api_key','Agent API HTTP 401','Arrearage','insufficient balance']):
                    status('blocked_api_auth_or_quota',item['episode_id'],error=error);refresh(plan);return
            refresh(plan)
        refresh(plan)
        progress=B.read(OUT/'progress.json')
        status('complete_official_non_vlm_pending_default_dino_review' if progress['completed_episodes']==plan['num_episodes'] else 'finished_with_recorded_episode_exceptions',
            completed_episodes=progress['completed_episodes'],technical_exceptions=len(errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    configure();plan=prepare()
    if args.action=='run':
        try:run(plan)
        except Exception as exc:status('queue_stopped_with_exception',error=repr(exc));raise
