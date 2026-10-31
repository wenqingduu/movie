#!/usr/bin/env python3
"""Serial remaining126 rerun; exact window binding, disk admission, no repair agent."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import qwen140_multiface_weak_episode_batch as batch
import qwen140_weak4_aggregate as aggregate
from qwen140_cs_face_crop_review import build as crop_review
from qwen140_pilot_route_agent import config as api_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/entitybench_qwen140_multiface_remaining126_20261030'
COMPLETED = ROOT/'outputs/entitybench_qwen140_partial_rerun_full140_20261030/selection_manifest.json'
GIB = 1024**3
ASSET_BYTES_PER_ROLE = 97.13458806818181 * 1024**2
OTHER_BYTES_PER_SHOT = 8.455428685897436 * 1024**2
GEN_SECONDS_PER_SHOT = 308.01385053071857
ASSET_SECONDS_PER_ROLE = 128.31349926993929
EVAL_SECONDS_PER_SHOT = 11.610185679963703
RESERVE_BYTES = 8*GIB


def tag(episode):
    return episode['episode_id'].split('__')[2]


def select_episodes(manifest, previous):
    excluded = {e['episode_id'] for e in previous['episodes'] if e['edition']=='rerun'}
    assert len(excluded)==14
    selected = [e for e in manifest['episodes'] if e['episode_id'] not in excluded]
    assert len(selected)==126 and sum(len(e['shots']) for e in selected)==2316
    assert len({e['episode_id'] for e in selected})==126
    return selected, excluded


def roles(item):
    return sorted({n for s in item['episode']['shots'] for n in s['entity_schedule'].get('characters',[])})


def status(state, episode=None, **values):
    batch.write(OUT/'status.json',{'updated_utc':batch.now(),'status':state,'active_episode':episode,
        'expected_episodes':126,'expected_shots':2316,'repair_agent_enabled':False,**values})


def configure():
    original = batch.environment
    def environment(directory):
        env=original(directory)
        # A run can occur in several windows. Each subprocess sees exactly its
        # own run->tag mapping, rather than a lossy cohort-wide dictionary.
        env['QWEN140_COHORT_CONFIG']=str(Path(directory)/'cohort_config.json')
        return env
    batch.OUT=OUT;batch.environment=environment;batch.status=status
    aggregate.OUT=OUT
    batch.CODE=list(dict.fromkeys(batch.CODE+[
        'pretest/qwen140_multiface_remaining_episode_batch.py','pretest/qwen140_weak4_aggregate.py',
        'pretest/qwen140_cs_face_crop_review.py','pretest/qwen140_single_pass_shot.py',
        'pretest/prepare_entitybench_character_assets.py','pretest/qwen140_pilot_qwen.py',
        'pretest/qwen140_reaggregate_partial_reruns.py']))


def index(plan):
    rows=[]
    for item in plan['episodes']:
        d=Path(item['output_root']);rel=d.relative_to(OUT)
        n=len(list(d.glob('episodes/*/shot_*/result.json')))
        label='生成＋官方非VLM完成，视频待审' if (d/'complete.json').exists() else '待执行／运行中'
        if not (d/'complete.json').exists() and (d/'technical_exception.json').exists():label='技术异常留档，见日志'
        rows.append(f'<tr><td>{html.escape(item["episode_id"])}</td><td>{n}/{len(item["episode"]["shots"])}</td><td>{label}</td><td><a href="{rel}/review/index.html">前后视频</a> · <a href="{rel}/report.md">指标</a> · <a href="{rel}/review/cs_face_crops/index.html">实际人物裁剪</a></td></tr>')
    (OUT/'index.html').write_text('<meta charset="utf-8"><h1>剩余126集／2316shot：多人3D集成重跑</h1><p>已完成14集保留；当前126集从首shot统一使用冻结方案。单卡串行生成与官方非VLM评测，修复Agent关闭。</p><p>预计约10天（9–12天），需要持续GPU/API与足够存储。剩余任务预估新增105GB；空间不足时在下一集开始前等待，不删除历史产物。</p><p><a href="status.json">实时阶段</a> · <a href="eta.json">耗时及存储估计</a> · <a href="report.md">本批已完成子集指标</a> · <a href="full140_snapshot/index.html">最新140集汇总</a> · <a href="cohort.json">完整队列</a></p><table border="1" cellpadding="6"><tr><th>完整episode_id</th><th>shot进度</th><th>状态</th><th>结果</th></tr>'+''.join(rows)+'</table>')


def prepare():
    if (OUT/'cohort.json').exists():return batch.read(OUT/'cohort.json')
    manifest=batch.read(batch.BASE/'manifest.json');previous=batch.read(COMPLETED)
    assert previous['original_manifest_sha256']==batch.sha(batch.BASE/'manifest.json')
    episodes,excluded=select_episodes(manifest,previous)
    items=[];total_roles=0
    for ep in episodes:
        assert batch.sha(ep['episode_path'])==ep['episode_sha256']
        hits=list((batch.BASE/'evaluation/official_parallel_v2/.shards').glob(f'worker_*/results/{ep["episode_id"]}.json'))
        assert len(hits)==1
        original=batch.read(hits[0])
        scores={k:(original['vbench_means'] if k in original['vbench_means'] else original['metrics'])[k]['value'] for k in batch.METRICS}
        d=OUT/'experiments'/ep['episode_id']
        item={'episode':ep,'episode_id':ep['episode_id'],'scores':scores,'baseline_result':str(hits[0]),
            'baseline_result_sha256':batch.sha(hits[0]),'output_root':str(d)}
        descriptions={v['name']:v['description'] for v in ep['entity_descriptions'].values()}
        assert all(name in descriptions and descriptions[name] for name in roles(item))
        total_roles+=len(roles(item));items.append(item)
        batch.write(d/'cohort_config.json',{'run_tags':{ep['run']:tag(ep)}},frozen=True)
        batch.write(d/'style_specs.json',{'scope':'One consistent photographic medium per episode, same pipeline as completed cohorts; no new per-episode visual-style validation claimed',
            'episodes':{f'{ep["run"]}__{tag(ep)}':{'medium':'photorealistic live-action film',
                'evidence':'Fixed current-pipeline style constraint applied to character assets, keyframes and videos'}}},frozen=True)
        item['cohort_config_sha256']=batch.sha(d/'cohort_config.json')
        item['style_specs_sha256']=batch.sha(d/'style_specs.json')
    assert total_roles==909
    frozen=[]
    for name in batch.CODE:
        p=ROOT/name;digest=batch.sha(p);frozen.append({'path':str(p),'sha256':digest})
        target=OUT/'runtime/frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    cfg=api_config();agent_model=cfg.get('SHOT_AGENT_MODEL',cfg['PORTRAIT_QWEN_MODEL'])
    plan={'created_utc':batch.now(),'num_episodes':126,'num_shots':2316,'total_role_identities':909,
        'episodes':items,'excluded_completed_episode_ids':sorted(excluded),
        'excluded_snapshot':str(COMPLETED),'excluded_snapshot_sha256':batch.sha(COMPLETED),
        'baseline_manifest_sha256':batch.sha(batch.BASE/'manifest.json'),
        'baseline_generation_sha256':batch.sha(batch.BASE/'generation_report.json'),
        'v7_template_sha256':batch.sha(batch.TEMPLATE),'frozen_code':frozen,
        'experiment_revision':'remaining126_current_multiface_pipeline_20261030',
        'unified_version_from_first_shot':True,'agent_model':agent_model,'style':'photorealistic live-action film',
        'selection_policy':'All remaining exact episode IDs, original manifest order; no score-based filter; keep all fourteen completed editions',
        'window_binding':'Each episode has its own one-run/one-window config; repeated run names never collapse windows',
        'reference_policy':'Fresh episode-local clean role-only portrait and matching Gaussian; same current policy as completed cohorts',
        'failure_policy':'Normal per-role/all-role 3D gate skips reuse Control. Technical episode errors are recorded; proceed to other episodes. Stop for a global API auth/quota error. No repair Agent, source patch, threshold relaxation or model substitution.',
        'disk_policy':{'reserve_bytes':RESERVE_BYTES,'episode_estimate_safety_factor':1.25,'poll_seconds':60,
            'action':'Wait before admitting next episode until sufficient space; retain all original/history outputs; no automatic cleanup'},
        'evaluation_scope':'Complete episode official strict/skip_vlm; six VBench plus entity/DINO/boundary, preserve legitimate non-applicable null metrics, zero VLM calls'}
    batch.write(OUT/'cohort.json',plan,frozen=True)
    now=datetime.now(timezone.utc)
    hours=(2316*(GEN_SECONDS_PER_SHOT+EVAL_SECONDS_PER_SHOT)+909*ASSET_SECONDS_PER_ROLE)/3600
    forecast_bytes=909*ASSET_BYTES_PER_ROLE+2316*OTHER_BYTES_PER_SHOT
    batch.write(OUT/'eta.json',{'created_utc':now.isoformat(),'measured_generation_seconds_per_shot':GEN_SECONDS_PER_SHOT,
        'measured_asset_seconds_per_role':ASSET_SECONDS_PER_ROLE,'measured_evaluation_seconds_per_shot':EVAL_SECONDS_PER_SHOT,
        'basis':'Prior completed focus4/extra5 stage timings:111 fresh generation shots,55 prepared episode-local identities,117 evaluated shots',
        'central_hours':hours,'range_days':[9,12],
        'estimated_finish_beijing':[(now+timedelta(days=v)).astimezone(timezone(timedelta(hours=8))).isoformat() for v in [9,12]],
        'estimated_new_output_bytes':int(forecast_bytes),'new_output_gib':forecast_bytes/GIB,
        'free_gib_at_prepare':shutil.disk_usage(OUT).free/GIB,'asset_mib_per_role':ASSET_BYTES_PER_ROLE/1024**2,
        'other_output_mib_per_shot':OTHER_BYTES_PER_SHOT/1024**2,
        'condition':'Assumes continuous single RTX4090 48GB availability, API access and storage expansion; disk waits and technical exceptions extend time'})
    index(plan);status('prepared',completed_episodes=0,completed_shots=0)
    return plan


def disk_required(item):
    d=Path(item['output_root'])
    done=len(list(d.glob('episodes/*/shot_*/result.json')))
    remaining=max(0,len(item['episode']['shots'])-done)
    assets=0 if (d/'face_injection_policy.json').exists() else len(roles(item))*ASSET_BYTES_PER_ROLE
    return RESERVE_BYTES+int(1.25*(assets+remaining*OTHER_BYTES_PER_SHOT))


def wait_for_space(item):
    needed=disk_required(item)
    while shutil.disk_usage(OUT).free < needed:
        free=shutil.disk_usage(OUT).free
        status('waiting_for_disk_space',item['episode_id'],free_gib=free/GIB,
            needed_gib=needed/GIB,stage='before_next_episode',reason='Keep history and wait for free space; no automatic deletion')
        time.sleep(60)


def summarize(item):
    d=Path(item['output_root']);eid=item['episode_id']
    comparison=batch.read(d/'evaluation/weak4_v1/comparison.json')['rows'][0]['metrics']
    raw=batch.read(d/'evaluation/weak4_v1/results'/f'{eid}.json')
    manifest=batch.read(d/'evaluation/weak4_v1/results/run_manifest.json')
    assert manifest['strict'] and manifest['skip_vlm'] and manifest['vlm_call_stats']['total_calls']==0
    coverage={};lines=[f'# {eid}：原始与多人3D集成','',
        '完整episode，官方strict/skip_vlm，VLM调用0；cs_face为缺VLM保真度门控的人物区域DINO。','',
        '| 指标↑ | 原始 | 当前 | 差值 |','|---|---:|---:|---:|']
    for group in ['vbench_means','metrics']:
        for name,values in comparison[group].items():
            if name not in batch.LABELS:continue
            stat=raw[group][name];assert stat['n_failed']==0,(name,stat)
            assert stat['value'] is not None or stat['n_evaluated']==0,(name,stat)
            coverage[name]=stat;a,b=values['baseline'],values['candidate']
            formatted=lambda v:f'{v:.4f}' if v is not None else '未评分／不适用'
            delta=f'{b-a:+.4f}' if a is not None and b is not None else '—'
            lines.append(f'| {batch.LABELS[name]} | {formatted(a)} | {formatted(b)} | {delta} |')
    assert len(coverage)==12
    episode=batch.read(d/'episodes'/eid/'episode_result.json');assert len(episode['shots'])==len(item['episode']['shots'])
    injections=[]
    for index,r in enumerate(episode['shots']):
        assert batch.sha(r['video'])==r['video_sha256']
        assert batch.sha(r['last_frame'])==r['last_sha256']
        if index and not r['cut']:
            assert r['parent_video_sha256']==episode['shots'][index-1]['video_sha256']
            assert r['parent_tail_sha256']==episode['shots'][index-1]['last_sha256']
        if r.get('injection_result'):
            inj=batch.read(r['injection_result'])
            if inj['injection_applied']:assert any(v.get('injected') and v.get('residual_norm',0)>0 for v in inj['step_log'])
            else:assert batch.sha(inj['control'])==batch.sha(inj['treatment'])
            injections.append({'shot':r['shot_key'],'injected':inj['applied_characters'],
                'skipped':inj['skipped_characters'],'target_failures':inj['target_failures']})
    batch.write(d/'evaluation_validation.json',{'coverage':coverage,'vlm_calls':0,'actual_injections':injections,
        'visual_review':'Pending complete human video review; null non-applicable metrics preserved'},frozen=True)
    lines+=['','素材身份、路由、提示词及注入范围同时变化，非3D插件单独消融。完整视频人工审阅待完成。','',
        '[前后视频](review/index.html)']
    (d/'report.md').write_text('\n'.join(lines)+'\n')
    batch.write(d/'complete.json',{'episode_id':eid,'completed_utc':batch.now(),
        'generation_and_official_non_vlm':'complete','null_metrics':[k for k,v in coverage.items() if v['value'] is None],
        'video_visual_review':'pending'},frozen=True)
    try:crop_review(item)
    except Exception as exc:batch.write(d/'crop_review_exception.json',{'error':repr(exc),'official_scores_unchanged':True})


def update_aggregates():
    ok=True
    for label,function in [('cohort',aggregate.aggregate),('full140',None)]:
        try:
            if function:function()
            else:
                log=OUT/'logs/full140_snapshot.log';log.parent.mkdir(exist_ok=True)
                with log.open('a') as stream:
                    subprocess.run([batch.EVAL,'-u','pretest/qwen140_reaggregate_partial_reruns.py',
                        '--additional-cohort',str(OUT/'cohort.json'),'--out',str(OUT/'full140_snapshot')],
                        cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        except Exception as exc:
            ok=False;batch.write(OUT/(label+'_aggregation_exception.json'),{'recorded_utc':batch.now(),'error':repr(exc),
                'action':'Preserve individual complete results; no score/model changes or automatic source repair'})
    return ok


def run(plan):
    OUT.mkdir(exist_ok=True)
    with (OUT/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for item in plan['frozen_code']:assert batch.sha(item['path'])==item['sha256'],item['path']
        assert batch.sha(batch.BASE/'manifest.json')==plan['baseline_manifest_sha256']
        assert batch.sha(COMPLETED)==plan['excluded_snapshot_sha256']
        probe=subprocess.run([batch.QWEN,'-c','import torch; assert torch.cuda.is_available()'],capture_output=True,timeout=60)
        assert probe.returncode==0,'CUDA unavailable; retain prepared queue'
        errors=[];index(plan)
        for item in plan['episodes']:
            d=Path(item['output_root'])
            if (d/'complete.json').exists():continue
            assert batch.sha(d/'cohort_config.json')==item['cohort_config_sha256']
            assert batch.sha(d/'style_specs.json')==item['style_specs_sha256']
            assert batch.read(d/'cohort_config.json')['run_tags']=={item['episode']['run']:tag(item['episode'])}
            wait_for_space(item)
            cfg=api_config();assert cfg.get('SHOT_AGENT_MODEL',cfg['PORTRAIT_QWEN_MODEL'])==plan['agent_model']
            try:
                env=batch.assets(item,plan)
                if not (d/'rerun_manifest.json').exists():
                    for action in ['prepare','refs']:
                        batch.command(item,action,[batch.WAN,'-u','pretest/qwen140_episode_single_pass.py',action,'--run',item['episode']['run']],env)
                if not (d/'episodes'/item['episode_id']/'episode_result.json').exists():
                    batch.command(item,'generation',[batch.WAN,'-u','pretest/qwen140_episode_single_pass.py','run-remaining','--run',item['episode']['run']],env)
                for action in ['prepare','run','baseline-aggregate','compare']:
                    batch.command(item,'evaluation_'+action,[batch.EVAL,'-u','pretest/qwen140_integrated_episode_eval.py',
                        action,'--name','weak4_v1','--runs',item['episode']['run']],env)
                summarize(item)
                update_aggregates()
            except Exception as exc:
                generation_log=d/'logs/generation.log'
                tail=generation_log.read_text(errors='replace')[-6000:] if generation_log.exists() else ''
                entry={'recorded_utc':batch.now(),'episode_id':item['episode_id'],'error':repr(exc),
                    'action':'Preserve valid prefix and continue other episodes; repair Agent disabled'}
                errors.append(entry);batch.write(d/'technical_exception.json',entry);batch.write(OUT/'technical_exceptions.json',errors)
                if any(word in tail for word in ['insufficient_quota','invalid_api_key','Agent API HTTP 401']):
                    status('blocked_api_auth_or_quota',item['episode_id'],technical_exceptions=errors,
                        reason='External API unavailable; stop new admissions, keep all caches')
                    index(plan);return
            index(plan)
        ok=update_aggregates();done=[i for i in plan['episodes'] if (Path(i['output_root'])/'complete.json').exists()]
        status('complete_non_vlm_pending_video_visual_review' if len(done)==126 and ok else 'batch_finished_with_recorded_exceptions',
            completed_episodes=len(done),completed_shots=sum(len(i['episode']['shots']) for i in done),
            technical_exceptions=len(errors),aggregation_complete=ok)
        index(plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','aggregate']);args=parser.parse_args()
    configure();plan=prepare()
    if args.action=='run':
        try:run(plan)
        except Exception as exc:
            status('batch_setup_exception_recorded',error=repr(exc));raise
    elif args.action=='aggregate':update_aggregates()
    else:print(OUT/'cohort.json')
