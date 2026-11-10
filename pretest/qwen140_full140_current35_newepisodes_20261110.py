"""Input-bound current35 + newly completed whole episodes; preserve all old media.

Official per-episode scoring is reused with SHA/provenance validation; only the
new canonical visibility gates need GPU inference. Localization review is explicit.
"""
from __future__ import annotations
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import qwen140_original40_priority_context_20261109 as history
import qwen140_full140_latest_20261106 as frozen
import qwen140_rerun_region_diagnostic as diag
import qwen140_badshot_reviewed_dino_20261104 as review

ROOT, BASE = frozen.ROOT, frozen.BASE
PRIOR = ROOT/'outputs/entitybench_original40_priority_remaining_20261109/paused8_benchmark_20261109'
OUT = ROOT/'outputs/entitybench_full140_current35_newepisodes_20261110'
REGION = OUT/'visible_region'
read, write, sha, immutable = diag.read, diag.write, diag.sha, diag.immutable


def status(stage, **kw):
    write(OUT/'status.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),vlm_calls=0,**kw))


def prepare():
    if (OUT/'snapshot.json').exists():
        p=read(OUT/'snapshot.json')
        for r in p['bindings']:assert sha(r['path'])==r['sha256'],r['path']
        return p
    original=read(history.batch.OUT/'baseline_snapshot.json')
    additions={e['episode_id']:e for e in read(OUT/'completed_episodes_at_request.json')['episodes']}
    oldmedia=read(PRIOR/'replacement_manifest.json')['videos']
    assert read(PRIOR/'status.json')['stage']=='complete_all_available_metrics_with_reviewed_dino'
    bindings={};episodes=[];newitems=[];media={};superseded=[]
    def bind(path):
        path=str(Path(path).resolve());bindings[path]=sha(path)
    common=read(BASE/'evaluation/official_parallel_v2/run_manifest.json')
    for item in original['episodes']:
        ep,eid=item['episode'],item['episode_id']
        previous=PRIOR/'affected/results'/(eid+'.json')
        if not previous.exists():previous=Path(item['result'])
        before=read(previous);bind(previous)
        if eid in additions:
            source=additions[eid];directory=Path(source['output_root'])
            result=directory/'evaluation/weak4_v1/results'/(eid+'.json')
            manifest=read(result.parent/'run_manifest.json')
            for field in ['config','python_version','libraries','model_checkpoints','pillars','evaluation_scope']:
                assert manifest[field]==common[field],(eid,field)
            assert manifest['strict'] and manifest['skip_vlm'] and not manifest['allow_config_override']
            assert manifest['vlm_call_stats']['total_calls']==0
            generation=read(directory/'episodes'/eid/'episode_result.json')
            assert len(generation['shots'])==len(ep['shots'])
            assert (directory/'complete.json').exists()
            for i,(shot,g) in enumerate(zip(ep['shots'],generation['shots'])):
                assert shot['shot_key']==g['shot_key'] and shot['cut']==g['cut']
                assert sha(g['video'])==g['video_sha256'] and sha(g['last_frame'])==g['last_sha256']
                if i and not shot['cut']:
                    assert g['parent_video_sha256']==generation['shots'][i-1]['video_sha256']
                    assert g['parent_tail_sha256']==generation['shots'][i-1]['last_sha256']
                key=eid+'/'+g['shot_key'];previous_media=oldmedia[key]
                if previous_media.get('cumulative_replaced',previous_media.get('replaced',False)):superseded.append(key)
                media[key]=dict(source=g['video'],sha256=g['video_sha256'],replacement_kind='new_complete_episode')
            for file in [result,result.parent/'run_manifest.json',directory/'complete.json',directory/'episodes'/eid/'episode_result.json']:
                bind(file)
        else:
            result=previous;directory=Path(item['output_root'])
            for shot in ep['shots']:
                key=eid+'/'+shot['shot_key'];r=oldmedia[key]
                assert sha(r['source'])==r['sha256']
                media[key]=dict(source=r['source'],sha256=r['sha256'],replacement_kind='retained_current35')
        raw=read(result)
        assert raw['num_shots']==before['num_shots']==len(ep['shots'])
        for group in ['vbench_means','metrics']:
            for name,v in raw[group].items():
                if name in frozen.J.full.agg.LABELS:assert v['n_failed']==0,(eid,name)
        row=dict(episode_id=eid,episode=ep,result=str(result.resolve()),result_sha256=sha(result),
            before_result=str(previous.resolve()),before_result_sha256=sha(previous),
            output_root=str(directory),artifacts=str(result.resolve().parent/'.work/eval_artifacts'/eid),
            edition='new_complete_episode' if eid in additions else 'retained_current35')
        episodes.append(row)
        if eid in additions:newitems.append(row)
    assert len(episodes)==140 and len(media)==2491
    assert len(newitems)==7 and sum(len(e['episode']['shots']) for e in newitems)==77
    for key,r in media.items():
        eid,sk=key.split('/');a,b=map(int,sk.split(':'))
        link=OUT/'candidate_media'/eid/f'{a:03d}_{b:03d}.mp4';link.parent.mkdir(parents=True,exist_ok=True)
        assert not link.exists();link.symlink_to(Path(r['source']).resolve());r['canonical']=str(link)
    for file in [Path(__file__),OUT/'completed_episodes_at_request.json',PRIOR/'replacement_manifest.json',
        PRIOR/'official_aggregates.json',PRIOR/'visible_region/report.json',PRIOR/'visible_region/review/decisions.json',
        history.batch.OUT/'baseline_snapshot.json',BASE/'runtime/evaluate_benchmark_official.py',
        Path(diag.__file__),Path(review.__file__)]:bind(file)
    snapshot=dict(created_utc=datetime.now(timezone.utc).isoformat(),episodes=episodes,media=media,
        new_complete_episodes=newitems,replaced_new_episodes=len(newitems),replaced_new_shots=77,
        retained_shots=2414,previous_dynamic_selected=35,dynamic_overlay_superseded_by_new_whole_episodes=superseded,
        bindings=[dict(path=k,sha256=v) for k,v in bindings.items()],
        scope='140 episode-macro; replace seven complete new episodes over current35, keep other133 episodes byte-identical. Whole-episode precedence over an older independent overlay; no incomplete prefixes, no score selection.')
    immutable(OUT/'snapshot.json',snapshot)
    immutable(OUT/'replacement_manifest.json',dict(videos=media,new_whole_episodes=list(additions),
        superseded_previous_dynamic_shots=superseded,baseline=str(PRIOR),scope=snapshot['scope']))
    immutable(REGION/'input_manifest.json',dict(episodes=newitems,selected_episode_count=len(newitems),
        selected_shot_count=77,scope='New complete episodes only; unchanged contexts reuse exact current35 canonical gates.'))
    shutil.copyfile(PRIOR/'visible_region/required_span_validation.json',REGION/'required_span_validation.json')
    status('prepared_complete7_shots77')
    return snapshot


def official():
    p=prepare();mod=diag.gate.load_official_pure();groups={'before':[],'after':[]}
    details=[]
    for item in p['episodes']:
        ep,eid=item['episode'],item['episode_id']
        old,new=read(item['before_result']),read(item['result'])
        for label,raw in [('before',old),('after',new)]:groups[label].append(mod._episode_result_from_json(eid,ep['difficulty'],raw))
        details.append(dict(episode_id=eid,before=old['metrics'],after=new['metrics'],edition=item['edition']))
    aggregates={k:mod.aggregate_results(rows,read(BASE/'split_full140.json')) for k,rows in groups.items()}
    assert aggregates['before']==read(PRIOR/'official_aggregates.json')['after']
    table=[dict(metric=k,label=label,before=frozen.J.full.agg.agg_metric(aggregates['before'],k)['mean'],
        after=frozen.J.full.agg.agg_metric(aggregates['after'],k)['mean']) for k,label in frozen.J.full.agg.LABELS.items()]
    write(OUT/'official_aggregates.json',aggregates)
    write(OUT/'official_comparison.json',dict(metrics=table,episodes=140,shots=2491,new_episodes=7,new_shots=77,
        default_reviewed_dino='pending_localization_review',vlm_calls=0))
    exact={k:sum(r.vbench_means['dynamic_degree'].value for r in rows)/140 for k,rows in groups.items()}
    write(OUT/'dynamic_exact.json',exact)
    text='# 当前35条替换版＋7个新整集：完整140集官方指标\n\n'+p['scope']+'\n\n| 指标 | 35条版 | 加入7集后 | 变化 |\n|---|---:|---:|---:|\n'
    for r in table:text+=f"| {r['label']} | {r['before']:.4f} | {r['after']:.4f} | {r['after']-r['before']:+.4f} |\n"
    text+='\nstrict/skip_vlm，VLM调用0；cs_face/cs_object缺fidelity gate，partial_without_vlm。默认人物DINO待新裁剪定位复核。\n'
    (OUT/'official_report.md').write_text(text)
    status('official12_complete_default_dino_pending',new_complete_episodes=7,new_shots=77)


def baseline_maps():
    rows,rejected,_=history.maps()
    for file in (PRIOR/'visible_region/extraction').glob('*/*.json'):
        for row in read(file)['rows']:rows[diag.key(row)]=row
    reference=read(PRIOR/'visible_region/report.json')
    rejected={diag.key(r):r for r in reference['carried_exclusions']+reference['new_exclusions']}
    assert len(rows)==4991
    for key,r in rejected.items():assert rows[key]['original_crop_sha256']==r['original_crop_sha256']
    return rows,rejected,reference


def dino(final=False):
    p=prepare();old,rejected,reference=baseline_maps();new=copy.deepcopy(old)
    for item in p['new_complete_episodes']:
        ep,eid=item['episode'],item['episode_id']
        for shot in ep['shots']:
            sk=shot['shot_key'];data=read(REGION/'extraction'/eid/(sk.replace(':','_')+'.json'))
            artifact=Path(item['artifacts'])/'intra_shot'/sk.replace(':','_')/'_eval.json'
            assert data['evaluation_sha256']==sha(artifact)
            assert data['video_sha256']==p['media'][eid+'/'+sk]['sha256']
            assert data['protocol_sha256']==sha(REGION/'protocol_v3_cuda.json')
            assert {r['character'] for r in data['rows']}==set(shot['entity_schedule'].get('characters',[]))
            for row in data['rows']:
                if row['original_crop']:assert sha(row['original_crop'])==row['original_crop_sha256']
                row['edition']='rerun' if row['original_crop_sha256']!=old[diag.key(row)]['original_crop_sha256'] else 'unchanged'
                new[diag.key(row)]=row
    carry={k:r for k,r in rejected.items() if new[k]['original_crop_sha256']==r['original_crop_sha256']}
    decisions=[]
    if final:
        manifest=read(REGION/'review/manifest.json');saved=read(REGION/'review/decisions.json')
        assert saved['manifest_sha256']==sha(REGION/'review/manifest.json')
        expected={r['review_id']:r for r in manifest['rows']}
        assert len(saved['decisions'])==len(expected) and {r['review_id'] for r in saved['decisions']}==set(expected)
        for r in saved['decisions']:
            ref=expected[r['review_id']];assert diag.key(ref)==diag.key(r)
            assert ref['original_crop_sha256']==r['original_crop_sha256']==new[diag.key(r)]['original_crop_sha256']
            assert r['decision'] in ['keep','exclude'] and r['reason'] and r['evidence']
            if r['decision']=='exclude':carry[diag.key(r)]=r
        decisions=saved['decisions']
    mod=diag.gate.load_official_pure();groups={'before':[],'after':[]};details=[];scored=[]
    check={r['episode_id']:r['after'] for r in reference['per_episode']}
    changed={e['episode_id'] for e in p['new_complete_episodes']}
    for item in p['episodes']:
        ep,eid=item['episode'],item['episode_id']
        before,_=diag.compute(ep,read(item['before_result']),old,rejected,mod)
        expected=check[eid];actual=before.metrics['cs_face'].to_json()
        assert actual['n_evaluated']==expected['n_evaluated']
        assert (actual['value'] is None and expected['value'] is None) or abs(actual['value']-expected['value'])<1e-10
        after,entities=diag.compute(ep,read(item['result']),new,carry,mod)
        groups['before'].append(before);groups['after'].append(after)
        details.append(dict(episode_id=eid,before=actual,after=after.metrics['cs_face'].to_json()))
        if eid in changed:
            for name,data in entities.items():
                for app in data['per_shot_sim_to_centroid']:scored.append({**new[eid,name,app['shot']],'similarity':app['sim_to_centroid']})
    aggregates={k:mod.aggregate_results(v,read(BASE/'split_full140.json')) for k,v in groups.items()}
    for tier in ['all','easy','medium','hard']:assert aggregates['before'][tier]['cs_face']==reference['after'][tier]['cs_face']
    coverage={}
    for label,rows,rej in [('before',old,rejected),('after',new,carry)]:
        coverage[label]=dict(scheduled_role_shots=len(rows),canonical_present=sum(r['original_status']=='present' for r in rows.values()),
            auto_eligible=sum(r['kept'] for r in rows.values()),retained=sum(r['kept'] and k not in rej for k,r in rows.items()),
            scored_appearances=sum(v.metrics['cs_face'].n_evaluated for v in groups[label]),
            valid_episodes=sum(v.metrics['cs_face'].value is not None for v in groups[label]),
            gate_reasons=dict(Counter(r['reason'] for r in rows.values() if not r['kept'])))
    report=dict(metric='cs_face_visible_single_face_reviewed' if final else 'cs_face_visible_single_face_auto',
        status='complete_with_localization_review' if final else 'automatic_pending_localization_review',
        **aggregates,per_episode=details,coverage=coverage,new_review_items=len(decisions),
        carried_exclusions=[r for k,r in rejected.items() if k in carry and carry[k]==r],
        new_exclusions=[r for r in decisions if r['decision']=='exclude'],vlm_calls=0)
    write(REGION/('report.json' if final else 'automatic_results.json'),report)
    if not final:
        write(REGION/'scored_crops.json',scored);review.OUT=REGION
        n=review.review_sheets(scored,p['new_complete_episodes']);status('official_complete_default_dino_review_pending',review_items=n)
        return
    table=[dict(metric=report['metric'],label='人物DINO（可见单脸＋定位复核）',
        before=aggregates['before']['all']['cs_face']['mean'],after=aggregates['after']['all']['cs_face']['mean'])]+read(OUT/'official_comparison.json')['metrics']
    write(OUT/'complete_comparison.json',dict(metrics=table,coverage=coverage,episodes=140,shots=2491,new_episodes=7,new_shots=77,vlm_calls=0))
    text='# 当前35条替换版＋7个新整集：完整140集指标\n\n仅覆盖完整7集77shot，其余133集2414shot维持当前35版；无分数筛选，原文件保留。\n\n| 指标↑ | 当前35条版 | 加入7集后 | 变化 |\n|---|---:|---:|---:|\n'
    for r in table:text+=f"| {r['label']} | {r['before']:.4f} | {r['after']:.4f} | {r['after']-r['before']:+.4f} |\n"
    text+='\n官方12项复用已验证的完整episode评分，官方函数重新汇总140集；默认DINO对新裁剪应用同可见单脸门控、定位复核后重新计算质心。strict/skip_vlm，VLM0，raw cs_face/cs_object无fidelity gate，partial_without_vlm。复核为助手视觉复核，非外部人工签核。\n'
    (OUT/'complete_report.md').write_text(text)
    write(OUT/'completion_validation.json',dict(episodes=140,shots=2491,new_episodes=7,new_shots=77,unchanged_shots=2414,
        full140_before_official_reproduced=True,full140_before_reviewed_dino_reproduced=True,
        report_sha256=sha(OUT/'complete_report.md'),manifest_sha256=sha(OUT/'replacement_manifest.json'),
        review_manifest_sha256=sha(REGION/'review/manifest.json'),decisions_sha256=sha(REGION/'review/decisions.json'),
        coverage=coverage,vlm_calls=0,original_media_overwritten=False))
    status('complete_all_available_metrics_with_reviewed_dino',new_episodes=7,new_shots=77)


def resume_generation():
    file=OUT/'generation_schedule_checkpoint.json'
    if not file.exists():return
    receipt=read(file);row=receipt['supervisor'];stat=Path('/proc')/str(row['pid'])/'stat'
    if stat.exists():
        fields=stat.read_text().split()
        if fields[21]==row['start_ticks']:
            os.kill(row['pid'],signal.SIGCONT)
            receipt.update(state='supervisor_resumed',resumed_utc=datetime.now(timezone.utc).isoformat())
            write(file,receipt)


def extract():
    p=prepare()
    try:
        # Supervisor is paused; let its existing episode child finish all writes.
        while True:
            workers=[]
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():continue
                try:
                    args=(proc/'cmdline').read_bytes().split(b'\0')
                    if args and b'python' in args[0] and any(a.endswith(b'qwen140_never_rerun_resume_20261109.py') for a in args) and b'episode' in args:workers.append(proc.name)
                except OSError:pass
            if not workers:break
            status('waiting_current_episode_generation_checkpoint',worker_pids=workers)
            time.sleep(10)
        diag.OUT=REGION;diag.extract('cuda')
    finally:resume_generation()
    dino()


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='official':official()
    elif action=='extract':extract()
    elif action=='automatic':dino()
    elif action=='final':dino(True)
    elif action=='resume-generation':resume_generation()
    else:raise ValueError(action)
