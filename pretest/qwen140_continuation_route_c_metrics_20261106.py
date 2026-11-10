"""Original-vs-current metrics for frozen407 original-continuation route-C shots.

Reuse complete episode scores and reviewed canonical-region DINO. Never generate
media, resume a queue, modify a gate or estimate missing scores from episode means.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import qwen140_full140_latest_20261106 as snapshot

ROOT, BASE = snapshot.ROOT, snapshot.BASE
OUT = ROOT/'outputs/entitybench_continuation_route_c_407_20261106'
read, write, sha = snapshot.read, snapshot.write, snapshot.sha
LABELS = {'cs_face_visible_single_face_reviewed': '人物区域DINO（可见单脸＋定位纠错）',
          **snapshot.J.full.agg.LABELS}


def prepare():
    data = read(snapshot.OUT/'snapshot.json')
    targets = [r for r in read(ROOT/'outputs/entitybench_dynamic_degree_analysis_20261106/shots.json')
               if r['route']=='C' and not r['cut']]
    assert len(targets)==407 and len({(r['episode_id'],r['shot']) for r in targets})==407
    plan = dict(episodes=len({r['episode_id'] for r in targets}), target_shots=407,
                snapshot_sha256=sha(snapshot.OUT/'snapshot.json'), targets=targets,
                scope='Original full benchmark vs current57-complete-episode+141-shot snapshot; equal407 target-shot means, not full benchmark or route-only causal ablation.')
    snapshot.immutable(OUT/'plan.json',plan)
    return data,plan


def maps(data):
    old,old_reject=snapshot.diag.historical()
    _,_,current,rejected,_=snapshot.paired.load_maps()
    for directory in [snapshot.J.OUT/'visible_region', snapshot.REGION]:
        review=read(directory/'review/decisions.json')
        assert review['manifest_sha256']==sha(directory/'review/manifest.json')
        if directory==snapshot.REGION:
            for item in data['new_complete_episodes']:
                for p in (directory/'extraction'/item['episode_id']).glob('*.json'):
                    for r in read(p)['rows']:current[snapshot.diag.key(r)]=r
        for r in review['decisions']:
            if r['decision']=='exclude' and current[snapshot.diag.key(r)]['original_crop_sha256']==r['original_crop_sha256']:
                rejected[snapshot.diag.key(r)]=r
    rejected={k:r for k,r in rejected.items() if current[k]['original_crop_sha256']==r['original_crop_sha256']}
    return {snapshot.diag.key(r):r for r in old},old_reject,current,rejected


def cached():
    data,plan=prepare(); old,old_reject,current,rejected=maps(data)
    lookup={r['episode_id']:r for r in data['episodes']}
    mod=snapshot.diag.gate.load_official_pure(); pools={}; checks=[]; values=[]
    published=snapshot.diag.gate.OUT/'remaining_low_score_review/recomputed/results'
    for eid in sorted({r['episode_id'] for r in plan['targets']}):
        item=lookup[eid];ep=item['episode'];pools[eid]={}
        for edition,path,gate,exclude in [('before',item['original_result'],old,old_reject),
                                        ('after',item['result'],current,rejected)]:
            raw=read(path); result,entities=snapshot.diag.compute(ep,raw,gate,exclude,mod)
            reference=(read(published/(eid+'.json'))['metrics']['cs_face'] if edition=='before' else
                       next(r['current'] for r in read(snapshot.REGION/'report.json')['per_episode'] if r['episode_id']==eid))
            actual=result.metrics['cs_face'].to_json()
            assert actual['n_evaluated']==reference['n_evaluated']
            assert (actual['value'] is None and reference['value'] is None) or abs(actual['value']-reference['value'])<1e-10
            shot_dino={}
            for name,entity in entities.items():
                for app in entity['per_shot_sim_to_centroid']:
                    shot_dino.setdefault(app['shot'],[]).append(app['sim_to_centroid'])
            pools[eid][edition]=(raw,Path(path).parent/'.work',shot_dino)
            checks.append(dict(episode_id=eid,edition=edition,result=str(path),sha256=sha(path),reviewed_episode_reproduced=True))
    for target in plan['targets']:
        eid,sk=target['episode_id'],target['shot']; row=dict(episode_id=eid,run=target['run'],shot=sk)
        for edition in ['before','after']:
            raw,work,dino=pools[eid][edition]
            value=snapshot.J.full.per_shot(raw,work,eid,sk)
            value['cs_face_visible_single_face_reviewed']=math.fsum(dino[sk])/len(dino[sk]) if sk in dino else None
            value['cs_transition_boundary']=None
            row[edition]=value
        assert row['before']['dynamic_degree']==target['original']
        assert row['after']['dynamic_degree']==target['current']
        values.append(row)
    coverage={}
    keys={(r['episode_id'],r['shot']) for r in plan['targets']}
    for edition,gate,exclude in [('before',old,old_reject),('after',current,rejected)]:
        g=[(k,r) for k,r in gate.items() if (k[0],k[2]) in keys]
        coverage[edition]=dict(scheduled_role_shots=len(g),canonical_present=sum(r['original_status']=='present' for _,r in g),
            auto_eligible=sum(r['kept'] for _,r in g),post_review_retained=sum(r['kept'] and k not in exclude for k,r in g),
            scored_appearances=sum(len(pools[eid][edition][2].get(sk,[])) for eid,sk in keys))
    write(OUT/'cached_shots.json',values)
    write(OUT/'cache_validation.json',dict(checks=checks,coverage=coverage,context_episodes=plan['episodes'],new_neural_scoring_calls=0,vlm_calls=0))
    summarize()


def summarize():
    _,plan=prepare(); rows=read(OUT/'cached_shots.json');pairs=read(OUT/'boundary.json') if (OUT/'boundary.json').exists() else {}
    sample=read(OUT/'boundary_sample.json') if (OUT/'boundary_sample.json').exists() else None
    boundary_keys=set(sample['keys']) if sample else set()
    for row in rows:
        key=row['episode_id']+'/'+row['shot']
        if key in pairs and key in boundary_keys:
            for edition in ['before','after']:row[edition]['cs_transition_boundary']=pairs[key][edition]['score']
    table=[]
    for name,label in LABELS.items():
        v={side:[r[side][name] for r in rows if r[side].get(name) is not None] for side in ['before','after']}
        means={side:math.fsum(g)/len(g) if g else None for side,g in v.items()}
        common=[r for r in rows if all(r[side].get(name) is not None for side in ['before','after'])]
        matched={side:math.fsum(r[side][name] for r in common)/len(common) if common else None for side in ['before','after']}
        table.append(dict(metric=name,label=label,**means,delta=means['after']-means['before'] if all(x is not None for x in means.values()) else None,
                          before_n=len(v['before']),after_n=len(v['after']),paired_n=len(common),common_support=matched))
    complete=bool(boundary_keys) and boundary_keys<=set(pairs)
    report=dict(status='complete_407_cached_metrics_and_random16_boundary' if complete else 'cached_metrics_complete_boundary_sample_pending',
                target_shots=407,context_episodes=plan['episodes'],scope=plan['scope'],metrics=table,
                dino_coverage=read(OUT/'cache_validation.json')['coverage'],vlm_calls=0,boundary_pairs_completed=len(boundary_keys&set(pairs)),
                boundary_scope='Uniform deterministic random16 of407; probes excluded. Not a full407 boundary mean.',
                unavailable='VLM action/alignment/fidelity/LLM metrics remain null; no VLM calls; raw cs_face/object have no fidelity gate.')
    write(OUT/'comparison.json',report);write(OUT/'shots.json',rows)
    with (OUT/'shots.csv').open('w',newline='') as stream:
        cols=['episode_id','run','shot']+[f'{n}_{side}' for n in LABELS for side in ['before','after','delta']]
        writer=csv.DictWriter(stream,fieldnames=cols);writer.writeheader()
        for row in rows:
            d={k:row[k] for k in ['episode_id','run','shot']}
            for n in LABELS:
                a,b=row['before'].get(n),row['after'].get(n)
                d.update({n+'_before':a,n+'_after':b,n+'_delta':b-a if a is not None and b is not None else None})
            writer.writerow(d)
    text='# 原接续／当前C重建：407镜头前后指标\n\n'+plan['scope']+'\n\n人物/物体中心使用各自完整episode上下文；只汇总407目标镜头。前=原始完整benchmark，不是v2替换前集成版。条件指标缺失不填0；原输入对应的历史定位复核按像素SHA复用，本次不因选C而新增挑选/纠错。\n\n| 指标↑ | 原版 | 当前 | 差值 | 有效shot 原/新 |\n|---|---:|---:|---:|---:|\n'
    for r in table:
        def fmt(v):return '待补算' if v is None else f'{v:.4f}'
        label=r['label']+'（随机16镜头抽查）' if r['metric']=='cs_transition_boundary' else r['label']
        text+=f"| {label} | {fmt(r['before'])} | {fmt(r['after'])} | {fmt(r['delta'])} | {r['before_n']}/{r['after_n']} |\n"
    text+='\n边界按原官方函数相邻视频末帧/首帧DINO计算，仅目标shot的入边；按用户“稍微抽查”范围，随机种子20261106从407等概率无放回选16个，CPU原FP32算法补算，两个联调样本不进入抽查均值。该行不能当407镜头全量边界结果。其余12行覆盖全部407目标，缺失不填0。身份条件均值需结合有效数量与同一有效shot交集，JSON保存common_support。其他VLM依赖指标未评，调用0。\n'
    (OUT/'report.md').write_text(text)
    return report


def boundary(limit=None):
    import os
    os.environ.update(DINO_MODEL_PATH='/root/autodl-tmp/models/dinov2-base',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    from transformers import AutoImageProcessor,AutoModel
    # Memory-efficient loading changes allocation only; same FP32 weights/forward.
    model=AutoModel.from_pretrained(os.environ['DINO_MODEL_PATH'],low_cpu_mem_usage=True).to('cpu').eval()
    processor=AutoImageProcessor.from_pretrained(os.environ['DINO_MODEL_PATH'])
    mod=snapshot.J.full.scorer();mod._MODELS['dino']=(model,processor,'cpu')
    data,plan=prepare();lookup={r['episode_id']:r['episode'] for r in data['episodes']}
    pairs=read(OUT/'boundary.json') if (OUT/'boundary.json').exists() else {}
    selected=set(read(OUT/'boundary_sample.json')['keys'])
    started=time.monotonic();count=0
    for r in plan['targets']:
        eid,sk=r['episode_id'],r['shot'];key=eid+'/'+sk
        if key not in selected:continue
        if key in pairs:continue
        ep=lookup[eid];i=next(i for i,s in enumerate(ep['shots']) if s['shot_key']==sk)
        assert i and not ep['shots'][i]['cut'];prev=ep['shots'][i-1]['shot_key']
        value={}
        for side in ['before','after']:
            if side=='before':
                # Original generation report stores its canonical output video.
                a=BASE/'videos'/eid/('shot_'+prev.replace(':','_'))/'video.mp4'
                b=BASE/'videos'/eid/('shot_'+sk.replace(':','_'))/'video.mp4'
            else:a=Path(data['media'][eid+'/'+prev]['source']);b=Path(data['media'][eid+'/'+sk]['source'])
            score=mod.compute_transition_score(str(a),str(b),1,'cpu')
            value[side]=dict(score=score,previous=str(a),current=str(b),previous_sha256=sha(a),current_sha256=sha(b))
        pairs[key]=value;write(OUT/'boundary.json',pairs);count+=1
        write(OUT/'boundary_progress.json',dict(completed=len(selected&set(pairs)),expected=len(selected),probe_cache_excluded=len(set(pairs)-selected),elapsed_seconds=time.monotonic()-started,device='cpu',generation_resumed=False,utc=datetime.now(timezone.utc).isoformat()))
        print('boundary sample',len(selected&set(pairs)),'/',len(selected),round(time.monotonic()-started,1),'seconds',flush=True)
        if limit and count>=limit:break
    summarize()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cached','boundary','summary']);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.action=='cached':cached()
    elif a.action=='boundary':boundary(a.limit)
    else:summarize()
