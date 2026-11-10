"""Full140 snapshot from scored complete episodes and accepted141 shot overlay.

Reuse official scores; new complete episodes need CPU visibility detection and
score-hidden localization review before publishing the default DINO row.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

import qwen140_v2_shots_only_report_20261105 as paired
import qwen140_rerun_region_diagnostic as diag

J = paired.job
ROOT, BASE = J.ROOT, J.BASE
OUT = ROOT / 'outputs/entitybench_full140_latest57_v2_141_20261106'
REGION = OUT / 'new_complete_visible_region'
read, write, sha, immutable = J.read, J.write, J.sha, J.immutable


def status(stage, **kw):
    write(OUT/'status.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),vlm_calls=0,**kw))


def prepare():
    if (OUT/'snapshot.json').exists():
        return read(OUT/'snapshot.json')
    original=read(BASE/'manifest.json');before=read(J.OUT/'baseline_selection.json')
    selected={e['episode_id']:e for e in before['episodes']}
    target_contexts={e['episode_id'] for e in read(J.OUT/'plan.json')['episodes']}
    queue=read(ROOT/'outputs/entitybench_v2_remaining_episodes_20261105/plan.json')
    lookup={e['episode_id']:e for e in read(ROOT/'outputs/entitybench_qwen140_multiface_remaining126_20261030/cohort.json')['episodes']}
    new={e['episode_id']:e for e in queue['episodes'] if (Path(e['output_root'])/'complete.json').exists()}
    assert len(new)==6 and not (set(new)&target_contexts)
    media=copy.deepcopy(before['videos']);media.update(read(J.OUT/'replacement_manifest.json')['videos'])
    rows=[];newitems=[]
    for ep in original['episodes']:
        eid=ep['episode_id']
        if eid in new:
            root=Path(new[eid]['output_root']);result=root/'evaluation/weak4_v1/results'/(eid+'.json');edition='new_complete_v2_episode'
            generation=read(root/'episodes'/eid/'episode_result.json')
            assert len(generation['shots'])==len(ep['shots'])
            for s in generation['shots']:
                media[eid+'/'+s['shot_key']]=dict(source=s['video'],sha256=s['video_sha256'])
        elif eid in target_contexts:
            result=J.OUT/'affected/results'/(eid+'.json');root=J.OUT/'episode_media';edition='previous_complete_with_v2_shot_overlay'
        else:
            result=Path(selected[eid]['selected_result']);root=Path(selected[eid]['source_root']);edition=selected[eid]['edition']
        row=dict(episode_id=eid,episode=ep,output_root=str(root),result=str(result),result_sha256=sha(result),
            original_result=selected[eid]['baseline_result'],original_result_sha256=sha(selected[eid]['baseline_result']),
            edition=edition,artifacts=str(result.parent/'.work/eval_artifacts'/eid))
        rows.append(row)
        if eid in new:newitems.append(row)
    assert len(rows)==140 and len(media)==2491
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),episodes=rows,new_complete_episodes=newitems,
        replaced_complete_episodes=57,v2_independent_shots=141,media=media,shots=2491,
        scope='140 episode-macro snapshot:57 complete reruns,83 original episodes;141 accepted v2 shot overlay on prior complete reruns. No incomplete episode prefixes.',
        target_review_sha256=sha(J.OUT/'visible_region/review/decisions.json'))
    immutable(OUT/'snapshot.json',p)
    return p


def official(p):
    mod=diag.gate.load_official_pure();split=read(BASE/'split_full140.json')
    common=read(BASE/'evaluation/official_parallel_v2/run_manifest.json');manifests={};groups={'original':[],'current':[]}
    for v in p['media'].values():assert sha(v['source'])==v['sha256']
    for row in p['episodes']:
        ep=row['episode'];eid=ep['episode_id'];source=Path(row['result']);assert sha(source)==row['result_sha256']
        # Resolve symlinks to the actual worker's provenance, not a synthetic merged directory.
        provenance=source.resolve().parent/'run_manifest.json'
        if str(provenance) not in manifests:
            m=read(provenance)
            assert m['strict'] and m['skip_vlm'] and not m['allow_config_override'] and m['vlm_call_stats']['total_calls']==0
            for field in ['config','python_version','libraries','model_checkpoints','pillars','evaluation_scope']:assert m[field]==common[field],(source,field)
            manifests[str(provenance)]=sha(provenance)
        raw=read(source);assert raw['num_shots']==len(ep['shots'])
        for metric in J.full.agg.LABELS:assert J.full.agg.metric(raw,metric)['n_failed']==0
        groups['current'].append(mod._episode_result_from_json(eid,ep['difficulty'],raw))
        groups['original'].append(mod._episode_result_from_json(eid,ep['difficulty'],read(row['original_result'])))
    results={k:mod.aggregate_results(v,split) for k,v in groups.items()}
    write(OUT/'official_aggregates.json',results)
    write(OUT/'official_validation.json',dict(episodes=140,shots=2491,scoring_manifests=manifests,
        media_shas_verified=2491,new_neural_scoring_calls=0,vlm_calls=0))
    table=[dict(metric=k,label=label,original=J.full.agg.agg_metric(results['original'],k)['mean'],
        current=J.full.agg.agg_metric(results['current'],k)['mean']) for k,label in J.full.agg.LABELS.items()]
    write(OUT/'official_comparison.json',dict(metrics=table,snapshot=p['scope'],default_dino='pending_new_complete_episode_localization_review'))
    status('official12_complete_default_dino_pending',episodes=140,new_complete_episodes=6)


def extract(p):
    diag.immutable(REGION/'input_manifest.json',dict(episodes=p['new_complete_episodes'],
        selected_episode_count=len(p['new_complete_episodes']),selected_shot_count=sum(len(e['episode']['shots']) for e in p['new_complete_episodes']),
        scope='Only newly complete episode canonical face gates; reuse exact prior reviewed141-context gates elsewhere.'))
    shutil.copyfile(J.PRIOR/'visible_region/required_span_validation.json',REGION/'required_span_validation.json')
    diag.OUT=REGION;diag.extract('cpu')


def dino(p,final=False):
    _,_,rows,rejected,_=paired.load_maps()
    target_review=read(J.OUT/'visible_region/review/decisions.json')
    assert target_review['manifest_sha256']==sha(J.OUT/'visible_region/review/manifest.json')
    for r in target_review['decisions']:
        if r['decision']=='exclude' and rows[diag.key(r)]['original_crop_sha256']==r['original_crop_sha256']:rejected[diag.key(r)]=r
    for e in p['new_complete_episodes']:
        files=list((REGION/'extraction'/e['episode_id']).glob('*.json'));assert len(files)==len(e['episode']['shots'])
        for f in files:
            for r in read(f)['rows']:rows[diag.key(r)]=r
    rejected={k:r for k,r in rejected.items() if rows[k]['original_crop_sha256']==r['original_crop_sha256']}
    decisions=[]
    if final:
        manifest=read(REGION/'review/manifest.json');review=read(REGION/'review/decisions.json')
        assert review['manifest_sha256']==sha(REGION/'review/manifest.json')
        expected={r['review_id']:r for r in manifest['rows']}
        assert len(review['decisions'])==len(expected) and {r['review_id'] for r in review['decisions']}==set(expected)
        for r in review['decisions']:
            old=expected[r['review_id']];assert diag.key(old)==diag.key(r) and old['original_crop_sha256']==r['original_crop_sha256']
            assert r['decision'] in ['keep','exclude'] and r['reason'] and r['evidence']
            if r['decision']=='exclude':rejected[diag.key(r)]=r
        decisions=review['decisions']
    mod=diag.gate.load_official_pure();original=[];current=[];details=[];scored=[]
    published=diag.gate.OUT/'remaining_low_score_review/recomputed/results'
    newids={e['episode_id'] for e in p['new_complete_episodes']}
    for item in p['episodes']:
        ep=item['episode'];eid=ep['episode_id'];raw=read(item['result'])
        old=mod._episode_result_from_json(eid,ep['difficulty'],read(published/(eid+'.json')))
        value,entities=diag.compute(ep,raw,rows,rejected,mod)
        original.append(old);current.append(value)
        details.append(dict(episode_id=eid,original=old.metrics['cs_face'].to_json(),current=value.metrics['cs_face'].to_json()))
        if eid in newids:
            for name,group in entities.items():
                for app in group['per_shot_sim_to_centroid']:
                    scored.append({**rows[eid,name,app['shot']],'edition':'rerun','similarity':app['sim_to_centroid']})
    split=read(BASE/'split_full140.json');a=mod.aggregate_results(original,split);b=mod.aggregate_results(current,split)
    reference=read(J.full.SNAPSHOT/'reviewed_dino_full140.json')['baseline']
    for t in ['all','easy','medium','hard']:assert a[t]['cs_face']==reference[t]['cs_face']
    report=dict(status='complete_with_localization_review' if final else 'automatic_pending_localization_review',
        original=a,current=b,per_episode=details,scored_appearances=sum(x.metrics['cs_face'].n_evaluated for x in current),
        valid_episodes=sum(x.metrics['cs_face'].value is not None for x in current),new_review_items=len(decisions),
        exclusions=list(rejected.values()),vlm_calls=0)
    write(REGION/('report.json' if final else 'automatic_report.json'),report)
    if not final:
        paired.review.OUT=REGION;write(REGION/'scored_crops.json',scored)
        count=paired.review.review_sheets(scored,p['new_complete_episodes'])
        status('localization_review_required',items=count,official12_complete=True)
    else:
        table=read(OUT/'official_comparison.json')['metrics']
        table=[dict(metric='cs_face_visible_single_face_reviewed',label='人物区域DINO（可见单脸＋定位纠错）',
            original=a['all']['cs_face']['mean'],current=b['all']['cs_face']['mean'])]+table
        write(OUT/'comparison.json',dict(episodes=140,shots=2491,replaced_complete_episodes=57,
            v2_independent_shots=141,metrics=table,default_dino='complete_with_localization_review',
            default_dino_valid_episodes=report['valid_episodes'],scope=p['scope'],vlm_calls=0))
        text='# 当前完整benchmark：57整集重跑＋141个v2独立镜头\n\n'+p['scope']+'\n\n官方逐集评分复用并SHA核验，按原split集间等权聚合。默认DINO补充新完成6集可见单脸检测和定位复核。VLM调用0；raw cs_face/cs_object无VLM保真度门控。\n\n| 指标↑ | 原始完整benchmark | 当前完整140集 | 差值 |\n|---|---:|---:|---:|\n'
        for row in table:text+=f"| {row['label']} | {row['original']:.4f} | {row['current']:.4f} | {row['current']-row['original']:+.4f} |\n"
        text+=f"\n默认DINO有效episode {report['valid_episodes']}/140，计分appearance {report['scored_appearances']}；新定位样本复核{len(decisions)}个。未完成episode前缀不纳入；保留原版补齐140/2491。\n"
        (OUT/'report.md').write_text(text)
        status('complete_all_available_metrics',episodes=140,shots=2491,new_review_items=len(decisions))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','final']);args=parser.parse_args()
    p=prepare()
    if args.action=='run':
        official(p);extract(p);dino(p)
    else:dino(p,True)
