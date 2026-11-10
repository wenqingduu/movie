"""Reviewed region-DINO and shot-only paired tables for the frozen141 pairs."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

import qwen140_v2_shots_only_eval_20261105 as job
import qwen140_badshot_reviewed_dino_20261104 as review

OUT, PRIOR, ROOT = job.OUT, job.PRIOR, job.ROOT
read, write, sha = job.read, job.write, job.sha


def load_maps():
    before, before_rejected, exclusions = review.before_state()
    new = {}
    for path in (PRIOR/'visible_region/extraction').glob('*/*.json'):
        for row in read(path)['rows']:
            new[review.diag.key(row)] = row
    for path in (OUT/'visible_region/extraction').glob('*/*.json'):
        for row in read(path)['rows']:
            new[review.diag.key(row)] = row
    assert set(new) == set(before)
    p = PRIOR/'visible_region/review/decisions.json'
    old_review = read(p)
    assert old_review['manifest_sha256'] == sha(PRIOR/'visible_region/review/manifest.json')
    exclusions += [r for r in old_review['decisions'] if r['decision']=='exclude']
    rejected = {review.diag.key(r):r for r in exclusions if
        new[review.diag.key(r)]['original_crop_sha256']==r['original_crop_sha256']}
    reviewed_exact = {(review.diag.key(r),r['original_crop_sha256']) for r in old_review['decisions']}
    return before, before_rejected, new, rejected, reviewed_exact


def compute(final=False):
    plan=job.prepare(); data=read(OUT/'replacement_manifest.json')
    before, before_rejected, new, rejected, reviewed_exact = load_maps()
    decisions=[]
    if final:
        manifest=read(OUT/'visible_region/review/manifest.json')
        manual=read(OUT/'visible_region/review/decisions.json')
        assert manual['manifest_sha256']==sha(OUT/'visible_region/review/manifest.json')
        expected={r['review_id']:r for r in manifest['rows']}
        assert len(manual['decisions'])==len(expected)
        assert {r['review_id'] for r in manual['decisions']}==set(expected)
        for row in manual['decisions']:
            ref=expected[row['review_id']]
            assert review.diag.key(ref)==review.diag.key(row)
            assert ref['original_crop_sha256']==row['original_crop_sha256']
            assert row['decision'] in ['keep','exclude'] and row['reason'] and row['evidence']
            if row['decision']=='exclude':
                rejected[review.diag.key(row)]=row
        decisions=manual['decisions']
    mod=review.diag.gate.load_official_pure()
    selected={e['episode_id']:e for e in read(OUT/'baseline_selection.json')['episodes']}
    old_check={e['episode_id']:e['current'] for e in read(job.full.SNAPSHOT/'reviewed_dino_full140.json')['per_episode']}
    scored=[]; per_shot={}; coverage={'before':{},'after':{}}
    scored_counts={'before':0,'after':0}
    for ep in plan['episodes']:
        eid=ep['episode_id'];raws={'before':read(selected[eid]['selected_result']),
            'after':read(OUT/'affected/results'/(eid+'.json'))}
        for edition, rows, rejected_rows in [('before',before,before_rejected),('after',new,rejected)]:
            metric, entities = review.diag.compute(ep,raws[edition],rows,rejected_rows,mod)
            if edition=='before':
                expected=old_check[eid]
                assert metric.metrics['cs_face'].n_evaluated==expected['n_evaluated']
                value=metric.metrics['cs_face'].value
                assert (value is None and expected['value'] is None) or abs(value-expected['value'])<1e-8
            for character, group in entities.items():
                for app in group['per_shot_sim_to_centroid']:
                    key=eid+'/'+app['shot']
                    if key not in data['targets']:
                        continue
                    k=(eid,character,app['shot']);r=rows[k]
                    per_shot.setdefault(key,{}).setdefault(edition,[]).append(app['sim_to_centroid'])
                    scored_counts[edition]+=1
                    if edition=='after' and (k,r['original_crop_sha256']) not in reviewed_exact and (
                            r['original_crop_sha256']!=before[k]['original_crop_sha256']):
                        scored.append({**r,'edition':'rerun','similarity':app['sim_to_centroid']})
    for edition,rows,reject in [('before',before,before_rejected),('after',new,rejected)]:
        subset=[r for k,r in rows.items() if k[0]+'/'+k[2] in data['targets']]
        coverage[edition]=dict(scheduled_role_shots=len(subset),present=sum(r['original_status']=='present' for r in subset),
            auto_eligible=sum(r['kept'] for r in subset),
            retained=sum(r['kept'] and review.diag.key(r) not in reject for r in subset),
            scored_appearances=scored_counts[edition],
            valid_target_shots=sum(bool(v.get(edition)) for v in per_shot.values()))
    report=dict(metric='cs_face_visible_single_face_reviewed' if final else 'cs_face_visible_single_face_auto',
        status='complete_with_localization_review' if final else 'automatic_pending_localization_review',
        scope='141 target-shot values from full43-episode centroids; no episode/full-benchmark aggregation.',
        coverage=coverage,carried_exact_exclusions=list(rejected.values()),new_review_items=len(decisions),
        per_shot={key:{edition:math.fsum(values)/len(values) for edition,values in editions.items()}
            for key,editions in per_shot.items()},vlm_calls=0)
    write(OUT/'visible_region'/('report.json' if final else 'automatic_report.json'),report)
    if not final:
        review.OUT=OUT/'visible_region'
        # Prior exact decisions retain their role/shot/crop binding. Review only
        # newly evaluated target crops, with the unchanged low60+random20 design.
        count=review.review_sheets(scored,[{'episode_id':e['episode_id'],'episode':e} for e in plan['episodes']])
        write(OUT/'visible_region/review_protocol.json',dict(
            reference='skills/evaluate-face-injection-v2/references/visible_single_face_region_dino.md',
            prior_review_sha256=sha(PRIOR/'visible_region/review/decisions.json'),
            new_review_items=count,selection='New target crops only, prior exact reviews reused; lowest60+random20, all if fewer80.'))
        job.status('localization_review_required',items=count,official_complete=True)
    return report


def summarize(final=False):
    comparison=read(OUT/'target_shot_comparison.json')
    dino=read(OUT/'visible_region'/('report.json' if final else 'automatic_report.json'))
    scope33={(r['episode_id'],r['shot']) for r in read(job.BATCH/'scope33_20261105.json')['selected']}
    for row in comparison['rows']:
        key=row['episode_id']+'/'+row['shot']
        for edition in ['before','after']:
            row[edition]['cs_face_visible_single_face_reviewed']=dino['per_shot'].get(key,{}).get(edition) if final else None
        row['group']='new33' if (row['episode_id'],row['shot']) in scope33 else 'earlier108'
    assert len(comparison['rows'])==141
    labels={'cs_face_visible_single_face_reviewed':'人物一致性DINO（可见单脸＋定位纠错）',**job.full.agg.LABELS}
    tables={}
    for group in ['all141','new33','earlier108']:
        rows=[r for r in comparison['rows'] if group=='all141' or r['group']==group]
        table=[]
        for name,label in labels.items():
            values={edition:[r[edition].get(name) for r in rows if r[edition].get(name) is not None] for edition in ['before','after']}
            means={edition:math.fsum(v)/len(v) if v else None for edition,v in values.items()}
            paired=[r for r in rows if r['before'].get(name) is not None and r['after'].get(name) is not None]
            common={edition:math.fsum(r[edition][name] for r in paired)/len(paired) if paired else None for edition in ['before','after']}
            table.append(dict(metric=name,label=label,**means,delta=means['after']-means['before'] if all(v is not None for v in means.values()) else None,
                before_n=len(values['before']),after_n=len(values['after']),paired_n=len(paired),
                common_support_before=common['before'],common_support_after=common['after']))
        tables[group]=dict(shots=len(rows),metrics=table)
    result=dict(status='complete_all_available_metrics' if final else 'official_complete_default_dino_pending',
        scope='Equal target-shot means. v1=each original integrated video; v2=all141 completed independent reruns. Not a benchmark or episode-macro score.',
        context='43 complete episodes supply centroids; only target shots enter means. Incoming non-cut edges for target shots supply boundary values.',
        conditional_coverage='DINO/missing metrics retain independent valid populations; paired-common-support means are supplementary in JSON.',
        other_unavailable_metrics='Appearance/action/LLM consistency require VLM and remain null; strict skip_vlm, calls0.',
        tables=tables,dino_coverage=dino['coverage'],vlm_calls=0)
    write(OUT/('comparison.json' if final else 'comparison_official_pending_dino.json'),result)
    write(OUT/('shots.json' if final else 'shots_official_pending_dino.json'),comparison)
    with (OUT/('shots.csv' if final else 'shots_official_pending_dino.csv')).open('w',newline='') as f:
        names=['episode_id','shot','group']+[f'{m}_{e}' for m in labels for e in ['before','after','delta']]
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for row in comparison['rows']:
            flat={k:row[k] for k in ['episode_id','shot','group']}
            for m in labels:
                a,b=row['before'].get(m),row['after'].get(m)
                flat.update({m+'_before':a,m+'_after':b,m+'_delta':b-a if a is not None and b is not None else None})
            writer.writerow(flat)
    text='# v2独立重跑镜头与v1：全部可用指标\n\n只汇总目标shot，不汇总完整benchmark。v1为同镜头重跑前集成版；人物中心使用完整episode上下文，边界仅目标镜头的接续入边。strict/skip_vlm，VLM调用0。\n\n'
    for group,title in [('all141','全部141个'),('new33','刚完成33个'),('earlier108','此前108个')]:
        text+=f'## {title}\n\n| 指标↑ | v1 | v2 | 差值 | 有效镜头数v1/v2 |\n|---|---:|---:|---:|---:|\n'
        for row in tables[group]['metrics']:
            def fmt(v):return '待评／不适用' if v is None else f'{v:.4f}'
            text+='| '+row['label']+' | '+fmt(row['before'])+' | '+fmt(row['after'])+' | '+fmt(row['delta'])+f" | {row['before_n']}/{row['after_n']} |\n"
        text+='\n'
    text+='所有目标均保留，包括动态仍0及降分镜头。目标原先按低动态筛选，不能把这里的改善率解释为全benchmark改善。B包含过渡；raw cs_face/cs_object缺VLM保真度门控，默认DINO是条件诊断，缺失不填0。JSON另存同一有效镜头交集的补充均值。\n'
    if not final:
        text+='\n默认人物DINO尚待真实定位复核，未用auto/raw填充最终行。\n'
    (OUT/('report.md' if final else 'report_official_pending_dino.md')).write_text(text)
    if final:
        job.status('complete_all_available_target_shot_metrics',target_shots=141,review_items=dino['new_review_items'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['auto','final']);args=parser.parse_args()
    compute(args.action=='final');summarize(args.action=='final')
