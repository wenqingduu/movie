"""Cumulative cumulative28 + priority1277 dynamic gains, complete-context evaluation.

Input-bound adapter. Unfiltered attempt table is separate from dynamic-only media
selection. New localization decisions must be frozen before action `final`.
"""
from __future__ import annotations

import copy
from collections import Counter
import math
from pathlib import Path
import sys

import qwen140_original40_priority_remaining_20261109 as batch
import qwen140_full140_latest_20261106 as frozen
import qwen140_localization_repair_gpu_20261108 as baseline_maps
import qwen140_v2_shots_only_eval_20261105 as execution

ROOT, BASE, PRIOR = batch.ROOT, batch.BASE, batch.BASE28
OUT, REGION = batch.OUT / 'replacement', batch.OUT / 'replacement/visible_region'
read, write, sha, immutable = batch.read, batch.write, batch.sha, batch.immutable
diag = frozen.diag


def status(stage, **kw):
    write(OUT / 'status.json', dict(stage=stage, utc=batch.core.now(), vlm_calls=0, api_calls=0,
          quality_gate_enabled=False, old_media_overwritten=False, large_queue_resumed=False, **kw))


def link(src, dst):
    src, dst = Path(src).resolve(), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert dst.resolve() == src
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())


def prepare():
    if (OUT / 'plan.json').exists():
        p = read(OUT / 'plan.json')
        for b in p['bindings']:
            assert sha(b['path']) == b['sha256'], b['path']
        return p
    generation = batch.prepare()
    snapshot = read(batch.OUT / 'baseline_snapshot.json')
    selected, decisions, attempt_rows, bindings = {}, [], [], {}
    for c in generation['cases']:
        folder = batch.OUT / 'cases' / c['case_id']
        r, s = read(folder / 'generation.json'), read(folder / 'scores.json')
        assert r['video_sha256'] == s['video_sha256'] == sha(r['video'])
        assert r['prompt'] == c['prompt'] and r['input_sha256'] == c['input_sha256'] and r['seed'] == c['seed']
        assert r['plan_sha256'] == sha(batch.OUT / 'plan.json')
        assert s['before'] == c['baseline_scores'] and c['frozen_benchmark_scores']['dynamic_degree'] == 0
        assert all(math.isfinite(s['after'][d]) for d in batch.core.DIMS)
        assert s['after']['dynamic_degree'] in (0, 1)
        key = c['episode_id'] + '/' + c['shot']
        chosen = s['after']['dynamic_degree'] > c['frozen_benchmark_scores']['dynamic_degree']
        decision = dict(case_id=c['case_id'], source_case=c['source_case'], episode_id=c['episode_id'], shot=c['shot'],
                        selected=chosen, before_dynamic=0, after_dynamic=s['after']['dynamic_degree'], other_quality_gates_applied=False)
        decisions.append(decision)
        attempt_rows.append(dict(**decision, before=c['frozen_benchmark_scores'], same_prompt4=c['same_prompt4_reference']['scores'] if c['same_prompt4_reference'] else None,
                                 after=s['after'], video=r['video'], video_sha256=r['video_sha256']))
        if chosen:
            selected[key] = dict(**c, selected_video=r['video'], selected_video_sha256=r['video_sha256'], after_scores=s['after'])
        for f in [folder / 'generation.json', folder / 'scores.json', Path(s['source_score'])]:
            bindings[str(f)] = sha(f)
    write(batch.OUT / 'comparison.json', dict(attempted=len(attempt_rows), selected=len(selected), complete=True, quality_gate_enabled=False,
          scope=generation['scope'], rows=attempt_rows,
          means={d:{side:sum(r[side][d] for r in attempt_rows)/len(attempt_rows) for side in ['before','after']} for d in batch.core.DIMS}))
    affected = {c['episode_id'] for c in selected.values()}
    episodes = [e['episode'] for e in snapshot['episodes'] if e['episode_id'] in affected]
    common = read(BASE / 'evaluation/official_parallel_v2/run_manifest.json')
    cache = dict(dimensions=batch.core.DIMS, evaluator_sha256=sha(execution.full.EVALUATOR), shots={})
    previous = read(PRIOR / 'replacement_manifest.json')['videos']
    media, score_data = {}, {}
    for item in snapshot['episodes']:
        ep, eid = item['episode'], item['episode_id']
        source = Path(item['result'])
        assert sha(source) == item['result_sha256']
        bindings[str(source)] = sha(source)
        if eid in affected:
            provenance = source.resolve().parent / 'run_manifest.json'
            m = read(provenance)
            for field in ['config','python_version','libraries','model_checkpoints','pillars','evaluation_scope']:
                assert m[field] == common[field], (eid, field)
            assert m['strict'] and m['skip_vlm'] and not m['allow_config_override'] and m['vlm_call_stats']['total_calls'] == 0
            bindings[str(provenance)] = sha(provenance)
        records = []
        for shot in ep['shots']:
            sk = shot['shot_key']; key = eid + '/' + sk
            c = selected.get(key)
            old = snapshot['media'][key]
            video = c['selected_video'] if c else old['source']
            digest = c['selected_video_sha256'] if c else old['sha256']
            assert sha(video) == digest
            a,b = map(int, sk.split(':'))
            canonical = OUT / 'candidate_media' / eid / f'{a:03d}_{b:03d}.mp4'
            link(video, canonical)
            media[key] = dict(source=video, sha256=digest, canonical=str(canonical),
                              before_source=old['source'], before_sha256=old['sha256'], replaced=bool(c),
                              cumulative_replaced=bool(c) or previous[key].get('cumulative_replaced', previous[key]['replaced']))
            records.append(dict(shot_key=sk, cut=shot['cut'], video=video, video_sha256=digest))
            if eid not in affected:
                continue
            if c:
                scores, sources = c['after_scores'], [dict(path=str(batch.OUT/'cases'/c['case_id']/'scores.json'),
                    sha256=sha(batch.OUT/'cases'/c['case_id']/'scores.json'))]
            else:
                scores, sources = {}, []
                for dim in batch.core.DIMS:
                    path = source.parent / '.work' / f'{eid}_{dim}_eval_results.json'
                    if str(path) not in score_data:
                        score_data[str(path)] = read(path); bindings[str(path)] = sha(path)
                    hits = [r['video_results'] for r in score_data[str(path)][dim][1] if Path(r['video_path']).stem == f'{a:03d}_{b:03d}']
                    assert len(hits) == 1
                    scores[dim] = float(hits[0]); sources.append(dict(path=str(path),sha256=bindings[str(path)]))
            assert all(math.isfinite(scores[d]) for d in batch.core.DIMS)
            cache['shots'][key] = dict(reuse=True, video_sha256=digest, scores=scores, score_sources=sources)
        immutable(OUT/'episode_media/episodes'/eid/'episode_result.json', dict(episode_id=eid,shots=records))
    for f in [Path(__file__), batch.OUT/'plan.json', batch.OUT/'baseline_snapshot.json', Path(execution.__file__),
              execution.full.EVALUATOR, ROOT/'pretest/qwen140_vbench_verified_cache_20261105.py']:
        bindings[str(f)] = sha(f)
    bound = [dict(path=k,sha256=v) for k,v in bindings.items()]
    cache['bindings'] = bound
    immutable(OUT/'vbench_cache_manifest.json', cache)
    immutable(OUT/'replacement_manifest.json', dict(videos=media,decisions=decisions,attempted=len(attempt_rows),selected=len(selected),
        previous_selected=28,cumulative_selected=28+len(selected),quality_gate_enabled=False,
        baseline_cumulative28_manifest_sha256=sha(PRIOR/'replacement_manifest.json')))
    p = dict(episodes=episodes,recompute_episode_ids=[e['episode_id'] for e in episodes], affected_episodes=len(episodes),
             complete_context_shots=sum(len(e['shots']) for e in episodes),target_cases=selected,target_shots=len(selected),
             preliminary_candidate_shots=len(selected),bindings=bound,workers=2,threads=4,
             scope='Cumulative full140: keep prior28 and append only priority1277 dynamic0_to1 gains. All1277 attempts retained, no other quality rejection; default DINO requires score-hidden review.')
    immutable(OUT/'plan.json',p)
    counts = {e['episode_id']:len(e['episode']['shots']) for e in snapshot['episodes']}
    before = read(PRIOR/'dynamic_exact.json')['after']
    delta = sum(1/counts[c['episode_id']]/140 for c in selected.values())
    immutable(OUT/'dynamic_exact.json',dict(before=before,after=before+delta,delta=delta,
        delta_percentage_points=delta*100,selected=len(selected),cumulative_selected=28+len(selected),episodes=140))
    status('selected_dynamic_gains_context_pending',selected=len(selected),affected_episodes=len(episodes))
    return p


def aggregate(p):
    snapshot=read(batch.OUT/'baseline_snapshot.json');mod=diag.gate.load_official_pure()
    changed=set(p['recompute_episode_ids']);groups={'before':[],'after':[]};receipts=[]
    for item in snapshot['episodes']:
        ep,eid=item['episode'],item['episode_id'];old=read(item['result'])
        path=OUT/'affected/results'/(eid+'.json')
        new=read(path) if eid in changed else old
        groups['before'].append(mod._episode_result_from_json(eid,ep['difficulty'],old))
        groups['after'].append(mod._episode_result_from_json(eid,ep['difficulty'],new))
        if eid in changed:
            receipt=path.resolve().parent/'.work'/(eid+'_vbench_reuse_receipt.json');r=read(receipt)
            assert r['cache_manifest_sha256']==sha(OUT/'vbench_cache_manifest.json')
            assert r['unchanged_shots_reused']==len(ep['shots']) and r['changed_shots_computed']==0
            receipts.append(dict(path=str(receipt),sha256=sha(receipt)))
    values={k:mod.aggregate_results(v,read(BASE/'split_full140.json')) for k,v in groups.items()}
    assert values['before']==read(PRIOR/'official_aggregates.json')['after']
    exact=read(OUT/'dynamic_exact.json')
    for label,rows in groups.items():
        value=sum(r.vbench_means['dynamic_degree'].value for r in rows)/140
        assert abs(value-exact[label])<1e-12
    table=[dict(metric=k,label=label,before=frozen.J.full.agg.agg_metric(values['before'],k)['mean'],
                after=frozen.J.full.agg.agg_metric(values['after'],k)['mean']) for k,label in frozen.J.full.agg.LABELS.items()]
    write(OUT/'official_aggregates.json',values)
    write(OUT/'official_comparison.json',dict(metrics=table,default_reviewed_dino='pending_localization_review'))
    write(OUT/'official_completion_validation.json',dict(episodes=140,shots=2491,recomputed_episodes=len(changed),
        complete_context_shots=p['complete_context_shots'],receipts=receipts,vlm_calls=0,scope='partial_without_vlm',exact_dynamic_verified=True))
    text='# 新增1277条尝试：累计替换版官方完整140集\n\n基线为此前已完成cumulative28版本；只替换本批动态0→1，无其他质量门控。\n\n| 指标 | cumulative28基线 | 新累计版本 |\n|---|---:|---:|\n'
    for r in table:text+=f"| {r['label']} | {r['before']:.4f} | {r['after']:.4f} |\n"
    text+='\n默认人物DINO待定位复核；官方四位舍入，动态精确增益见dynamic_exact.json。strict/skip_vlm、VLM0、raw cs_face/cs_object缺保真门控，partial_without_vlm。\n'
    (OUT/'official_report.md').write_text(text)


def maps():
    rows=baseline_maps.gate_maps()['current']
    for directory in [batch.BASE20, PRIOR]:
        for path in (directory/'visible_region/extraction').glob('*/*.json'):
            for r in read(path)['rows']:rows[diag.key(r)]=r
    ref=read(PRIOR/'visible_region/report.json')
    rejected={diag.key(r):r for r in ref['carried_exclusions']+ref['new_exclusions']}
    assert len(rows)==4991
    for k,r in rejected.items():assert rows[k]['original_crop_sha256']==r['original_crop_sha256']
    return rows,rejected,ref


def dino(final=False):
    import qwen140_badshot_reviewed_dino_20261104 as review
    p=prepare();snapshot=read(batch.OUT/'baseline_snapshot.json')
    old,rejected,reference=maps();new=copy.deepcopy(old)
    media=read(OUT/'replacement_manifest.json')['videos']
    for ep in p['episodes']:
        for shot in ep['shots']:
            path=REGION/'extraction'/ep['episode_id']/(shot['shot_key'].replace(':','_')+'.json')
            data=read(path)
            sk=shot['shot_key'];key=ep['episode_id']+'/'+sk
            artifact=OUT/'affected/results/.work/eval_artifacts'/ep['episode_id']/'intra_shot'/sk.replace(':','_')/'_eval.json'
            assert data['evaluation_sha256']==sha(artifact)
            assert data['video_sha256']==media[key]['sha256']
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
    check={r['episode_id']:r['after'] for r in reference['per_episode']};changed=set(p['recompute_episode_ids'])
    for item in snapshot['episodes']:
        ep,eid=item['episode'],item['episode_id'];raw=read(item['result'])
        a,_=diag.compute(ep,raw,old,rejected,mod)
        expected=check[eid];actual=a.metrics['cs_face'].to_json()
        assert actual['n_evaluated']==expected['n_evaluated']
        assert (actual['value'] is None and expected['value'] is None) or abs(actual['value']-expected['value'])<1e-10
        value,entities=diag.compute(ep,read(OUT/'affected/results'/(eid+'.json')) if eid in changed else raw,new,carry,mod)
        groups['before'].append(a);groups['after'].append(value)
        details.append(dict(episode_id=eid,before=actual,after=value.metrics['cs_face'].to_json()))
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
        n=review.review_sheets(scored,[dict(episode_id=e['episode_id'],episode=e) for e in p['episodes']])
        status('official_full140_complete_default_dino_review_pending',review_items=n,
               selected=p['target_shots'],cumulative_selected=28+p['target_shots'])
    else:
        table=[dict(metric=report['metric'],label='人物一致性DINO（可见单脸＋定位纠错）',
               before=aggregates['before']['all']['cs_face']['mean'],after=aggregates['after']['all']['cs_face']['mean'])]+read(OUT/'official_comparison.json')['metrics']
        write(OUT/'complete_comparison.json',dict(metrics=table,coverage=coverage,episodes=140,shots=2491,selected=p['target_shots'],cumulative_selected=28+p['target_shots'],vlm_calls=0))
        text='# 新增1277条尝试及累计动态替换完整140集\n\n比较基线：此前cumulative28版本。官方非VLM12项＋原区域可见单脸定位复核DINO；VLM调用0，保真门控缺失。\n\n| 指标 | cumulative28基线 | 新累计版本 |\n|---|---:|---:|\n'
        for r in table:text+=f"| {r['label']} | {r['before']:.4f} | {r['after']:.4f} |\n"
        text+='\n所有24尝试和下降保留，仅动态增长决定媒体替换；原视频不覆盖。默认DINO覆盖和精确动态增益见JSON。\n'
        (OUT/'complete_report.md').write_text(text)
        write(OUT/'completion_validation.json',dict(report_sha256=sha(OUT/'complete_report.md'),review_sha256=sha(REGION/'review/decisions.json'),
            manifest_sha256=sha(OUT/'replacement_manifest.json'),vlm_calls=0))
        status('complete_all_available_metrics_with_reviewed_dino',selected=p['target_shots'],cumulative_selected=28+p['target_shots'])
        batch.status('complete_all_available_metrics_with_reviewed_dino',selected=p['target_shots'],cumulative_selected=28+p['target_shots'])


def run():
    p=prepare();oldout,oldfile=execution.OUT,execution.__file__
    try:
        execution.OUT,execution.__file__=OUT,str(Path(__file__))
        if p['affected_episodes']:
            execution.evaluate(p);execution.dino_prepare(p)
        aggregate(p);dino()
    except Exception as error:
        status('stopped_context_error_preserve_all_media',error=repr(error));raise
    finally:execution.OUT,execution.__file__=oldout,oldfile


if __name__=='__main__':
    if sys.argv[1]=='scorer':
        import qwen140_vbench_verified_cache_20261105 as adapter
        adapter.CACHE=OUT/'vbench_cache_manifest.json';sys.argv.pop(1);adapter.main()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='final':dino(True)
    else:raise ValueError(sys.argv[1])
