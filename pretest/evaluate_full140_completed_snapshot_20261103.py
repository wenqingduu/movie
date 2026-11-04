"""Freeze complete reruns and aggregate a mixed full140 snapshot without media generation."""
from __future__ import annotations
import argparse, copy, csv, json, math
from datetime import datetime, timezone
from pathlib import Path
import qwen140_reaggregate_partial_reruns as agg
import qwen140_rerun_region_diagnostic as diag
ROOT=agg.ROOT
OUT=ROOT/'outputs/entitybench_full140_completed41_snapshot_20261103'
PRIOR=ROOT/'outputs/entitybench_completed35_comparison_20261102'
read,write,sha=agg.read,agg.write,agg.sha

def prepare():
    OUT.mkdir(exist_ok=True)
    path=OUT/'cohort_snapshot.json'
    if path.exists(): return read(path)
    prior=read(PRIOR/'cohort_snapshot.json')
    entries=copy.deepcopy(prior['episodes'])
    for e in entries: e['comparison_group']='previous35'
    ids={e['episode_id'] for e in entries}
    progress=ROOT/'outputs/entitybench_episode_repairs_20261103/progress.json'
    source={e['episode_id']:e for e in read(agg.BASE/'manifest.json')['episodes']}
    for row in read(progress)['episodes']:
        eid=row['episode_id'];root=Path(row['output_root'])
        if eid in ids or not (root/'complete.json').exists():continue
        complete=read(root/'complete.json');assert complete['generation_and_official_non_vlm']=='complete'
        ev=root/'evaluation/weak4_v1';res=ev/'results'/(eid+'.json')
        entries.append({'episode_id':eid,'episode':source[eid],'output_root':str(root),
            'evaluation':str(ev),'result':str(res),'result_sha256':sha(res),'comparison_group':'new6'})
        ids.add(eid)
    assert len(entries)==41 and sum(len(e['episode']['shots']) for e in entries)==670
    snap={'created_utc':datetime.now(timezone.utc).isoformat(),'selected_episodes':41,'selected_shots':670,
        'original_episodes':140,'original_shots':2491,'unchanged_episodes':99,'unchanged_shots':1821,
        'selection':'All complete episodes at frozen request snapshot; include declines; no partial shot prefixes; accepted latest run594 retained.',
        'queue_progress_sha256':sha(progress),'prior35_snapshot_sha256':sha(PRIOR/'cohort_snapshot.json'),'episodes':entries}
    diag.immutable(path,snap)
    return snap

def official():
    snap=prepare()
    frozen=read(ROOT/'outputs/entitybench_rerun30_latest_evaluation_20261101/frozen_inputs.json')
    for s in frozen['sources']:assert sha(s['path'])==s['sha256'],s['path']
    selected={}
    for e in snap['episodes']:
        assert sha(e['result'])==e['result_sha256']
        selected[e['episode_id']]={'root':Path(e['output_root']),'evaluation':Path(e['evaluation']),
          'cohort':str(OUT/'cohort_snapshot.json'),'group':e['comparison_group'],
          'generation_revision':e.get('generation_revision','Current multiface pipeline; technical adapter retains generation settings'),
          'version_note':'Latest accepted whole episode; historical prompt revisions retained in provenance'}
    agg.replacements=lambda additional_cohort=None:selected
    agg.main(OUT/'official')
    write(OUT/'status.json',{'stage':'official_complete_reviewed_dino_pending','episodes':140,'replaced_episodes':41,'replaced_shots':670})

def final():
    snap=prepare();mod=diag.gate.load_official_pure();split=read(agg.BASE/'split_full140.json')
    for item in read(PRIOR/'validation.json')['sources']:assert sha(item['path'])==item['sha256'],item['path']
    published=diag.gate.OUT/'remaining_low_score_review/recomputed/results'
    previous=[ROOT/'outputs/entitybench_rerun30_latest_evaluation_20261101/visible_region/report.json',
              PRIOR/'new5_visible_region/report.json',OUT/'new6_visible_region/report.json']
    replacements={};sources=[]
    for p in previous:
        r=read(p);assert r.get('status',r.get('stage'))=='complete_with_localization_review'
        for row in r['per_episode']:
            assert row['episode_id'] not in replacements
            replacements[row['episode_id']]=row
        sources.append({'path':str(p),'sha256':sha(p)})
    assert set(replacements)=={e['episode_id'] for e in snap['episodes']}
    entries={e['episode_id']:e for e in snap['episodes']}
    baseline=[];current=[];rows=[]
    for ep in read(agg.BASE/'manifest.json')['episodes']:
        eid=ep['episode_id'];p=published/(eid+'.json');raw=read(p)
        a=mod._episode_result_from_json(eid,ep['difficulty'],raw);baseline.append(a)
        b=copy.deepcopy(a)
        if eid in replacements:
            rep=replacements[eid]
            assert sha(entries[eid]['result'])==entries[eid]['result_sha256']
            if 'official_result_sha256' in rep:assert rep['official_result_sha256']==sha(entries[eid]['result'])
            assert rep['baseline']['value']==a.metrics['cs_face'].value or (rep['baseline']['value'] is not None and abs(rep['baseline']['value']-a.metrics['cs_face'].value)<1e-10)
            b.metrics['cs_face']=mod.MetricValue(**rep['current'])
        current.append(b)
        rows.append({'episode_id':eid,'difficulty':ep['difficulty'],'edition':'rerun' if eid in replacements else 'original',
          'baseline':a.metrics['cs_face'].to_json(),'current':b.metrics['cs_face'].to_json()})
        sources.append({'path':str(p),'sha256':sha(p)})
    before=mod.aggregate_results(baseline,split);after=mod.aggregate_results(current,split)
    historical=read(diag.gate.OUT/'remaining_low_score_review/recomputed/report.json')
    for tier in ['all','easy','medium','hard']:
        assert before[tier]['cs_face']==historical['after'][tier]['cs_face'],tier
    dino={'metric':'cs_face_visible_single_face_reviewed','status':'complete_with_localization_review',
      'baseline':before,'current':after,'per_episode':rows,'sources':sources,'vlm_calls':0}
    write(OUT/'reviewed_dino_full140.json',dino)
    official_report=read(OUT/'official/comparison.json');table=official_report['metrics']
    av=[r.metrics['cs_face'].value for r in baseline if r.metrics['cs_face'].value is not None]
    bv=[r.metrics['cs_face'].value for r in current if r.metrics['cs_face'].value is not None]
    drow={'metric':'cs_face_visible_single_face_reviewed','label':'人物一致性DINO（可见单脸＋定位纠错）',
          'baseline':before['all']['cs_face']['mean'],'current':after['all']['cs_face']['mean'],
          'delta':math.fsum(bv)/len(bv)-math.fsum(av)/len(av)}
    table=[drow]+table
    coverage={}
    oldrows,rejected=diag.historical();oldmap={diag.key(r):r for r in oldrows};newmap=dict(oldmap)
    reject=dict(rejected)
    for folder in [ROOT/'outputs/entitybench_rerun30_latest_evaluation_20261101/visible_region',PRIOR/'new5_visible_region',OUT/'new6_visible_region']:
        manifest=read(folder/'input_manifest.json')
        for item in manifest['episodes']:
            eid=item['episode_id']
            for k in [k for k in reject if k[0]==eid]:del reject[k]
            for p in (folder/'extraction'/eid).glob('*.json'):
                data=read(p)
                for r in data['rows']:
                    k=diag.key(r);newmap[k]=r
                    if k in rejected and rejected[k]['original_crop_sha256']==r['original_crop_sha256']:reject[k]=rejected[k]
        dec=read(folder/'review/decisions.json');assert dec['manifest_sha256']==sha(folder/'review/manifest.json')
        for r in dec['decisions']:
            if r['decision']=='exclude':reject[diag.key(r)]=r
    assert len(newmap)==len(oldmap)==4991
    coverage={'scheduled_role_shots':4991,'baseline_present':sum(r['original_status']=='present' for r in oldmap.values()),
      'current_present':sum(r['original_status']=='present' for r in newmap.values()),
      'baseline_auto_eligible':sum(r['kept'] for r in oldmap.values()),'current_auto_eligible':sum(r['kept'] for r in newmap.values()),
      'baseline_reviewed_retained':sum(r['kept'] and k not in rejected for k,r in oldmap.items()),
      'current_reviewed_retained':sum(r['kept'] and k not in reject for k,r in newmap.items()),
      'baseline_scored':sum(r.metrics['cs_face'].n_evaluated for r in baseline),
      'current_scored':sum(r.metrics['cs_face'].n_evaluated for r in current),
      'baseline_valid_episodes':len(av),'current_valid_episodes':len(bv)}
    report={'snapshot':{k:v for k,v in snap.items() if k!='episodes'},'metrics':table,'dino_coverage':coverage,
      'review_scope':'Previous30 review80 plus previous5 review80 reused by same inputs; newly complete6 review80. Historical original3 exclusions remain only on matching crops.',
      'scope':'Mixed full140,41 complete reruns plus99 unchanged originals; strict/skip_vlm, raw cs_face/cs_object provisional without fidelity gate.',
      'vlm_calls':0,'new_official_inference_calls':0,'new_dino_embedding_calls':0}
    write(OUT/'report.json',report)
    oldtable=read(OUT/'official/per_episode_comparison.json')['episodes']
    with (OUT/'per_episode_comparison.csv').open('w',newline='') as f:
        fields=['episode_id','run','difficulty','edition','shots']+[n+'_'+s for n in [*agg.LABELS,drow['metric']] for s in ['baseline','current','delta']]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for raw,d in zip(oldtable,rows):
            assert raw['episode_id']==d['episode_id']
            flat={k:raw[k] for k in fields[:5]}
            for name,pair in raw['metrics'].items():
                for s in ['baseline','current','delta']:flat[name+'_'+s]=pair[s]
            a,b=d['baseline']['value'],d['current']['value']
            for s,v in [('baseline',a),('current',b),('delta',b-a if a is not None and b is not None else None)]:flat[drow['metric']+'_'+s]=v
            writer.writerow(flat)
    lines=['# 完整140集：替换41集后的结果','',
      '固定本次请求快照：41集670shot采用已完成重跑（含新版run594），其余99集1821shot沿用原版；合计140集2491shot。包含所有已完成版本和下降案例，不纳入未完成集前缀。',
      '', '| 指标↑ | 原始完整benchmark | 替换41集后 | 差值 |','|---|---:|---:|---:|']
    for r in table:
        label='官方原始人物区域DINO cs_face' if r['metric']=='cs_face' else r['label']
        lines.append(f"| {label} | {r['baseline']:.4f} | {r['current']:.4f} | {r['delta']:+.4f} |")
    lines+=['',f"默认DINO覆盖：原版{len(av)}/140集、{coverage['baseline_scored']}次计分；当前{len(bv)}/140集、{coverage['current_scored']}次计分。按有效episode等权，缺失保留null；默认DINO是条件诊断，不等于参考人脸身份准确率。",
      '', '新6集使用CPU RetinaFace，按既有协议做80项分数隐藏定位复查后重算质心；此前35集复查及未替换原版结果保留。原all/easy/medium/hard官方聚合完全复现；视频、尾帧、父链及配置SHA已核验，无新增官方评分/DINO特征推理、视频生成或付费API。',
      '', '官方范围为strict/skip_vlm、partial_without_vlm，VLM调用0；raw cs_face/cs_object没有VLM保真度门控。边界一致性仅适用的137集参与均值。此次是混合全量结果，不是140集统一新版重跑，也不是单独3D注入消融。完整视频视觉审核仍待完成。',
      '', '[140集逐集CSV](per_episode_comparison.csv) · [JSON及覆盖](report.json) · [原官方指标、all/easy/medium/hard](official/evaluation/current_official_aggregate.json) · [完整输入及视频映射](official/selection_manifest.json) · [新增6集定位审核](new6_visible_region/review/decisions.json)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'validation.json',{'snapshot_sha256':sha(OUT/'cohort_snapshot.json'),
      'adapter_sha256':sha(Path(__file__)),'original_reviewed_baseline_reproduced':True,
      'official_validation':read(OUT/'official/validation.json'),'reviewed_dino_sources':sources,
      'dino_coverage':coverage,'vlm_calls':0})
    write(OUT/'status.json',{'stage':'complete_official_and_reviewed_dino','episodes':140,'shots':2491,
      'replaced_episodes':41,'replaced_shots':670,'completed_utc':datetime.now(timezone.utc).isoformat()})
    print('\n'.join(lines[:22]))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','official','final']);args=parser.parse_args()
    {'prepare':prepare,'official':official,'final':final}[args.action]()
