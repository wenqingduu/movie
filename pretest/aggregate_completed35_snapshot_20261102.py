"""Compare only the 35 complete reruns available at this request's snapshot.

Read-only reuse of verified scores; no inference, generation, VLM or API calls.
Use the exact official deserializer/episode macro aggregator. Do not mix the
30 available reviewed-DINO episodes into a misleading 35-episode current mean.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import qwen140_reaggregate_partial_reruns as agg
import evaluate_entitybench_qwen140_region_face_gate as gate

ROOT = Path('/root/autodl-tmp/movie')
OUT = ROOT/'outputs/entitybench_completed35_comparison_20261102'
OLD = ROOT/'outputs/entitybench_rerun30_latest_evaluation_20261101'
QUEUE = ROOT/'outputs/entitybench_csface_priority_20261101'
read, write, sha = agg.read, agg.write, agg.sha
REVIEWED = gate.OUT/'remaining_low_score_review/recomputed/results'


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'cohort_snapshot.json').exists():
        frozen = read(OUT/'cohort_snapshot.json')
        entries = frozen['episodes']
    else:
        progress = read(QUEUE/'progress.json')
        complete = [e for e in progress['episodes'] if e['complete_official_non_vlm']]
        assert len(complete) == 5, 'Use a new snapshot adapter if cohort grows'
        entries = copy.deepcopy(read(OLD/'visible_region/input_manifest.json')['episodes'])
        for e in entries:
            e['comparison_group'] = 'previous30'
        source = read(agg.BASE/'manifest.json')
        by_id = {e['episode_id']: e for e in source['episodes']}
        for item in complete:
            eid = item['episode_id']; root = Path(item['output_root'])
            evaluation = root/'evaluation/weak4_v1'
            result = evaluation/'results'/(eid+'.json')
            entries.append({'episode_id': eid, 'episode': by_id[eid],
                'output_root': str(root), 'evaluation': str(evaluation),
                'result': str(result), 'result_sha256': sha(result),
                'comparison_group': 'new5'})
        assert len(entries) == len({e['episode_id'] for e in entries}) == 35
        frozen = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'selected_episodes': 35, 'selected_shots': sum(len(i['episode']['shots']) for i in entries),
            'selection': 'Previous30 latest including repaired run594 plus5 newly complete episodes; exclude all unfinished prefixes.',
            'queue_progress_sha256': sha(QUEUE/'progress.json'), 'episodes': entries}
        assert frozen['selected_shots'] == 520
        write(OUT/'cohort_snapshot.json', frozen)

    # Revalidate the prior30 frozen evaluation instead of trusting displayed text.
    prior = read(OLD/'frozen_inputs.json')
    assert sha(OLD/'visible_region/input_manifest.json') == prior['manifest_sha256']
    for source in prior['sources']:
        assert sha(source['path']) == source['sha256'], source['path']

    common = read(agg.ORIGINAL/'run_manifest.json')
    original_inputs = read(agg.ORIGINAL/'input_fingerprint.json')['inputs']
    baseline_jobs = {(r['episode_id'],r['shot_key']):r for r in read(agg.BASE/'generation_report.json')['jobs']}
    mod = gate.load_official_pure()
    baseline, current, reviewed_baseline, per_episode, sources = {}, {}, {}, [], []
    videos_verified = 0
    for item in entries:
        eid, ep = item['episode_id'], item['episode']
        root, ev = Path(item['output_root']), Path(item['evaluation'])
        assert sha(ep['episode_path']) == ep['episode_sha256']
        result = read(item['result'])
        assert sha(item['result']) == item['result_sha256']
        old = read(agg.ORIGINAL/(eid+'.json'))
        assert result['episode_id'] == old['episode_id'] == eid
        assert result['num_shots'] == old['num_shots'] == len(ep['shots'])
        manifest = read(ev/'results/run_manifest.json')
        assert manifest['strict'] and manifest['skip_vlm'] and not manifest['allow_config_override']
        assert manifest['vlm_call_stats']['total_calls'] == 0
        for field in ['config','python_version','libraries','model_checkpoints','pillars','evaluation_scope']:
            assert manifest[field] == common[field], (eid,field)
        inputs = read(ev/'input_manifest.json')
        assert inputs['evaluator_sha256'] == sha(agg.BASE/'runtime/evaluate_benchmark_partial.py')
        generation_file = root/'episodes'/eid/'episode_result.json'
        generated = read(generation_file)['shots']
        assert len(generated) == len(ep['shots']) == len(inputs['videos'])
        for j, (shot, record) in enumerate(zip(ep['shots'], generated)):
            key = eid+'/'+shot['shot_key']
            assert record['shot_key'] == shot['shot_key'] and record['cut'] == shot['cut']
            assert sha(record['video']) == record['video_sha256'] == inputs['videos'][key]['sha256']
            assert sha(record['last_frame']) == record['last_sha256']
            scene, num = map(int,shot['shot_key'].split(':'))
            canonical = root/'official_results'/eid/f'{scene:03d}_{num:03d}.mp4'
            assert canonical.resolve() == Path(record['video']).resolve(), canonical
            job = baseline_jobs[eid,shot['shot_key']]
            assert sha(job['output_video']) == job['video_sha256'] == original_inputs['videos'][key]
            assert record['baseline_video_sha256'] == job['video_sha256']
            if j and not shot['cut']:
                assert record['parent_video_sha256'] == generated[j-1]['video_sha256']
                assert record['parent_tail_sha256'] == generated[j-1]['last_sha256']
            videos_verified += 1
        for raw in [old,result]:
            for name in agg.LABELS:
                m = agg.metric(raw,name)
                assert m['n_failed'] == 0, (eid,name)
                assert m['value'] is None or math.isfinite(m['value'])
        baseline[eid] = mod._episode_result_from_json(eid,ep['difficulty'],old)
        current[eid] = mod._episode_result_from_json(eid,ep['difficulty'],result)
        reviewed_raw = read(REVIEWED/(eid+'.json'))
        reviewed_baseline[eid] = mod._episode_result_from_json(eid,ep['difficulty'],reviewed_raw)
        metric_values = {name:{'baseline':agg.metric(old,name)['value'],
                              'current':agg.metric(result,name)['value']}
                         for name in agg.LABELS}
        for value in metric_values.values():
            value['delta'] = value['current']-value['baseline'] if all(value[k] is not None for k in ['current','baseline']) else None
        per_episode.append({'episode_id':eid, 'run':ep['run'], 'difficulty':ep['difficulty'],
            'shots':len(ep['shots']), 'group':item['comparison_group'], 'metrics':metric_values})
        for path in [Path(item['result']), agg.ORIGINAL/(eid+'.json'), ev/'results/run_manifest.json',
                     ev/'input_manifest.json', generation_file, REVIEWED/(eid+'.json')]:
            sources.append({'path':str(path),'sha256':sha(path)})

    groups = {}
    for group in ['all35','new5','previous30']:
        subset = [e for e in entries if group=='all35' or e['comparison_group']==group]
        ids = [e['episode_id'] for e in subset]
        split = {tier:[e['episode_id'] for e in subset if e['episode']['difficulty']==tier]
                 for tier in ['easy','medium','hard']}
        a = mod.aggregate_results([baseline[eid] for eid in ids],split)
        b = mod.aggregate_results([current[eid] for eid in ids],split)
        dino = mod.aggregate_results([reviewed_baseline[eid] for eid in ids],split)['all']['cs_face']
        rows = []
        for name,label in agg.LABELS.items():
            if name=='cs_face':label='官方原始人物区域DINO cs_face'
            av,bv = agg.agg_metric(a,name),agg.agg_metric(b,name)
            rows.append({'metric':name,'label':label,'baseline':av,'current':bv,
                'delta':bv['mean']-av['mean'] if av['mean'] is not None and bv['mean'] is not None else None})
            # Independent macro calculation, not a weighted mean across shots.
            for raw, expected in [(baseline,av),(current,bv)]:
                vals = [(raw[eid].vbench_means if name in raw[eid].vbench_means else raw[eid].metrics)[name].value
                        for eid in ids]
                vals = [v for v in vals if v is not None]
                assert expected['n_episodes_evaluated'] == len(vals)
                if vals:
                    precise = sum(vals)/len(vals)
                    assert round(precise,4) == expected['mean']
                    rows[-1]['baseline_unrounded' if raw is baseline else 'current_unrounded'] = precise
            if 'baseline_unrounded' in rows[-1] and 'current_unrounded' in rows[-1]:
                rows[-1]['delta'] = rows[-1]['current_unrounded']-rows[-1]['baseline_unrounded']
        groups[group] = {'episodes':len(ids),'shots':sum(len(e['episode']['shots']) for e in subset),
            'split':split,'baseline':a,'current':b,'metrics':rows,
            'default_reviewed_dino':{'baseline':dino, 'current':None,
                'status':'pending_new5_face_gate_and_localization_review' if group!='previous30' else 'see_verified_previous30',
                'complete_current_episodes':30 if group=='all35' else 0 if group=='new5' else 30}}

    previous_report = read(OLD/'official_report.json')
    assert groups['previous30']['baseline'] == previous_report['baseline']
    assert groups['previous30']['current'] == previous_report['current']
    existing_dino = read(OLD/'visible_region/report.json')
    groups['previous30']['default_reviewed_dino'] = {
        'baseline':existing_dino['baseline']['all']['cs_face'],
        'current':existing_dino['current']['all']['cs_face'],
        'status':'complete_with_localization_review','complete_current_episodes':30}
    reviewed_new5_path=OUT/'new5_visible_region/report.json'
    reviewed_new5=read(reviewed_new5_path) if reviewed_new5_path.exists() else None
    dino_complete=reviewed_new5 is not None
    if dino_complete:
        assert reviewed_new5['status']=='complete_with_localization_review'
        expected={e['episode_id']:e for e in entries if e['comparison_group']=='new5'}
        assert {r['episode_id'] for r in reviewed_new5['per_episode']}==set(expected)
        for row in reviewed_new5['per_episode']:
            assert row['official_result_sha256']==expected[row['episode_id']]['result_sha256']
        current_dino_by_id={r['episode_id']:r['current']
                           for r in existing_dino['per_episode']+reviewed_new5['per_episode']}
        for group in ['all35','new5']:
            ids=[e['episode_id'] for e in entries if group=='all35' or e['comparison_group']=='new5']
            dr=[]
            for eid in ids:
                ep=next(e['episode'] for e in entries if e['episode_id']==eid)
                rec=mod.EpisodeResult(eid,ep['difficulty'],len(ep['shots']))
                rec.metrics['cs_face']=mod.MetricValue(**current_dino_by_id[eid])
                dr.append(rec)
            groups[group]['default_reviewed_dino']['current']=mod.aggregate_results(dr,groups[group]['split'])['all']['cs_face']
            groups[group]['default_reviewed_dino']['status']='complete_with_localization_review'
            groups[group]['default_reviewed_dino']['complete_current_episodes']=len(ids)
        sources += [{'path':str(p),'sha256':sha(p)} for p in [reviewed_new5_path,
                     OUT/'new5_visible_region/review/manifest.json',
                     OUT/'new5_visible_region/review/decisions.json']]
    assert abs(groups['previous30']['default_reviewed_dino']['baseline']['mean']-
               mod.aggregate_results([reviewed_baseline[e['episode_id']] for e in entries if e['comparison_group']=='previous30'],
                                     groups['previous30']['split'])['all']['cs_face']['mean']) < 1e-12
    for row in per_episode:
        row['metrics']['cs_face_visible_single_face_reviewed'] = {
            'baseline':reviewed_baseline[row['episode_id']].metrics['cs_face'].value,
            'current':None,'delta':None,'status':'pending_new_crop_review'}
    prior_dino = {r['episode_id']:r for r in existing_dino['per_episode']+
                 (reviewed_new5['per_episode'] if dino_complete else [])}
    for row in per_episode:
        if row['group']=='previous30' or dino_complete:
            m = row['metrics']['cs_face_visible_single_face_reviewed']
            m['current'] = prior_dino[row['episode_id']]['current']['value']
            m['delta'] = m['current']-m['baseline'] if m['current'] is not None and m['baseline'] is not None else None
            m['status'] = 'complete_with_localization_review'
    detector_calls=0
    if dino_complete:
        for path in (OUT/'new5_visible_region/extraction').glob('*/*.json'):
            samples=[r['sample_idx'] for r in read(path)['rows'] if r['original_status']=='present']
            detector_calls += max(samples)-min(samples)+1 if samples else 0
    report = {'created_utc':datetime.now(timezone.utc).isoformat(), 'scope':frozen['selection'],
        'aggregation':'Frozen official deserializer and valid-episode equal-weight aggregation',
        'groups':groups,'per_episode':per_episode, 'vlm_calls':0,
        'new_official_metric_inference_calls':0,'new_dino_embedding_inference_calls':0,
        'cpu_retinaface_forward_calls':detector_calls,
        'limitations':([] if dino_complete else ['Default reviewed DINO current35/new5 pending; do not average only30 valid current episodes as35.'])+[
                       'Official strict/skip_vlm partial_without_vlm; cs_face/cs_object lack VLM fidelity gate.',
                       'Whole-pipeline original-vs-rerun comparison, not an isolated3D injection ablation.']}
    if dino_complete:
        report['dino_coverage']={k:existing_dino['coverage'][k]+reviewed_new5['coverage'][k]
            for k in ['scheduled_role_shots','baseline_present','current_present','baseline_auto_eligible',
                      'current_auto_eligible','baseline_reviewed_retained','current_reviewed_retained',
                      'baseline_scored','current_scored']}
        report['dino_review']={'previous30_review_items':existing_dino['review_items'],
            'new5_review_items':reviewed_new5['review_items'],
            'previous30_exclusions':existing_dino['new_exclusions'],
            'new5_exclusions':reviewed_new5['new_exclusions'],
            'unchanged30_decisions_reused_by_verified_inputs':True}
    write(OUT/'report.json',report)
    write(OUT/'per_episode_comparison.json',per_episode)
    for name, rows in [('per_episode_comparison.csv',per_episode),
                       ('new5_per_episode_comparison.csv',[r for r in per_episode if r['group']=='new5'])]:
        fields=['episode_id','run','difficulty','shots','group']
        for metric in [*agg.LABELS,'cs_face_visible_single_face_reviewed']:
            fields += [metric+'_'+suffix for suffix in ['baseline','current','delta']]
        with (OUT/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:
                flat={key:row[key] for key in ['episode_id','run','difficulty','shots','group']}
                flat.update({metric+'_'+suffix:values[suffix]
                             for metric,values in row['metrics'].items() for suffix in ['baseline','current','delta']})
                writer.writerow(flat)
    lines=['# 已完成35集：原版与当前重跑的同集比较','',
        '固定此次请求时已完成的35集／520shot：此前30集343shot（含新版run594）＋本轮新增5集177shot。当前run264及所有中断前缀不纳入；不是全140集。',
        '', '使用冻结官方反序列化和聚合函数，对同一批episode逐集等权，缺失保留null。所有输入视频、接续父链、评分文件和配置重新核验；官方12项复用已验证评分。默认DINO仅新增CPU人脸检测，原代表人物区域和DINO特征保持不变；没有重跑视频、DINO特征提取或付费API。',
        '', '| 指标↑ | 原版同35集 | 重跑35集 | 差值 | 原版新增同5集 | 重跑新增5集 | 差值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'待完成' if v is None else f'{v:.4f}'
    fmt_delta=lambda v:'—' if v is None else f'{v:+.4f}'
    if dino_complete:
        cells=[]
        for group in ['all35','new5']:
            d=groups[group]['default_reviewed_dino'];a=d['baseline']['mean'];b=d['current']['mean']
            cells += [fmt(a),fmt(b),fmt_delta(b-a) if a is not None and b is not None else '—']
        lines.append('| **人物一致性DINO（可见单脸＋定位纠错）** | '+' | '.join(cells)+' |')
    else:
        lines.append('| **人物一致性DINO（可见单脸＋定位纠错）** | '+
            fmt(groups['all35']['default_reviewed_dino']['baseline']['mean'])+' | 待补新增5集 | — | '+
            fmt(groups['new5']['default_reviewed_dino']['baseline']['mean'])+' | 待完成 | — |')
    for a,b in zip(groups['all35']['metrics'],groups['new5']['metrics']):
        lines.append(f"| {a['label']} | {fmt(a['baseline']['mean'])} | {fmt(a['current']['mean'])} | {fmt_delta(a['delta'])} | {fmt(b['baseline']['mean'])} | {fmt(b['current']['mean'])} | {fmt_delta(b['delta'])} |")
    if dino_complete:
        lines += ['', f"默认DINO已补完新增5集177shot的CPU RetinaFace筛选，并对{reviewed_new5['review_items']}项做分数隐藏的定位复查，确认排除{len(reviewed_new5['new_exclusions'])}项后重算角色质心；此前30集的相同输入与80项复查保持原结果。该指标是条件诊断，保留真实身份/画风失败与未决情况。覆盖及决策见JSON。"]
    else:
        lines += ['', '默认DINO的新增5集尚未做新视频可见单脸筛选和定位复核，不能把原版0.7863或官方raw cs_face混入该行。已完成同口径的此前30集仍为0.8022→0.7991（−0.0031），见原报告；这不是35集结果。']
    lines += ['', '官方指标差值由未四舍五入的逐集均值计算，展示值保留四位小数，因此尾位可能与展示值直接相减不同。',
        '', '## 本轮新增5集','', '| Episode | Shot数 | 默认DINO 原→新 | 官方raw cs_face 原→新 | 角色出现率 原→新 | 场景出现率 原→新 |', '|---|---:|---:|---:|---:|---:|']
    for r in per_episode:
        if r['group']!='new5':continue
        brief=lambda n:fmt(r['metrics'][n]['baseline'])+' → '+fmt(r['metrics'][n]['current'])
        lines.append(f"| {r['episode_id']} | {r['shots']} | {brief('cs_face_visible_single_face_reviewed')} | {brief('cs_face')} | {brief('intra_character_presence')} | {brief('intra_location_presence')} |")
    lines += ['', '全部12项官方指标为strict/skip_vlm，VLM调用0。cs_face与cs_object缺少VLM保真度门控，仍为partial_without_vlm；原版与重跑的设置一致。默认DINO为独立条件诊断。',
        '', '[35集逐集CSV](per_episode_comparison.csv) · [新增5集CSV](new5_per_episode_comparison.csv) · [JSON及有效集数](report.json) · [核验](validation.json)',
        '', '使用技能：[/root/.codex/skills/evaluate-face-injection-v2/SKILL.md](/root/.codex/skills/evaluate-face-injection-v2/SKILL.md)。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'validation.json', {'episodes':len(entries),'shots':videos_verified,
        'original_videos_verified':videos_verified,'current_videos_and_tail_hashes_verified':videos_verified,
        'original30_aggregate_exactly_reproduced':True,'canonical_mapping_and_parent_chain_verified':True,
        'zero_metric_failures':True,'vlm_calls':0,'new_official_metric_inference_calls':0,
        'new_dino_embedding_inference_calls':0,'cpu_retinaface_forward_calls':detector_calls,
        'default_current35_dino_pending':not dino_complete,'official_function_sha256':sha(gate.OFFICIAL),
        'adapter_sha256':sha(Path(__file__)),'snapshot_sha256':sha(OUT/'cohort_snapshot.json'),
        'sources':sources})
    write(OUT/'status.json',{'stage':'complete_official_and_reviewed_dino' if dino_complete else 'official_complete_dino_pending',
        'updated_utc':datetime.now(timezone.utc).isoformat(),'episodes':35,'shots':520,
        'new_complete_episodes':5,'new_complete_shots':177,'vlm_calls':0})
    if dino_complete:
        write(OUT/'new5_visible_region/status.json',{'stage':'complete_with_localization_review',
            'processed_shots':177,'expected_shots':177,'reviewed_items':reviewed_new5['review_items'],
            'exclusions':len(reviewed_new5['new_exclusions']),
            'report_sha256':sha(reviewed_new5_path)})
    print('\n'.join(lines[:25]))


if __name__=='__main__':
    main()
