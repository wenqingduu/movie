"""Evaluate only the saved 30 reruns, including accepted run594 3:4/3:5.

Reuse verified official whole-episode scores; apply the standing v2 visible
single-face gate, freeze localization review, and recompute region centroids.
No media generation, Control ablation, full140 replacement, VLM or paid calls.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import qwen140_rerun_region_diagnostic as diag
import qwen140_reaggregate_partial_reruns as agg

ROOT = Path('/root/autodl-tmp/movie')
OUT = ROOT/'outputs/entitybench_rerun30_latest_evaluation_20261101'
OLD = ROOT/'outputs/entitybench_rerun_visible_region_identity_20261101'
DIAG = OUT/'visible_region'
read, write, sha = diag.read, diag.write, diag.sha


def status(stage, **kw):
    write(OUT/'status.json', {'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(), **kw})


def prepare():
    if (DIAG/'input_manifest.json').exists():
        validate(); return read(DIAG/'input_manifest.json')
    plan = read(OUT/'cohort_plan.json'); source = read(plan['source_manifest'])
    assert sha(plan['source_manifest']) == plan['source_manifest_sha256']
    assert len(source['episodes']) == 30 and source['selected_shot_count'] == 343
    episodes = copy.deepcopy(source['episodes']); sources = []; video_count = 0
    common = read(agg.ORIGINAL/'run_manifest.json')
    for item in episodes:
        eid = item['episode_id']; ep = item['episode']
        assert sha(ep['episode_path']) == ep['episode_sha256']
        sources.append({'path': ep['episode_path'], 'sha256': ep['episode_sha256']})
        if eid == plan['override_episode_id']:
            item['previous_result'] = item['result']; item['output_root'] = plan['latest594_root']
            item['evaluation'] = str(Path(item['output_root'])/'evaluation/official')
            item['result'] = str(Path(item['evaluation'])/'results'/(eid+'.json'))
            item['result_sha256'] = sha(item['result'])
            item['artifacts'] = str(Path(item['evaluation'])/'results/.work/eval_artifacts'/eid)
            item['generation_revision'] = 'Original rerun with accepted3:4/3:5 repair; other8 shots unchanged'
        generation_path = Path(item['output_root'])/'episodes'/eid/'episode_result.json'
        generation = read(generation_path)
        inputs_path = Path(item['evaluation'])/'input_manifest.json'; inputs = read(inputs_path)
        manifest_path = Path(item['evaluation'])/'results/run_manifest.json'; manifest = read(manifest_path)
        assert manifest['strict'] and manifest['skip_vlm'] and not manifest['allow_config_override']
        assert manifest['vlm_call_stats']['total_calls'] == 0
        for field in ['config', 'python_version', 'libraries', 'model_checkpoints', 'pillars', 'evaluation_scope']:
            assert manifest[field] == common[field], (eid, field)
        assert inputs['evaluator_sha256'] == sha(agg.BASE/'runtime/evaluate_benchmark_partial.py')
        assert sha(item['result']) == item['result_sha256']
        result = read(item['result']); assert result['num_shots'] == len(ep['shots']) == len(generation['shots'])
        for name in agg.LABELS:
            assert agg.metric(result, name)['n_failed'] == 0, (eid, name)
        for index, (shot, record) in enumerate(zip(ep['shots'], generation['shots'])):
            sk = shot['shot_key']; assert record['shot_key'] == sk and record['cut'] == shot['cut']
            assert sha(record['video']) == record['video_sha256'] == inputs['videos'][eid+'/'+sk]['sha256']
            assert sha(record['last_frame']) == record['last_sha256']
            if index and not shot['cut']:
                previous = generation['shots'][index-1]
                assert record['parent_video_sha256'] == previous['video_sha256']
                assert record['parent_tail_sha256'] == previous['last_sha256']
            assert (Path(item['artifacts'])/'intra_shot'/sk.replace(':','_')/'_eval.json').is_file()
            video_count += 1
        for p in [generation_path, inputs_path, manifest_path, Path(item['result'])]:
            sources.append({'path': str(p), 'sha256': sha(p)})
        baseline = agg.ORIGINAL/(eid+'.json'); sources.append({'path': str(baseline), 'sha256': sha(baseline)})
    assert len({i['episode_id'] for i in episodes}) == 30 and video_count == 343
    DIAG.mkdir(exist_ok=True)
    manifest = {'selected_episode_count': 30, 'selected_shot_count': 343, 'episodes': episodes,
        'scope': 'Only30 complete rerun episode IDs. Latest run594 overrides accepted3:4/3:5. No full140 evaluation or aggregation.'}
    write(DIAG/'input_manifest.json', manifest)
    shutil.copy2(OLD/'required_span_validation.json', DIAG/'required_span_validation.json')
    imported = []
    for item in episodes:
        eid = item['episode_id']; media = read(Path(item['output_root'])/'episodes'/eid/'episode_result.json')
        media = {r['shot_key']: r for r in media['shots']}
        for shot in item['episode']['shots']:
            sk = shot['shot_key']; cache = OLD/'extraction'/eid/(sk.replace(':','_')+'.json')
            if not cache.exists(): continue
            data = read(cache); evaluation = Path(item['artifacts'])/'intra_shot'/sk.replace(':','_')/'_eval.json'
            if data['evaluation_sha256'] != sha(evaluation) or data['video_sha256'] != media[sk]['video_sha256']: continue
            source_protocol = next(p for p in OLD.glob('protocol*.json') if sha(p) == data['protocol_sha256'])
            protocol = read(source_protocol)
            assert protocol['config'] == diag.gate.CONFIG and protocol['detector_sha256'] == sha(diag.faces.WEIGHT)
            for row in data['rows']:
                if row['original_crop']: assert sha(row['original_crop']) == row['original_crop_sha256']
            dest = DIAG/'extraction'/eid/cache.name; shutil.copy2(cache, dest) if dest.parent.exists() else (dest.parent.mkdir(parents=True), shutil.copy2(cache, dest))
            for row in data['rows']:
                context = OLD/'contexts'/eid/(sk.replace(':','_')+f"_f{row['sample_idx']}.jpg")
                if context.exists():
                    target = DIAG/'contexts'/eid/context.name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(context, target)
            imported.append({'path': str(dest), 'source': str(cache), 'sha256': sha(cache),
                             'protocol': str(source_protocol), 'protocol_sha256': sha(source_protocol)})
    code = [Path(__file__).resolve(), ROOT/'pretest/qwen140_rerun_region_diagnostic.py',
            ROOT/'pretest/evaluate_entitybench_qwen140_region_face_gate.py',
            ROOT/'pretest/evaluate_entitybench_face_only_dino_qwen140.py',
            agg.BASE/'runtime/evaluate_benchmark_partial.py', agg.BASE/'runtime/evaluate_benchmark_official.py']
    for p in code: sources.append({'path': str(p), 'sha256': sha(p)})
    write(OUT/'frozen_inputs.json', {'sources': sources, 'manifest_sha256': sha(DIAG/'input_manifest.json'),
        'imported_detector_caches': imported, 'vlm_calls': 0, 'video_count': video_count})
    status('inputs_verified', episodes=30, shots=343, cached_detection_shots=len(imported))
    return manifest


def validate():
    frozen = read(OUT/'frozen_inputs.json')
    assert sha(DIAG/'input_manifest.json') == frozen['manifest_sha256']
    for item in frozen['sources']: assert sha(item['path']) == item['sha256'], item['path']
    for item in frozen['imported_detector_caches']:
        assert sha(item['path']) == item['sha256']
        assert sha(item['protocol']) == item['protocol_sha256']


def official_report():
    p = read(DIAG/'input_manifest.json'); mod = diag.gate.load_official_pure()
    old, new = [], []; split = {t: [] for t in ['easy','medium','hard']}; per_episode = []
    for item in p['episodes']:
        ep = item['episode']; eid = ep['episode_id']; tier = ep['difficulty']; split[tier].append(eid)
        a, b = read(agg.ORIGINAL/(eid+'.json')), read(item['result'])
        old.append(mod._episode_result_from_json(eid, tier, a)); new.append(mod._episode_result_from_json(eid, tier, b))
        values = {name: {'baseline': agg.metric(a,name)['value'], 'current': agg.metric(b,name)['value']}
                  for name in agg.LABELS}
        per_episode.append({'episode_id': eid, 'run': ep['run'], 'shots': len(ep['shots']), 'metrics': values})
    result = {'scope': p['scope'], 'baseline': mod.aggregate_results(old, split),
        'current': mod.aggregate_results(new, split), 'per_episode': per_episode, 'vlm_calls': 0}
    write(OUT/'official_report.json', result); write(OUT/'split.json', split)
    return result


def score(final=False):
    validate(); p = read(DIAG/'input_manifest.json'); diag.OUT = DIAG
    old, historical = diag.historical(); oldmap = {diag.key(r): r for r in old}; currentmap = {}
    for item in p['episodes']:
        eid = item['episode_id']; files = list((DIAG/'extraction'/eid).glob('*.json'))
        assert len(files) == len(item['episode']['shots']), (eid,len(files))
        for f in files:
            data = read(f)
            for row in data['rows']:
                if row['original_crop']: assert sha(row['original_crop']) == row['original_crop_sha256']
                currentmap[diag.key(row)] = row
    selected_ids = {i['episode_id'] for i in p['episodes']}
    oldmap = {k: v for k,v in oldmap.items() if k[0] in selected_ids}
    assert set(oldmap) == set(currentmap)
    carry = {k:v for k,v in historical.items() if k in currentmap and v['original_crop_sha256'] == currentmap[k]['original_crop_sha256']}
    decisions = {}; review = None
    if final:
        review = read(DIAG/'review/decisions.json'); selection = read(DIAG/'review/manifest.json')
        assert review['manifest_sha256'] == sha(DIAG/'review/manifest.json')
        assert len(review['decisions']) == len(selection['rows'])
        by_id = {r['review_id']: r for r in selection['rows']}
        assert {r['review_id'] for r in review['decisions']} == set(by_id)
        for row in review['decisions']:
            original = by_id[row['review_id']]
            assert diag.key(row) == diag.key(original)
            assert row['original_crop_sha256'] == original['original_crop_sha256']
            assert row['decision'] in ['keep','exclude'] and row['evidence'] and row['reason']
            if row['decision'] == 'exclude': decisions[diag.key(row)] = row
    rejected = {**carry, **decisions}; mod = diag.gate.load_official_pure()
    base_results, current_results, scored, details = [], [], [], []
    for item in p['episodes']:
        ep = item['episode']; eid = ep['episode_id']
        a, ae = diag.compute(ep, read(agg.ORIGINAL/(eid+'.json')), oldmap, historical, mod)
        b, be = diag.compute(ep, read(item['result']), currentmap, rejected, mod)
        original = read(diag.gate.OUT/'remaining_low_score_review/recomputed/results'/(eid+'.json'))['metrics']['cs_face']
        assert a.metrics['cs_face'].n_evaluated == original['n_evaluated']
        if original['value'] is not None: assert abs(a.metrics['cs_face'].value-original['value']) < 1e-10
        base_results.append(a); current_results.append(b)
        for name,data in be.items():
            for app in data['per_shot_sim_to_centroid']:
                row = dict(currentmap[eid,name,app['shot']]); row['similarity'] = app['sim_to_centroid']; scored.append(row)
        details.append({'episode_id': eid, 'baseline': a.metrics['cs_face'].to_json(),
                        'current': b.metrics['cs_face'].to_json(), 'current_entities': be})
    split = read(OUT/'split.json')
    report = {'metric': 'cs_face_visible_single_face_reviewed' if final else 'cs_face_visible_single_face_auto',
        'stage': 'complete_with_localization_review' if final else 'automatic_pending_localization_review',
        'episodes': 30, 'shots': 343, 'baseline': mod.aggregate_results(base_results,split),
        'current': mod.aggregate_results(current_results,split), 'per_episode': details,
        'coverage': {'scheduled_role_shots': len(currentmap),
            'baseline_present': sum(r['original_status']=='present' for r in oldmap.values()),
            'current_present': sum(r['original_status']=='present' for r in currentmap.values()),
            'baseline_auto_eligible': sum(r['kept'] for r in oldmap.values()),
            'current_auto_eligible': sum(r['kept'] for r in currentmap.values()),
            'baseline_reviewed_retained': sum(r['kept'] and k not in historical for k,r in oldmap.items()),
            'current_reviewed_retained': sum(r['kept'] and k not in rejected for k,r in currentmap.items()),
            'baseline_scored': sum(r.metrics['cs_face'].n_evaluated for r in base_results),
            'current_scored': len(scored), 'current_gate_reasons': dict(Counter(r['reason'] for r in currentmap.values()))},
        'review_items': len(review['decisions']) if review else 0, 'new_exclusions': list(decisions.values()),
        'historical_exclusions_carried_by_same_crop_sha': len(carry),
        'canonical_person_region_pixels_and_dino_unchanged': True, 'dino_inference_calls': 0, 'vlm_calls': 0,
        'baseline_same30_original_episodes_not_full140_0_7863': True,
        'manifest_sha256': sha(DIAG/'input_manifest.json')}
    write(DIAG/('report.json' if final else 'automatic_report.json'), report)
    if not final:
        write(DIAG/'scored_crops.json', scored); diag.prepare_review(scored)
        descriptions = {i['episode_id']:i['episode']['entity_descriptions'] for i in p['episodes']}
        write(DIAG/'review/role_descriptions.json', descriptions)
        status('manual_localization_review_required', items=80, episodes=30, shots=343)
    else: publish(report)
    print(json.dumps({'stage': report['stage'], 'baseline': report['baseline']['all']['cs_face'],
                      'current': report['current']['all']['cs_face'], 'coverage': report['coverage']}), flush=True)


def publish(dino):
    raw = read(OUT/'official_report.json'); rows = []
    rows.append({'metric': dino['metric'], 'label':'人物一致性DINO（可见单脸＋定位纠错）',
        'baseline': dino['baseline']['all']['cs_face'], 'current': dino['current']['all']['cs_face']})
    for name,label in agg.LABELS.items():
        if name=='cs_face': label='官方原始人物区域DINO cs_face（单独保留）'
        rows.append({'metric':name, 'label':label, 'baseline':agg.agg_metric(raw['baseline'],name),
                     'current':agg.agg_metric(raw['current'],name)})
    write(OUT/'report.json', {'scope': raw['scope'], 'episodes':30,'shots':343,'metrics':rows,
        'dino_coverage':dino['coverage'],'localization_review_items':dino['review_items'],
        'localization_exclusions':dino['new_exclusions'],'vlm_calls':0})
    lines=['# 已重跑30集的最新评估','', '仅30个已完成重跑episode／343shot；run594/i75_j98纳入用户接受的新版3:4、3:5，其他镜头按各自已完成重跑视频。不是全140集。',
        '', '原版列为相同30集的原始完整benchmark子集，按集等权聚合；不把全140集的0.7863作为本表基线。29集核验并复用官方完整评分，更新run594整集重新评分；人物DINO补做可见单脸与定位复查。',
        '', '| 指标↑ | 原版同30集 | 最新重跑30集 | 差值 | 有效集（原/新） |', '|---|---:|---:|---:|---:|']
    for r in rows:
        a,b=r['baseline'],r['current'];av,bv=a['mean'],b['mean']
        delta=f'{bv-av:+.4f}' if av is not None and bv is not None else '—'
        fmt=lambda v:f'{v:.4f}' if v is not None else '—'
        lines.append(f"| {r['label']} | {fmt(av)} | {fmt(bv)} | {delta} | {a['n_episodes_evaluated']}/{b['n_episodes_evaluated']} |")
    lines+=['', '人物一致性默认行为项目条件诊断，保留官方原代表人物区域与DINO特征，仅筛选可见单脸及确认定位错误后重算质心；不删生成身份/画风失败。其余指标为官方strict/skip_vlm，VLM调用0；缺VLM保真度门控，官方cs_face/cs_object仍为partial_without_vlm。',
        '', f"覆盖：{json.dumps(dino['coverage'],ensure_ascii=False)}。定位复查{dino['review_items']}项，确认排除{len(dino['new_exclusions'])}项；决策见visible_region/review/decisions.json。",
        '', '[逐集CSV](per_episode_comparison.csv) · [逐集JSON](per_episode_comparison.json) · [默认DINO完整报告](visible_region/report.json) · [官方12项报告](official_report.json)',
        '', '旧输入与官方原评分均未覆盖；30集不是按本轮分数筛选的正向案例，生成版本历史修订仍保留。未完成全部视频人工审阅，不把指标完成等同于画面全面符合剧情。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    by_id={r['episode_id']:r for r in dino['per_episode']};per_episode=[];flat=[]
    for row in raw['per_episode']:
        row=copy.deepcopy(row);d=by_id[row['episode_id']]
        row['metrics'][dino['metric']]={'baseline':d['baseline']['value'],'current':d['current']['value']}
        per_episode.append(row); f={k:row[k] for k in ['episode_id','run','shots']}
        for name,pair in row['metrics'].items():
            for key,value in pair.items():f[name+'_'+key]=value
        flat.append(f)
    write(OUT/'per_episode_comparison.json',per_episode)
    with (OUT/'per_episode_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    status('complete_non_vlm_and_reviewed_dino',episodes=30,shots=343,vlm_calls=0,reviewed_crops=dino['review_items'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','extract','score','final']);args=parser.parse_args()
    if args.action=='prepare':prepare();official_report()
    elif args.action=='extract':
        validate();diag.OUT=DIAG;diag.extract('cuda')
    else:score(args.action=='final')
