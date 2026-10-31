#!/usr/bin/env python3
"""Validate and aggregate full140 with complete multiface rerun episodes replaced.

No media generation or neural inference. The mixed snapshot is deliberately
distinct from a uniform full140 rerun of the new method.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140'
ORIGINAL = BASE / 'evaluation/official_parallel_v2'
OUT = ROOT / 'outputs/entitybench_qwen140_partial_rerun_full140_20261030'
COHORTS = ['entitybench_qwen140_multiface_weak4_20261028',
           'entitybench_qwen140_multiface_focus4_20261029',
           'entitybench_qwen140_multiface_overnight_extra_20261029']
PILOT = ROOT / 'outputs/entitybench_qwen140_run700_multiface_episode_20261028'
LABELS = {'subject_consistency':'主体一致性', 'temporal_flickering':'时间稳定性',
    'motion_smoothness':'运动平滑度', 'dynamic_degree':'动态程度',
    'aesthetic_quality':'美学质量', 'imaging_quality':'成像质量（0–100）',
    'intra_character_presence':'角色出现率', 'intra_object_presence':'物体出现率',
    'intra_location_presence':'场景出现率', 'cs_face':'人物区域cs_face',
    'cs_object':'物体区域cs_object', 'cs_transition_boundary':'接续边界一致性'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def replacements(additional_cohort=None):
    found = {}
    for name in COHORTS:
        root = ROOT/'outputs'/name
        plan = read(root/'cohort.json')
        for item in plan['episodes']:
            eid = item['episode_id'];directory = Path(item['output_root'])
            assert eid not in found, 'Duplicate replacement: ' + eid
            assert (directory/'complete.json').is_file(), 'Incomplete replacement: ' + eid
            found[eid] = {'root':directory, 'evaluation':directory/'evaluation/weak4_v1',
                'cohort':str(root/'cohort.json'), 'group':name,
                'generation_revision':plan.get('experiment_revision', plan.get('resume_revision')),
                'version_note':'First weak4 retained earlier prefixes across documented prompt/matching revisions'
                    if name == COHORTS[0] else 'Unified frozen pipeline from first shot'}
    pilot = read(PILOT/'rerun_manifest.json');eid = pilot['episode_id']
    assert eid not in found
    found[eid] = {'root':PILOT, 'evaluation':PILOT/'evaluation/run700_multiface_episode_v1',
        'cohort':str(PILOT/'rerun_manifest.json'), 'group':'run700_multiface_pilot',
        'generation_revision':'Pilot with retained first-six prefix and later all-face matching fix',
        'version_note':'Pilot is not uniformly the latest scheduling/matching revision'}
    assert len(found) == 14
    if additional_cohort is not None:
        plan = read(additional_cohort)
        assert set(plan['excluded_completed_episode_ids']) == set(found)
        for item in plan['episodes']:
            eid = item['episode_id'];directory = Path(item['output_root'])
            assert eid not in found, 'Duplicate replacement: ' + eid
            if not (directory/'complete.json').is_file():continue
            found[eid] = {'root':directory, 'evaluation':directory/'evaluation/weak4_v1',
                'cohort':str(additional_cohort), 'group':'remaining126',
                'generation_revision':plan['experiment_revision'],
                'version_note':'Current frozen pipeline, whole-episode rerun'}
    return found


def metric(raw, name):
    return (raw['vbench_means'] if name in raw['vbench_means'] else raw['metrics'])[name]


def agg_metric(report, name):
    return report['all']['vbench'][name] if name in report['all']['vbench'] else report['all'][name]


def main(out, additional_cohort=None):
    out.mkdir(parents=True, exist_ok=True)
    write(out/'status.json', {'status':'validating_inputs_and_aggregating', 'neural_inference':False})
    manifest = read(BASE/'manifest.json');split = read(BASE/'split_full140.json')
    episodes = manifest['episodes'];ids = {e['episode_id'] for e in episodes}
    assert len(ids) == len(episodes) == 140
    assert sum(len(e['shots']) for e in episodes) == 2491
    tiers = ('easy','medium','hard')
    assert {eid for tier in tiers for eid in split[tier]} == ids
    assert sum(len(split[tier]) for tier in tiers) == 140
    if 'all' in split:assert len(split['all']) == 140 and set(split['all']) == ids
    selected = replacements(additional_cohort);assert set(selected) <= ids
    replaced_count = len(selected)
    expected_changed_shots = sum(len(ep['shots']) for ep in episodes if ep['episode_id'] in selected)
    generation = read(BASE/'generation_report.json')
    jobs = {(j['episode_id'],j['shot_key']):j for j in generation['jobs']}
    assert len(jobs) == len(generation['jobs']) == 2491
    original_fingerprint = read(ORIGINAL/'input_fingerprint.json')['inputs']
    evaluator = BASE/'runtime/evaluate_benchmark_partial.py'
    assert sha(evaluator) == original_fingerprint['evaluator_sha256']
    assert sha(BASE/'manifest.json') == original_fingerprint['manifest_sha256']
    assert sha(BASE/'split_full140.json') == original_fingerprint['split_sha256']
    common = read(ORIGINAL/'run_manifest.json')
    checked_manifests = set()
    hash_cache = {}
    def verify(path, expected):
        key = str(Path(path).resolve())
        if key not in hash_cache:hash_cache[key] = sha(path)
        assert hash_cache[key] == expected, path
    def check_scoring_manifest(path):
        if str(path) in checked_manifests:return
        data = read(path)
        assert data['strict'] and data['skip_vlm'] and not data['allow_config_override']
        assert data['vlm_call_stats']['total_calls'] == 0
        for field in ['config','python_version','libraries','model_checkpoints','pillars','evaluation_scope']:
            assert data[field] == common[field], (str(path),field,'Scoring provenance changed')
        checked_manifests.add(str(path))
    check_scoring_manifest(ORIGINAL/'run_manifest.json')
    old_rows, current_rows, inventory, media = [], [], [], {}
    originals_raw, current_raw = [], []
    changed_shots = 0
    for index,ep in enumerate(episodes,1):
        eid = ep['episode_id'];tier = ep['difficulty']
        assert eid in split[tier]
        verify(ep['episode_path'],ep['episode_sha256'])
        assert original_fingerprint['scripts'][eid] == ep['episode_sha256']
        hits = list((ORIGINAL/'.shards').glob(f'worker_*/results/{eid}.json'))
        assert len(hits) == 1, (eid,'Missing or duplicate baseline result')
        baseline_path = hits[0];baseline_raw = read(baseline_path)
        assert sha(ORIGINAL/(eid+'.json')) == sha(baseline_path)
        check_scoring_manifest(baseline_path.parent/'run_manifest.json')
        version = selected.get(eid)
        root = version['root'] if version else BASE
        evaluation = version['evaluation'] if version else ORIGINAL
        candidate_path = evaluation/'results'/(eid+'.json') if version else baseline_path
        candidate_raw = read(candidate_path)
        assert candidate_raw['episode_id'] == baseline_raw['episode_id'] == eid
        assert candidate_raw['num_shots'] == baseline_raw['num_shots'] == len(ep['shots'])
        for raw in (baseline_raw,candidate_raw):
            for name in LABELS:
                stat = metric(raw,name)
                assert stat['n_failed'] == 0, (eid,name,stat)
                if stat['value'] is not None:assert math.isfinite(stat['value'])
        generated = inputs = None
        if version:
            check_scoring_manifest(evaluation/'results/run_manifest.json')
            inputs = read(evaluation/'input_manifest.json')
            assert inputs['source_manifest_sha256'] == sha(BASE/'manifest.json')
            assert inputs['source_generation_report_sha256'] == sha(BASE/'generation_report.json')
            assert inputs['evaluator_sha256'] == sha(evaluator)
            generated = read(root/'episodes'/eid/'episode_result.json')['shots']
            assert len(generated) == len(ep['shots']) == len(inputs['videos'])
            changed_shots += len(generated)
        for j,s in enumerate(ep['shots']):
            key = (eid,s['shot_key']);job = jobs[key]
            assert job['cut'] == s['cut']
            assert original_fingerprint['videos'][eid+'/'+s['shot_key']] == job['video_sha256']
            verify(job['output_video'],job['video_sha256'])
            if not s['cut']:
                assert j > 0
                previous = jobs[eid,ep['shots'][j-1]['shot_key']]
                assert job['parent_video_sha256'] == previous['video_sha256']
                assert job['input_sha256'] == previous['last_frame_sha256']
            if generated is not None:
                record = generated[j];info = inputs['videos'][eid+'/'+s['shot_key']]
                assert record['shot_key'] == s['shot_key'] and record['cut'] == s['cut']
                assert record['baseline_video_sha256'] == info['baseline_sha256'] == job['video_sha256']
                assert record['video_sha256'] == info['sha256']
                assert Path(record['video']).resolve() == Path(info['path']).resolve()
                verify(info['path'],info['sha256']);verify(record['last_frame'],record['last_sha256'])
                if not s['cut']:
                    assert record['parent_video_sha256'] == generated[j-1]['video_sha256']
                    assert record['parent_tail_sha256'] == generated[j-1]['last_sha256']
                path,digest = info['path'],info['sha256']
            else:path,digest = job['output_video'],job['video_sha256']
            scene,shot = map(int,s['shot_key'].split(':'))
            link = out/'official_results'/eid/f'{scene:03d}_{shot:03d}.mp4'
            link.parent.mkdir(parents=True,exist_ok=True)
            if link.exists():assert link.is_symlink() and link.resolve() == Path(path).resolve()
            else:link.symlink_to(Path(path).resolve())
            media[eid+'/'+s['shot_key']] = {'path':str(link),'source':path,'sha256':digest,
                'original_sha256':job['video_sha256'],'edition':'rerun' if version else 'original'}
        for label,path in [('baseline',baseline_path),('current',candidate_path)]:
            dest = out/'evaluation'/label/'results'/(eid+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
        old_rows.append((eid,tier,baseline_raw));current_rows.append((eid,tier,candidate_raw))
        originals_raw.append(baseline_raw);current_raw.append(candidate_raw)
        inventory.append({'episode_id':eid,'run':ep['run'],'difficulty':tier,'shots':len(ep['shots']),
            'edition':'rerun' if version else 'original','source_root':str(root),
            'baseline_result':str(baseline_path),'baseline_result_sha256':sha(baseline_path),
            'selected_result':str(candidate_path),'selected_result_sha256':sha(candidate_path),
            'evaluation_manifest_sha256':sha(evaluation/'results/run_manifest.json') if version else sha(baseline_path.parent/'run_manifest.json'),
            'input_manifest_sha256':sha(evaluation/'input_manifest.json') if version else None,
            'generation_revision':version.get('generation_revision') if version else 'original_full140',
            'version_note':version.get('version_note') if version else 'Unchanged original result'})
        if index%20 == 0:print(f'Verified {index}/140 episodes',flush=True)
    assert changed_shots == expected_changed_shots and len(media) == 2491
    spec = importlib.util.spec_from_file_location('frozen_full140_reaggregate',evaluator)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    before = module.aggregate_results([module._episode_result_from_json(*v) for v in old_rows],split)
    after = module.aggregate_results([module._episode_result_from_json(*v) for v in current_rows],split)
    original_report_path = ORIGINAL/'report_qwen21_native_reference_a14b_4step_continuation_full140_no_vlm.json'
    original_report = read(original_report_path)
    for group in ['all','easy','medium','hard']:
        assert before[group] == original_report[group], 'Baseline aggregate does not reproduce: ' + group
    independent = {};table = []
    for name,label in LABELS.items():
        a,b = agg_metric(before,name),agg_metric(after,name)
        av = [metric(r,name)['value'] for r in originals_raw if metric(r,name)['value'] is not None]
        bv = [metric(r,name)['value'] for r in current_raw if metric(r,name)['value'] is not None]
        exact_a,exact_b = math.fsum(av)/len(av),math.fsum(bv)/len(bv)
        assert abs(exact_a-a['mean']) <= .000051 and abs(exact_b-b['mean']) <= .000051
        assert a['n_episodes_evaluated'] == len(av) and b['n_episodes_evaluated'] == len(bv)
        independent[name] = {'baseline_exact_episode_mean':exact_a,'current_exact_episode_mean':exact_b,
            'baseline_valid_episode_count':len(av),'current_valid_episode_count':len(bv)}
        table.append({'metric':name,'label':label,'baseline':a['mean'],'current':b['mean'],'delta':b['mean']-a['mean']})
    write(out/'split_full140.json',split)
    write(out/'evaluation/baseline_official_aggregate.json',before)
    write(out/'evaluation/current_official_aggregate.json',after)
    selection={'kind':'full140_snapshot_after_partial_episode_reruns','num_episodes':140,'num_shots':2491,
        'rerun_episodes':replaced_count,'rerun_shots':changed_shots,
        'unchanged_original_episodes':140-replaced_count,'unchanged_original_shots':2491-changed_shots,
        'additional_cohort':str(additional_cohort) if additional_cohort else None,
        'scope':'Mixed snapshot; replace all completed current multiface editions by exact episode_id regardless of gains. Not a uniform new-method full140 run.',
        'episodes':inventory,'videos':media,'original_manifest_sha256':sha(BASE/'manifest.json'),
        'original_generation_report_sha256':sha(BASE/'generation_report.json'),'original_report_sha256':sha(original_report_path),
        'evaluator_sha256':sha(evaluator),'script_sha256':sha(Path(__file__)),
        'source_scoring_manifests':sorted(checked_manifests),'neural_inference':False,'vlm_calls':0}
    write(out/'selection_manifest.json',selection)
    write(out/'comparison.json',{'scope':selection['scope'],'metrics':table,'independent_mean_validation':independent})
    per_episode = []
    csv_rows = []
    for row,a,b in zip(inventory,originals_raw,current_raw):
        values = {name:{'baseline':metric(a,name)['value'],'current':metric(b,name)['value'],
            'delta':metric(b,name)['value']-metric(a,name)['value']
                if metric(a,name)['value'] is not None and metric(b,name)['value'] is not None else None}
            for name in LABELS}
        per_episode.append({**row,'metrics':values})
        flat = {k:row[k] for k in ['episode_id','run','difficulty','edition','shots']}
        for name,pair in values.items():
            for field,value in pair.items():flat[name+'_'+field]=value
        csv_rows.append(flat)
    write(out/'per_episode_comparison.json',{'scope':selection['scope'],'episodes':per_episode})
    with (out/'per_episode_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    runtime = out/'runtime';runtime.mkdir(exist_ok=True)
    shutil.copy2(__file__,runtime/Path(__file__).name)
    for name in ['evaluate_benchmark_partial.py','evaluate_benchmark_official.py','skip_vlm.patch','evaluation_provenance.json']:
        if (BASE/'runtime'/name).exists():shutil.copy2(BASE/'runtime'/name,runtime/name)
    write(out/'validation.json',{'verified_utc':datetime.now(timezone.utc).isoformat(),
        'episode_count':140,'shot_count':2491,'replaced_episode_count':replaced_count,'replaced_shot_count':changed_shots,
        'baseline_aggregate_reproduced_exactly_for_all_tiers':True,'no_scoring_configuration_or_model_differences':True,
        'unique_media_sources_hashed':len(hash_cache),'scoring_manifests_verified':len(checked_manifests),
        'all_original_and_selected_video_hashes_verified':True,'canonical_links_verified':True,
        'original_and_rerun_parent_chains_verified':True,'vlm_calls':0,'no_neural_inference':True,
        'visual_review':'Full video visual review pending; unchanged evaluation scope partial_without_vlm'})
    lines=['# 完整140集：替换已完成重跑后的当前结果','',
        f'范围：140集／2491shot，其中{replaced_count}集／{changed_shots}shot采用完整多人3D集成重跑，其余{140-replaced_count}集／{2491-changed_shots}shot沿用原始结果。',
        '按原始full140 split，由冻结官方实现重新解序列化并集等权聚合；原始all/easy/medium/hard结果已完全复现。复用相同配置与媒体SHA核验通过的逐集评分，本次不重复神经推理；VLM调用0。','',
        ('**这是部分重跑后的混合全量快照。** ' if replaced_count < 140 else '**140集均已有多人3D集成重跑结果。** ')
        + '替换纳入全部已完成的多人3D版本，包含下降案例；run700与首批weak4保留已记录的历史前缀/修订，不声称生成版本完全统一。','',
        '| 指标↑ | 原始140集 | 替换重跑后的140集 | 差值 |','|---|---:|---:|---:|']
    for row in table:lines.append(f"| {row['label']} | {row['baseline']:.4f} | {row['current']:.4f} | {row['delta']:+.4f} |")
    lines += ['', '边界一致性仅137集适用，按官方有效episode分母处理；其余指标的有效数见comparison.json和官方聚合。缺少VLM保真度门控，cs_face/cs_object属于partial_without_vlm，不能当作已验证的人脸身份准确率。完整人工视频审阅待完成。','',
        f'## 已替换的{replaced_count}集','', '| episode | shot数 | 前后视频／指标 |','|---|---:|---|']
    html_rows=[]
    for row in inventory:
        if row['edition'] != 'rerun':continue
        relative = os.path.relpath(Path(row['source_root']),out)
        lines.append(f"| {row['episode_id']} | {row['shots']} | [视频]({relative}/review/index.html) · [指标]({relative}/report.md) |")
        html_rows.append(f'<tr><td>{html.escape(row["episode_id"])}</td><td>{row["shots"]}</td><td><a href="{relative}/review/index.html">前后视频</a> · <a href="{relative}/report.md">指标</a></td></tr>')
    lines += ['', '[全部140集逐集前后指标CSV](per_episode_comparison.csv) · [逐集JSON](per_episode_comparison.json)。', '',
        '原始140集媒体、评分及报告保持原位。完整来源表与2491个视频映射见selection_manifest.json；完整性核验见validation.json。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'index.html').write_text(f'<meta charset="utf-8"><h1>完整140集：{replaced_count}集替换为重跑结果</h1><p>140集2491shot；{replaced_count}集{changed_shots}shot重跑＋{140-replaced_count}集{2491-changed_shots}shot原版。生成版本前缀见来源清单。官方非VLM口径，未重复神经推理，VLM调用0。</p><p><a href="report.md">全部指标对比</a> · <a href="per_episode_comparison.csv">140集逐集对比CSV</a> · <a href="comparison.json">指标与分母JSON</a> · <a href="selection_manifest.json">完整来源及视频映射</a> · <a href="validation.json">核验记录</a></p><table border="1" cellpadding="7"><tr><th>已替换episode</th><th>shot数</th><th>结果</th></tr>'+''.join(html_rows)+'</table>')
    write(out/'status.json',{'status':'complete_reaggregation_only','completed_utc':datetime.now(timezone.utc).isoformat(),
        'episodes':140,'shots':2491,'replaced_episodes':replaced_count,'replaced_shots':changed_shots,'vlm_calls':0})
    print(json.dumps(table,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--additional-cohort',type=Path)
    args=parser.parse_args()
    try:main(args.out.resolve(), args.additional_cohort.resolve() if args.additional_cohort else None)
    except Exception as exc:
        write(args.out/'status.json',{'status':'validation_failed','error':repr(exc),'neural_inference':False})
        raise
