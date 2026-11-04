"""All completed reruns: paired 3D-injection diagnostics, no figure filter.

This only evaluates saved generated keyframes. Route B evaluates its generated
transition target, not the later decoded main-video start. No media regeneration.
Separate bounded IF workers precede one DINO worker; no repair agent or API.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/movie')
OUT = ROOT/'outputs/entitybench_firstframe_ablation_rerun30_20261101'
FIG = ROOT/'outputs/rerun_firstframe_identity_candidates_20261101'
SOURCE = ROOT/'outputs/entitybench_rerun_visible_region_identity_20261101/input_manifest.json'


def read(p): return json.loads(Path(p).read_text())
def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def frozen(p, value):
    if Path(p).exists(): assert read(p) == value, f'Frozen input changed: {p}'
    else: write(p, value)
def token(job): return sha(job['result'])
def state_path(job): return OUT/'insightface'/f"{job['id']}.json"
def dino_path(job, name):
    return OUT/'dino'/f"{job['id']}_{hashlib.sha256(name.encode()).hexdigest()[:12]}.json"


def initialize():
    source = read(SOURCE); jobs = []; counts = Counter(); slots = []
    for item in source['episodes']:
        ep = item['episode']; base = Path(item['output_root'])/'episodes'/ep['episode_id']
        generated = read(base/'episode_result.json')
        media = {s['shot_key']:s for s in generated['shots']}
        for shot in ep['shots']:
            key = shot['shot_key']; actual = media[key]
            record = base/f"shot_{key.replace(':','_')}"/'frame/injection/result.json'
            slot = {'episode_id':ep['episode_id'],'run':ep['run'],'shot':key,
                    'route':actual['source_kind'],'skip_reason':actual.get('injection_skip_reason')}
            counts['all_shots'] += 1
            if not record.exists():
                slot['scope'] = 'no_independent_generated_pair'
                counts['without_generated_pair'] += 1; slots.append(slot); continue
            r = read(record)
            for b in ('control','treatment'): assert sha(r[b]) == r[b+'_sha256']
            slot.update(scope='injected' if r['injection_applied'] else 'control_reuse',
                        result=str(record),result_sha256=sha(record),
                        skipped_characters=r.get('skipped_characters',[]),
                        target_failures=r.get('target_failures',[]))
            counts['saved_pairs'] += 1
            if not r['injection_applied']:
                assert r['control_sha256'] == r['treatment_sha256']
                counts['whole_pair_control_reuse'] += 1; slots.append(slot); continue
            assert {t['name'] for t in r['targets']} == set(r['applied_characters'])
            refs = {t['name']:sha(t['reference_image']) for t in r['targets']}
            for t in r['targets']:
                model = read(t['facelift_result'])
                if model.get('source_reference_sha256'):
                    assert model['source_reference_sha256'] == refs[t['name']]
            job = {**slot,'id':hashlib.sha256((ep['episode_id']+'|'+key).encode()).hexdigest()[:20],
                   'control':r['control'],'treatment':r['treatment'],
                   'control_sha256':r['control_sha256'],'treatment_sha256':r['treatment_sha256'],
                   'prepared':r['prepared'],'prepared_sha256':sha(r['prepared']),
                   'targets':r['targets'],'reference_sha256':refs,
                   'stage':'transition_target' if actual['source_kind']=='single_pass_route_B' else 'shot_start'}
            jobs.append(job);slots.append(slot)
            counts['injected_images'] += 1; counts['injected_role_pairs'] += len(r['targets'])
    assert len(source['episodes']) == 30 and counts['all_shots'] == 343
    frozen(OUT/'manifest.json', {'source':str(SOURCE),'source_sha256':sha(SOURCE),
        'episodes':30,'counts':dict(counts),'jobs':jobs,'all_shot_slots':slots,
        'selection':'Every applied role in every complete rerun; no face-size or outcome filter',
        'primary_scope':'Actual-injection paired generated keyframes, per role-image mean',
        'control_reuse_scope':'Identical saved images, zero paired delta; not counted as actual injections; absolute scores not inferred',
        'no_independent_pair_scope':'A inherits decoded previous tail; two legacy unpaired frames retained in coverage',
        'B_scope':'Generated transition-target image, not actual decoded main-shot start',
        'official_metrics_modified':False,'api_calls':0})
    sources = [Path(__file__),ROOT/'pretest/qwen140_rerun_figure_screen_cpu.py',
        ROOT/'pretest/evaluate_qwen_multiface_pair.py',ROOT/'pretest/evaluate_qwen_reference_dino.py',
        ROOT/'multishot/multi_face_reference.py']
    frozen(OUT/'protocol.json', {'manifest_sha256':sha(OUT/'manifest.json'),
        'source_sha256':{str(p):sha(p) for p in sources},'device':'cpu',
        'recognition_model':str(ROOT/'third_party/PuLID/models/antelopev2/glintr100.onnx'),
        'recognition_sha256':sha(ROOT/'third_party/PuLID/models/antelopev2/glintr100.onnx'),
        'detector_sha256':sha(ROOT/'third_party/PuLID/models/antelopev2/scrfd_10g_bnkps.onnx'),
        'face_confidence':.5,'matching':'Original prepared target roster, spatial one-to-one IoU>=.25, never final identity maximum',
        'reference_face':'Largest reference-portrait detection',
        'dino_model':'/root/autodl-tmp/models/dinov2-base','face_expansion':.35,
        'dino_definition':'Original DinoEncoder normalized pooled/CLS feature cosine to same episode-local reference; not official cs_face',
        'missing':'Record failures; no zero imputation, no score-based exclusion',
        'cached_metrics':'Validate exact input paths and current result/image/prepared hashes; figures use same frozen evaluator definitions'})
    print(json.dumps(dict(counts)), flush=True)


def check_job(job):
    assert token(job) == job['result_sha256']
    assert sha(job['prepared']) == job['prepared_sha256']
    for b in ('control','treatment'): assert sha(job[b]) == job[b+'_sha256']
    for t in job['targets']: assert sha(t['reference_image']) == job['reference_sha256'][t['name']]


def load_cache_index():
    index = {}
    for folder in [FIG, ROOT/'outputs/firstframe_identity_figure_multiface_candidates_20261030',
                   ROOT/'outputs/firstframe_identity_figure_candidates_20261030']:
        for file in folder.glob('diagnostics/*/identity_metrics.json'):
            r = read(file)
            if r.get('characters') and r.get('control') and r.get('treatment'):
                index[str(Path(r['control']).resolve()),str(Path(r['treatment']).resolve())] = (file,r)
    boxes = {}
    for file in (FIG/'diagnostics').glob('*/row_state.json'):
        row = read(file)
        if row.get('reference_bbox'): boxes[row['reference_sha256']] = row['reference_bbox']
    return index, boxes


def face_batch(batch_size):
    import cv2
    import numpy as np
    from PIL import Image
    import qwen140_rerun_figure_screen_cpu as adapter
    cv2.setNumThreads(1)
    jobs = read(OUT/'manifest.json')['jobs']
    todo = [j for j in jobs if not state_path(j).exists()]
    if not todo: return
    multi = {'cv2':cv2,'np':np,'Image':Image}
    adapter.selected_definitions(ROOT/'multishot/multi_face_reference.py',{'detected_faces'},multi)
    pair = {'argparse':argparse,'Path':Path,'np':np,'Image':Image,'json':json,
            'detected_faces':multi['detected_faces']}
    adapter.selected_definitions(ROOT/'pretest/evaluate_qwen_multiface_pair.py',
        {'_embedding','_bbox','_iou','_match_by_bbox','run'},pair)
    cache, boxes = load_cache_index()
    app = adapter.lean_face_app()
    for job in todo[:batch_size]:
        check_job(job)
        state = {'job_id':job['id'],'input_result_sha256':job['result_sha256']}
        try:
            key = str(Path(job['control']).resolve()),str(Path(job['treatment']).resolve())
            if key in cache:
                file, metrics = cache[key]
                state['cached_metric_source'] = str(file); state['cached_metric_sha256'] = sha(file)
            else:
                metrics = pair['run'](argparse.Namespace(prepared=Path(job['prepared']),control=Path(job['control']),
                    treatment=Path(job['treatment']),output=OUT/'raw_if'/f"{job['id']}.json",min_face_confidence=.5),app=app)
            assert {c['name'] for c in metrics['characters']} == {t['name'] for t in job['targets']}
            state['metrics'] = metrics; state['reference_boxes'] = {}
            for target in job['targets']:
                reference_sha = job['reference_sha256'][target['name']]
                if reference_sha not in boxes:
                    faces = multi['detected_faces'](app,Image.open(target['reference_image']).convert('RGB'),.5)
                    if not faces: raise RuntimeError(f"No reference face: {target['name']}")
                    face = max(faces,key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    boxes[reference_sha] = [float(v) for v in face.bbox]
                state['reference_boxes'][target['name']] = boxes[reference_sha]
        except Exception as exc:
            state['error'] = repr(exc)
        write(state_path(job),state)
        completed = sum(state_path(j).exists() for j in jobs)
        write(OUT/'status.json',{'phase':'insightface','completed_images':completed,'expected_images':len(jobs),
            'episode_id':job['episode_id'],'shot':job['shot'],'last_error':state.get('error')})
        print(json.dumps(read(OUT/'status.json'),ensure_ascii=False),flush=True)


def dino():
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    import qwen140_rerun_figure_screen_cpu as adapter
    torch.set_num_threads(2)
    namespace = {'np':np,'Image':Image,'Path':Path,'torch':torch,'F':F,
                 'AutoImageProcessor':AutoImageProcessor,'AutoModel':AutoModel}
    adapter.selected_definitions(ROOT/'pretest/evaluate_qwen_reference_dino.py',{'DinoEncoder','_crop'},namespace)
    original = AutoModel.from_pretrained
    def loader(*args,**kwargs):
        kwargs['low_cpu_mem_usage'] = True
        return original(*args,**kwargs)
    AutoModel.from_pretrained = loader
    encoder = namespace['DinoEncoder'](Path('/root/autodl-tmp/models/dinov2-base'),'cpu',1)
    reference_vectors = {};cache = {}; completed = 0
    for file in (FIG/'diagnostics').glob('*/row_state.json'):
        row = read(file); metric = file.parent/'dino_first_frame.json'
        if metric.exists(): cache[row['control_sha256'],row['treatment_sha256'],row['reference_sha256']] = (metric,read(metric))
    for job in read(OUT/'manifest.json')['jobs']:
        check_job(job); state = read(state_path(job))
        assert state['input_result_sha256'] == job['result_sha256']
        if state.get('error'): continue
        characters = {c['name']:c for c in state['metrics']['characters']}
        for target in job['targets']:
            name = target['name']; dest = dino_path(job,name); completed += 1
            if dest.exists():
                assert read(dest)['input_result_sha256'] == job['result_sha256']; continue
            row = {'job_id':job['id'],'name':name,'input_result_sha256':job['result_sha256']}
            try:
                ref_sha = job['reference_sha256'][name]; key = job['control_sha256'],job['treatment_sha256'],ref_sha
                if key in cache:
                    file, cached = cache[key]; row.update(control=cached['control'],treatment=cached['treatment'],
                        delta=cached['delta'],cached_metric_source=str(file),cached_metric_sha256=sha(file))
                else:
                    crop = namespace['_crop']
                    if ref_sha not in reference_vectors:
                        reference_vectors[ref_sha] = encoder.encode([crop(Image.open(target['reference_image']).convert('RGB'),state['reference_boxes'][name],.35)])[0]
                    vectors = encoder.encode([crop(Image.open(job[b]).convert('RGB'),characters[name]['detections'][b]['bbox'],.35) for b in ('control','treatment')])
                    a,b = [float(reference_vectors[ref_sha]@v) for v in vectors]
                    row.update(control=a,treatment=b,delta=b-a)
            except Exception as exc: row['error'] = repr(exc)
            write(dest,row)
            write(OUT/'status.json',{'phase':'dino','completed_role_pairs':completed,'expected_role_pairs':595,
                'episode_id':job['episode_id'],'shot':job['shot'],'name':name,'last_error':row.get('error')})
            if completed%5 == 0: print(json.dumps(read(OUT/'status.json'),ensure_ascii=False),flush=True)


def summarize():
    manifest = read(OUT/'manifest.json'); rows = []; pending = []; failures = []
    for job in manifest['jobs']:
        if not state_path(job).exists(): pending.append(job['id']); continue
        check_job(job); state = read(state_path(job))
        assert state['input_result_sha256'] == job['result_sha256']
        if state.get('error'):
            failures.append({'episode_id':job['episode_id'],'shot':job['shot'],'characters':[t['name'] for t in job['targets']],
                             'stage':'insightface','error':state['error']});continue
        for c in state['metrics']['characters']:
            dest = dino_path(job,c['name'])
            if not dest.exists(): pending.append(job['id']+'|'+c['name']);continue
            d = read(dest)
            if d.get('error'):
                failures.append({'episode_id':job['episode_id'],'shot':job['shot'],'name':c['name'],'stage':'dino','error':d['error']});continue
            rows.append({'episode_id':job['episode_id'],'run':job['run'],'shot':job['shot'],'character':c['name'],
                'route':job['route'],'keyframe_stage':job['stage'],
                'insightface_control':c['control_cosine'],'insightface_injected':c['treatment_cosine'],'insightface_delta':c['delta'],
                'reference_dino_control':d['control'],'reference_dino_injected':d['treatment'],'reference_dino_delta':d['delta']})
    def mean(items):
        return {k:statistics.mean(r[k] for r in items) for k in rows[0] if k.startswith(('insightface_','reference_dino_'))} if items else {}
    by_episode = defaultdict(list)
    for r in rows: by_episode[r['episode_id']].append(r)
    report = {'status':'incomplete' if pending else 'complete_with_unscorable_pairs' if failures else 'complete',
        'manifest_sha256':sha(OUT/'manifest.json'),'coverage':manifest['counts'],
        'jointly_scored_role_pairs':len(rows),'pending_count':len(pending),'failures':failures,
        'primary_role_pair_micro':mean(rows),'episode_macro':mean([mean(v) for v in by_episode.values()]),
        'per_episode':{k:{'scored_role_pairs':len(v),'means':mean(v)} for k,v in by_episode.items()},
        'primary_scope':manifest['primary_scope'],'control_reuse_scope':manifest['control_reuse_scope'],
        'selection':manifest['selection'],'B_scope':manifest['B_scope'],
        'metrics':'Reference-anchored first-keyframe IF and face DINO cosine; not official cs_face or video metrics',
        'official_metrics_modified':False}
    write(OUT/'report.json',report)
    if rows:
        with (OUT/'per_role_pair.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text = ['# 30集全部已生成关键帧：无注入/3D注入消融','',
        f"状态：{report['status']}。共30集343shot；331个已保存生成配对，311张实际注入、595个角色配对。20张整图沿用Control；10个A复用上一实际尾帧，2张早期路径没有独立3D配对。",'',
        f"两项均成功计分的角色配对：{len(rows)}；待完成：{len(pending)}；不可评测记录：{len(failures)}。",'',
        '所有角色配对均在分数之前确定；不使用绘图用128px筛选、6步优先或正向筛选。主表对两项都可计算的同一批实际注入角色—图像配对等权平均；负增益保留。整图Control复用的差值为0，但未凭空填入绝对相似度。', '',
        'B路线评测的是所生成过渡目标关键帧，C/硬切评测新首帧。不是全部视频实际首帧或完整视频消融；相同条件指每个关键帧自身Control与Treatment，并非重新跑了整条无注入episode。','']
    if not pending and rows:
        m = mean(rows); text += ['| 设置 | InsightFace ↑ | 参考脸DINO ↑ |','|---|---:|---:|',
            f"| Control（无3D注入） | {m['insightface_control']:.4f} | {m['reference_dino_control']:.4f} |",
            f"| 3D注入 | {m['insightface_injected']:.4f} | {m['reference_dino_injected']:.4f} |",
            f"| 配对差值 | {m['insightface_delta']:+.4f} | {m['reference_dino_delta']:+.4f} |",'']
    text += ['模型与阈值、源文件SHA在protocol.json；逐角色明细在per_role_pair.csv，全部不可评测及缺失原因在report.json。DINO为同集参考脸扩展0.35后的特征余弦，不能与0.7863人物区域诊断或官方cs_face混用。', '',
        '使用evaluate-face-injection skill（/root/.codex/skills/evaluate-face-injection/SKILL.md）的配对规则；只读原媒体、原门控和原官方指标，不调用API，不重新生成。']
    (OUT/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'status':report['status'],'scored':len(rows),'pending':len(pending),'failures':len(failures)},ensure_ascii=False),flush=True)
    return report


def run():
    initialize(); jobs = read(OUT/'manifest.json')['jobs']
    while any(not state_path(j).exists() for j in jobs):
        subprocess.run([sys.executable,'-u',__file__,'if','--batch-size','4'],check=True,cwd=ROOT)
    subprocess.run([sys.executable,'-u',__file__,'dino'],check=True,cwd=ROOT)
    report = summarize();write(OUT/'status.json',{'phase':report['status'],'scored':report['jointly_scored_role_pairs'],'failures':len(report['failures'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('action',choices=['init','if','dino','report','run']);parser.add_argument('--batch-size',type=int,default=4)
    args = parser.parse_args(); OUT.mkdir(exist_ok=True)
    if args.action == 'init': initialize()
    elif args.action == 'if': face_batch(args.batch_size)
    elif args.action == 'dino': dino()
    elif args.action == 'report': summarize()
    else: run()
