"""Memory-bounded, sequential IF/DINO adapter for the figure-assets skill.

Uses the exact original paired metric functions, no diffusion pipeline imports.
Run IF and DINO as separate processes on the current 2GiB CPU container.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path('/root/autodl-tmp/movie')
OUT=ROOT/'outputs/rerun_firstframe_identity_candidates_20261101'
DIAG=ROOT/'outputs/entitybench_rerun_visible_region_identity_20261101'
sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']=''


def selected_definitions(path,names,namespace):
    nodes=[n for n in ast.parse(path.read_text()).body
           if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)


def module():
    skill=ROOT/'skills/movie-firstframe-figure-assets/scripts/select_pairs.py'
    spec=importlib.util.spec_from_file_location('bounded_skill_selector',skill)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def lean_face_app():
    import onnxruntime as ort
    import onnx
    from insightface.model_zoo import model_zoo
    from insightface.app.face_analysis import FaceAnalysis
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False
    original=model_zoo.PickableInferenceSession
    def session(path,**kwargs):
        kwargs['sess_options']=options
        return original(path,**kwargs)
    model_zoo.PickableInferenceSession=session
    graph_meta=json.loads((OUT/'arcface_graph_metadata.json').read_text())
    recognition_path=ROOT/'third_party/PuLID/models/antelopev2/glintr100.onnx'
    assert hashlib.sha256(recognition_path.read_bytes()).hexdigest()==graph_meta['model_sha256']
    original_load=onnx.load
    def graph_header(path,*args,**kwargs):
        if Path(path).resolve()==recognition_path.resolve():
            return types.SimpleNamespace(graph=types.SimpleNamespace(node=[types.SimpleNamespace(name=n) for n in graph_meta['first_eight_node_names']]))
        return original_load(path,*args,**kwargs)
    onnx.load=graph_header
    try:
        app=FaceAnalysis.__new__(FaceAnalysis);app.models={}
        # FaceAnalysis normally constructs landmark/gender models before
        # discarding them. Load only the same requested detector and recognizer.
        for name in ['scrfd_10g_bnkps.onnx','glintr100.onnx']:
            model=model_zoo.get_model(str(ROOT/'third_party/PuLID/models/antelopev2'/name),providers=['CPUExecutionProvider'])
            app.models[model.taskname]=model
        app.det_model=app.models['detection']
        # Providers are explicitly CPU. Nonnegative ctx avoids rebuilding each
        # already-CPU ORT session a second time in prepare's set_providers call.
        app.prepare(ctx_id=0,det_size=(640,640))
        assert all(m.session.get_providers()==['CPUExecutionProvider'] for m in app.models.values())
        return app
    finally:
        model_zoo.PickableInferenceSession=original
        onnx.load=original_load


def face_phase(batch_size=8):
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(1)
    skill=module()
    multi={'cv2':cv2,'np':np,'Image':Image}
    selected_definitions(ROOT/'multishot/multi_face_reference.py',{'detected_faces'},multi)
    pair={'argparse':argparse,'Path':Path,'np':np,'Image':Image,'json':json,
          'detected_faces':multi['detected_faces']}
    selected_definitions(ROOT/'pretest/evaluate_qwen_multiface_pair.py',
                         {'_embedding','_bbox','_iou','_match_by_bbox','run'},pair)
    roots=[Path(i['output_root']) for i in json.loads((DIAG/'input_manifest.json').read_text())['episodes']]
    rows,excluded=skill.candidates(roots,128,'any')
    # Figure priority is fixed by geometry and schedule, before reading final
    # scores. Keep the entire candidate population and partial pilot diagnostics.
    pool=list(rows)
    rows=[r for r in rows if r['preview_face_height']>=224 or r.get('active_injection_steps')==6]
    for row in pool:
        if row not in rows:excluded.append({'id':row['id'],'reason':'Deferred from resource-bounded figure priority; not selected by outcome'})
    skill.write(OUT/'input_candidates.json',{'candidates':rows,'excluded':excluded,
        'all_128px_candidates':pool,'selection_filter_only':True,'target_mode':'any','min_preview_face_height':128,
        'priority':'preview face height>=224 or saved six-step injection; fixed without final metrics'})
    done_rows=[];todo=[]
    for row in rows:
        state=OUT/'diagnostics'/row['id']/'row_state.json'
        if state.exists():done_rows.append(skill.read(state))
        else:todo.append(row)
    if not todo:return finish_face_phase(skill,rows,done_rows,excluded)
    app=lean_face_app();cached_frames={};reference_boxes={};errors=[]
    original_get=app.get;inference_cache={}
    def cached_get(image,*args,**kwargs):
        token=(image.shape,hashlib.sha256(image.tobytes()).digest())
        if token not in inference_cache:inference_cache[token]=original_get(image,*args,**kwargs)
        return inference_cache[token]
    app.get=cached_get
    for index,row in enumerate(todo[:batch_size]):
        path=OUT/'diagnostics'/row['id']/'identity_metrics.json'
        try:
            if path.exists():metrics=skill.read(path)
            else:
                framekey=(row['control_sha256'],row['treatment_sha256'],row['prepared'])
                metrics=cached_frames.get(framekey)
                if metrics is None:
                    metrics=pair['run'](argparse.Namespace(prepared=Path(row['prepared']),control=Path(row['control']),
                        treatment=Path(row['treatment']),output=path,min_face_confidence=.5),app=app)
                    cached_frames[framekey]=metrics
                else:skill.write(path,metrics)
            row['insightface']=next(c for c in metrics['characters'] if c['name']==row['name'])
            if row['reference'] not in reference_boxes:
                faces=multi['detected_faces'](app,Image.open(row['reference']).convert('RGB'),.5)
                face=max(faces,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                reference_boxes[row['reference']]=[float(v) for v in face.bbox]
            row['reference_bbox']=reference_boxes[row['reference']]
        except Exception as exc:
            row['error']=repr(exc);errors.append({'id':row['id'],'error':repr(exc)})
        skill.write(OUT/'diagnostics'/row['id']/'row_state.json',row)
        done_rows.append(row)
        print(f'CPU InsightFace {len(done_rows)}/{len(rows)}',flush=True)
        skill.write(OUT/'progress.json',{'phase':'insightface','done':len(done_rows),'expected':len(rows),'errors':errors})
    if len(done_rows)<len(rows):return
    finish_face_phase(skill,rows,done_rows,excluded)


def finish_face_phase(skill,rows,done_rows,excluded):
    errors=[{'id':r['id'],'error':r['error']} for r in done_rows if r.get('error')]
    ranked=sorted([r for r in done_rows if 'insightface' in r],key=lambda r:-r['insightface']['delta'])
    skill.write(OUT/'ranking_insightface.json',ranked)
    skill.write(OUT/'if_phase.json',{'evaluated':len(ranked),'screened':len(rows),'excluded_count':len(excluded),'errors':errors})
    skill.write(OUT/'progress.json',{'phase':'insightface_complete','evaluated':len(ranked)})


def dino_phase():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor,AutoModel
    torch.set_num_threads(2)
    skill=module();namespace={'np':np,'Image':Image,'Path':Path,'torch':torch,'F':F,
                             'AutoImageProcessor':AutoImageProcessor,'AutoModel':AutoModel}
    selected_definitions(ROOT/'pretest/evaluate_qwen_reference_dino.py',{'DinoEncoder','_crop'},namespace)
    stub=types.ModuleType('pretest.evaluate_qwen_reference_dino');stub._crop=namespace['_crop'];sys.modules[stub.__name__]=stub
    original_loader=AutoModel.from_pretrained
    def memory_bounded_loader(*args,**kwargs):
        kwargs['low_cpu_mem_usage']=True
        return original_loader(*args,**kwargs)
    AutoModel.from_pretrained=memory_bounded_loader
    encoder=namespace['DinoEncoder'](Path('/root/autodl-tmp/models/dinov2-base'),'cpu',1)
    ranked=skill.read(OUT/'ranking_insightface.json');crop=namespace['_crop']
    for index,row in enumerate(ranked[:48]):
        path=OUT/'diagnostics'/row['id']/'dino_first_frame.json'
        try:
            if path.exists():row['dino']=skill.read(path)
            else:
                boxes={'reference':row['reference_bbox'],**{b:row['insightface']['detections'][b]['bbox'] for b in ('control','treatment')}}
                vectors=encoder.encode([crop(Image.open(row[b]).convert('RGB'),boxes[b],.35) for b in ('reference','control','treatment')])
                a,b=float(vectors[0]@vectors[1]),float(vectors[0]@vectors[2])
                row['dino']={'control':a,'treatment':b,'delta':b-a,'reference_bbox':boxes['reference'],
                    'definition':'Reference-anchored DINOv2 CLS cosine, face boxes expanded0.35; not official cs_face'}
                skill.write(path,row['dino'])
        except Exception as exc:row['dino_error']=repr(exc)
        print(f'CPU DINO {index+1}/{min(48,len(ranked))}',flush=True)
        skill.write(OUT/'progress.json',{'phase':'dino','done':index+1,'expected':min(48,len(ranked))})
    positives=[r for r in ranked if r['insightface']['delta']>0 and r.get('dino',{}).get('delta',-1)>0]
    selected=positives[:16]
    for index,row in enumerate(selected,1):
        slug=f"{index:02d}_{row['run']}_shot_{row['shot_key'].replace(':','_')}_{row['name'].lower()}"
        row['material_folder']=str((OUT/'candidates'/slug).resolve());skill.package(row,Path(row['material_folder']))
    summary=skill.read(OUT/'if_phase.json')
    skill.write(OUT/'selection.json',{**summary,'kind':'illustrative_paired_firstframe_identity_selection',
        'target_mode':'any','min_preview_face_height':128,'selected':selected,'all_ranked':ranked,
        'insightface_positive_count':sum(r['insightface']['delta']>0 for r in ranked),
        'dino_scope':'Top48 IF deltas only','official_metrics_modified':False,
        'note':'Selected positive examples, manual review pending; no cross-shot identity claim from a single frame'})
    skill.write(OUT/'progress.json',{'phase':'complete','evaluated':len(ranked),'selected':len(selected)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['if','dino']);parser.add_argument('--worker',action='store_true');parser.add_argument('--batch-size',type=int,default=8);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    sources=[Path(__file__),ROOT/'pretest/evaluate_qwen_multiface_pair.py',ROOT/'pretest/evaluate_qwen_reference_dino.py',
             ROOT/'multishot/multi_face_reference.py',ROOT/'skills/movie-firstframe-figure-assets/scripts/select_pairs.py']
    (OUT/'cpu_adapter_provenance.json').write_text(json.dumps({'version':3,'source_sha256':{
        str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'device':'cpu',
        'algorithm':'Exact existing function/class definitions; same ONNX weights, CPU arena disabled and duplicate graph parsing avoided; IF/DINO separate; no diffusion imports',
        'official_metrics_modified':False},indent=2))
    if args.action=='dino':dino_phase()
    elif args.worker:face_phase(args.batch_size)
    else:
        while True:
            subprocess.run([sys.executable,'-u',__file__,'if','--worker','--batch-size',str(args.batch_size)],check=True)
            state=json.loads((OUT/'progress.json').read_text())
            if state['phase']=='insightface_complete':break
