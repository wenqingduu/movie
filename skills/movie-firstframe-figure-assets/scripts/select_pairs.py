#!/usr/bin/env python3
"""Read-only selection of paired Qwen first frames; CPU identity diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def candidates(roots, minimum, target_mode='single'):
    from PIL import Image
    rows, excluded, seen = [], [], set()
    for root in roots:
        files = set(root.glob('firstframes/*/shot_*/result.json'))
        files.update(root.glob('experiments/*/episodes/*/shot_*/frame/injection/result.json'))
        files.update(root.glob('episodes/*/shot_*/frame/injection/result.json'))
        files.update(root.glob('*/shots/shot_*/result.json'))
        for file in sorted(files):
            record = read(file)
            if not record.get('injection_applied'):
                excluded.append({'result': str(file), 'reason': 'No actual injection'})
                continue
            targets = record.get('targets', [])
            if not targets or (target_mode == 'single' and len(targets) != 1) or (target_mode == 'multi' and len(targets) < 2):
                excluded.append({'result': str(file), 'reason': 'Outside requested prepared-target count', 'target_mode': target_mode})
                continue
            try:
                for branch in ('control', 'treatment'):
                    assert sha(record[branch]) == record[branch + '_sha256'], branch + ' hash mismatch'
                width, image_height = Image.open(record['control']).size
                assert Image.open(record['treatment']).size == (width, image_height)
            except Exception as exc:
                excluded.append({'result': str(file), 'reason': 'Integrity or missing artifact: ' + repr(exc)})
                continue
            for target in targets:
                if target['name'] not in record.get('applied_characters', []):
                    continue
                bbox = target['bbox']
                height = bbox[3] - bbox[1]
                if height < minimum:
                    excluded.append({'result': str(file), 'name': target['name'], 'reason': 'Below illustrative face-height filter', 'height': height})
                    continue
                try:
                    ref = target['reference_image']
                    reference_sha = sha(ref)
                    face3d = read(target['facelift_result'])
                    if face3d.get('source_reference_sha256'):
                        assert face3d['source_reference_sha256'] == reference_sha, '3D identity reference mismatch'
                    fingerprint = hashlib.sha256('|'.join([record['control_sha256'], record['treatment_sha256'], reference_sha]).encode()).hexdigest()
                    if fingerprint in seen:
                        excluded.append({'result': str(file), 'reason': 'Identical C/T/reference triple already included'})
                        continue
                    seen.add(fingerprint)
                    rows.append({'id': fingerprint[:20], 'source_root': str(root), 'result': str(file),
                        'run': record['run'], 'shot_key': record['shot_key'], 'name': target['name'],
                        'episode_scope': str(file.parent.parent), 'prepared': record['prepared'],
                        'control': record['control'], 'treatment': record['treatment'], 'reference': ref,
                        'reference_sha256': reference_sha, 'control_sha256': record['control_sha256'],
                        'treatment_sha256': record['treatment_sha256'], 'preview_face_height': height,
                        'prepared_target_count': len(targets),
                        'image_size': [width, image_height], 'preview_face_height_fraction': height / image_height,
                        'target_bbox': bbox, 'target': target, 'seed': record['seed'], 'prompt': record['prompt'],
                        'protocol': record.get('protocol'), 'kind': record.get('kind'),
                        'active_injection_steps': target.get('active_injection_steps')})
                except Exception as exc:
                    excluded.append({'result': str(file), 'name': target['name'], 'reason': 'Integrity or missing artifact: ' + repr(exc)})
    return rows, excluded


def face_app(project):
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    # This InsightFace version drops sess_options in get_model; inject it at
    # its session constructor within this diagnostic process, not on disk.
    original = model_zoo.PickableInferenceSession
    def bounded_session(model_path, **kwargs):
        kwargs['sess_options'] = options
        return original(model_path, **kwargs)
    model_zoo.PickableInferenceSession = bounded_session
    try:
        app = FaceAnalysis(name='antelopev2', root=str(project / 'third_party/PuLID'),
            allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    finally:
        model_zoo.PickableInferenceSession = original
    app.prepare(ctx_id=-1, det_size=(640, 640))
    assert all(model.session.get_session_options().intra_op_num_threads == 2 for model in app.models.values())
    return app


def dino_scores(row, encoder, app):
    from PIL import Image
    from multishot.multi_face_reference import detected_faces
    from pretest.evaluate_qwen_reference_dino import _crop
    faces = detected_faces(app, Image.open(row['reference']).convert('RGB'), .5)
    if not faces:
        raise ValueError('No reference face')
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    boxes = {'reference': [float(v) for v in face.bbox]}
    metrics = row['insightface']
    crops = [_crop(Image.open(row['reference']).convert('RGB'), boxes['reference'], .35)]
    for branch in ('control', 'treatment'):
        boxes[branch] = metrics['detections'][branch]['bbox']
        crops.append(_crop(Image.open(row[branch]).convert('RGB'), boxes[branch], .35))
    vectors = encoder.encode(crops)
    a, b = float(vectors[0] @ vectors[1]), float(vectors[0] @ vectors[2])
    return {'control': a, 'treatment': b, 'delta': b - a,
            'definition': 'Reference-anchored DINOv2 CLS cosine, face boxes expanded 0.35; not official cs_face',
            'reference_bbox': boxes['reference']}


def package(row, dest):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    def copy(source, name):
        source = Path(source)
        target = dest / name
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        manifest.append({'file': name, 'source': str(source), 'sha256': sha(target)})
    for field in ('reference', 'control', 'treatment'):
        copy(row[field], field + '.png')
    copy(row['target']['harmonized_reference'], 'aligned_3d.png')
    copy(row['target']['mask'], 'injection_mask.png')
    boxes = [row['insightface']['detections'][b]['bbox'] for b in ('control', 'treatment')]
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    dx, dy = (x2-x1)*.25, (y2-y1)*.25
    width, height = row['image_size']
    roi = [max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
           min(width, math.ceil(x2+dx)), min(height, math.ceil(y2+dy))]
    for branch in ('control', 'treatment'):
        Image.open(row[branch]).convert('RGB').crop(roi).save(dest / (branch + '_face.png'))
    from pretest.evaluate_qwen_reference_dino import _crop
    _crop(Image.open(row['reference']).convert('RGB'), row['dino']['reference_bbox'], .35).save(dest / 'reference_face.png')
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 18) if Path(font_path).exists() else ImageFont.load_default()
    sheet = Image.new('RGB', (1200, 850), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"{row['run']} / {row['shot_key']} / {row['name']}   Paired first-frame identity", fill='black', font=font)
    entries = [('reference_face.png','Identity reference'), ('control_face.png','Control face'),
               ('treatment_face.png','Injected face'), ('aligned_3d.png','Aligned 3D guide'),
               ('control.png','Control full frame'), ('treatment.png','Injected full frame')]
    for index,(name,label) in enumerate(entries):
        x,y=(index%3)*400,(index//3)*360+45
        draw.text((x+10,y),label,fill='black',font=font)
        im=Image.open(dest/name).convert('RGB');im=ImageOps.contain(im,(380,315))
        sheet.paste(im,(x+(400-im.width)//2,y+30+(315-im.height)//2))
    a=row['insightface'];b=row['dino']
    draw.text((12,780),f"InsightFace: {a['control_cosine']:.4f} -> {a['treatment_cosine']:.4f} ({a['delta']:+.4f})",fill='black',font=font)
    draw.text((12,810),f"Reference DINO: {b['control']:.4f} -> {b['treatment']:.4f} ({b['delta']:+.4f})",fill='black',font=font)
    sheet.save(dest/'comparison.jpg',quality=95)
    write(dest/'evidence.json', {**row, 'shared_display_roi': roi, 'copied_originals': manifest,
        'crop_note': 'C/T shown using identical union ROI and equal magnification; originals unmodified',
        'visual_review': 'pending', 'selection_scope': 'Illustrative positive case, not a benchmark aggregate'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/movie'))
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--min-face-height',type=float,default=128)
    p.add_argument('--target-mode',choices=['single','multi','any'],default='single')
    p.add_argument('--dino-top',type=int,default=24)
    p.add_argument('--top',type=int,default=8)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    # CPU-only diagnostics must not compete with an active GPU generation queue.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['MULTISHOT_INSIGHTFACE_PROVIDERS']='CPUExecutionProvider'
    os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
    sys.path.insert(0,str(args.project));sys.path.insert(0,str(args.project/'pretest'))
    import torch
    import cv2
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    from pretest.evaluate_qwen_multiface_pair import run
    rows,excluded=candidates([r.resolve() for r in args.roots],args.min_face_height,args.target_mode)
    write(args.out/'input_candidates.json',{'candidates':rows,'excluded':excluded,
        'selection_filter_only':True,'target_mode':args.target_mode,'min_preview_face_height':args.min_face_height})
    app=face_app(args.project)
    errors=[]
    for index,row in enumerate(rows):
        path=args.out/'diagnostics'/row['id']/'identity_metrics.json'
        try:
            if path.exists():metrics=read(path)
            else:metrics=run(SimpleNamespace(prepared=Path(row['prepared']),control=Path(row['control']),
                    treatment=Path(row['treatment']),output=path,min_face_confidence=.5),app=app)
            metric=next(c for c in metrics['characters'] if c['name']==row['name'])
            row['insightface']=metric
        except Exception as exc:
            row['error']=repr(exc);errors.append({'id':row['id'],'error':repr(exc)})
        if (index+1)%10==0:print(f'CPU InsightFace {index+1}/{len(rows)}',flush=True)
        write(args.out/'progress.json',{'phase':'insightface','done':index+1,'expected':len(rows),'errors':errors})
    ranked=sorted([r for r in rows if 'insightface' in r],key=lambda r:r['insightface']['delta'],reverse=True)
    write(args.out/'ranking_insightface.json',ranked)
    from pretest.evaluate_qwen_reference_dino import DinoEncoder
    encoder=DinoEncoder(Path('/root/autodl-tmp/models/dinov2-base'),'cpu',6)
    for index,row in enumerate(ranked[:args.dino_top]):
        try:
            path=args.out/'diagnostics'/row['id']/'dino_first_frame.json'
            row['dino']=read(path) if path.exists() else dino_scores(row,encoder,app)
            if not path.exists():write(path,row['dino'])
        except Exception as exc:row['dino_error']=repr(exc)
        print(f'CPU DINO {index+1}/{min(len(ranked),args.dino_top)}',flush=True)
    # Both signs positive for an illustration; keep all negatives/disagreements in the ranking.
    positives=[r for r in ranked if r['insightface']['delta']>0 and r.get('dino',{}).get('delta',-1)>0]
    selected=positives[:args.top]
    for index,row in enumerate(selected,1):
        slug=f"{index:02d}_{row['run']}_shot_{row['shot_key'].replace(':','_')}_{row['name'].lower()}"
        row['material_folder']=str((args.out/'candidates'/slug).resolve());package(row,Path(row['material_folder']))
    write(args.out/'selection.json',{'kind':'illustrative_paired_firstframe_identity_selection',
        'evaluated_count':len(ranked),'screened_count':len(rows),'excluded_count':len(excluded),
        'target_mode':args.target_mode,
        'min_preview_face_height':args.min_face_height,'insightface_positive_count':sum(r['insightface']['delta']>0 for r in ranked),
        'dino_scope':f'Top {args.dino_top} by InsightFace delta only; not all candidates',
        'selected':selected,'all_ranked':ranked,'errors':errors,'official_metrics_modified':False,
        'note':'Large-face positive examples deliberately selected for explanation; not representative performance. Manual visual review pending. First-frame reference likeness is not proof of cross-shot consistency.'})
    write(args.out/'progress.json',{'phase':'complete','evaluated':len(ranked),'selected':len(selected),'errors':errors})
    print(json.dumps({'out':str(args.out),'evaluated':len(ranked),'selected':len(selected)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
