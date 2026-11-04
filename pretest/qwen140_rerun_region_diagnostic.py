"""Frozen mixed-full140 visible-face diagnostic; never edits official inputs.

RetinaFace, thresholds, tracking, original canonical pixels, saved DINO Gram
matrices and official centroid arithmetic match the original 0.7863 diagnostic.
Only new complete reruns require face detection. Historical manual exclusions
apply by crop SHA, not by role name to a regenerated crop. New manual decisions
are frozen before final scoring. Figure assets are a separate diagnostic scope.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import evaluate_entitybench_qwen140_region_face_gate as gate
import evaluate_entitybench_face_only_dino_qwen140 as faces

ROOT = Path('/root/autodl-tmp/movie')
BASE = gate.RUN
OUT = ROOT/'outputs/entitybench_rerun_visible_region_identity_20261101'
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
read, write, sha = gate.read, gate.write, gate.sha


def immutable(path, data):
    path = Path(path)
    if path.exists():
        assert read(path) == data, f'Frozen data changed: {path}'
    else:
        write(path, data)


def inputs():
    p = read(OUT/'input_manifest.json')
    for i in p['episodes']:
        assert sha(i['result']) == i['result_sha256'], i['episode_id']
    return p


def decode(path):
    reader = iio.get_reader(path, 'ffmpeg', ffmpeg_params=['-threads', '1'])
    try:
        frames = list(reader.iter_data())
    finally:
        reader.close()
    assert len(frames) >= 5
    # Exact official extract_frames sampling, including B videos of new length.
    step = max(1, (len(frames)-3)/4)
    indexes = [min(int(1+i*step),len(frames)-1) for i in range(5)]
    return [frames[i] for i in indexes], indexes


def detection_cache(model, path, anchors):
    found = []
    import torch
    frames, indexes = decode(path)
    required=[r['frame_idx'] for r in anchors if r['status']=='present']
    span=range(min(required),max(required)+1) if required else range(0)
    for sample,frame in enumerate(frames):
        if sample not in span:
            found.append([])
            continue
        with torch.inference_mode():
            det = model.detect_faces(frame[:, :, ::-1].copy(), conf_threshold=.9,
                                     nms_threshold=.4, use_origin_size=True)
        found.append([{'bbox':[float(v) for v in face[:4]], 'confidence':float(face[4]),
                       'landmarks':[float(v) for v in face[5:]]}
                      for face in det if min(face[2]-face[0],face[3]-face[1]) >= 24])
    return frames, indexes, faces.build_tracks(found)


def extract(device='cpu'):
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    from facexlib.detection import init_detection_model
    p = inputs()
    old_gate = BASE/'evaluation/character_region_visible_face_gate_20261025/gate_decisions.json'
    protocol = {'created_from_manifest_sha256':sha(OUT/'input_manifest.json'),
        'config':gate.CONFIG,'tracking_config':faces.CONFIG,'detector_sha256':sha(faces.WEIGHT),
        'original_gate_sha256':sha(old_gate),'script_sha256':sha(__file__),
        'sampling':'Exact official five-frame sampling using actual decoded frame count; supports B stitched videos',
        'device':device,'vlm_calls':0,'official_metric_changes':False,
        'optimization':'Only detect the contiguous canonical-claim sample interval; no claimed sample or between-sample tracking step omitted',
        'optimization_validation_sha256':sha(OUT/'required_span_validation.json')}
    checked=read(OUT/'required_span_validation.json')
    assert checked['original_slots_verified']==4991 and not checked['different_decisions']
    protocol_file=OUT/f'protocol_v3_{device}.json'
    immutable(protocol_file,protocol)
    if device=='cuda':assert torch.cuda.is_available(),'CUDA unavailable; keep all caches'
    model = init_detection_model('retinaface_resnet50', half=False, device=device,
                                 model_rootpath=str(faces.WEIGHT.parent))
    started = time.monotonic(); count = 0
    for item in p['episodes']:
        ep = item['episode']; eid = ep['episode_id']; d = Path(item['output_root'])
        result = read(d/'episodes'/eid/'episode_result.json')
        media = {r['shot_key']:r for r in result['shots']}
        for shot in ep['shots']:
            sk = shot['shot_key']; dest=OUT/'extraction'/eid/(sk.replace(':','_')+'.json')
            evaluation = Path(item['artifacts'])/'intra_shot'/sk.replace(':','_')/'_eval.json'
            eval_sha = sha(evaluation); video=media[sk]['video']; video_sha=media[sk]['video_sha256']
            if dest.exists():
                cached=read(dest)
                assert cached['evaluation_sha256']==eval_sha and cached['video_sha256']==video_sha
                for r in cached['rows']:
                    if r['original_crop']:assert sha(r['original_crop'])==r['original_crop_sha256']
                count+=1;continue
            assert sha(video)==video_sha
            anchors=read(evaluation)['presence']['character']['items']
            assert {r['name'] for r in anchors}==set(shot['entity_schedule'].get('characters',[]))
            frames,indexes,tracks=detection_cache(model,video,anchors)
            rows=[]
            for anchor in anchors:
                path=anchor.get('canonical_path'); sample=anchor.get('frame_idx')
                region=faces.canonical_bbox(frames[sample],path) if path else None
                r={'episode_id':eid,'tier':ep['difficulty'],'character':anchor['name'],'shot':sk,
                   'original_status':anchor['status'],'original_crop':path,
                   'original_crop_sha256':sha(path) if path else None,'sample_idx':sample,
                   'region':region,'kept':False,'reason':'original_not_present',
                   'candidates':[],'face_track_id':None,'video_path':video,'edition':'rerun'}
                if anchor['status']=='present':
                    assert region is not None
                    for track in tracks:
                        for det in track['items']:
                            if det['sample_idx']!=sample:continue
                            overlap=gate.containment(det['bbox'],region)
                            if overlap>=.8:
                                r['candidates'].append({'track_id':track['track_id'],'bbox':det['bbox'],
                                    'face_coverage':overlap,'confidence':det['confidence']})
                    n=len(r['candidates']);r['kept']=n==1
                    r['reason']='no_complete_face_inside_region' if n==0 else 'multiple_faces_inside_region' if n>1 else 'unique_visible_face'
                    if n==1:r['face_track_id']=r['candidates'][0]['track_id']
                rows.append(r)
            claims=defaultdict(list)
            for r in rows:
                if r['kept']:claims[r['face_track_id']].append(r)
            for group in claims.values():
                if len(group)>1:
                    for r in group:r.update(kept=False,reason='same_face_track_claimed_by_multiple_roles')
            for k in {r['sample_idx'] for r in rows if r['kept']}:
                path=OUT/'contexts'/eid/(sk.replace(':','_')+f'_f{k}.jpg')
                path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(frames[k]).save(path,quality=94)
            write(dest,{'episode_id':eid,'shot':sk,'rows':rows,'tracks':tracks,
                'actual_frame_indices':indexes,'evaluation_sha256':eval_sha,'video_sha256':video_sha,
                'protocol_sha256':sha(protocol_file)})
            count+=1
            progress={'stage':f'extracting_{device}_retinaface','processed_shots':count,
                'expected_shots':p['selected_shot_count'],'elapsed_seconds':time.monotonic()-started,
                'episode_id':eid,'shot':sk}
            write(OUT/'status.json',progress)
            if count%5==0:print(json.dumps(progress),flush=True)
    write(OUT/'status.json',{'stage':'face_extraction_complete','processed_shots':count,
                            'expected_shots':p['selected_shot_count']})


def key(r):
    return r['episode_id'],r['character'],r['shot']


def historical():
    base=BASE/'evaluation/character_region_visible_face_gate_20261025'
    rows=read(base/'gate_decisions.json')['rows']
    review=read(base/'remaining_low_score_review/review_decisions.json')
    rejected={key(r):r for r in review['decisions'] if r['decision']=='exclude'}
    assert len(rejected)==3
    return rows,rejected


def recover(ep,raw,rows,mod):
    eid=ep['episode_id'];order=[s['shot_key'] for s in ep['shots']];pos={s:i for i,s in enumerate(order)}
    pools={}; baseline=[]
    for k,data in raw['per_entity_scores'].items():
        if not k.startswith('face_'):continue
        name=k[5:];apps=sorted(data['per_shot_sim_to_centroid'],key=lambda a:pos[a['shot']]);n=len(apps)
        pairs=[a for a in raw['gap_decay_data'] if a['entity_type']=='character' and a['entity_name']==name]
        assert len(pairs)==n*(n-1)//2
        gram=np.eye(n);q=0
        for a in range(n):
            for b in range(a+1,n):
                assert pairs[q]['gap']==abs(pos[apps[b]['shot']]-pos[apps[a]['shot']])
                gram[a,b]=gram[b,a]=pairs[q]['similarity'];q+=1
        features=gate.gram_to_features(gram);c=features.mean(0);c/=np.linalg.norm(c);sims=features@c
        assert np.max(np.abs(sims-np.array([a['sim_to_centroid'] for a in apps])))<2e-6
        baseline.extend(sims.tolist());pool=[]
        for app,v in zip(apps,features):
            r=rows[eid,name,app['shot']]
            pool.append(mod.EntityCrop(name,'character',name,app['shot'],r['sample_idx'],tuple(r['region']),
                r['original_crop'],app['clip_text_sim'],v))
        pools[name]=pool
    expected=raw['metrics']['cs_face']['value']
    assert (not baseline)==(expected is None)
    if baseline:assert abs(float(np.mean(baseline))-expected)<2e-6
    return pools


def compute(ep,raw,rows,rejected,mod,only_keys=None):
    pools=recover(ep,raw,rows,mod);subset={}
    for name,crops in pools.items():
        subset[name]=[c for c in crops if rows[ep['episode_id'],name,c.shot_key]['kept']
            and (ep['episode_id'],name,c.shot_key) not in rejected
            and (only_keys is None or (ep['episode_id'],name,c.shot_key) in only_keys)]
    value,entities,n=mod.compute_cross_shot_consistency(subset,'character',shot_order=[s['shot_key'] for s in ep['shots']])
    result=mod.EpisodeResult(ep['episode_id'],ep['difficulty'],len(ep['shots']))
    result.metrics['cs_face']=mod.MetricValue(value=value,n_evaluated=n)
    return result,entities


def score(final=False):
    p=inputs();manifest=read(BASE/'manifest.json');selected={i['episode_id']:i for i in p['episodes']}
    for source in read(gate.OUT/'plan.json')['inputs']:
        assert sha(source['path'])==source['sha256'],source['path']
    for item in p['episodes']:
        generation=read(Path(item['output_root'])/'episodes'/item['episode_id']/'episode_result.json')
        for shot in generation['shots']:
            assert sha(shot['video'])==shot['video_sha256'],shot['video']
    old,rejected=historical();oldmap={key(r):r for r in old};newmap=dict(oldmap)
    protocols=set()
    for file in OUT.glob('protocol*.json'):
        protocol=read(file)
        assert protocol['config']==gate.CONFIG
        assert protocol['detector_sha256']==sha(faces.WEIGHT)
        assert protocol['created_from_manifest_sha256']==sha(OUT/'input_manifest.json')
        assert protocol['device'] in ['cpu','cuda']
        protocols.add(sha(file))
    for eid,item in selected.items():
        files=sorted((OUT/'extraction'/eid).glob('*.json'))
        assert len(files)==len(item['episode']['shots']),(eid,len(files))
        for f in files:
            data=read(f);assert data['protocol_sha256'] in protocols
            for r in data['rows']:newmap[key(r)]=r
    assert len(oldmap)==len(newmap)==4991
    # Original manual rejections may only transfer with identical canonical pixels.
    carry={k:v for k,v in rejected.items() if newmap[k]['original_crop_sha256']==v['original_crop_sha256']}
    manual={}
    if final:
        review=read(OUT/'review/decisions.json');selection=read(OUT/'review/manifest.json')
        assert review['manifest_sha256']==sha(OUT/'review/manifest.json')
        assert {r['review_id'] for r in review['decisions']}=={r['review_id'] for r in selection['rows']}
        assert len(review['decisions'])==80
        manual={key(r):r for r in review['decisions'] if r['decision']=='exclude'}
        for k,r in manual.items():assert r['original_crop_sha256']==newmap[k]['original_crop_sha256']
        assert all(newmap[k].get('edition')=='rerun' for k in manual)
    allreject={**carry,**manual};mod=gate.load_official_pure();base_results=[];current_results=[];scored=[];details=[]
    paired_a=[];paired_b=[];paired_ids=[]
    for ep in manifest['episodes']:
        eid=ep['episode_id'];original=read(gate.EVAL/f'{eid}.json')
        current=read(selected[eid]['result']) if eid in selected else original
        a,ae=compute(ep,original,oldmap,rejected,mod);b,be=compute(ep,current,newmap,allreject,mod)
        baseline_check=read(gate.OUT/'remaining_low_score_review/recomputed/results'/f'{eid}.json')['metrics']['cs_face']
        assert a.metrics['cs_face'].n_evaluated==baseline_check['n_evaluated']
        if a.metrics['cs_face'].value is not None:assert abs(a.metrics['cs_face'].value-baseline_check['value'])<1e-10
        base_results.append(a);current_results.append(b)
        ent=[]
        for name,data in be.items():
            for app in data['per_shot_sim_to_centroid']:
                r=dict(newmap[eid,name,app['shot']]);r['similarity']=app['sim_to_centroid'];scored.append(r)
            olddata=ae.get(name)
            ent.append({'name':name,'old':olddata,'new':data})
        details.append({'episode_id':eid,'replaced':eid in selected,'baseline':a.metrics['cs_face'].to_json(),
            'current':b.metrics['cs_face'].to_json(),'entities':ent})
        path=OUT/('final' if final else 'automatic')/'results'/f'{eid}.json'
        write(path,{'episode_id':eid,'tier':ep['difficulty'],'num_shots':len(ep['shots']),
            'metrics':{'cs_face':b.metrics['cs_face'].to_json()},'per_entity_scores':be})
        if eid in selected:
            common={k for k in oldmap if k[0]==eid and oldmap[k]['kept'] and newmap[k]['kept'] and k not in rejected and k not in allreject}
            pa,_=compute(ep,original,oldmap,rejected,mod,common);pb,_=compute(ep,current,newmap,allreject,mod,common)
            paired_a.append(pa);paired_b.append(pb);paired_ids.append(eid)
    split={t:[e['episode_id'] for e in manifest['episodes'] if e['difficulty']==t] for t in ['easy','medium','hard']}
    a=mod.aggregate_results(base_results,split);b=mod.aggregate_results(current_results,split)
    assert a['all']['cs_face']==read(gate.OUT/'remaining_low_score_review/recomputed/report.json')['after']['all']['cs_face']
    paired_split={t:[e['episode_id'] for e in manifest['episodes'] if e['difficulty']==t and e['episode_id'] in paired_ids] for t in split}
    report={'scope':'visible-single-face original-character-region diagnostic, not official cs_face',
        'status':'complete_with_manual_review' if final else 'automatic_complete_pending_manual_review',
        'selected_episodes':p['selected_episode_count'],'selected_shots':p['selected_shot_count'],
        'baseline':a,'current':b,'baseline_coverage':{'kept_crops':sum(r['kept'] and key(r) not in rejected for r in old),
            'scored_appearances':sum(r.metrics['cs_face'].n_evaluated for r in base_results)},
        'current_coverage':{'kept_crops':sum(r['kept'] and key(r) not in allreject for r in newmap.values()),
            'scored_appearances':len(scored),'scheduled_role_shots':4991},
        'historical_rejections_carried_by_identical_crop_sha':len(carry),
        'new_review_exclusions':list(manual.values()),'per_episode':details,
        'new_review_scope':'80 scored crops from replaced episodes only; preserve historical crop-SHA decisions on all unchanged inputs, never import exclusions by role name',
        'replaced_episode_matched_role_shot_intersection':{'baseline':mod.aggregate_results(paired_a,paired_split),
            'current':mod.aggregate_results(paired_b,paired_split),'scope':'Same role-shot IDs in each complete rerun; not identical pixels or 3D-only ablation'},
        'vlm_calls':0,'dino_inference_calls':0,'detector':'FP32 RetinaFace R50 on new five-frame samples; device and source pinned in protocol files; original cached detections reused',
        'baseline_reproduced':True,'manifest_sha256':sha(OUT/'input_manifest.json'),
        'input_validation':'Original frozen diagnostic sources, new episode results and generated video SHA reverified; canonical crop SHA verified during extraction/resume',
        'limitations':'Historical 0.7863 includes a later-known unresolved R051 association. Original and current gate coverage/centroids differ. New review is localization-only; do not delete generated style/identity failures.'}
    write(OUT/('report.json' if final else 'automatic_report.json'),report)
    write(OUT/'scored_crops.json',scored)
    print(json.dumps({'stage':report['status'],'baseline':a['all']['cs_face'],'current':b['all']['cs_face'],
        'coverage':report['current_coverage']},ensure_ascii=False),flush=True)
    if not final:prepare_review(scored)


def prepare_review(scored):
    ordered=sorted([r for r in scored if r.get('edition')=='rerun'],key=lambda r:(r['similarity'],key(r)))
    assert len(ordered)>=80
    low=ordered[:60]
    chosen=low+random.Random(20261025).sample(ordered[60:],20)
    random.Random(20261101).shuffle(chosen)
    rows=[]
    for i,r in enumerate(chosen,1):
        x={k:v for k,v in r.items() if k!='similarity'};x['review_id']=f'N{i:03d}';rows.append(x)
    immutable(OUT/'review/manifest.json',{'selection':'Changed complete reruns only: lowest60 + seed20261025 random20; shuffled seed20261101, display scores hidden; unchanged original reviews retained by SHA',
        'scored_crops_sha256':sha(OUT/'scored_crops.json'),'rows':rows})
    manifest=read(BASE/'manifest.json');epmap={e['episode_id']:e for e in manifest['episodes']}
    for start in range(0,len(rows),6):
        group=rows[start:start+6];canvas=Image.new('RGB',(1600,1290),'#ededed');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(group):
            x=(i%2)*800;y=(i//2)*430;ep=epmap[r['episode_id']]
            title=f"{r['review_id']} {ep['run']} {r['shot']} target:{r['character']} {r.get('edition','original')}"
            draw.text((x+6,y+4),title,font=FONT,fill='black')
            crop=Image.open(r['original_crop']).convert('RGB');thumb=ImageOps.contain(crop,(180,285));canvas.paste(thumb,(x+607,y+36))
            if r.get('edition')=='rerun':context=OUT/'contexts'/r['episode_id']/(r['shot'].replace(':','_')+f"_f{r['sample_idx']}.jpg")
            else:
                oldcache=read(faces.OUT/'extraction/A'/f"{r['episode_id']}.json")
                original=next(v for v in oldcache['rows'] if v['entity']==r['character'] and v['shot_key']==r['shot'])
                context=None
                # Diagnostic face representative may differ from canonical time;
                # decode exact official canonical sample for the manual review.
                frames,_=decode(original['video_path']);im=Image.fromarray(frames[r['sample_idx']])
                context=OUT/'review/contexts'/(r['review_id']+'.jpg');context.parent.mkdir(exist_ok=True);im.save(context,quality=94)
            im=Image.open(context).convert('RGB');mark=ImageDraw.Draw(im);mark.rectangle(r['region'],outline='red',width=3)
            for candidate in r['candidates']:mark.rectangle(candidate['bbox'],outline='yellow',width=2)
            im=ImageOps.contain(im,(590,315));canvas.paste(im,(x+5,y+35))
            shot=next(s for s in ep['shots'] if s['shot_key']==r['shot']);text=shot['prompt']
            # Prompt lines are evidence, not a computed score.
            for j in range(3):draw.text((x+6,y+350+j*20),text[j*100:(j+1)*100],font=FONT,fill='black')
        canvas.save(OUT/'review'/f'page_{start//6+1:02d}.jpg',quality=95)
    write(OUT/'status.json',{'stage':'manual_localization_review_required','items':len(rows),'pages':14})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['extract','score','final']);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.action=='extract':extract(args.device)
    else:score(args.action=='final')
