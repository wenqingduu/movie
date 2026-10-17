"""Face-only DINOv2 centroid adapter for Qwen full140, isolated from original scores.

Uses RetinaFace detection only, never InsightFace/ArcFace identity features.
Original EntityBench reports and generated media are immutable inputs. Role
association is spatial and conservative; no DINO score participates in mapping.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment

PROJECT=Path('/root/autodl-tmp/movie')
ROOT=PROJECT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140'
OUT=ROOT/'evaluation/face_only_dino'
HELPER_PATH=PROJECT/'pretest/evaluate_entitybench_corrected_character_crops.py'
spec=importlib.util.spec_from_file_location('face_only_shared_helpers',HELPER_PATH)
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
ROOTS={'A':ROOT}
helper.NAMES={'A':'Qwen full140'}
helper.scoredir=lambda root: root/'evaluation/official_parallel_v2'
helper.OFFICIAL=ROOT/'runtime/evaluate_benchmark_official.py'
def original_reports():
 p=ROOT/'evaluation/official_parallel_v2/report_qwen21_native_reference_a14b_4step_continuation_full140_no_vlm.json'
 return [dict(path=str(p),sha256=helper.sha(p))]
helper.original_reports=original_reports
WEIGHT=Path('/root/autodl-tmp/movie/.venv/lib/python3.10/site-packages/facexlib/weights/detection_Resnet50_Final.pth')
CONFIG=dict(face_detector='facexlib RetinaFace ResNet50',detector_confidence=.9,nms_threshold=.4,
 native_min_width=24,native_min_height=24,crop_padding=.1,sampling_indices=[1,20,40,59,79],
 mapping_min_geometry=.35,mapping_relative_ambiguity=.15,track_min_similarity=.3,
 track_relative_ambiguity=.15,track_max_gap=1,
 candidate_selection='face detection confidence × official sharpness_norm × official area_norm; no identity/cosine selection',
 role_source='benchmark schedule and original canonical character-region position; weak regions allowed as spatial anchors; single-role sole-face fallback; conflicting roles rejected',
 embedding='pinned official dinov2-base CLS, L2 normalize, original evaluation environment',
 aggregation='original per-episode character centroid pooling; episode macro and separate instance-weighted diagnostic',
 alignment=False,insightface=False,vlm=False)

read=helper.read;write=helper.write;sha=helper.sha

def guard():
 import shutil
 assert shutil.disk_usage(ROOT).free>=10*1024**3,'Less than 10GiB free; expand and resume; preserve old outputs'

def validate_media(plan):
 for ep in plan['episodes']:
  for shot in ep['shots']:
   assert sha(shot['video_path'])==shot['video_sha256']
   for anchor in shot['anchors']:
    if anchor['canonical_path']:assert sha(anchor['canonical_path'])==anchor['canonical_sha256']

def model_sources():
 paths=[p for p in Path('/root/autodl-tmp/models/dinov2-base').rglob('*') if p.is_file() and '.cache' not in p.parts]
 return {str(p):sha(p) for p in paths}

def initialize():
 if (OUT/'plan.json').exists():
  p=read(OUT/'plan.json');assert p['config']==CONFIG and p['script_sha256']==sha(__file__)
  assert p['helper_sha256']==sha(HELPER_PATH) and p['detector_weights_sha256']==sha(WEIGHT)
  assert p['model_sources']==model_sources()
  validate_media(p)
  return p
 episodes=[];inputs=[]
 for route,root in ROOTS.items():
  manifest=read(root/'manifest.json');gen=read(root/'generation_report.json')
  assert gen['status']=='complete' and len(gen['jobs'])==2491 and len(manifest['episodes'])==140
  jobs={(j['episode_id'],j['shot_key']):j for j in gen['jobs']}
  inputs.extend(dict(path=str(path),sha256=sha(path)) for path in [root/'manifest.json',root/'generation_report.json'])
  for ep in manifest['episodes']:
   assert sha(ep['episode_path'])==ep['episode_sha256']
   art=helper.artifacts(root,ep['episode_id']);shots=[]
   for s in ep['shots']:
    j=jobs[(ep['episode_id'],s['shot_key'])];p=art/'intra_shot'/s['shot_key'].replace(':','_')/'_eval.json'
    ev=read(p);items=ev['presence']['character']['items']
    assert set(x['name'] for x in items)==set(j['entity_schedule']['characters'])
    anchors=[]
    for item in items:
     cp=item.get('canonical_path');anchors.append(dict(entity=item['name'],original_status=item['status'],
      original_sample_idx=item.get('frame_idx'),canonical_path=cp,canonical_sha256=sha(cp) if cp else None))
    shots.append(dict(shot_key=s['shot_key'],shot_index=s['shot_index'],prompt=j['prompt'],cut=j['cut'],
     video_path=j['output_video'],video_sha256=j['video_sha256'],anchors=anchors))
   episodes.append(dict(route=route,episode_id=ep['episode_id'],tier=ep['difficulty'],shots=shots))
 plan=dict(version=1,model_sources=model_sources(),scope='face_only_dino_qwen_full140_without_vlm',config=CONFIG,
  script_sha256=sha(__file__),helper_sha256=sha(HELPER_PATH),official_sha256=helper.OFFICIAL_SHA,
  detector_weights_path=str(WEIGHT),detector_weights_sha256=sha(WEIGHT),
  original_reports=helper.original_reports(),inputs=inputs,episodes=episodes)
 validate_media(plan)
 write(OUT/'plan.json',plan);return plan

def intersection(a,b):
 return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def area(a):return max(0,a[2]-a[0])*max(0,a[3]-a[1])
def iou(a,b):
 inter=intersection(a,b);return inter/max(1,area(a)+area(b)-inter)
def track_similarity(a,b):
 aw=max(1,a[2]-a[0]);ah=max(1,a[3]-a[1]);bw=max(1,b[2]-b[0]);bh=max(1,b[3]-b[1])
 if not .33<=bw/aw<=3 or not .33<=bh/ah<=3:return 0.
 ac=np.array([(a[0]+a[2])/2,(a[1]+a[3])/2]);bc=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
 dist=float(np.linalg.norm(ac-bc))/max(aw,ah,bw,bh)
 if dist>2.5:return 0.
 return max(iou(a,b),.65*math.exp(-dist))

def build_tracks(detections):
 tracks=[]
 for fi,dets in enumerate(detections):
  eligible=[t for t in tracks if fi-t['items'][-1]['sample_idx']<=CONFIG['track_max_gap']]
  assigned=set()
  if eligible and dets:
   scores=np.array([[track_similarity(t['items'][-1]['bbox'],d['bbox']) for d in dets] for t in eligible])
   ti,di=linear_sum_assignment(-scores)
   for tr,det in zip(ti.tolist(),di.tolist()):
    score=scores[tr,det]
    alternative=max([v for k,v in enumerate(scores[tr]) if k!=det]+[v for k,v in enumerate(scores[:,det]) if k!=tr]+[0.])
    if score<CONFIG['track_min_similarity'] or (alternative>0 and alternative>=score*(1-CONFIG['track_relative_ambiguity'])):continue
    eligible[tr]['items'].append(dict(**dets[det],sample_idx=fi));assigned.add(det)
  for k,d in enumerate(dets):
   if k not in assigned:tracks.append(dict(track_id=len(tracks),items=[dict(**d,sample_idx=fi)]))
 return tracks

def canonical_bbox(frame,path):
 crop=np.asarray(Image.open(path).convert('RGB'));h,w=crop.shape[:2]
 if h>frame.shape[0] or w>frame.shape[1]:raise RuntimeError('Canonical crop larger than frame')
 match=cv2.matchTemplate(frame,crop,cv2.TM_SQDIFF)
 _,_,loc,_=cv2.minMaxLoc(match);x,y=loc
 if not np.array_equal(frame[y:y+h,x:x+w],crop):
  # Floating point cancellation in TM_SQDIFF can displace equal minima.
  for idx in np.argsort(match.ravel())[:20]:
   yy,xx=np.unravel_index(idx,match.shape)
   if np.array_equal(frame[yy:yy+h,xx:xx+w],crop):x,y=int(xx),int(yy);break
  else:raise RuntimeError(f'Could not recover exact canonical crop position: {path}')
 return [x,y,x+w,y+h]

def geometry(face,region):
 fw=max(1,face[2]-face[0]);fh=max(1,face[3]-face[1]);rw=max(1,region[2]-region[0])
 horizontal=max(0,min(face[2],region[2])-max(face[0],region[0]))/fw
 if horizontal<.5:return 0.
 vertical=max(0,min(face[3],region[3])-max(face[1],region[1]))/fh
 if vertical>0:base=.5+.5*min(1,vertical)
 elif face[3]<=region[1] and region[1]-face[3]<=1.5*fh:base=.65/(1+(region[1]-face[3])/fh)
 else:return 0.
 xdist=abs((face[0]+face[2]-region[0]-region[2])/2)/rw
 return float(base*horizontal/(1+.25*xdist))

def decode(path):
 reader=iio.get_reader(path,'ffmpeg',ffmpeg_params=['-threads','1']);frames=[];count=0
 try:
  for frame in reader:
   if count in CONFIG['sampling_indices']:frames.append(frame)
   count+=1
 finally:reader.close()
 assert count==81 and len(frames)==5
 return frames

def extract():
 plan=initialize();OUT.mkdir(parents=True,exist_ok=True)
 for inp in plan['inputs']+plan['original_reports']:assert sha(inp['path'])==inp['sha256']
 import torch
 torch.set_num_threads(4)
 from facexlib.detection import init_detection_model
 model=init_detection_model('retinaface_resnet50',half=False,device='cuda',model_rootpath=str(WEIGHT.parent))
 mod=helper.official();beg=time.monotonic();all_rows=[];processed=0
 for ep in plan['episodes']:
  eid=ep['episode_id'];route=ep['route'];epcache=OUT/'extraction'/route/(eid+'.json')
  if epcache.exists():
   saved=read(epcache);assert saved['plan_sha256']==sha(OUT/'plan.json')
   for r in saved['rows']:
    if r['crop_path']:assert sha(r['crop_path'])==r['crop_sha256']
   all_rows.extend(saved['rows']);processed+=len(ep['shots']);continue
  rows=[];shot_audits=[]
  for shot in ep['shots']:
   guard()
   assert sha(shot['video_path'])==shot['video_sha256'];sk=shot['shot_key']
   frames=decode(shot['video_path']);detections=[];small=0
   for frame in frames:
    with torch.inference_mode():det=model.detect_faces(frame[:,:,::-1].copy(),conf_threshold=.9,nms_threshold=.4,use_origin_size=True)
    accepted=[]
    for face in det:
     box=[float(x) for x in face[:4]]
     if box[2]-box[0]<24 or box[3]-box[1]<24:small+=1;continue
     accepted.append(dict(bbox=box,confidence=float(face[4]),landmarks=[float(x) for x in face[5:]]))
    detections.append(accepted)
   tracks=build_tracks(detections);choices=[]
   for anchor in shot['anchors']:
    region=None;matches=[];fi=anchor['original_sample_idx']
    if anchor['canonical_path']:
     assert sha(anchor['canonical_path'])==anchor['canonical_sha256']
     region=canonical_bbox(frames[fi],anchor['canonical_path'])
     for track in tracks:
      same=[d for d in track['items'] if d['sample_idx']==fi]
      if same:
       strength=geometry(same[0]['bbox'],region)
       if strength>=CONFIG['mapping_min_geometry']:matches.append((strength,track['track_id']))
    matches.sort(reverse=True);selected=None;reason='no_spatially_matched_face'
    if matches:
     if len(matches)>1 and matches[1][0]>=matches[0][0]*(1-CONFIG['mapping_relative_ambiguity']):reason='ambiguous_multiple_faces_in_character_region'
     else:selected=matches[0][1];reason='canonical_region_spatial_match'
    if selected is None and len(shot['anchors'])==1:
     if len(tracks)==1:selected=tracks[0]['track_id'];reason='single_scheduled_role_single_face_track'
     elif fi is not None and len(detections[fi])==1:
      chosen=next(t for t in tracks if any(d['sample_idx']==fi and d['bbox']==detections[fi][0]['bbox'] for d in t['items']))
      selected=chosen['track_id'];reason='single_scheduled_role_single_face_at_canonical_sample'
    if not tracks:reason='no_detectable_face_at_minimum_size'
    choices.append(dict(anchor=anchor,region=region,track_id=selected,reason=reason,mapping_candidates=matches))
   conflicts=Counter(c['track_id'] for c in choices if c['track_id'] is not None)
   for c in choices:
    if c['track_id'] is not None and conflicts[c['track_id']]>1:c['track_id']=None;c['reason']='same_face_track_conflicts_between_scheduled_roles'
    anchor=c['anchor'];safe=''.join(ch if ch.isalnum() else '_' for ch in anchor['entity'])
    rid=f"{route}__{eid}__{sk.replace(':','_')}__{safe}";best=None
    if c['track_id'] is not None:
     for d in tracks[c['track_id']]['items']:
      bbox=[max(0,int(d['bbox'][0])),max(0,int(d['bbox'][1])),min(frames[0].shape[1],int(math.ceil(d['bbox'][2]))),min(frames[0].shape[0],int(math.ceil(d['bbox'][3])))]
      crop=mod.crop_entity(frames[d['sample_idx']],bbox,.1)
      comp=mod.compute_selection_score(d['confidence'],crop,frames[0].shape)
      if best is None or comp['selection']>best['selection']:
       best=dict(sample_idx=d['sample_idx'],bbox=bbox,crop=crop,selection=comp['selection'],components=comp,face_confidence=d['confidence'])
    crop_path=None;crop_sha=None
    if best:
     folder=OUT/'crops'/route/eid;folder.mkdir(parents=True,exist_ok=True);path=folder/(sk.replace(':','_')+'__'+safe+'.png')
     Image.fromarray(best['crop']).save(path);crop_path=str(path);crop_sha=sha(path)
     frame_path=folder/(sk.replace(':','_')+'__'+safe+'__context.jpg')
     Image.fromarray(frames[best['sample_idx']]).save(frame_path,quality=90)
     best={k:v for k,v in best.items() if k!='crop'};best['context_path']=str(frame_path)
    row=dict(id=rid,route=route,episode_id=eid,entity=anchor['entity'],shot_key=sk,shot_index=shot['shot_index'],
     original_character_status=anchor['original_status'],original_region= c['region'],
     video_path=shot['video_path'],video_sha256=shot['video_sha256'],
     crop_path=crop_path,crop_sha256=crop_sha,selected=best,status='mapped_face' if best else 'missing',reason=c['reason'],
     track_id=c['track_id'],mapping_candidates=c['mapping_candidates'],small_detections_in_shot=small,
     extra_label=eid.split('__')[1])
    rows.append(row)
   shot_audits.append(dict(shot_key=sk,n_face_detections=[len(x) for x in detections],small_detections=small,tracks=tracks))
   processed+=1
   if processed%20==0:
    status=dict(phase='extracting_face_crops',processed_videos=processed,total_videos=2491,elapsed_seconds=time.monotonic()-beg)
    write(OUT/'status.json',status);print(status,flush=True)
  write(epcache,dict(plan_sha256=sha(OUT/'plan.json'),route=route,episode_id=eid,rows=rows,shot_audits=shot_audits));all_rows.extend(rows)
  print('Episode extraction saved',route,eid,len(rows),'scheduled',sum(r['crop_path'] is not None for r in rows),'mapped faces',flush=True)
 mapped=[r for r in all_rows if r['crop_path']];pages=[];(OUT/'review').mkdir(parents=True,exist_ok=True)
 for start in range(0,len(mapped),48):
  batch=mapped[start:start+48];page=len(pages)+1;path=OUT/'review'/f'page_{page:03d}.jpg'
  helper.sheet(batch,path,f'FACE ONLY DINO | page {page} | identity scores hidden; selected crops before embedding')
  pages.append(dict(page=page,path=str(path),ids=[r['id'] for r in batch]))
 write(OUT/'face_crop_manifest.json',dict(plan_sha256=sha(OUT/'plan.json'),rows=all_rows,pages=pages,
  counts={route:dict(Counter(r['status'] for r in all_rows if r['route']==route)) for route in ROOTS}))
 write(OUT/'status.json',dict(phase='face_crops_complete_pending_visual_review',processed_videos=2491,total_videos=2491,
  scheduled_role_shots=len(all_rows),mapped_faces=len(mapped),review_pages=len(pages),elapsed_seconds=time.monotonic()-beg))
 print(read(OUT/'status.json'),flush=True)

def validate_review(manifest,review,manifest_sha256):
 assert review['face_crop_manifest_sha256']==manifest_sha256
 assert review.get('reviewer') and review.get('reviewed_at')
 assert set(review['reviewed_pages'])=={p['page'] for p in manifest['pages']}
 mapped={r['id'] for r in manifest['rows'] if r['crop_path']}
 assert len(mapped)==sum(bool(r['crop_path']) for r in manifest['rows']), 'Duplicate crop IDs'
 assert len(review['reviewed_ids'])==len(set(review['reviewed_ids']))
 assert set(review['reviewed_ids'])==mapped, 'Every crop requires a recorded visual decision'
 assert set(review['rejected_ids']).issubset(mapped)
 assert all(review.get('rejection_reasons',{}).get(rid) for rid in review['rejected_ids'])
 assert review['criteria']=='face-localization only; retain generated identity/style drift; no DINO-score filtering'

def score():
 guard()
 plan=initialize();manifest=read(OUT/'face_crop_manifest.json');review=read(OUT/'review/decisions.json')
 validate_review(manifest,review,sha(OUT/'face_crop_manifest.json'))
 frozen=dict(face_crop_manifest_sha256=sha(OUT/'face_crop_manifest.json'),review_sha256=sha(OUT/'review/decisions.json'),
  reviewed_pages=review['reviewed_pages'],reviewed_ids=review['reviewed_ids'],rejected_ids=review['rejected_ids'])
 if (OUT/'frozen_review.json').exists():assert read(OUT/'frozen_review.json')==frozen
 else:write(OUT/'frozen_review.json',frozen)
 mod=helper.official();embdir=OUT/'embeddings';embdir.mkdir(exist_ok=True)
 rows=manifest['rows'];rejected=review['rejected_ids']
 for r in rows:
  if not r['crop_path'] or r['id'] in rejected:continue
  guard()
  assert sha(r['crop_path'])==r['crop_sha256'];p=embdir/(r['crop_sha256']+'.npy')
  if not p.exists():np.save(p,mod.compute_dino_embedding(np.asarray(Image.open(r['crop_path']).convert('RGB')),'cuda'))
  r['_embedding']=np.load(p)
 split={tier:[e['episode_id'] for e in plan['episodes'] if e['route']=='A' and e['tier']==tier] for tier in ['easy','medium','hard']}
 reports={}
 for route in ROOTS:
  episodes=[];role_reasons=Counter();n_sched=0;n_valid=0
  for ep in [e for e in plan['episodes'] if e['route']==route]:
   rs=[r for r in rows if r['route']==route and r['episode_id']==ep['episode_id']];pools=defaultdict(list);n_sched+=len(rs)
   for r in rs:
    if '_embedding' not in r:role_reasons['review_rejected_false_face' if r['id'] in rejected else r['reason']]+=1;continue
    n_valid+=1;pools[r['entity']].append(mod.EntityCrop(entity_name=r['entity'],entity_type='character',entity_description=r['entity'],
      shot_key=r['shot_key'],frame_idx=r['selected']['sample_idx'],bbox=tuple(r['selected']['bbox']),crop_path=r['crop_path'],clip_text_sim=0.,dino_embedding=r['_embedding']))
   val,entities,n=mod.compute_cross_shot_consistency(dict(pools),'character',shot_order=[s['shot_key'] for s in ep['shots']])
   result=mod.EpisodeResult(episode_id=ep['episode_id'],tier=ep['tier'],num_shots=len(ep['shots']))
   result.metrics['cs_face']=mod.MetricValue(value=val,n_evaluated=n,n_failed=0,n_skipped=sum('_embedding' not in r for r in rs));episodes.append(result)
   write(OUT/'results'/route/(ep['episode_id']+'.json'),dict(episode_id=ep['episode_id'],tier=ep['tier'],num_shots=result.num_shots,
    crop_type='face_only',metrics={'cs_face':dict(value=val,n_evaluated=n,n_failed=0,n_skipped=result.metrics['cs_face'].n_skipped)},
    per_entity_scores={f'face_{k}':v for k,v in entities.items()}))
  ag=mod.aggregate_results(episodes,split);valid=[e.metrics['cs_face'] for e in episodes if e.metrics['cs_face'].value is not None]
  total=sum(m.n_evaluated for m in valid)
  reports[route]=dict(name=helper.NAMES[route],aggregation=ag,scheduled_role_shots=n_sched,valid_face_role_shots=n_valid,
   face_coverage=n_valid/n_sched,cs_face_appearances=total,n_missing_by_reason=dict(role_reasons),
   instance_weighted_raw=sum(m.value*m.n_evaluated for m in valid)/total if total else None)
 support={route:{(r['episode_id'],r['entity'],r['shot_key']) for r in rows if r['route']==route and '_embedding' in r} for route in ROOTS}
 common=set.intersection(*support.values());common_report={}
 for route in ROOTS:
  epvals=[];sims_all=[]
  for ep in [e for e in plan['episodes'] if e['route']==route]:
   pools=defaultdict(list)
   for r in rows:
    if r['route']==route and r['episode_id']==ep['episode_id'] and (r['episode_id'],r['entity'],r['shot_key']) in common:pools[r['entity']].append(r['_embedding'])
   sims=[]
   for vectors in pools.values():
    if len(vectors)<2:continue
    mat=np.stack(vectors);center=mat.mean(0);center/=np.linalg.norm(center);sims.extend((mat@center).tolist())
   if sims:epvals.append(float(np.mean(sims)));sims_all.extend(sims)
  common_report[route]=dict(episode_macro_mean=float(np.mean(epvals)) if epvals else None,n_episodes=len(epvals),
   n_appearances=len(sims_all),instance_weighted_raw=float(np.mean(sims_all)) if sims_all else None)
 report=dict(status='complete',metric='cs_face',crop_type='face_only',scope='face_only_DINO_centroid_non_vlm_not_original_character_region_protocol',
  formula_source_sha256=helper.OFFICIAL_SHA,plan_sha256=sha(OUT/'plan.json'),frozen_review_sha256=sha(OUT/'frozen_review.json'),
  vlm_calls=0,insightface_used=False,routes=reports,common_support=dict(n_role_shots=len(common),routes=common_report))
 write(OUT/'report.json',report)
 for inp in plan['inputs']+plan['original_reports']:assert sha(inp['path'])==inp['sha256']
 independent_audit()
 write(OUT/'status.json',dict(phase='face_only_dino_complete',episodes_per_route=140,videos_per_route=2491,vlm_calls=0,insightface_used=False))
 print(json.dumps({k:{'cs_face':r['aggregation']['all']['cs_face'],'coverage':r['face_coverage'],'missing':r['n_missing_by_reason']} for k,r in reports.items()},ensure_ascii=False),flush=True)

def independent_audit():
 manifest=read(OUT/'face_crop_manifest.json');report=read(OUT/'report.json');plan=read(OUT/'plan.json')
 review=read(OUT/'review/decisions.json');rows=manifest['rows'];results=[]
 validate_media(plan)
 assert len(plan['episodes'])==140 and sum(len(e['shots']) for e in plan['episodes'])==2491
 assert report['vlm_calls']==0 and report['insightface_used'] is False
 rejected=set(review['rejected_ids']);covered=0;appearances=0;missing=Counter();valid_episodes=0
 for ep in plan['episodes']:
  rs=[r for r in rows if r['episode_id']==ep['episode_id']];pools=defaultdict(list)
  assert len(rs)==sum(len(s['anchors']) for s in ep['shots'])
  for r in rs:
   if not r['crop_path'] or r['id'] in rejected:
    missing['review_rejected_false_face' if r['id'] in rejected else r['reason']]+=1;continue
   assert sha(r['crop_path'])==r['crop_sha256']
   vector=np.load(OUT/'embeddings'/(r['crop_sha256']+'.npy'))
   assert np.isfinite(vector).all() and abs(np.linalg.norm(vector)-1)<1e-5
   pools[r['entity']].append(vector);covered+=1
  sims=[]
  for vectors in pools.values():
   if len(vectors)<2:continue
   matrix=np.stack(vectors);centroid=matrix.mean(axis=0);centroid/=np.linalg.norm(centroid)
   sims.extend((matrix@centroid).tolist())
  expected=float(np.mean(sims)) if sims else None
  cached=read(OUT/'results/A'/(ep['episode_id']+'.json'))['metrics']['cs_face']
  assert (cached['value'] is None)==(expected is None)
  if expected is not None:
   assert abs(cached['value']-expected)<1e-6;valid_episodes+=1
  assert cached['n_evaluated']==len(sims)
  appearances+=len(sims);results.append(dict(episode_id=ep['episode_id'],score=expected))
 actual=report['routes']['A']
 assert actual['scheduled_role_shots']==len(rows) and actual['valid_face_role_shots']==covered
 assert actual['cs_face_appearances']==appearances and actual['n_missing_by_reason']==dict(missing)
 values=[r['score'] for r in results if r['score'] is not None]
 pooled=float(np.mean(values)) if values else None
 assert actual['aggregation']['all']['cs_face']['mean']==pooled or abs(actual['aggregation']['all']['cs_face']['mean']-pooled)<1e-6
 write(OUT/'independent_validation.json',dict(status='passed',episodes=140,videos=2491,
  scheduled_role_shots=len(rows),mapped_face_role_shots=covered,coverage=covered/len(rows),
  valid_episodes=valid_episodes,appearances=appearances,missing_by_reason=dict(missing),
  report_sha256=sha(OUT/'report.json'),frozen_review_sha256=sha(OUT/'frozen_review.json')))


def main():
 global ROOT,OUT,ROOTS
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','score']);p.add_argument('--output-root',type=Path,default=ROOT);a=p.parse_args()
 ROOT=a.output_root.resolve();OUT=ROOT/'evaluation/face_only_dino';ROOTS={'A':ROOT}
 helper.OFFICIAL=ROOT/'runtime/evaluate_benchmark_official.py'
 from audit_entitybench_qwen21_full140 import verify_adapters
 verify_adapters(ROOT)
 {'extract':extract,'score':score}[a.stage]()
if __name__=='__main__':main()
