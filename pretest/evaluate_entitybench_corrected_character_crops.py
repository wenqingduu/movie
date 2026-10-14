"""Auditable, human-reviewed character localization correction on frozen videos.

Never edits original media, crops or reports. The official DINO/centroid formulas
are reused; correcting inputs makes this a separate non-VLM diagnostic metric.
Review decisions must precede computation of corrected DINO scores.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT=Path('/root/autodl-tmp/movie')
OUT=PROJECT/'outputs/entitybench_cs_face_corrected_character_crops_20ep_20261014'
ROOTS={
 'A':PROJECT/'outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep',
 'B':PROJECT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep',
 'C':PROJECT/'outputs/entitybench_flux_portraitagent_photo_i2v_a14b_4step_continuation_20ep'}
NAMES={'A':'SDXL→FLUX','B':'Qwen','C':'FLUX→FLUX'}
OFFICIAL=ROOTS['C']/'runtime/evaluate_benchmark_official.py'
OFFICIAL_SHA='2334aad4b61cae2a0ea38cc2ee9ae85125b6df4e128610903654fd2b49607bb4'
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)

def read(p):return json.loads(Path(p).read_text())
def write(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scoredir(root):return root/'evaluation'/('official' if root==ROOTS['A'] else 'official_parallel_v2')
def artifacts(root,eid):
 p=list(scoredir(root).glob('**/eval_artifacts/'+eid));assert len(p)==1
 return p[0]
def original_reports():
 result=[]
 comp=read(PROJECT/'outputs/entitybench_paper_flux_qwen_portraitflux_photo20_comparison/comparison.json')
 for info in comp['reports'].values():
  for field in ['source','archive']:
   assert sha(info[field])==info['sha256']
   result.append(dict(path=info[field],sha256=info['sha256']))
 return result
def official():
 assert sha(OFFICIAL)==OFFICIAL_SHA
 for key,path in dict(DINO_MODEL_PATH='/root/autodl-tmp/models/dinov2-base',
  CLIP_MODEL_PATH='/root/autodl-tmp/models/clip-vit-base-patch32',
  GDINO_CONFIG='/root/autodl-tmp/models/entitybench_groundingdino/GroundingDINO_SwinT_OGC.py',
  GDINO_WEIGHTS='/root/autodl-tmp/models/entitybench_groundingdino/groundingdino_swint_ogc.pth').items():os.environ[key]=path
 spec=importlib.util.spec_from_file_location('corrected_crop_official',OFFICIAL)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 import torch
 torch.set_num_threads(4)
 return mod

def sheet(rows,path,title,per_page=48):
 cols=6;cellw=230;cellh=200;nr=(len(rows)+cols-1)//cols
 canvas=Image.new('RGB',(cols*cellw,45+nr*cellh),'#171a1e');d=ImageDraw.Draw(canvas)
 d.text((7,7),title,font=FONT,fill='white')
 for i,r in enumerate(rows):
  x=i%cols*cellw;y=45+i//cols*cellh
  d.text((x+4,y),f"{r['id']} {r['entity']} {r['shot_key']}",font=FONT,fill='#8addff')
  im=Image.open(r['crop_path']).convert('RGB');im.thumbnail((cellw-8,cellh-45))
  canvas.paste(im,(x+4,y+24))
  d.text((x+4,y+cellh-20),r.get('extra_label',''),font=FONT,fill='white')
 canvas.save(path,quality=92)

def prepare():
 assert not (OUT/'input_manifest.json').exists(),'Use existing immutable input manifest.'
 OUT.mkdir(parents=True,exist_ok=True);rows=[];episodes=[];fingerprints=[];counts={}
 for route,root in ROOTS.items():
  manifest=read(root/'manifest.json');gen=read(root/'generation_report.json')
  assert gen['status']=='complete' and len(gen['jobs'])==363 and len(manifest['episodes'])==20
  jobs={(j['episode_id'],j['shot_key']):j for j in gen['jobs']}
  fingerprints.extend([dict(path=str(root/'manifest.json'),sha256=sha(root/'manifest.json')),dict(path=str(root/'generation_report.json'),sha256=sha(root/'generation_report.json'))])
  original_count=0;present_count=0
  for ep in manifest['episodes']:
   eid=ep['episode_id'];assert sha(ep['episode_path'])==ep['episode_sha256']
   if route=='A':episodes.append(dict(episode_id=eid,tier=ep['difficulty'],episode_path=ep['episode_path']))
   desc={v['name']:v.get('description','') for v in ep['entity_descriptions'].values() if isinstance(v,dict)}
   result_path=scoredir(root)/(eid+'.json');res=read(result_path)
   fingerprints.append(dict(path=str(result_path),sha256=sha(result_path)))
   original_count+=res['metrics']['cs_face']['n_evaluated']
   art=artifacts(root,eid)
   for shot in ep['shots']:
    sk=shot['shot_key'];eval_path=art/'intra_shot'/sk.replace(':','_')/'_eval.json'
    ev=read(eval_path);job=jobs[(eid,sk)]
    assert sha(job['output_video'])==job['video_sha256']
    for item in ev['presence']['character']['items']:
     if item['status']!='present':continue
     present_count+=1;path=Path(item['canonical_path'])
     rows.append(dict(id=f"{route}{present_count:04d}",route=route,episode_id=eid,
      entity=item['name'],description=desc[item['name']],shot_key=sk,shot_index=shot['shot_index'],
      cut=shot['cut'],prompt=job['prompt'],scheduled_characters=job['entity_schedule']['characters'],
      crop_path=str(path),crop_sha256=sha(path),original_sample_idx=item['frame_idx'],
      original_clip_sim=item['clip_sim'],video_path=job['output_video'],video_sha256=job['video_sha256'],
      extra_label=eid.split('__')[1],review_status='pending'))
  counts[route]=dict(original_present=present_count,original_cs_face_appearances=original_count)
 pages=[];(OUT/'review/original').mkdir(parents=True,exist_ok=True)
 for start in range(0,len(rows),48):
  batch=rows[start:start+48];page=len(pages)+1;path=OUT/'review/original'/f'page_{page:03d}.jpg'
  sheet(batch,path,f'Original character crops | page {page} | scores hidden | keep appearance drift')
  pages.append(dict(page=page,path=str(path),ids=[r['id'] for r in batch]))
 data=dict(protocol='manual_localization_correction_v1',scope='same20_3routes_no_vlm_no_insightface_no_generation',
  eligibility='original status=present only; weak/absent population not expanded',
  review_rule='Review localization only. Retain rendered appearance/gender/style drift. Repair non-person or wrong-location crops using original five sampling points. Ambiguous target absence recorded, never invented.',
  sampling='official extract_frames(video,5), for81frames indices1,20,40,59,79',
  replacement_detector_query='person',original_detection_thresholds=dict(box=.25,text=.2),
  replacement_selection='official CLIP(full original description) × sharpness × crop area; require CLIP>=0.2; manual spatial role verification before DINO scoring',
  crop_pad_ratio=.1,official_source_sha256=OFFICIAL_SHA,script_sha256=sha(__file__),
  original_reports=original_reports(),fingerprints=fingerprints,episodes=episodes,counts=counts,rows=rows,pages=pages)
 write(OUT/'input_manifest.json',data)
 write(OUT/'review/original_decisions.json',dict(reviewed_pages=[],decisions={}))
 print('Prepared',len(rows),'original present crops;',len(pages),'pages',flush=True)

def validate(data):
 assert sha(OFFICIAL)==data['official_source_sha256']
 for item in data['fingerprints']+data['original_reports']:assert sha(item['path'])==item['sha256'],item['path']
 for row in data['rows']:assert sha(row['crop_path'])==row['crop_sha256'],row['id']

def reviewed_rows(data):
 review=read(OUT/'review/original_decisions.json')
 assert set(review['reviewed_pages'])=={p['page'] for p in data['pages']},'All original pages must be visually reviewed.'
 assert set(review['decisions'])=={r['id'] for r in data['rows']},'Each original present crop requires a decision.'
 for r in data['rows']:
  d=review['decisions'][r['id']];assert d['action'] in ['keep','repair','unresolved']
  yield r,d

def repair():
 data=read(OUT/'input_manifest.json');validate(data);pairs=list(reviewed_rows(data));mod=official()
 targets=[r for r,d in pairs if d['action']=='repair'];cache=OUT/'repair/candidates.json'
 saved=read(cache) if cache.exists() else {};by_video={};beg=time.monotonic()
 for row in targets:
  if row['id'] in saved:continue
  assert sha(row['video_path'])==row['video_sha256']
  key=(row['episode_id'],row['shot_key'],row['route'])
  if key not in by_video:by_video[key]=mod.extract_frames(row['video_path'],5)
  frames=by_video[key];candidates=[];folder=OUT/'repair'/row['id'];folder.mkdir(parents=True,exist_ok=True)
  for fi,frame in enumerate(frames):
   # Reuse original detector implementation/settings with an explicit person query.
   dets=mod.ground_one_entity_in_frame(frame,row['entity'],'person','character','cuda',.25,.20,True)
   Image.fromarray(frame).save(folder/f'frame_{fi}.jpg',quality=93)
   for di,det in enumerate(dets):
    crop=mod.crop_entity(frame,det['bbox'],.1)
    if crop.size==0:continue
    clip=mod.compute_clip_text_sim(crop,row['description'],'cuda')
    components=mod.compute_selection_score(clip,crop,frame.shape)
    path=folder/f'candidate_f{fi}_d{di}.png';Image.fromarray(crop).save(path)
    candidates.append(dict(id=f"{row['id']}_f{fi}_d{di}",crop_path=str(path),crop_sha256=sha(path),
      sample_idx=fi,bbox=list(det['bbox']),clip_sim=clip,selection=components['selection'],
      components=components,grounding_confidence=det['confidence'],frame_path=str(folder/f'frame_{fi}.jpg'),
      frame_shape=list(frame.shape)))
  candidates.sort(key=lambda c:c['selection'],reverse=True)
  saved[row['id']]=dict(row=row,candidates=candidates)
  write(cache,saved)
  # Display top six candidates plus the full five original sampling frames.
  w=1500;h=760;canvas=Image.new('RGB',(w,h),'#181b21');d=ImageDraw.Draw(canvas)
  d.text((8,8),f"{row['id']} {row['entity']} {row['episode_id']} shot {row['shot_key']}",font=FONT,fill='white')
  d.text((8,29),'Spatial role/localization review; no DINO scores; top6 person candidates; original sampled context below',font=FONT,fill='white')
  for i,c in enumerate(candidates[:6]):
   im=Image.open(c['crop_path']);im.thumbnail((244,230));canvas.paste(im,(i*250,62))
   d.text((i*250+2,299),c['id'],font=FONT,fill='#8addff')
   d.text((i*250+2,318),f"CLIP {c['clip_sim']:.3f}",font=FONT,fill='white')
  for i,frame in enumerate(frames):
   im=Image.fromarray(frame);im.thumbnail((298,240));canvas.paste(im,(i*300,371))
   d.text((i*300+3,615),f'sample {i}',font=FONT,fill='white')
  im=Image.open(row['crop_path']);im.thumbnail((240,100));canvas.paste(im,(10,645))
  d.text((262,645),'Original suspect crop',font=FONT,fill='white')
  text=row['description']+' | '+row['prompt']
  for i in range(4):d.text((262,666+i*20),text[i*130:(i+1)*130],font=FONT,fill='white')
  canvas.save(folder/'review.jpg',quality=93)
  print('Repair proposals',row['id'],len(candidates),'candidates',flush=True)
  if len(by_video)>12:by_video.clear()
 write(OUT/'repair/status.json',dict(status='proposals_complete',targets=len(targets),elapsed_seconds=time.monotonic()-beg))

def score():
 data=read(OUT/'input_manifest.json');validate(data);pairs=list(reviewed_rows(data))
 repairs=read(OUT/'repair/candidates.json') if (OUT/'repair/candidates.json').exists() else {}
 decisions=read(OUT/'review/repair_decisions.json') if (OUT/'review/repair_decisions.json').exists() else {}
 review_sha=sha(OUT/'review/original_decisions.json');resolved=[]
 for row,d in pairs:
  chosen=None;reason=d['reason'];action=d['action'];clip=row['original_clip_sim'];fi=row['original_sample_idx']
  if action=='keep':chosen=row['crop_path']
  elif action=='repair':
   assert row['id'] in decisions and row['id'] in repairs,row['id'];rd=decisions[row['id']]
   reason=reason+'; '+rd['reason']
   if rd['candidate_id'] is not None:
    c=next(c for c in repairs[row['id']]['candidates'] if c['id']==rd['candidate_id'])
    assert c['clip_sim']>=.2,'No threshold relaxation.'
    assert sha(c['crop_path'])==c['crop_sha256'];chosen=c['crop_path'];clip=c['clip_sim'];fi=c['sample_idx'];action='repaired'
   else:action='unresolved'
  resolved.append(dict(**row,decision=action,reason=reason,selected_crop_path=chosen,selected_crop_sha256=sha(chosen) if chosen else None,selected_clip_sim=clip,selected_sample_idx=fi))
 correction_manifest=dict(input_manifest_sha256=sha(OUT/'input_manifest.json'),review_sha256=review_sha,
  repair_review_sha256=sha(OUT/'review/repair_decisions.json') if decisions else None,rows=resolved)
 if (OUT/'correction_manifest.json').exists():assert read(OUT/'correction_manifest.json')==correction_manifest
 else:write(OUT/'correction_manifest.json',correction_manifest)
 # Only after all manual input decisions are frozen is corrected DINO computed.
 mod=official();embedding_cache=OUT/'embeddings';embedding_cache.mkdir(exist_ok=True)
 for r in resolved:
  if not r['selected_crop_path']:continue
  cache=embedding_cache/(r['selected_crop_sha256']+'.npy')
  if not cache.exists():np.save(cache,mod.compute_dino_embedding(np.asarray(Image.open(r['selected_crop_path']).convert('RGB')),'cuda'))
  emb=np.load(cache);assert abs(float(np.linalg.norm(emb))-1)<1e-5
  r['_embedding']=emb
 route_results={};all_entities=[]
 split={tier:[e['episode_id'] for e in data['episodes'] if e['tier']==tier] for tier in ['easy','medium','hard']}
 for route,root in ROOTS.items():
  episodic=[];count=Counter(r['decision'] for r in resolved if r['route']==route)
  for ep in data['episodes']:
   eid=ep['episode_id'];rs=[r for r in resolved if r['route']==route and r['episode_id']==eid]
   crops=defaultdict(list)
   for r in rs:
    if '_embedding' not in r:continue
    crops[r['entity']].append(mod.EntityCrop(entity_name=r['entity'],entity_type='character',
     entity_description=r['description'],shot_key=r['shot_key'],frame_idx=r['selected_sample_idx'],
     bbox=(0,0,0,0),crop_path=r['selected_crop_path'],clip_text_sim=r['selected_clip_sim'],dino_embedding=r['_embedding']))
   order=[s['shot_key'] for s in read(root/'manifest.json')['episodes'] if s['episode_id']==eid for s in s['shots']]
   val,entities,n=mod.compute_cross_shot_consistency(dict(crops),'character',shot_order=order)
   original=read(scoredir(root)/(eid+'.json'))
   out=mod.EpisodeResult(episode_id=eid,tier=ep['tier'],num_shots=original['num_shots'])
   skipped=sum(r['decision']=='unresolved' for r in rs)
   out.metrics['cs_face']=mod.MetricValue(value=val,n_evaluated=n,n_failed=0,n_skipped=skipped)
   out.per_entity_scores={f'face_{k}':v for k,v in entities.items()}
   payload=dict(episode_id=eid,tier=ep['tier'],num_shots=out.num_shots,
    metrics={'cs_face':dict(value=val,n_evaluated=n,n_failed=0,n_skipped=skipped)},
    per_entity_scores=out.per_entity_scores,scope='localization_corrected_non_vlm_conditional',
    original_cs_face=original['metrics']['cs_face'])
   write(OUT/'results'/route/(eid+'.json'),payload);episodic.append(out)
   all_entities.extend(dict(route=route,episode_id=eid,entity=k,**v) for k,v in entities.items())
  agg=mod.aggregate_results(episodic,split)
  nonnull=[e.metrics['cs_face'] for e in episodic if e.metrics['cs_face'].value is not None]
  weighted=sum(m.value*m.n_evaluated for m in nonnull)/sum(m.n_evaluated for m in nonnull)
  eligible_before=data['counts'][route]['original_present'];nvalid=sum(r['selected_crop_path'] is not None for r in resolved if r['route']==route)
  route_results[route]=dict(name=NAMES[route],counts=dict(count),original_present=eligible_before,
   valid_localization_crops=nvalid,localization_coverage=nvalid/eligible_before,
   aggregate=agg,instance_weighted_raw_diagnostic=weighted,n_cs_face_appearances=sum(m.n_evaluated for m in nonnull))
 # Matched support diagnostic: same character/shot indices in all three routes.
 supports={route:{(r['episode_id'],r['entity'],r['shot_key']) for r in resolved if r['route']==route and '_embedding' in r} for route in ROOTS}
 common=set.intersection(*supports.values());common_report={}
 for route in ROOTS:
  values=[];total=0;summed=0
  for ep in data['episodes']:
   pools=defaultdict(list)
   for r in resolved:
    if r['route']==route and r['episode_id']==ep['episode_id'] and (r['episode_id'],r['entity'],r['shot_key']) in common:pools[r['entity']].append(r['_embedding'])
   sims=[]
   for embs in pools.values():
    if len(embs)<2:continue
    embs=np.stack(embs);cent=embs.mean(0);cent/=np.linalg.norm(cent);sims.extend((embs@cent).tolist())
   if sims:values.append(float(np.mean(sims)));total+=len(sims);summed+=sum(sims)
  common_report[route]=dict(episode_macro_mean=float(np.mean(values)) if values else None,instance_weighted_raw=summed/total if total else None,n_episodes=len(values),n_appearances=total)
 report=dict(status='complete',metric='cs_face_localization_corrected',
  scope='conditional_on_verified_character_localization_without_vlm_not_paper_table_reproduction',
  crop_type='original character region, not face-only',original_cs_face_unmodified=True,
  vlm_calls=0,insightface_used=False,official_formula_source_sha256=OFFICIAL_SHA,
  correction_manifest_sha256=sha(OUT/'correction_manifest.json'),routes=route_results,
  common_support=dict(n_character_shot_instances=len(common),routes=common_report),entities=all_entities)
 write(OUT/'report.json',report)
 # Verify archived/source original reports did not change after all computations.
 for r in data['original_reports']:assert sha(r['path'])==r['sha256']
 print(json.dumps({k:{'cs_face':v['aggregate']['all']['cs_face'],'counts':v['counts'],'weighted':v['instance_weighted_raw_diagnostic']} for k,v in route_results.items()},ensure_ascii=False),flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','repair','score']);args=parser.parse_args()
 {'prepare':prepare,'repair':repair,'score':score}[args.stage]()
if __name__=='__main__':main()
