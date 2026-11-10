"""Paired scene-only evaluation of a frozen completed background-enhanced cohort.
Original grounding/crop/CLIP/thresholds; no VLM, generation, crop repair or centroid metric.
Pause the current generation worker after its next persisted shot, score, resume in finally.
"""
from __future__ import annotations
import argparse,importlib.util,json,os,signal,subprocess,sys,time,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
ROOT=Path('/root/autodl-tmp/movie')
GEN=ROOT/'outputs/entitybench_all_zero_motion_background_20261108'
OUT=GEN/'scene_presence_completed_cohort_20261108'
BASE=ROOT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140'
OFFICIAL=BASE/'runtime/evaluate_benchmark_official.py'
INVENTORY=ROOT/'outputs/entitybench_cached_matching_audit_20261107/inventory.json'
SNAP=ROOT/'outputs/entitybench_full140_latest57_v2_141_20261106/snapshot.json'
EVAL='/root/autodl-tmp/envs/entitybench-eval/bin/python'
GEN_WORKER=69726

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def state(stage,**kw):write(OUT/'status.json',dict(stage=stage,utc=now(),gpu_worker=GEN_WORKER,api_calls=0,vlm_calls=0,media_generation=False,**kw))
def immutable(p,x):
 if Path(p).exists():assert read(p)==x,str(p)
 else:write(p,x)

def prepare():
 if (OUT/'plan.json').exists():
  p=read(OUT/'plan.json')
  for b in p['bindings']:assert sha(b['path'])==b['sha256'],b['path']
  return p
 gp=read(GEN/'plan.json');snapshot=read(SNAP);eps={x['episode_id']:x['episode'] for x in snapshot['episodes']};inv=read(INVENTORY);items={}
 for r in inv['rows']:
  if r['kind']!='location':continue
  for ref in r['references']:
   if ref['edition']=='current':items[r['episode_id'],r['shot'],r['name']]=(r,ref)
 bindings={str(f):sha(f) for f in [GEN/'plan.json',SNAP,INVENTORY,OFFICIAL,Path(__file__),BASE/'evaluation/official_parallel_v2/run_manifest.json']};cases=[]
 for c in gp['cases']:
  path=Path(c['output']).parent/'result.json'
  if not c.get('background_cue') or not path.exists():continue
  r=read(path);assert sha(r['video'])==r['video_sha256'];assert sha(c['baseline_video'])==c['baseline_video_sha256']
  ep=eps[c['episode_id']];shot=next(x for x in ep['shots'] if x['shot_key']==c['shot']);schedule={'characters':[],'objects':[],'places':shot['entity_schedule'].get('places',[])}
  old=[]
  for name in schedule['places']:
   item,ref=items[c['episode_id'],c['shot'],name];assert ref['video_sha256']==c['baseline_video_sha256'];assert sha(ref['evaluation'])==ref['evaluation_sha256'];assert sha(item['crop'])==item['crop_sha256']
   old.append(dict(name=name,status=ref['status'],clip_sim=ref['clip_sim'],frame_idx=item['sample_idx'],crop=item['crop'],crop_sha256=item['crop_sha256'],evaluation=ref['evaluation'],evaluation_sha256=ref['evaluation_sha256']))
   bindings[ref['evaluation']]=ref['evaluation_sha256']
  bindings[str(path)]=sha(path)
  cases.append(dict(case_id=c['case_id'],episode_id=c['episode_id'],run=c['run'],shot=c['shot'],video=r['video'],video_sha256=r['video_sha256'],
   before_video=c['baseline_video'],before_video_sha256=c['baseline_video_sha256'],schedule=schedule,entity_descriptions=ep['entity_descriptions'],before=old,
   background_cue=c['background_cue'],all_generated_included=True,original_description=c['official_description']))
 assert cases
 p=dict(created_utc=now(),cases=cases,total=len(cases),generation_plan_sha256=sha(GEN/'plan.json'),
  scope='All completed background-enhanced shots at request-time freeze; all included irrespective of dynamic/VBench outcome. Paired target-shot scene presence only, not a full benchmark update or isolated background ablation.',
  bindings=[dict(path=k,sha256=v) for k,v in bindings.items()],sampling='Original extract_frames(n_frames=5, strict=True)',
  thresholds=read(BASE/'evaluation/official_parallel_v2/run_manifest.json')['config'],vlm_calls=0,api_calls=0,grounding_function='Unmodified pinned ground_shot_unified with places-only schedule',
  baseline_validation='Re-ground first baseline video with original sampled frames and require identical status/sample/crop SHA and CLIP within1e-5; no threshold or box changes.')
 immutable(OUT/'plan.json',p);state('prepared',expected=len(cases));return p

def load():
 m=read(BASE/'evaluation/official_parallel_v2/run_manifest.json')
 for k in ['GDINO_CONFIG','GDINO_WEIGHTS','DINO_MODEL_PATH','CLIP_MODEL_PATH']:os.environ[k]=m['model_checkpoints'][k]['path']
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',VBENCH_CACHE_DIR='/root/autodl-tmp/models/entitybench_vbench')
 spec=importlib.util.spec_from_file_location('scene_presence_frozen_official',OFFICIAL);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 import torch,cv2
 torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert mod.BENCHMARK_CONFIG==m['config'];mod.get_grounding_dino('cuda',strict=True);mod.get_clip_model('cuda');mod.get_dino_model('cuda')
 return mod

def ground(mod,c,video,out):
 frames=mod.extract_frames(video,n_frames=5,strict=True);cfg=mod.BENCHMARK_CONFIG
 result=mod.ground_shot_unified(c['shot'],frames,c['schedule'],c['entity_descriptions'],str(out),'cuda',
  {'box_threshold':cfg['gdino_box_threshold'],'text_threshold':cfg['gdino_text_threshold']},cfg['clip_match_threshold'])
 items=[]
 for name in c['schedule']['places']:
  r=result['per_entity'][name];canonical=r['canonical'];items.append(dict(name=name,status=r['status'],clip_sim=canonical.clip_text_sim if canonical else None,
   frame_idx=canonical.frame_idx if canonical else None,crop=r['canonical_path'],crop_sha256=sha(r['canonical_path']) if r['canonical_path'] else None,
   canonical_bbox=list(canonical.bbox) if canonical else None,selection=r['canonical_score_components'],n_detections=len(r['all_detections'])))
 return items

def score():
 p=prepare();start=time.monotonic();mod=load()
 import torch
 torch.cuda.reset_peak_memory_stats()
 proof=OUT/'validation/baseline_replay.json'
 if not proof.exists():
  c=p['cases'][0];new=ground(mod,c,c['before_video'],OUT/'validation/canonical_crops');comparisons=[]
  for a,b in zip(c['before'],new):
   assert a['name']==b['name'] and a['status']==b['status'] and a['frame_idx']==b['frame_idx'],(a,b)
   assert a['crop_sha256']==b['crop_sha256'],(a,b)
   assert abs(a['clip_sim']-b['clip_sim'])<1e-5,(a,b)
   comparisons.append(dict(name=a['name'],same_status=True,same_sample=True,same_crop_sha256=True,clip_absolute_error=abs(a['clip_sim']-b['clip_sim'])))
  immutable(proof,dict(plan_sha256=sha(OUT/'plan.json'),case_id=c['case_id'],comparisons=comparisons))
 for index,c in enumerate(p['cases'],1):
  folder=OUT/'cases'/c['case_id'];path=folder/'result.json'
  if path.exists():
   r=read(path);assert r['video_sha256']==c['video_sha256'] and r['plan_sha256']==sha(OUT/'plan.json')
   continue
  assert sha(c['video'])==c['video_sha256'];items=ground(mod,c,c['video'],folder/'canonical_crops')
  n=len(items);before=sum(x['status']=='present' for x in c['before'])/n;after=sum(x['status']=='present' for x in items)/n
  immutable(path,dict(case_id=c['case_id'],episode_id=c['episode_id'],run=c['run'],shot=c['shot'],before=c['before'],after=items,
   before_presence=before,after_presence=after,delta=after-before,n_total=n,video=c['video'],video_sha256=c['video_sha256'],
   before_video=c['before_video'],before_video_sha256=c['before_video_sha256'],plan_sha256=sha(OUT/'plan.json')))
  state('scoring_scene_presence',completed=index,expected=p['total'],elapsed_seconds=time.monotonic()-start,gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
  print(c['case_id'],index,'/',p['total'],before,'->',after,flush=True)
 report(p,start,torch.cuda.max_memory_allocated()/1024**3)

def report(p,start,peak):
 rows=[read(OUT/'cases'/c['case_id']/'result.json') for c in p['cases']];n=len(rows);eps=defaultdict(list)
 for r in rows:eps[r['episode_id']].append(r)
 means={}
 for side in ['before','after']:
  means[side]=dict(shot_mean=sum(r[side+'_presence'] for r in rows)/n,
   selected_episode_macro=sum(sum(r[side+'_presence'] for r in a)/len(a) for a in eps.values())/len(eps),
   scheduled_instance_mean=sum(x['status']=='present' for r in rows for x in r[side])/sum(r['n_total'] for r in rows),
   status_counts=dict(Counter(x['status'] for r in rows for x in r[side])))
 summary=dict(scope=p['scope'],shots=n,episodes=len(eps),scheduled_instances=sum(r['n_total'] for r in rows),means=means,
  improved=sum(r['delta']>0 for r in rows),unchanged=sum(r['delta']==0 for r in rows),declined=sum(r['delta']<0 for r in rows),
  elapsed_seconds=time.monotonic()-start,gpu_peak_allocated_gib=peak,plan_sha256=sha(OUT/'plan.json'),
  validation=read(OUT/'validation/baseline_replay.json'),full_benchmark_updated=False,final_replacement_eligibility='pending_all_metrics_and_visual_review')
 immutable(OUT/'summary.json',summary)
 lines=['# 已完成背景增强组：场景出现率配对评测','',p['scope'],'',f'冻结{n}shot，{len(eps)}个episode，{summary["scheduled_instances"]}个调度场景实例；全部纳入，不按动态或质量结果选择。','',
  '| 聚合 | 重跑前 | 动作＋背景强化后 |','|---|---:|---:|',
  *[f'| {k} | {means["before"][k]:.6f} | {means["after"][k]:.6f} |' for k in ['shot_mean','selected_episode_macro','scheduled_instance_mean']],
  '',f'提升{summary["improved"]}，不变{summary["unchanged"]}，下降{summary["declined"]}。',
  '', '子集特意来自基线场景weak镜头，基线通常接近0；不能与完整140集0.6238直接比较。前后同时改变动作词和背景词，没有同一shot仅动作强化的配对视频，不能将全部变化归因于背景句。',
  '', '使用原始5帧采样、GroundingDINO、代表裁剪选择、CLIP/100与.20阈值；一个原视频已回放验证状态/采样/裁剪SHA相同，CLIP误差<1e-5。这里只计算独立shot场景出现，不生成跨镜头指标，不修改任何正式benchmark或筛选资格。',
  '', '| case | episode/run | shot | before | after | delta |','|---|---|---|---:|---:|---:|']
 lines += [f'| {r["case_id"]} | {r["run"]}/{r["episode_id"].split("__")[2]} | {r["shot"]} | {r["before_presence"]:.4f} | {r["after_presence"]:.4f} | {r["delta"]:+.4f} |' for r in rows]
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');state('complete_scene_only',completed=n,expected=n,summary=summary)

def active(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return None
 s=(p/'stat').read_text().split()
 if s[2]=='Z':return None
 return (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()

def schedule():
 p=prepare();existing=set((GEN/'cases').glob('*/result.json'));paused=False
 try:
  state('waiting_for_next_persisted_generation_shot',expected=p['total'])
  while active(GEN_WORKER):
   cmd=active(GEN_WORKER)
   if not cmd:break
   assert 'qwen140_all_zero_background_20261108.py worker' in cmd
   if set((GEN/'cases').glob('*/result.json'))-existing:
    os.kill(GEN_WORKER,signal.SIGSTOP);paused=True;break
   time.sleep(.5)
  if not paused:raise RuntimeError('Generation worker exited before pause; leave scoring queued for explicit safe reschedule')
  time.sleep(2)
  info=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits']).decode().strip()
  state('generation_paused_scoring_scene_only',expected=p['total'],gpu_before_evaluation=info)
  env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
  with (OUT/'scorer.log').open('a') as log:subprocess.run([EVAL,'-u',str(Path(__file__)),'score'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 except Exception as e:
  state('stopped_scene_evaluation_error',error=repr(e));raise
 finally:
  if paused and active(GEN_WORKER):
   os.kill(GEN_WORKER,signal.SIGCONT)
   write(OUT/'generation_resume.json',dict(utc=now(),pid=GEN_WORKER,resumed=True,official_outputs_unchanged=True))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','schedule','score']);args=a.parse_args()
 if args.action=='prepare':print(json.dumps(dict(total=prepare()['total'],out=str(OUT))))
 elif args.action=='score':score()
 else:schedule()
