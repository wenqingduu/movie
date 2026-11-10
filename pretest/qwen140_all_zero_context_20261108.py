"""Dependency job: after all candidates, score complete qualifying contexts.

No generation, no PID-bound handoff, no retries, no publication before human
localization/action review. Reuse only input-verified per-video VBench scores.
"""
from __future__ import annotations
import argparse,math,os,shutil,sys,time
from pathlib import Path
from collections import defaultdict
import qwen140_all_zero_motion_20261108 as batch
import qwen140_full140_latest_20261106 as f
import qwen140_v2_shots_only_eval_20261105 as execution

ROOT=batch.ROOT;OUT=batch.OUT/'context_evaluation';BASE=batch.BASE
read,write,sha,immutable=batch.read,batch.write,batch.sha,batch.immutable

def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,utc=batch.now(),vlm_calls=0,api_calls=0,media_generation=False,final_publication_pending=True,**kw))
def link(src,dst):
 src=Path(src).resolve();dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert dst.resolve()==src
 else:dst.symlink_to(src,target_is_directory=src.is_dir())

def prepare():
 if (OUT/'plan.json').exists():
  p=read(OUT/'plan.json')
  for x in p['bindings']:assert sha(x['path'])==x['sha256']
  return p
 p=batch.prepare();assert read(batch.OUT/'status.json')['generated']==1333
 selected={c['episode_id']+'/'+c['shot']:c for c in p['cases'] if read(Path(c['output']).parent/'vbench.json')['vbench_pass']}
 snapshot=read(batch.SNAP);items=[x for x in snapshot['episodes'] if x['episode_id'] in {k.split('/')[0] for k in selected}]
 bindings={};cache=dict(dimensions=batch.DIMS,evaluator_sha256=sha(execution.full.EVALUATOR),shots={})
 expected=read(BASE/'evaluation/official_parallel_v2/run_manifest.json');media={};rawcache={};proofcache={}
 for item in items:
  eid=item['episode_id'];ep=item['episode'];records=[];source=Path(item['result']);manifest=source.resolve().parent/'run_manifest.json';m=read(manifest)
  for field in ['config','python_version','libraries','model_checkpoints','pillars','evaluation_scope']:assert m[field]==expected[field],(eid,field)
  assert m['strict'] and m['skip_vlm'] and m['vlm_call_stats']['total_calls']==0
  bindings[str(source)]=sha(source);bindings[str(manifest)]=sha(manifest)
  for shot in ep['shots']:
   sk=shot['shot_key'];key=eid+'/'+sk;c=selected.get(key);old=snapshot['media'][key]
   if c:
    recordpath=Path(c['output']).parent/'result.json';record=read(recordpath);video=record['video'];vsha=record['video_sha256']
    scorepath=Path(c['output']).parent/'vbench.json';scores=read(scorepath)['after'];bindings[str(scorepath)]=sha(scorepath);bindings[str(recordpath)]=sha(recordpath)
    score_sources=[dict(path=str(scorepath),sha256=sha(scorepath))]
   else:
    video,vsha=old['source'],old['sha256'];scores={};score_sources=[];scene,num=map(int,sk.split(':'))
    for dim in batch.DIMS:
     path=source.parent/'.work'/f'{eid}_{dim}_eval_results.json'
     if str(path) not in proofcache:proofcache[str(path)]=read(path);bindings[str(path)]=sha(path)
     matches=[r['video_results'] for r in proofcache[str(path)][dim][1] if Path(r['video_path']).stem==f'{scene:03d}_{num:03d}'];assert len(matches)==1
     scores[dim]=float(matches[0]);score_sources.append(dict(path=str(path),sha256=bindings[str(path)]))
   assert sha(video)==vsha and all(math.isfinite(v) for v in scores.values())
   a,b=map(int,sk.split(':'));canonical=OUT/'candidate_media'/eid/f'{a:03d}_{b:03d}.mp4';link(video,canonical)
   media[key]=dict(source=video,sha256=vsha,before_source=old['source'],before_sha256=old['sha256'],candidate=bool(c))
   records.append(dict(shot_key=sk,cut=shot['cut'],video=video,video_sha256=vsha))
   cache['shots'][key]=dict(reuse=True,video_sha256=vsha,scores=scores,score_sources=score_sources)
  immutable(OUT/'episode_media/episodes'/eid/'episode_result.json',dict(episode_id=eid,shots=records))
 for path in [Path(__file__),Path(execution.__file__),ROOT/'pretest/qwen140_vbench_verified_cache_20261105.py',
              batch.OUT/'plan.json',batch.OUT/'selection_policy.json',batch.SNAP,execution.full.EVALUATOR]:bindings[str(path)]=sha(path)
 cache['bindings']=[dict(path=k,sha256=v) for k,v in bindings.items()];immutable(OUT/'vbench_cache_manifest.json',cache)
 plan=dict(episodes=[x['episode'] for x in items],recompute_episode_ids=[x['episode_id'] for x in items],
  affected_episodes=len(items),complete_context_shots=sum(len(x['episode']['shots']) for x in items),preliminary_candidate_shots=len(selected),
  target_cases=selected,bindings=cache['bindings'],workers=2,threads=4,source_generation_plan_sha256=sha(batch.OUT/'plan.json'),
  scope='Provisional all-VBench-pass sequence; not final selected benchmark. Full affected episode grounding/presence/centres/boundaries. Default reviewed DINO and action review required before final filtering.')
 immutable(OUT/'media.json',media);immutable(OUT/'plan.json',plan);status('prepared',affected_episodes=len(items),candidate_shots=len(selected));return plan

def diagnostic(p):
 import qwen140_localization_repair_gpu_20261108 as baseline
 import qwen140_badshot_reviewed_dino_20261104 as reviews
 maps=baseline.gate_maps()['current'];audit=read(baseline.AUDIT/'report.json');rej={f.diag.key(x):x for x in audit['applied_exclusions']['current']}
 selected=set(p['target_cases']);directory=OUT/'visible_region';newrows=dict(maps);scored=[];comparisons=[];entities=[]
 for ep in p['episodes']:
  eid=ep['episode_id']
  for path in (directory/'extraction'/eid).glob('*.json'):
   for r in read(path)['rows']:newrows[f.diag.key(r)]=r
  for k,row in list(newrows.items()):
   if k[0]==eid:row['edition']='rerun' if eid+'/'+k[2] in selected and row['original_crop_sha256']!=maps[k]['original_crop_sha256'] else 'unchanged'
 carry={k:r for k,r in rej.items() if newrows[k]['original_crop_sha256']==r['original_crop_sha256']}
 mod=f.diag.gate.load_official_pure();snap={x['episode_id']:x for x in read(batch.SNAP)['episodes']}
 for ep in p['episodes']:
  eid=ep['episode_id'];raw=read(OUT/'affected/results'/(eid+'.json'));value,per=f.diag.compute(ep,raw,newrows,carry,mod)
  write(directory/'automatic/results'/(eid+'.json'),dict(episode_id=eid,metrics={'cs_face':value.metrics['cs_face'].to_json()},per_entity_scores=per))
  for name,data in per.items():
   for app in data['per_shot_sim_to_centroid']:scored.append({**newrows[eid,name,app['shot']],'similarity':app['sim_to_centroid']})
  for shot in ep['shots']:
   sk=shot['shot_key'];key=eid+'/'+sk
   if key not in selected:continue
   oldfile=Path(snap[eid]['result']).parent/'.work/eval_artifacts'/eid/'intra_shot'/sk.replace(':','_')/'_eval.json'
   newfile=OUT/'affected/results/.work/eval_artifacts'/eid/'intra_shot'/sk.replace(':','_')/'_eval.json'
   before,after=read(oldfile),read(newfile);presence={};ok=True
   for kind in ['character','object','location']:
    a,b=before['presence'][kind],after['presence'][kind];x=a['n_present']/a['n_total'] if a['n_total'] else None;y=b['n_present']/b['n_total'] if b['n_total'] else None
    passed=(x is None and y is None) or (x is not None and y is not None and y>=x-1e-8);ok &=passed
    presence[kind]=dict(before=x,after=y,pass_gate=passed)
   comparisons.append(dict(episode_id=eid,shot=sk,presence=presence,presence_pass=ok,default_reviewed_dino='pending_localization_review',final_eligible=False))
 write(directory/'scored_crops.json',scored);reviews.OUT=directory
 reviews.review_sheets(scored,[dict(episode_id=e['episode_id'],episode=e) for e in p['episodes']])
 write(OUT/'entity_presence_comparison.json',comparisons)
 visual=[dict(case_id=c['case_id'],episode_id=c['episode_id'],shot=c['shot'],description=c['official_description'],before=c['baseline_video'],
  after=read(Path(c['output']).parent/'result.json')['video'],decision='pending') for c in p['target_cases'].values()]
 immutable(OUT/'visual_review/manifest.json',dict(generation_plan_sha256=sha(batch.OUT/'plan.json'),required_rules=batch.POLICY['visual_reject'],rows=visual))
 status('official_context_and_gate_complete_pending_human_review',candidate_shots=len(selected),presence_pass=sum(x['presence_pass'] for x in comparisons),
  localization_review_items=len(read(directory/'review/manifest.json')['rows']),action_review_items=len(visual),final_replacements=0,
  pending=['score_hidden_localization_decisions','visual_action_decisions','default_reviewed_dino_recompute','threshold_filter','joint_selected_sequence_recompute','publish_separate_selected_full140'])

def run():
 p=prepare()
 if not p['preliminary_candidate_shots']:
  status('complete_no_vbench_qualifiers',candidates_attempted=1333,final_replacements=0);return
 # Only the reviewed parallel execution pattern, never its old PID handoff/run.
 saved_out,saved_file=execution.OUT,execution.__file__
 try:
  execution.OUT=OUT;execution.__file__=str(Path(__file__));execution.evaluate(p);execution.dino_prepare(p)
 finally:
  # f.J aliases this module too; restore historical provenance before loading
  # the frozen baseline visibility maps for the new diagnostic.
  execution.OUT,execution.__file__=saved_out,saved_file
 diagnostic(p)

def wait_run():
 status('waiting_for_candidate_generation_and_vbench',generation_plan_sha256=sha(batch.OUT/'plan.json'))
 while True:
  s=read(batch.OUT/'status.json')
  if s['stage']=='all_candidates_and_vbench_complete_context_review_pending':break
  if s['stage']=='stopped_technical_error_preserve_outputs':
   status('stopped_upstream_error',upstream=s);return
  time.sleep(30)
 try:run()
 except Exception as e:status('stopped_context_error_outputs_preserved',error=repr(e));raise

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='scorer':
  import qwen140_vbench_verified_cache_20261105 as adapter
  adapter.CACHE=OUT/'vbench_cache_manifest.json';sys.argv.pop(1)
  if '--method_name' in sys.argv:sys.argv[sys.argv.index('--method_name')+1]='all_zero_motion_preliminary_context_20261108'
  adapter.main()
 else:
  parser=argparse.ArgumentParser();parser.add_argument('action',choices=['wait-run','prepare','run']);a=parser.parse_args()
  if a.action=='wait-run':wait_run()
  elif a.action=='prepare':prepare()
  else:run()
