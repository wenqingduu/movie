"""Versioned full140 media overlay: select only dynamic gains vs frozen current benchmark.
No other quality gate, new generation or old-media overwrite. Complete affected
contexts use verified original per-video VBench caches; default DINO review pending.
"""
from __future__ import annotations
import argparse,copy,math,sys,time
from pathlib import Path
from collections import defaultdict
import qwen140_original40_scope32_20261108 as trial
import qwen140_full140_latest_20261106 as frozen
import qwen140_v2_shots_only_eval_20261105 as execution
ROOT=trial.ROOT
OUT=ROOT/'outputs/entitybench_original40_dynamic_gain_selected20_20261109'
SOURCE=trial.OUT
SNAP=frozen.OUT/'snapshot.json'
BASE=frozen.BASE
read,write,sha,immutable=trial.read,trial.write,trial.sha,trial.immutable

def status(stage,**kw):
 write(OUT/'status.json',dict(stage=stage,utc=trial.core.now(),api_calls=0,vlm_calls=0,media_generation=False,
  quality_gate_enabled=False,selection_rule='dynamic increases vs frozen current benchmark only',old_media_overwritten=False,**kw))

def link(src,dst):
 src=Path(src).resolve();dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert dst.resolve()==src
 else:dst.symlink_to(src,target_is_directory=src.is_dir())

def prepare():
 if (OUT/'plan.json').exists():
  p=read(OUT/'plan.json')
  for b in p['bindings']:assert sha(b['path'])==b['sha256'],b['path']
  return p
 snapshot=read(SNAP);paired=read(SOURCE/'comparison.json');assert paired['complete'] and paired['scored']==32
 cases={c['case_id']:c for c in read(SOURCE/'plan.json')['cases']};selected={};decisions=[];bindings={}
 for row in paired['rows']:
  c=cases[row['case_id']];key=c['episode_id']+'/'+c['shot'];before=row['frozen_benchmark']['dynamic_degree'];after=row['after']['dynamic_degree']
  assert before in (0,1) and after in (0,1)
  chosen=after>before;path=SOURCE/'cases'/c['case_id']/'generation.json';record=read(path)
  assert sha(record['video'])==record['video_sha256'] and sha(c['input_path'])==c['input_sha256']
  assert row['frozen_benchmark']==c['frozen_benchmark_scores'] and before==0
  decision=dict(case_id=c['case_id'],episode_id=c['episode_id'],shot=c['shot'],selected=chosen,benchmark_before_dynamic=before,
   same_prompt_4step_dynamic=row['before']['dynamic_degree'],original40_dynamic=after,other_quality_gates_applied=False,
   reason='dynamic0_to1_vs_current_benchmark' if chosen else 'no_dynamic_gain_vs_current_benchmark')
  decisions.append(decision)
  if chosen:
   target={**c,'official_description':c['original_description'],'baseline_video':snapshot['media'][key]['source'],
    'baseline_video_sha256':snapshot['media'][key]['sha256'],'output':str(OUT/'selected_cases'/c['case_id']/'main'),
    'selected_video':record['video'],'selected_video_sha256':record['video_sha256'],'after_scores':row['after']}
   immutable(Path(target['output']).parent/'result.json',dict(episode_id=c['episode_id'],shot_key=c['shot'],video=record['video'],video_sha256=record['video_sha256']))
   selected[key]=target
   for f in [path,SOURCE/'cases'/c['case_id']/'scores.json']:bindings[str(f)]=sha(f)
 assert len(selected)==20 and len(decisions)==32
 items=[e for e in snapshot['episodes'] if e['episode_id'] in {k.split('/')[0] for k in selected}]
 media={};cache=dict(dimensions=trial.core.DIMS,evaluator_sha256=sha(execution.full.EVALUATOR),shots={});proofcache={}
 expected=read(BASE/'evaluation/official_parallel_v2/run_manifest.json')
 for item in snapshot['episodes']:
  eid=item['episode_id'];records=[];affected=any(k.startswith(eid+'/') for k in selected)
  source=Path(item['result']);assert sha(source)==item['result_sha256'];bindings[str(source)]=sha(source)
  if affected:
   provenance=source.resolve().parent/'run_manifest.json';m=read(provenance)
   for field in ['config','python_version','libraries','model_checkpoints','pillars','evaluation_scope']:assert m[field]==expected[field],(eid,field)
   assert m['strict'] and m['skip_vlm'] and m['vlm_call_stats']['total_calls']==0;bindings[str(provenance)]=sha(provenance)
  for shot in item['episode']['shots']:
   sk=shot['shot_key'];key=eid+'/'+sk;old=snapshot['media'][key];c=selected.get(key)
   video=c['selected_video'] if c else old['source'];vsha=c['selected_video_sha256'] if c else old['sha256'];assert sha(video)==vsha
   a,b=map(int,sk.split(':'));canonical=OUT/'candidate_media'/eid/f'{a:03d}_{b:03d}.mp4';link(video,canonical)
   media[key]=dict(source=video,sha256=vsha,canonical=str(canonical),before_source=old['source'],before_sha256=old['sha256'],replaced=bool(c))
   records.append(dict(shot_key=sk,cut=shot['cut'],video=video,video_sha256=vsha))
   if not affected:continue
   if c:
    scores=c['after_scores'];scorefile=SOURCE/'cases'/c['case_id']/'scores.json';sources=[dict(path=str(scorefile),sha256=sha(scorefile))]
   else:
    scores={};sources=[]
    for dim in trial.core.DIMS:
     path=source.parent/'.work'/f'{eid}_{dim}_eval_results.json'
     if str(path) not in proofcache:proofcache[str(path)]=read(path);bindings[str(path)]=sha(path)
     hits=[x['video_results'] for x in proofcache[str(path)][dim][1] if Path(x['video_path']).stem==f'{a:03d}_{b:03d}'];assert len(hits)==1
     scores[dim]=float(hits[0]);sources.append(dict(path=str(path),sha256=bindings[str(path)]))
   assert all(math.isfinite(v) for v in scores.values())
   cache['shots'][key]=dict(reuse=True,video_sha256=vsha,scores=scores,score_sources=sources)
  immutable(OUT/'episode_media/episodes'/eid/'episode_result.json',dict(episode_id=eid,shots=records))
 for path in [Path(__file__),SNAP,SOURCE/'plan.json',SOURCE/'comparison.json',execution.full.EVALUATOR,
              Path(execution.__file__),ROOT/'pretest/qwen140_vbench_verified_cache_20261105.py']:
  bindings[str(path)]=sha(path)
 cache['bindings']=[dict(path=k,sha256=v) for k,v in bindings.items()]
 immutable(OUT/'vbench_cache_manifest.json',cache)
 immutable(OUT/'replacement_manifest.json',dict(scope='Independent full140 selection; only20 dynamic improvements replace frozen current videos. Other2471 videos identical.',videos=media,attempted=32,selected=20,decisions=decisions,benchmark_baseline_snapshot_sha256=sha(SNAP)))
 plan=dict(episodes=[e['episode'] for e in items],recompute_episode_ids=[e['episode_id'] for e in items],affected_episodes=len(items),
  complete_context_shots=sum(len(e['episode']['shots']) for e in items),target_cases=selected,target_shots=20,preliminary_candidate_shots=20,
  bindings=cache['bindings'],workers=2,threads=4,source_generation_plan_sha256=sha(SOURCE/'plan.json'),
  scope='20 selected solely by dynamic0_to1 relative frozen140 baseline, complete affected contexts. No presence/quality/DINO rejection gate; report declines. Default DINO score needs localization review, not media eligibility gating.')
 immutable(OUT/'plan.json',plan)
 # Exact dynamic contribution: official episode equal weight, not20/2491.
 dynamic=read(ROOT/'outputs/entitybench_dynamic_degree_analysis_20261106/shots.json');epcounts={e['episode_id']:len(e['episode']['shots']) for e in snapshot['episodes']}
 assert len(dynamic)==2491;dlookup={r['episode_id']+'/'+r['shot']:r for r in dynamic}
 before=sum(sum(r['current'] for r in dynamic if r['episode_id']==eid)/n/140 for eid,n in epcounts.items())
 for key,c in selected.items():assert dlookup[key]['current']==0
 delta=sum(1/epcounts[c['episode_id']]/140 for c in selected.values())
 immutable(OUT/'dynamic_exact.json',dict(scope='Exact arithmetic update of per-shot binary dynamic scores under140-episode macro aggregation; no new whole-context score claimed.',
  before=before,after=before+delta,delta=delta,delta_percentage_points=delta*100,selected=20,attempted=32,episodes=140,
  media_mapping_verified=2491,all_other_metrics='pending complete affected-context evaluation'))
 status('media_selection_complete_context_evaluation_pending',selected_shots=20,affected_episodes=len(items),context_shots=plan['complete_context_shots'])
 return plan

def aggregate_official(p):
 snapshot=read(SNAP);mod=frozen.diag.gate.load_official_pure();split=read(BASE/'split_full140.json');before=[];after=[];changed=set(p['recompute_episode_ids'])
 for item in snapshot['episodes']:
  ep=item['episode'];eid=ep['episode_id'];old=read(item['result']);before.append(mod._episode_result_from_json(eid,ep['difficulty'],old))
  new=read(OUT/'affected/results'/(eid+'.json')) if eid in changed else old;after.append(mod._episode_result_from_json(eid,ep['difficulty'],new))
 a=mod.aggregate_results(before,split);b=mod.aggregate_results(after,split)
 reference=read(frozen.OUT/'official_aggregates.json');assert a==reference['current']
 table=[dict(metric=k,label=label,before=frozen.J.full.agg.agg_metric(a,k)['mean'],after=frozen.J.full.agg.agg_metric(b,k)['mean']) for k,label in frozen.J.full.agg.LABELS.items()]
 write(OUT/'official_aggregates.json',dict(before=a,after=b));write(OUT/'official_comparison.json',dict(metrics=table,scope=p['scope'],default_reviewed_dino='pending_localization_review'))
 text='# 20个动态改善镜头替换后的完整140集（官方非VLM）\n\n'+p['scope']+'\n\n| 指标 | 替换前 | 替换20条后 |\n|---|---:|---:|\n'
 for row in table:text+=f'| {row["label"]} | {row["before"]:.6f} | {row["after"]:.6f} |\n'
 text+='\n这是按动态筛选后的补充版本，32尝试/20选择/12保留原版；提示词和视频后端相对冻结基线均可能变化。原视频/分数保留；下降完整披露。人物默认可见单脸＋定位复核DINO另待审阅；raw cs_face不是人脸识别。\n'
 (OUT/'official_report.md').write_text(text)

def default_dino_review_material(p):
 import qwen140_localization_repair_gpu_20261108 as baseline
 import qwen140_badshot_reviewed_dino_20261104 as review
 f=frozen;maps=baseline.gate_maps()['current'];oldreject={f.diag.key(r):r for r in read(baseline.AUDIT/'report.json')['applied_exclusions']['current']}
 rows=copy.deepcopy(maps);directory=OUT/'visible_region';scored=[];auto=[];items=[]
 for ep in p['episodes']:
  eid=ep['episode_id']
  for path in (directory/'extraction'/eid).glob('*.json'):
   for row in read(path)['rows']:rows[f.diag.key(row)]=row
  for key,row in rows.items():
   if key[0]==eid:row['edition']='unchanged' if key in maps and row['original_crop_sha256']==maps[key]['original_crop_sha256'] else 'rerun'
 carry={k:r for k,r in oldreject.items() if k in rows and rows[k]['original_crop_sha256']==r['original_crop_sha256']}
 mod=f.diag.gate.load_official_pure()
 for ep in p['episodes']:
  eid=ep['episode_id'];raw=read(OUT/'affected/results'/(eid+'.json'));value,per=f.diag.compute(ep,raw,rows,carry,mod)
  auto.append(dict(episode_id=eid,metrics={'cs_face':value.metrics['cs_face'].to_json()},per_entity_scores=per))
  for name,data in per.items():
   for app in data['per_shot_sim_to_centroid']:scored.append({**rows[eid,name,app['shot']],'similarity':app['sim_to_centroid']})
  items.append(dict(episode_id=eid,episode=ep))
 write(directory/'automatic_results.json',auto);write(directory/'scored_crops.json',scored);review.OUT=directory;review.review_sheets(scored,items)
 status('official_full140_complete_default_dino_review_pending',selected_shots=20,affected_episodes=p['affected_episodes'],context_shots=p['complete_context_shots'],
  official_report=str(OUT/'official_report.md'),default_dino='automatic_only_pending_score_hidden_localization_review',media_selection_final=True,quality_gate_enabled=False)

def run():
 p=prepare();oldout,oldfile=execution.OUT,execution.__file__
 try:
  execution.OUT=OUT;execution.__file__=str(Path(__file__));execution.evaluate(p);execution.dino_prepare(p)
  execution.OUT,execution.__file__=oldout,oldfile
  aggregate_official(p);default_dino_review_material(p)
 except Exception as error:status('stopped_context_error_preserve_selected_media',error=repr(error));raise
 finally:execution.OUT,execution.__file__=oldout,oldfile

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='scorer':
  import qwen140_vbench_verified_cache_20261105 as adapter
  adapter.CACHE=OUT/'vbench_cache_manifest.json';sys.argv.pop(1);adapter.main()
 else:
  parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
  if args.action=='prepare':
   p=prepare();print({k:p[k] for k in ['target_shots','affected_episodes','complete_context_shots']})
  else:run()
