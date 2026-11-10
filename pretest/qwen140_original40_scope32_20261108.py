"""32-case original40 queue: preserve original8, append24 explicit dynamic0 cases.
Motion-enhanced prompts remain byte-identical to existing distilled4 controls.
No quality gates, new images, paidAPIs, whole-episode generation or benchmark replacement.
"""
from __future__ import annotations
import copy,fcntl,importlib.util,json,math,os,subprocess,sys,time
from pathlib import Path
import qwen140_original40_torch_rope_20261108 as trial
core=trial.original
ROOT=core.ROOT
PRIOR=trial.OUT
SOURCE=core.SOURCE
OUT=ROOT/'outputs/entitybench_original40_scope32_20261108'
ADDED=[
 'M0009','M0011',  # explicit camera pans
 'M0013','M0017','M0018','M0020','M0024','M0025','M0047','M0050',  # travel
 'M0104','M0106','M0113','M0115','M0116','M0107',  # continuous performance
 'M0125','M0143','M0146','M0150','M0171','M0191','M0194','M0198',  # explicit operations
]
read,write,sha,immutable=core.read,core.write,core.sha,core.immutable


def status(stage,**kw):
 done=list((OUT/'cases').glob('*/generation.json'));scored=list((OUT/'cases').glob('*/scores.json'))
 write(OUT/'status.json',dict(stage=stage,utc=core.now(),expected=32,generated=len(done),scored=len(scored),
  quality_gate_enabled=False,large_queue_resumed=False,benchmark_replaced=False,api_calls=0,**kw))


def prepare():
 if (OUT/'plan.json').exists():
  p=read(OUT/'plan.json')
  for b in p['bindings']:assert sha(b['path'])==b['sha256'],b['path']
  return p
 prior=read(PRIOR/'plan.json');source=read(SOURCE/'plan.json');lookup={c['case_id']:c for c in source['cases']};p=copy.deepcopy(prior)
 for c in p['cases']:
  c['inherit_first8']=True;c['frozen_benchmark_scores']=lookup[c['source_case']]['baseline_scores']
 cases=p['cases'];bindings={x['path']:x['sha256'] for x in prior['bindings']}
 for i,case_id in enumerate(ADDED,9):
  c=lookup[case_id];path=Path(c['output']).parent/'result.json';r=read(path);vp=Path(c['output']).parent/'vbench.json';v=read(vp);task=read(c['task_path'])
  assert not c.get('background_cue') and c['route']!='B' and v['after']['dynamic_degree']==0
  assert sha(r['video'])==r['video_sha256'] and sha(task['input_path'])==task['input_sha256']
  assert task==c['task'] and task['prompt']==c['new_prompt'] and task['frames']==81 and task['steps']==4 and task['last_frame_path'] is None
  cases.append(dict(case_id=f'N{i:02d}',source_case=case_id,episode_id=c['episode_id'],run=c['run'],shot=c['shot'],category=c['category'],
   original_description=c['official_description'],prompt=task['prompt'],input_path=task['input_path'],input_sha256=task['input_sha256'],
   seed=task['seed'],frames=81,fps=16,size=task['size'],distilled_video=r['video'],distilled_video_sha256=r['video_sha256'],
   distilled_scores=v['after'],distilled_score_source=str(vp),distilled_score_source_sha256=sha(vp),source_task=c['task_path'],
   source_task_sha256=sha(c['task_path']),baseline_is_dynamic_zero=True,inherit_first8=False,frozen_benchmark_scores=c['baseline_scores']))
  bindings[str(path)]=sha(path);bindings[str(vp)]=sha(vp);bindings[c['task_path']]=sha(c['task_path'])
 assert len(cases)==32 and len({c['source_case'] for c in cases})==32
 assert all(not lookup[c['source_case']].get('background_cue') for c in cases)
 for f in [Path(__file__),PRIOR/'plan.json',SOURCE/'plan.json']:bindings[str(f)]=sha(f)
 p.update(created_utc=core.now(),total=32,cases=cases,zero_controls=sum(c['baseline_is_dynamic_zero'] for c in cases),dynamic_positive_controls=sum(not c['baseline_is_dynamic_zero'] for c in cases),
  original8_plan=str(PRIOR/'plan.json'),original8_plan_sha256=sha(PRIOR/'plan.json'),original8_reused=True,new_cases=24,quality_gate_enabled=False,
  selection='Original8 preserved as controls, append24 completed distilled4 motion-enhanced dynamic0 references with no background cue or B-transition. Explicit descriptions manually checked:2 camera,8 travel,6 performance,8 operations. No selection on quality or future40-step scores.',
  scheduling='Wait original8 generation and6dim scores; reuse all8. Generate remaining24 in3 batches of8, release pipeline and score every batch. Do not resume paused1333 queue.',
  scope='32 purposively selected paired independent videos. Same motion-enhanced prompt, first-frame PNG, seed and video settings on both sides. All results reported without quality gates. Not a full benchmark replacement or isolated step-count/LoRA ablation.',
  bindings=[dict(path=k,sha256=v) for k,v in bindings.items()])
 immutable(OUT/'plan.json',p)
 immutable(OUT/'evaluation_policy.json',dict(user_confirmed_no_quality_gate=True,retain_all_declines=True,score_every_case=True,
  technical_integrity_checks=True,benchmark_replacement=False,motion_enhanced_prompts_unchanged=True,background_cues_added=False,total=32,new_generation=24,reuse=8))
 status('prepared_waiting_for_original8');return p


def configure():
 core.OUT=OUT;core.prepare=prepare;core.status=status;core.__file__=str(Path(__file__).resolve())


def inherit(p):
 previous=read(PRIOR/'comparison.json');by={r['case_id']:r for r in previous['rows']};assert len(by)==8
 for c in p['cases'][:8]:
  folder=OUT/'cases'/c['case_id'];source=PRIOR/'cases'/c['case_id']/'generation.json';r=read(source)
  assert r['plan_sha256']==sha(PRIOR/'plan.json') and sha(r['video'])==r['video_sha256']
  assert r['prompt']==c['prompt'] and r['input_sha256']==c['input_sha256'] and r['seed']==c['seed']
  immutable(folder/'generation.json',{**r,'plan_sha256':sha(OUT/'plan.json'),'reused_generation_source':str(source),
   'reused_generation_source_sha256':sha(source),'actual_generation_plan_sha256':r['plan_sha256']})
  for name in ['video.mp4','first.png','last.png']:
   dest=folder/name;src=PRIOR/'cases'/c['case_id']/name
   if not dest.exists():dest.symlink_to(src.resolve())
  row=by[c['case_id']];assert row['distilled4']==c['distilled_scores'] and row['original_video']==r['video']
  immutable(folder/'scores.json',dict(case_id=c['case_id'],before=row['distilled4'],after=row['original40'],
   video_sha256=r['video_sha256'],source_score=str(PRIOR/'comparison.json'),source_score_sha256=sha(PRIOR/'comparison.json'),reused=True,quality_filtered=False))


def evaluate(ids,tag):
 p=prepare();lookup={c['case_id']:c for c in p['cases']};cases=[lookup[i] for i in ids]
 official=core.BASE/'runtime/evaluate_benchmark_official.py';spec=importlib.util.spec_from_file_location('scope32_original_vbench',official);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 import torch
 torch.set_num_threads(4);os.environ['VBENCH_CACHE_DIR']='/root/autodl-tmp/models/entitybench_vbench';videos={}
 for i,c in enumerate(cases,1):
  r=read(OUT/'cases'/c['case_id']/'generation.json');assert sha(r['video'])==r['video_sha256'];videos[f'{i}:1']=r['video']
 folder=OUT/'evaluation'/tag;receipt=folder/'receipt.json'
 if receipt.exists():
  r=read(receipt);assert r['plan_sha256']==sha(OUT/'plan.json') and r['case_ids']==ids;scores=r['scores']
 else:
  scorer=mod.VBenchEvaluator('cuda',os.environ['VBENCH_CACHE_DIR'],str(folder),True)
  scores,failed=scorer.evaluate_episode_shots(videos,tag,str(folder));assert not failed
  immutable(receipt,dict(plan_sha256=sha(OUT/'plan.json'),case_ids=ids,scores=scores,video_sha256={k:sha(v) for k,v in videos.items()},official_sha256=sha(official)))
 for i,c in enumerate(cases,1):
  after=scores[f'{i}:1'];assert all(math.isfinite(after[d]) for d in core.DIMS)
  immutable(OUT/'cases'/c['case_id']/'scores.json',dict(case_id=c['case_id'],before=c['distilled_scores'],after=after,
   video_sha256=sha(videos[f'{i}:1']),source_score=str(receipt),source_score_sha256=sha(receipt),reused=False,quality_filtered=False))


def report():
 p=prepare();rows=[]
 for c in p['cases']:
  f=OUT/'cases'/c['case_id']/'scores.json'
  if not f.exists():continue
  s=read(f);r=read(OUT/'cases'/c['case_id']/'generation.json');assert s['video_sha256']==r['video_sha256']
  rows.append(dict(case_id=c['case_id'],source_case=c['source_case'],episode_id=c['episode_id'],run=c['run'],shot=c['shot'],category=c['category'],
   before=s['before'],after=s['after'],frozen_benchmark=c['frozen_benchmark_scores'],video=r['video'],before_video=c['distilled_video'],quality_filtered=False))
 if not rows:return
 means={d:{side:sum(r[side][d] for r in rows)/len(rows) for side in ['frozen_benchmark','before','after']} for d in core.DIMS}
 write(OUT/'comparison.json',dict(scope=p['scope'],expected=32,scored=len(rows),complete=len(rows)==32,rows=rows,means=means,
  dynamic_zero_to_one=sum(r['before']['dynamic_degree']==0 and r['after']['dynamic_degree']==1 for r in rows),quality_gate_enabled=False,benchmark_updated=False))
 lines=['# 32条原版40步配对实验（不设质量门控）','',f'已评分{len(rows)}/32；其余为pending，不填0。',p['scope'],'',
  '| 指标 | 冻结benchmark原视频 | 同词4步蒸馏 | 同词原版40步 |','|---|---:|---:|---:|']
 for d,v in means.items():lines.append(f'| {d} | {v["frozen_benchmark"]:.6f} | {v["before"]:.6f} | {v["after"]:.6f} |')
 lines += ['','冻结benchmark列的提示词与运动强化后可能不同；视频后端的直接配对只比较后两列。该表是当前已评分目标shot均值，不是完整episode或benchmark指标。全部结果包含下降项，不做成功筛选。','',
 '| case | source | 镜头 | 4步动态 | 40步动态 |','|---|---|---|---:|---:|']
 for r in rows:lines.append(f'| {r["case_id"]} | {r["source_case"]} | {r["run"]}/{r["episode_id"].split("__")[2]}/{r["shot"]} | {r["before"]["dynamic_degree"]:.0f} | {r["after"]["dynamic_degree"]:.0f} |')
 lines += ['','只评6项原VBench；人物出现/身份DINO与最终动作画面仍需另外评估，不自动覆盖benchmark。','']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def run():
 configure();p=prepare()
 with (OUT/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   while True:
    s=read(PRIOR/'status.json')
    if s['stage']=='complete_eight_backend_pairs':break
    if s['stage'].startswith('stopped_'):raise RuntimeError('Original8 stopped: '+str(s))
    status('waiting_for_original8_generation_and_evaluation',original8_generated=s['generated'],original8_stage=s['stage'])
    time.sleep(30)
   inherit(p);report()
   for start in range(8,32,8):
    cases=p['cases'][start:start+8];ids=[c['case_id'] for c in cases];tag=f'batch_{start//8:02d}'
    if any(not (OUT/'cases'/c['case_id']/'generation.json').exists() for c in cases):
     status('generating_next_original40_batch',case_ids=ids)
     with (OUT/(tag+'_generation.log')).open('a') as log:subprocess.run([core.PY,'-u',str(Path(__file__)),'worker',','.join(ids)],env=core.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
    if any(not (OUT/'cases'/c['case_id']/'scores.json').exists() for c in cases):
     status('evaluating_next_original40_batch',case_ids=ids)
     with (OUT/(tag+'_evaluation.log')).open('a') as log:subprocess.run([core.EVAL,'-u',str(Path(__file__)),'evaluate',','.join(ids),tag],env=core.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
    report();status('batch_complete',batch=tag)
   status('complete32_unfiltered_six_vbench',report=str(OUT/'report.md'))
  except Exception as error:status('stopped_error_outputs_preserved',error=repr(error));raise

if __name__=='__main__':
 configure();action=sys.argv[1]
 if action=='prepare':
  p=prepare();print(json.dumps(dict(total=p['total'],reuse=8,new_generation=24,output=str(OUT))))
 elif action=='run':run()
 elif action=='worker':
  full=prepare();ids=set(sys.argv[2].split(','));core.prepare=lambda:{**full,'cases':[c for c in full['cases'] if c['case_id'] in ids]};core.generate()
 elif action=='evaluate':evaluate(sys.argv[2].split(','),sys.argv[3])
 else:raise ValueError(action)
