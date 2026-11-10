"""Frozen completed original40 cohort: original6 VBench, verified first-case reuse, no selection."""
from __future__ import annotations
import importlib.util,os,signal,subprocess,sys,time
from pathlib import Path
import qwen140_original40_torch_rope_20261108 as job
job.configure()
OUT=job.OUT/'completed3_evaluation'
WORKER=101416

def state(stage,**kw):job.write(OUT/'status.json',dict(stage=stage,utc=job.original.now(),vlm_calls=0,api_calls=0,media_generation=False,**kw))
def prepare():
 path=OUT/'manifest.json'
 if path.exists():
  m=job.read(path)
  assert m['helper_sha256']==job.sha(__file__) and m['plan_sha256']==job.sha(job.OUT/'plan.json') and m['evaluator_sha256']==job.sha(m['evaluator'])
  return m
 p=job.prepare();rows=[]
 for c in p['cases']:
  f=job.OUT/'cases'/c['case_id']/'generation.json'
  if not f.exists():continue
  r=job.read(f);assert job.sha(r['video'])==r['video_sha256'] and job.sha(c['distilled_video'])==c['distilled_video_sha256']
  assert job.sha(c['distilled_score_source'])==c['distilled_score_source_sha256']
  rows.append(dict(case_id=c['case_id'],run=c['run'],shot=c['shot'],episode_id=c['episode_id'],video=r['video'],video_sha256=r['video_sha256'],
   before=c['distilled_scores'],before_video=c['distilled_video'],before_video_sha256=c['distilled_video_sha256'],
   score_source=c['distilled_score_source'],score_source_sha256=c['distilled_score_source_sha256'],elapsed_generation_seconds=r['elapsed_seconds']))
 assert len(rows)==3,'Freeze requested completed3 only; create another version for a later cohort'
 first=job.OUT/'first_case_evaluation/comparison.json';fm=job.OUT/'first_case_evaluation/manifest.json';old=job.read(first);manifest=job.read(fm)
 assert old['manifest_sha256']==job.sha(fm) and old['video_sha256']==rows[0]['video_sha256'] and old['before']==rows[0]['before']
 official=job.original.BASE/'runtime/evaluate_benchmark_official.py';assert manifest['evaluator_sha256']==job.sha(official)
 m=dict(created_utc=job.original.now(),rows=rows,total=3,plan_sha256=job.sha(job.OUT/'plan.json'),helper_sha256=job.sha(__file__),evaluator=str(official),
  evaluator_sha256=job.sha(official),first_cached_result=str(first),first_cached_sha256=job.sha(first),first_manifest=str(fm),first_manifest_sha256=job.sha(fm),
  scope='All3 currently completed predeclared original40 tests, unselected. Same exact input/prompt/seed/resolution/duration paired with distilled4; original6 VBench only. Not full benchmark or isolated step-count/LoRA ablation.')
 job.original.immutable(path,m);return m

def score():
 m=prepare();os.environ['VBENCH_CACHE_DIR']='/root/autodl-tmp/models/entitybench_vbench'
 spec=importlib.util.spec_from_file_location('completed3_original_vbench',m['evaluator']);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 import torch
 torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();start=time.monotonic();videos={str(i)+':1':r['video'] for i,r in enumerate(m['rows'][1:],2)}
 scorer=mod.VBenchEvaluator('cuda',os.environ['VBENCH_CACHE_DIR'],str(OUT/'runtime'),True)
 scores,failed=scorer.evaluate_episode_shots(videos,'original40_completed3_new2',str(OUT/'runtime'));assert not failed
 old=job.read(m['first_cached_result']);assert job.sha(m['first_cached_result'])==m['first_cached_sha256'];scores['1:1']=old['after'];rows=[]
 for i,r in enumerate(m['rows'],1):
  rows.append({**r,'after':scores[f'{i}:1'],'deltas':{k:scores[f'{i}:1'][k]-r['before'][k] for k in mod.VBENCH_DIMENSIONS}})
 means={k:{side:sum(r[side][k] for r in rows)/3 for side in ['before','after']} for k in mod.VBENCH_DIMENSIONS}
 result=dict(scope=m['scope'],rows=rows,means=means,computed_videos=2,verified_cached_videos=1,manifest_sha256=job.sha(OUT/'manifest.json'),
  elapsed_seconds=time.monotonic()-start,gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,vlm_calls=0,benchmark_updated=False)
 job.original.immutable(OUT/'comparison.json',result)
 lines=['# 已完成3条：原版40步与蒸馏4步配对评测','',m['scope'],'','| 指标 | 4步蒸馏 | 原版40步 | 变化 |','|---|---:|---:|---:|']
 for k,v in means.items():lines.append(f'| {k} | {v["before"]:.6f} | {v["after"]:.6f} | {v["after"]-v["before"]:+.6f} |')
 lines += ['','| case | 镜头 | 4步动态 | 40步动态 |','|---|---|---:|---:|']
 for r in rows:lines.append(f'| {r["case_id"]} | {r["run"]}/{r["shot"]} | {r["before"]["dynamic_degree"]:.0f} | {r["after"]["dynamic_degree"]:.0f} |')
 lines += ['','全部3条均纳入，未按动态或质量筛选。首条同SHA兼容缓存复用、另2条新评分。人物出现率/跨镜头DINO未计算，不能把视频内部主体一致性当作身份识别；不覆盖正式benchmark。','']
 (OUT/'report.md').write_text('\n'.join(lines));state('complete',completed=3,computed=2,reused=1)

def schedule():
 prepare();paused=False
 try:
  cmd=job.original.active(WORKER);assert cmd and 'qwen140_original40_torch_rope_20261108.py generate' in cmd
  os.kill(WORKER,signal.SIGSTOP);paused=True;time.sleep(2);state('scoring_completed3_generation_temporarily_paused')
  with (OUT/'scorer.log').open('a') as log:subprocess.run([job.original.EVAL,'-u',str(Path(__file__)),'score'],env=job.original.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
 except Exception as error:state('error',error=repr(error));raise
 finally:
  if paused and job.original.active(WORKER):
   os.kill(WORKER,signal.SIGCONT);job.write(OUT/'generation_resume.json',dict(utc=job.original.now(),pid=WORKER,resumed=True,large_queue_resumed=False))

if __name__=='__main__':
 if sys.argv[1]=='score':score()
 else:schedule()
