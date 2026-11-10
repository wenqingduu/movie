"""Score the completed first original40 clip immediately, then resume ongoing generation."""
from __future__ import annotations
import importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
import qwen140_original40_torch_rope_20261108 as job
job.configure()
OUT=job.OUT/'first_case_evaluation'
WORKER=101416

def state(stage,**kw):job.write(OUT/'status.json',dict(stage=stage,utc=job.original.now(),vlm_calls=0,api_calls=0,media_generation=False,**kw))
def prepare():
 p=job.prepare();c=p['cases'][0];r=job.read(job.OUT/'cases/N01/generation.json')
 assert job.sha(r['video'])==r['video_sha256'] and job.sha(c['distilled_video'])==c['distilled_video_sha256']
 m=dict(case_id='N01',source_case=c['source_case'],episode_id=c['episode_id'],shot=c['shot'],video=r['video'],video_sha256=r['video_sha256'],
  before=c['distilled_scores'],before_video=c['distilled_video'],before_video_sha256=c['distilled_video_sha256'],
  source_score=c['distilled_score_source'],source_score_sha256=c['distilled_score_source_sha256'],plan_sha256=job.sha(job.OUT/'plan.json'),
  evaluator=str(job.original.BASE/'runtime/evaluate_benchmark_official.py'),helper_sha256=job.sha(__file__))
 m['evaluator_sha256']=job.sha(m['evaluator']);job.original.immutable(OUT/'manifest.json',m);return m

def score():
 m=prepare();os.environ['VBENCH_CACHE_DIR']='/root/autodl-tmp/models/entitybench_vbench'
 spec=importlib.util.spec_from_file_location('first_original40_original_vbench',m['evaluator']);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 import torch
 torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 scorer=mod.VBenchEvaluator('cuda',os.environ['VBENCH_CACHE_DIR'],str(OUT/'runtime'),True)
 scores,failed=scorer.evaluate_episode_shots({'1:1':m['video']},'original40_firstcase',str(OUT/'runtime'));assert not failed
 after=scores['1:1'];deltas={k:after[k]-m['before'][k] for k in after}
 result=dict(before=m['before'],after=after,deltas=deltas,manifest_sha256=job.sha(OUT/'manifest.json'),video_sha256=m['video_sha256'],
  elapsed_seconds=time.monotonic()-start,gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
  scope='Single first-case original40 vs existing distilled4, same fixed image/prompt/seed. Six original VBench dimensions, not full benchmark or identity evaluation.')
 job.original.immutable(OUT/'comparison.json',result);state('complete',comparison=result)
 lines=['# 第一条：原版40步与4步蒸馏','',result['scope'],'','| 指标 | 4步蒸馏 | 原版40步 | 变化 |','|---|---:|---:|---:|']
 lines += [f'| {k} | {m["before"][k]:.6f} | {after[k]:.6f} | {deltas[k]:+.6f} |' for k in after]
 lines += ['','原版用时934.45秒；前后均81帧16fps，PNG/正提示词/种子相同。RoPE错误已解决；不由单条结论推断全部8条或benchmark。','']
 (OUT/'report.md').write_text('\n'.join(lines))

def schedule():
 prepare();paused=False
 try:
  cmd=job.original.active(WORKER);assert cmd and 'qwen140_original40_torch_rope_20261108.py generate' in cmd
  os.kill(WORKER,signal.SIGSTOP);paused=True;time.sleep(2)
  state('scoring_first_case_generation_temporarily_paused')
  with (OUT/'scorer.log').open('a') as log:subprocess.run([job.original.EVAL,'-u',str(Path(__file__)),'score'],env=job.original.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
 except Exception as e:state('error',error=repr(e));raise
 finally:
  if paused and job.original.active(WORKER):
   os.kill(WORKER,signal.SIGCONT)
   job.write(OUT/'generation_resume.json',dict(utc=job.original.now(),pid=WORKER,resumed=True,large_queue_resumed=False))

if __name__=='__main__':
 if sys.argv[1]=='score':score()
 else:schedule()
