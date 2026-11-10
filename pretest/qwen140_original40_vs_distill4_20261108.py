"""Eight frozen paired I2V backend comparisons, no benchmark replacement.
Original Wan40/CFG3.5/noLoRA vs existing LightX2V4/DMD2/CFGoff clips.
Pause only the batch scheduler; let its active worker finish, then resume in finally.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.util,json,math,os,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/autodl-tmp/movie')
SOURCE=ROOT/'outputs/entitybench_all_zero_motion_background_20261108'
OUT=ROOT/'outputs/entitybench_original40_vs_distill4_20261108'
BASE=ROOT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140'
LIGHT=Path('/root/autodl-tmp/LightX2V')
PY='/root/autodl-tmp/envs/lightx2v-trial/bin/python'
EVAL='/root/autodl-tmp/envs/entitybench-eval/bin/python'
IDS=['M0007','M0008','M0001','M0002','M0006','M0003','M0004','M0005']
PARENT=45449
WORKER=69726
DIMS=['subject_consistency','temporal_flickering','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n');t.replace(p)
def immutable(p,x):
 if Path(p).exists():assert read(p)==x,str(p)
 else:write(p,x)
def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,utc=now(),expected=8,
 generated=len(list((OUT/'cases').glob('*/generation.json'))),api_calls=0,whole_episode_generation=False,
 benchmark_replaced=False,**kw))
def env():return dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',DTYPE='BF16',SENSITIVE_LAYER_DTYPE='None')
def active(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return None
 state=(p/'stat').read_text().split()[2]
 if state=='Z':return None
 return (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()

def prepare():
 if (OUT/'plan.json').exists():
  p=read(OUT/'plan.json')
  for b in p['bindings']:assert sha(b['path'])==b['sha256'],b['path']
  return p
 source=read(SOURCE/'plan.json');lookup={c['case_id']:c for c in source['cases']};cases=[];bindings={}
 for i,case_id in enumerate(IDS,1):
  c=lookup[case_id];path=Path(c['output']).parent/'result.json';r=read(path);scorespath=Path(c['output']).parent/'vbench.json';v=read(scorespath)
  assert sha(r['video'])==r['video_sha256'] and sha(c['input_path'])==c['input_sha256']
  assert c['task']['steps']==4 and c['task']['frames']==81 and c['task']['last_frame_path'] is None and c['route']!='B'
  task=read(c['task_path']);assert task==c['task']
  cases.append(dict(case_id=f'N{i:02d}',source_case=case_id,episode_id=c['episode_id'],run=c['run'],shot=c['shot'],category=c['category'],
   original_description=c['official_description'],prompt=task['prompt'],input_path=task['input_path'],input_sha256=task['input_sha256'],
   seed=task['seed'],frames=81,fps=16,size=task['size'],distilled_video=r['video'],distilled_video_sha256=r['video_sha256'],
   distilled_scores=v['after'],distilled_score_source=str(scorespath),distilled_score_source_sha256=sha(scorespath),
   source_task=c['task_path'],source_task_sha256=sha(c['task_path']),distilled_generation=str(Path(r['video']).parent/'generation.json'),
   baseline_is_dynamic_zero=v['after']['dynamic_degree']==0))
  for f in [path,scorespath,Path(c['task_path'])]:bindings[str(f)]=sha(f)
 model=read(BASE/'manifest.json')['model_path']
 cfg=dict(infer_steps=40,num_frames=81,text_len=512,size=[480,832],self_attn_1_type='torch_sdpa',cross_attn_1_type='torch_sdpa',
  cross_attn_2_type='torch_sdpa',sample_guide_scale=[3.5,3.5],sample_shift=5.0,enable_cfg=True,cpu_offload=True,
  offload_granularity='model',boundary=.900,use_image_encoder=False,t5_cpu_offload=True,vae_cpu_offload=False,
  rms_norm_type='torch',t5_rms_norm_type='torch',layer_norm_type='torch',clean_cuda_cache=False,fps=16,modulate_type='torch')
 immutable(OUT/'original40_config.json',cfg)
 for f in [Path(__file__),SOURCE/'plan.json',BASE/'manifest.json',OUT/'original40_config.json',ROOT/'pretest/run_lightx2v_a14b_4step_qwen140.py',
  ROOT/'pretest/qwen140_technical_resume_v2_20261103.py',LIGHT/'lightx2v/pipeline.py',LIGHT/'lightx2v/models/runners/wan/wan_runner.py',
  LIGHT/'lightx2v/models/schedulers/wan/scheduler.py',LIGHT/'lightx2v/models/schedulers/wan/scheduler_factory.py',BASE/'runtime/evaluate_benchmark_official.py']:
  bindings[str(f)]=sha(f)
 p=dict(created_utc=now(),cases=cases,total=8,zero_controls=sum(c['baseline_is_dynamic_zero'] for c in cases),
  dynamic_positive_controls=sum(not c['baseline_is_dynamic_zero'] for c in cases),model_path=model,
  protocol='Same exact first-frame PNG, positive prompt, seed, resolution,81frames/16fps, BF16 and torch_sdpa. Compare original40/CFG3.5/noLoRA/WanScheduler/boundary.9 against existing4/DMD2/LoRA1022/CFGoff; empty negative prompt for original CFG. No image/person/3D regeneration, route change, following-shot generation, paidAPI, VLM or benchmark replacement.',
  limitation='Small purposive backend-recipe comparison, not an isolated step-count or distillation-weight ablation. Different CFG/scheduler are required original-vs-distilled recipes. Same seed does not itself prove identical latent trajectory. Report all8 and5 zero-baseline subset, not a full benchmark score.',
  first_case='N01: explicit rightward camera pan, existing distilled clip remains dynamic0. Only continue other7 after a valid81-frame first output.',
  config=str(OUT/'original40_config.json'),negative_prompt='',scheduling=dict(parent=PARENT,worker=WORKER,method='Stop idle parent now, let current GPU batch finish, use GPU exclusively for8 tests and original6dim evaluation, resume parent in finally.'),
  bindings=[dict(path=k,sha256=v) for k,v in bindings.items()])
 immutable(OUT/'plan.json',p);status('prepared',zero_controls=p['zero_controls'],dynamic_positive_controls=p['dynamic_positive_controls']);return p

def load(p):
 import torch
 from qwen140_technical_resume_v2_20261103 import install_t5_mmap
 install_t5_mmap();torch.set_num_threads(8)
 original_empty=torch.empty
 def pageable_empty(*args,**kw):
  if kw.get('pin_memory'):kw['pin_memory']=False
  return original_empty(*args,**kw)
 torch.empty=pageable_empty
 sys.path.insert(0,str(LIGHT));from lightx2v import LightX2VPipeline
 pipeline=LightX2VPipeline(model_path=p['model_path'],model_cls='wan2.2_moe',task='i2v',distill_method=None)
 pipeline.create_generator(config_json=p['config'])
 cfg=pipeline.runner.config
 assert cfg.get('distill_method') is None and not cfg.get('lora_configs')
 assert cfg['enable_cfg'] and cfg['infer_steps']==40 and cfg['sample_guide_scale']==[3.5,3.5]
 assert cfg['boundary']==.9 and cfg['feature_caching']=='NoCaching'
 assert type(pipeline.runner.scheduler).__name__=='WanScheduler'
 write(OUT/'resolved_runtime.json',dict(config=dict(cfg),scheduler_class=type(pipeline.runner.scheduler).__name__,
  torch=torch.__version__,cuda=torch.version.cuda,no_lora=True,config_sha256=sha(p['config']),plan_sha256=sha(OUT/'plan.json')))
 return pipeline

def generate():
 import torch
 from run_lightx2v_a14b_4step_qwen140 import decode_video
 p=prepare();start=time.monotonic();status('loading_original40_pipeline');pipe=load(p);load_seconds=time.monotonic()-start
 for c in p['cases']:
  folder=OUT/'cases'/c['case_id'];receipt=folder/'generation.json'
  if receipt.exists():
   r=read(receipt);assert r['plan_sha256']==sha(OUT/'plan.json') and sha(r['video'])==r['video_sha256'];continue
  assert sha(c['input_path'])==c['input_sha256'] and sha(c['distilled_video'])==c['distilled_video_sha256']
  folder.mkdir(parents=True,exist_ok=True);tmp=folder/'video.tmp.mp4';assert not tmp.exists()
  status('generating_original40',case_id=c['case_id'],source_case=c['source_case'],run=c['run'],shot=c['shot'],model_load_seconds=load_seconds)
  begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
  pipe.generate(seed=c['seed'],prompt=c['prompt'],negative_prompt='',image_path=c['input_path'],save_result_path=str(tmp));torch.cuda.synchronize()
  decoded=decode_video(tmp,folder/'first.png',folder/'last.png',81,16);tmp.replace(folder/'video.mp4')
  immutable(receipt,dict(case_id=c['case_id'],source_case=c['source_case'],episode_id=c['episode_id'],shot=c['shot'],
   video=str(folder/'video.mp4'),video_sha256=sha(folder/'video.mp4'),input_sha256=c['input_sha256'],seed=c['seed'],prompt=c['prompt'],
   negative_prompt='',decoded=decoded,elapsed_seconds=time.monotonic()-begin,model_load_seconds=load_seconds,
   cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,plan_sha256=sha(OUT/'plan.json'),
   original40_no_lora=True,config_sha256=sha(p['config'])))
  if c['case_id']=='N01':
   immutable(OUT/'first_case_validation.json',dict(case_id=c['case_id'],decoded=decoded,condition_unchanged=True,
    no_lora=True,original40_cfg35=True,elapsed_seconds=time.monotonic()-begin,plan_sha256=sha(OUT/'plan.json')))
  print('COMPLETE',c['case_id'],c['run'],c['shot'],'seconds',round(time.monotonic()-begin,2),flush=True)
 status('generation_complete_ready_for_evaluation',elapsed_seconds=time.monotonic()-start)

def evaluate():
 p=prepare();official=BASE/'runtime/evaluate_benchmark_official.py'
 spec=importlib.util.spec_from_file_location('original40_paired_original_vbench',official);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 import torch
 torch.set_num_threads(4);os.environ['VBENCH_CACHE_DIR']='/root/autodl-tmp/models/entitybench_vbench';videos={}
 for i,c in enumerate(p['cases'],1):
  r=read(OUT/'cases'/c['case_id']/'generation.json');assert sha(r['video'])==r['video_sha256'];assert sha(c['distilled_video'])==c['distilled_video_sha256'];videos[f'{i}:1']=r['video']
 receipt=OUT/'evaluation/receipt.json'
 if receipt.exists():
  r=read(receipt);assert r['plan_sha256']==sha(OUT/'plan.json');scores=r['scores']
 else:
  scorer=mod.VBenchEvaluator('cuda',os.environ['VBENCH_CACHE_DIR'],str(OUT/'evaluation'),True)
  scores,failed=scorer.evaluate_episode_shots(videos,'original40_vs_distill4_8pairs',str(OUT/'evaluation'));assert not failed
  immutable(receipt,dict(scores=scores,plan_sha256=sha(OUT/'plan.json'),official_sha256=sha(official),video_sha256={k:sha(v) for k,v in videos.items()},vlm_calls=0))
 rows=[]
 for i,c in enumerate(p['cases'],1):
  after=scores[f'{i}:1'];assert all(math.isfinite(after[d]) for d in DIMS)
  rows.append(dict(case_id=c['case_id'],source_case=c['source_case'],run=c['run'],shot=c['shot'],episode_id=c['episode_id'],category=c['category'],
   baseline_is_dynamic_zero=c['baseline_is_dynamic_zero'],distilled4=c['distilled_scores'],original40=after,
   distilled_video=c['distilled_video'],original_video=videos[f'{i}:1']))
 means={}
 for name,cohort in [('all8',rows),('distilled_dynamic0',[r for r in rows if r['baseline_is_dynamic_zero']])]:
  means[name]={d:{side:sum(r[side][d] for r in cohort)/len(cohort) for side in ['distilled4','original40']} for d in DIMS}
 write(OUT/'comparison.json',dict(rows=rows,means=means,scope=p['limitation'],benchmark_updated=False,character_dino='not_evaluated_not_a_centroid_cohort'))
 lines=['# 原版40步与蒸馏4步：8条配对视频','',p['protocol'],'',p['limitation'],'',
  '| 指标 | 蒸馏4步 | 原版40步 |','|---|---:|---:|']
 for d in DIMS:lines.append(f'| {d} | {means["all8"][d]["distilled4"]:.6f} | {means["all8"][d]["original40"]:.6f} |')
 lines += ['', '| case | 镜头 | 4步动态 | 40步动态 | 原版视频 |','|---|---|---:|---:|---|']
 for r in rows:lines.append(f'| {r["case_id"]} | {r["run"]}/{r["shot"]} | {r["distilled4"]["dynamic_degree"]:.0f} | {r["original40"]["dynamic_degree"]:.0f} | [video](cases/{r["case_id"]}/video.mp4) |')
 lines += ['','只评原6项VBench，不将结果当作人物身份/出现率/动作符合原文的证明。需看视频；不自动替换benchmark。','']
 (OUT/'report.md').write_text('\n'.join(lines));status('complete_eight_backend_pairs',report=str(OUT/'report.md'))

def schedule():
 p=prepare();resumable=False
 with (OUT/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   parent=active(PARENT);assert parent and 'qwen140_all_zero_background_20261108.py handoff-run' in parent
   os.kill(PARENT,signal.SIGSTOP);resumable=True
   status('waiting_for_active_generation_batch_to_finish',current_worker=WORKER,original_parent_paused=True)
   while True:
    cmd=active(WORKER)
    if not cmd:break
    assert 'qwen140_all_zero_background_20261108.py worker' in cmd
    time.sleep(15)
   expected=[f'M{i:04d}' for i in range(135,167)]
   assert all((SOURCE/'cases'/k/'result.json').exists() for k in expected),'Active batch incomplete: no original-model test started'
   # Generation worker has exited and released both GPU and host model memory.
   status('starting_exclusive_original40_trial')
   for action,python in [('generate',PY),('evaluate',EVAL)]:
    with (OUT/(action+'.log')).open('a') as log:subprocess.run([python,'-u',str(Path(__file__)),action],env=env(),stdout=log,stderr=subprocess.STDOUT,check=True)
  except Exception as e:
   status('stopped_test_error_outputs_preserved',error=repr(e));raise
  finally:
   if resumable and active(PARENT):
    os.kill(PARENT,signal.SIGCONT)
    write(OUT/'original_queue_resume.json',dict(utc=now(),parent=PARENT,resumed=True))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','schedule','generate','evaluate']);a=parser.parse_args()
 if a.action=='prepare':
  p=prepare();print(json.dumps(dict(output=str(OUT),cases=p['total'],dynamic_zero=p['zero_controls'],dynamic_positive=p['dynamic_positive_controls'])))
 elif a.action=='schedule':schedule()
 elif a.action=='generate':generate()
 else:evaluate()
