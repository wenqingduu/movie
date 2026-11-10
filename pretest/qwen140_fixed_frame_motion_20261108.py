"""Six isolated prompt-only I2V trials. No agent/API/image/episode generation.

Reuse each frozen current-v2 actual input image, model, seed and settings.
Only prompt changes. Outputs never enter the official benchmark automatically.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,fcntl
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path('/root/autodl-tmp/movie')
OUT=ROOT/'outputs/entitybench_fixed_frame_motion_prompt_20261108'
SNAP=ROOT/'outputs/entitybench_full140_latest57_v2_141_20261106/snapshot.json'
ROWS=ROOT/'outputs/entitybench_dynamic_degree_analysis_20261106/shots.json'
WORKER=ROOT/'pretest/qwen140_integrated_episodes.py'
PY='/root/autodl-tmp/envs/lightx2v-trial/bin/python'
EVAL='/root/autodl-tmp/envs/entitybench-eval/bin/python'
BASE=ROOT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140'
EDITS=[
 ('run1','i50_j84','18:1','walking',
  '画面外侧打电话的Tae-ho沿橱窗向画面右侧连续迈出三四步，双腿交替前进，身体明显走过窗框；手机始终贴耳，边走边说话。窗内黑衣Min-jun留在绿色卡座，偶尔望向窗外。保持原固定外景机位。'),
 ('run103','i30_j54','24:2','walking',
  'Anya沿长廊连续向前走三步，双腿交替迈步，双臂自然摆动，身体明显接近镜头；走到走廊中线才停下，再谨慎地转头查看两侧紧闭的门。保持原广角构图。'),
 ('run221','i0_j18','16:5','continuous_prop_motion',
  'Kenshin先将筷子双节棍带近脸部，随后连续熟练挥舞：前臂画出清楚的圆弧，棍端向两侧挥出再收回，接着继续旋转，而不是举着道具停留。保持得意表情，蓝色烟雾在脚边持续翻卷。'),
 ('run4','i80_j98','7:6','opening_door',
  'Liam穿着首帧中的消防员装束，蹲在焦黑沙发旁，握住消防栓柜门把手持续向外拉；手臂和上身一起后移，将半开的柜门拉到完全敞开，露出柜内空间。烟雾在他身旁翻卷。保持原机位。'),
 ('run146','i0_j46','7:1','one_off_arm_raise',
  'Casey怒视桌上的玩具恐龙，随后手臂从桌边迅速抬到肩膀上方，肘部和肩部一起完成清楚的上举动作；最后停在即将向下击打的姿势，玩具仍放在桌上。保持原中景机位。'),
 ('run264','i50_j98','1:6','strumming',
  '前景抱吉他的Silas持续对着麦克风演唱，右前臂连续上下扫过琴弦，左手在琴颈上换和弦，手臂动作贯穿演唱过程，身体自然随节奏轻微起伏。玻璃隔断上Caleb的倒影留在背景专注观看、轻微点头。保持原中景机位。'),
]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def immutable(p,x):
 if Path(p).exists():assert read(p)==x,str(p)
 else:write(p,x)
def env():return dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',DTYPE='BF16',SENSITIVE_LAYER_DTYPE='None')
def status(stage,**extra):write(OUT/'status.json',dict(stage=stage,utc=now(),expected=6,
 completed=sum((OUT/f'P{i:02d}/result.json').exists() for i in range(1,7)),episode_generation=False,api_calls=0,
 other_shots_unchanged=True,automatic_benchmark_replacement=False,**extra))

def prepare():
 if (OUT/'plan.json').exists():
  p=read(OUT/'plan.json');assert p['snapshot_sha256']==sha(SNAP)
  for x in p['source_files']:assert sha(x['path'])==x['sha256'],x['path']
  return p
 rows=read(ROWS);snap=read(SNAP);cases=[]
 for n,(run,window,shot,kind,action) in enumerate(EDITS,1):
  hits=[r for r in rows if (r['run'],r['episode_id'].split('__')[2],r['shot'])==(run,window,shot) and r['current']==0];assert len(hits)==1
  r=hits[0];source=Path(r['folder']);old=read(source/'main/task.json');result=read(source/'result.json')
  assert r['route']=='C' and old['last_frame_path'] is None and old['frames']==81 and old['steps']==4
  assert sha(old['input_path'])==old['input_sha256'] and sha(r['video'])==result['video_sha256']
  assert snap['media'][r['episode_id']+'/'+shot]['sha256']==result['video_sha256']
  prefix=old['prompt'].split('  ',1)[0];assert prefix!=old['prompt']
  new={**old,'prompt':prefix+'  '+action+' 单幅连续电影画面，无分屏、拼图或画风变化。'}
  assert {k:v for k,v in old.items() if k!='prompt'}=={k:v for k,v in new.items() if k!='prompt'}
  taskpath=OUT/f'P{n:02d}/main/task.json';immutable(taskpath,new)
  cases.append(dict(case_id=f'P{n:02d}',run=run,episode_id=r['episode_id'],shot=shot,kind=kind,original_description=r['prompt'],
   before_video=r['video'],before_sha256=result['video_sha256'],source_task=str(source/'main/task.json'),source_task_sha256=sha(source/'main/task.json'),
   input_image=old['input_path'],input_sha256=old['input_sha256'],old_prompt=old['prompt'],new_prompt=new['prompt'],
   task=str(taskpath),task_sha256=sha(taskpath),output=str(taskpath.parent)))
 p=dict(created_utc=now(),cases=cases,snapshot=str(SNAP),snapshot_sha256=sha(SNAP),
  source_files=[dict(path=str(x),sha256=sha(x)) for x in [Path(__file__),WORKER,ROOT/'pretest/qwen140_pilot_video.py',BASE/'manifest.json']],
  protocol='Prompt-only; identical actual input PNG/seed/model/4 steps/81 frames/fps/resolution/style/identity frame. Six predeclared current-v2 dynamic0 shots. No new agent routing, 3D assets/keyframes, transition API, following shots or full episodes. No retries or automatic replacement.',
  user_scope='User explicitly confirmed only rerun selected dynamic0 shots from their existing first frames; leave all other shots and episodes unchanged.',
  limits='Independent candidates: their changed tails may change a later splice boundary even though no following video file changes. No success guarantee; do not add unsupported camera motion or events.',
  expected_minutes=[15,25],evaluation='Original six VBench dimensions on12 paired videos; no full-benchmark/centroid evaluation or VLM.')
 immutable(OUT/'plan.json',p);status('prepared');prompts(p);return p

def prompts(p):
 text=['# 固定首帧的动作提示词强化试验','',p['protocol'],'',
  '仅试跑6个当前v2动态为0的shot。原实际首帧（含原3D注入）、身份文字、画风、种子、模型、4步、81帧均保留；只替换视频动作句。后台不会启动整个episode或改写其他shot，也不会自动替换benchmark。','',
  '规则：说明动作主体、可见的动作过程与结束状态；位移给出自然可见的连续步伐，操作说明手臂/物体的运动。原文只有抬手就不擅自增加击打，原文没有运镜就不加摇镜刷分。首帧若已是动作终点，提示词也可能无法修复，结果照实保留。','',
  '## 六项提示词','']
 for c in p['cases']:
  text+=['### '+c['case_id']+' '+c['run']+'/'+c['episode_id'].split('__')[2]+'/'+c['shot'],'','原视频提示词：','',c['old_prompt'],'','强化后：','',c['new_prompt'],'']
 text+=['## 运行与评估','',
  '- `plan.json`冻结输入SHA和全部新旧提示词；`P##/main/`保存独立视频、实际首尾帧、完整生成记录。','- 生成后对新旧12个视频计算相同原版6项VBench，`comparison.json`/`report.md`记录全部6例，包括动态仍为0或其他指标下降的例子。','- 不重跑人物素材、3D注入或Agent规划，不调用API；既有注入结果通过完全相同首帧继承。','- 这只是固定首帧的6例提示词试验，不能作为整集或全benchmark分数，暂不计算跨镜头中心。','']
 (OUT/'prompts.md').write_text('\n'.join(text));(OUT/'README.md').write_text('# 固定首帧动态提示词试跑\n\n见[prompts.md](prompts.md)、[status.json](status.json)、[plan.json](plan.json)。\n\n只有本目录内独立输出；原benchmark和其他shot保持不变。\n')

def run():
 p=prepare();start=time.monotonic()
 with (OUT/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   for c in p['cases']:
    folder=Path(c['output']);record=folder.parent/'result.json'
    assert sha(c['before_video'])==c['before_sha256'] and sha(c['input_image'])==c['input_sha256'] and sha(c['task'])==c['task_sha256']
    if not record.exists():
     status('generating_isolated_shot',case_id=c['case_id'],run=c['run'],shot=c['shot'],elapsed_seconds=time.monotonic()-start)
     if not (folder/'generation.json').exists():
      with (folder.parent/'generation.log').open('a') as log:
       subprocess.run([PY,'-u',str(WORKER),'worker-video','--task',c['task'],'--output',str(folder)],env=env(),stdout=log,stderr=subprocess.STDOUT,check=True)
     generation=read(folder/'generation.json');task=read(c['task'])
     digest=hashlib.sha256(json.dumps(task,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
     assert generation['condition_sha256']==digest and sha(folder/'video.mp4')==generation['video_sha256']
     write(record,dict(case_id=c['case_id'],episode_id=c['episode_id'],shot_key=c['shot'],video=str(folder/'video.mp4'),video_sha256=generation['video_sha256'],
      last_frame=str(folder/'last.png'),last_sha256=generation['last_sha256'],same_input_image=True,input_sha256=c['input_sha256'],plan_sha256=sha(OUT/'plan.json')))
    else:assert sha(read(record)['video'])==read(record)['video_sha256']
   status('generation_complete_evaluating_six_vbench',elapsed_seconds=time.monotonic()-start)
   with (OUT/'evaluation.log').open('a') as log:subprocess.run([EVAL,'-u',str(Path(__file__)),'evaluate'],env=env(),stdout=log,stderr=subprocess.STDOUT,check=True)
   for c in p['cases']:assert sha(c['before_video'])==c['before_sha256'] and sha(c['input_image'])==c['input_sha256']
   status('complete_six_shot_prompt_only_trial',report=str(OUT/'report.md'),elapsed_seconds=time.monotonic()-start)
  except Exception as error:
   status('stopped_error_outputs_preserved',error=repr(error),elapsed_seconds=time.monotonic()-start)
   raise

def evaluate():
 p=prepare();os.environ['VBENCH_CACHE_DIR']='/root/autodl-tmp/models/entitybench_vbench'
 official=BASE/'runtime/evaluate_benchmark_official.py';spec=importlib.util.spec_from_file_location('motion_trial_official',official);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 import torch
 torch.set_num_threads(4);videos={}
 for i,c in enumerate(p['cases'],1):
  result=read(Path(c['output']).parent/'result.json');assert sha(result['video'])==result['video_sha256']
  videos[f'{i}:1']=c['before_video'];videos[f'{i}:2']=result['video']
 receipt=OUT/'vbench_receipt.json'
 if receipt.exists():
  cached=read(receipt);assert cached['plan_sha256']==sha(OUT/'plan.json');values=cached['values']
 else:
  scorer=m.VBenchEvaluator(device='cuda',cache_dir=os.environ['VBENCH_CACHE_DIR'],output_dir=str(OUT/'evaluation'),strict=True)
  values,failed=scorer.evaluate_episode_shots(videos,'fixed_frame_motion_6_pairs',str(OUT/'evaluation'));assert not failed
  immutable(receipt,dict(plan_sha256=sha(OUT/'plan.json'),official_sha256=sha(official),values=values,vlm_calls=0))
 rows=[]
 for i,c in enumerate(p['cases'],1):
  before,after=values[f'{i}:1'],values[f'{i}:2'];assert before['dynamic_degree']==0
  rows.append(dict(case_id=c['case_id'],run=c['run'],shot=c['shot'],kind=c['kind'],before=before,after=after,
   input_sha256=c['input_sha256'],before_video=c['before_video'],after_video=videos[f'{i}:2']))
 dims=list(m.VBENCH_DIMENSIONS);table={d:dict(before=sum(x['before'][d] for x in rows)/6,after=sum(x['after'][d] for x in rows)/6) for d in dims}
 write(OUT/'comparison.json',dict(scope='Six predeclared paired independent prompt-only shot trials; target-shot equal mean, not episode/full benchmark. Official models/thresholds unchanged.',rows=rows,means=table,
  successes=sum(x['after']['dynamic_degree']==1 for x in rows),plan_sha256=sha(OUT/'plan.json'),vlm_calls=0,paid_api_calls=0))
 text=['# 固定首帧提示词强化：6个独立shot对照','',f"动态0→1：{sum(x['after']['dynamic_degree']==1 for x in rows)}/6。仅6个预选shot，不是整集或benchmark。",'',
  '| 指标↑ | 原v2 | 强化动作句 |','|---|---:|---:|']
 for d in dims:text.append(f"| {d} | {table[d]['before']:.6f} | {table[d]['after']:.6f} |")
 text+=['','| shot | 原动态 | 新动态 | 新视频 |','|---|---:|---:|---|']
 for x in rows:text.append(f"| {x['run']}/{x['shot']} | {x['before']['dynamic_degree']:.0f} | {x['after']['dynamic_degree']:.0f} | [{x['case_id']}]({x['case_id']}/main/video.mp4) |")
 text+=['','输入首帧、3D结果、种子、模型、步数、时长及原机位要求保留；只改视频动作文字。其他shot未生成/覆盖，benchmark未替换。跨镜头人物中心、人物出现率及拼接边界不属于这6项独立视频评估，不能宣称其他所有指标都不变。','',
  '新旧提示词见[prompts.md](prompts.md)，逐项完整数值见[comparison.json](comparison.json)。需要结合视频判断动作是否符合剧情；动态1只是光流过门槛。','']
 (OUT/'report.md').write_text('\n'.join(text))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','evaluate']);args=parser.parse_args()
 if args.action=='prepare':prepare()
 elif args.action=='run':run()
 else:evaluate()
