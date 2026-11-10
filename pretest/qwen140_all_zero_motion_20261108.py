"""All frozen dynamic0 shots: one prompt-only candidate each, then fixed quality gates.

No APIs, new frames/assets/routes, dependent-shot rollout, or in-place benchmark
replacement. Every attempt and rejection is retained. Context/DINO/visual review
is mandatory before publishing any selected replacement.
"""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,importlib.util,json,math,os,re,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
ROOT=Path('/root/autodl-tmp/movie')
OUT=ROOT/'outputs/entitybench_all_zero_fixed_frame_motion_20261108'
SNAP=ROOT/'outputs/entitybench_full140_latest57_v2_141_20261106/snapshot.json'
ROWS=ROOT/'outputs/entitybench_dynamic_degree_analysis_20261106/shots.json'
PILOT=ROOT/'outputs/entitybench_fixed_frame_motion_prompt_20261108'
BASE=ROOT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140'
OFFICIAL=BASE/'runtime/evaluate_benchmark_official.py'
PY='/root/autodl-tmp/envs/lightx2v-trial/bin/python'
EVAL='/root/autodl-tmp/envs/entitybench-eval/bin/python'
DIMS=['subject_consistency','temporal_flickering','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality']
POLICY=dict(user_confirmed=True,dynamic_degree_required=1,
 max_drop=dict(subject_consistency=.01,temporal_flickering=.01,motion_smoothness=.01,aesthetic_quality=.01,imaging_quality=1.,cs_face_visible_single_face_reviewed=.01),
 presence_max_drop=dict(intra_character_presence=0.,intra_object_presence=0.,intra_location_presence=0.),
 numerical_tolerance=1e-8,visual_required=True,visual_reject=['new_missing_character','new_identity_drift','split_screen','wrong_action'],
 missing='Both inapplicable/null: report not applicable. Previously measurable but now missing: reject. No zero imputation.',
 centroid='Recompute complete affected episode centres; no frozen-reference or old-centroid shortcut. Joint selected sequence must be rescored before publication.',
 publication='Separate quality-selected version; disclose all1333 attempts, candidate successes/rejections and selected counts. Preserve unselected and baseline results.')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def immutable(p,x):
 if Path(p).exists():assert read(p)==x,str(p)
 else:write(p,x)
def environment():return dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',DTYPE='BF16',SENSITIVE_LAYER_DTYPE='None')
def status(stage,**kw):
 generated=list((OUT/'cases').glob('*/result.json'));scored=list((OUT/'cases').glob('*/vbench.json'))
 passed=sum(read(p)['vbench_pass'] for p in scored)
 write(OUT/'status.json',dict(stage=stage,utc=now(),expected=1333,generated=len(generated),vbench_scored=len(scored),
 vbench_preliminary_pass=passed,final_replacements=0,final_selection_pending=True,api_calls=0,whole_episode_generation=False,
 unchanged_shots_regenerated=False,official_overwritten=False,**kw))

def strengthen(old,story):
 """Original wording preserved; short grounded process cue, no new plot/camera event."""
 cleaned,count=re.subn(r'图\s*[1-9]\d*\s*(?:中(?:的)?|所示的)?','',old)
 s=story.split('\n\n')[-1].lower()
 tests=[('camera',r'\b(?:camera (?:slowly )?(?:pans?|tracks?|dollies|zooms?|moves?|tilts?)|tracking shot|dolly (?:in|out)|pan(?:ning)? (?:left|right|across))\b',
  'Execute the camera movement already specified as a continuous change in framing, with the stated direction, pace and extent. Keep the described subject action visible.'),
 ('travel',r'\b(?:walks?|walking|runs?|running|rush(?:es|ing)?|strid(?:es|ing)|staggers?|stumbles?|backs? away|steps? (?:back|forward)|storms? in|march(?:es|ing)?|enters?|entering|approach(?:es|ing)?|climbs?|climbing|moves? (?:toward|across|away))\b',
  'Show the described travel through several visibly changing positions, with natural steps and body displacement at the stated pace. Any described stop or reaction follows the travel rather than replacing it.'),
 ('continuous_performance',r'\b(?:twirls?|swing(?:s|ing)?|strum(?:s|ming)?|danc(?:es|ing)|playing the|plays the|paddl(?:es|ing)|stirr?(?:s|ing))\b',
  'Keep the described performance progressing visibly through the shot: clear arm or body motion, with the corresponding prop movement and original rhythm. Preserve the stated ending.'),
 ('prop_or_arm',r'\b(?:raises?|raising|lifts?|lifting|pulls?|pulling|push(?:es|ing)?|opens?|opening|clos(?:es|ing)|reaches?|reaching|grabs?|grabbing|throws?|throwing|turns? (?:the|a) (?:knob|handle)|picks? up|puts? down|places? (?:a|the|his|her))\b',
  'Clearly show the specified arm/hand action and the corresponding object movement from the starting state toward the described endpoint. Complete only the stated action, at its original pace.'),
 ('gesture_or_dialogue',r'\b(?:gestur(?:es|ing)|nods?|nodding|speaks?|speaking|talks?|talking|argu(?:es|ing)|turns? (?:his|her|their) head|looks? (?:around|toward|at)|glances?)\b',
  'Let the already described gestures, looks and speech progress naturally in sequence, with readable changes between poses. Preserve their stated subtlety and the existing camera framing.'),
 ('environment',r'\b(?:swirls?|swirling|billows?|billowing|flickers?|flickering|sways?|swaying|drifts?|drifting|ripples?|rippling)\b',
  'Let the environmental motion explicitly described continue evolving visibly over time, while preserving the characters and scene layout.')]
 for category,pattern,cue in tests:
  if re.search(pattern,s):break
 else:
  category='other_or_static';cue='Progress only the changes already described. Subjects explicitly stated to be still or motionless retain that state; preserve the original framing and scene.'
 prompt=cleaned.rstrip()+'\n'+cue+' Keep the same character identities, clothing, props and visual style as the starting frame.'
 return prompt,category,count

def baseline_scores(item,sk,cache,bindings):
 eid=item['episode_id'];path=Path(item['result']);work=path.parent/'.work';result={}
 for dim in DIMS:
  f=work/f'{eid}_{dim}_eval_results.json'
  if str(f) not in cache:cache[str(f)]=read(f);bindings[str(f)]=sha(f)
  scene,num=map(int,sk.split(':'));stem=f'{scene:03d}_{num:03d}'
  scores=[v['video_results'] for v in cache[str(f)][dim][1] if Path(v['video_path']).stem==stem]
  assert len(scores)==1 and math.isfinite(scores[0]),(eid,sk,dim);result[dim]=float(scores[0])
 assert result['dynamic_degree']==0
 return result

def prepare():
 if (OUT/'plan.json').exists():
  p=read(OUT/'plan.json');assert p['snapshot_sha256']==sha(SNAP) and p['selection_policy']==POLICY
  for f in p['code_sources']:assert sha(f['path'])==f['sha256'],f['path']
  return p
 snap=read(SNAP);rows=[r for r in read(ROWS) if r['current']==0];assert len(rows)==1333
 eps={e['episode_id']:e for e in snap['episodes']};pilot=read(PILOT/'plan.json');pilot_results=read(PILOT/'comparison.json')
 reuse={(c['episode_id'],c['shot']):(c,v) for c,v in zip(pilot['cases'],pilot_results['rows'])}
 cache={};bindings={};cases=[]
 for row in rows:
  eid,sk=row['episode_id'],row['shot'];source=Path(row['folder']);key=eid+'/'+sk;video=row['video'];vsha=snap['media'][key]['sha256'];assert sha(video)==vsha
  if row['route']=='original':
   g=read(source/'generation.json');assert g['video_sha256']==vsha;first=g['input_image'];assert sha(first)==g['input_sha256']
   old=dict(episode_id=eid,shot_key=sk,segment='original_fixed_input',input_path=first,input_sha256=g['input_sha256'],
    last_frame_path=None,last_frame_sha256=None,prompt=g['prompt'],frames=81,fps=16,size=[480,832],steps=4,seed=g['seed'],
    model_revision=read(BASE/'manifest.json')['model_revision'],style_sha256=None)
   task_source=source/'generation.json'
  else:
   task_source=source/'main/task.json';old=read(task_source);assert sha(old['input_path'])==old['input_sha256']
  assert old['frames']==81 and old['steps']==4 and old['last_frame_path'] is None
  prompt,category,cleaned=strengthen(old['prompt'],row['prompt']);prior=reuse.get((eid,sk))
  if prior:prompt=prior[0]['new_prompt'];category='pilot_'+prior[0]['kind']
  new={**old,'prompt':prompt};assert [k for k in old if old[k]!=new[k]]==['prompt']
  transition=str(source/'transition/video.mp4') if row['route']=='B' else None
  if transition:assert Path(transition).exists()
  cases.append(dict(episode_id=eid,run=row['run'],window=eid.split('__')[2],shot=sk,route=row['route'],category=category,
   baseline_video=video,baseline_video_sha256=vsha,baseline_scores=baseline_scores(eps[eid],sk,cache,bindings),
   source_task=str(task_source),source_task_sha256=sha(task_source),input_path=old['input_path'],input_sha256=old['input_sha256'],
   official_description=row['prompt'],old_prompt=old['prompt'],new_prompt=prompt,removed_image_number_references=cleaned,
   transition=transition,transition_sha256=sha(transition) if transition else None,task=new,
   reuse_result=str(PILOT/prior[0]['case_id']/'result.json') if prior else None,reuse_scores=prior[1]['after'] if prior else None))
 rank={'camera':0,'travel':1,'continuous_performance':2,'prop_or_arm':3,'gesture_or_dialogue':4,'environment':5,'other_or_static':6}
 cases.sort(key=lambda c:(not bool(c['reuse_result']),rank.get(c['category'],-1),c['episode_id'],c['shot']))
 for n,c in enumerate(cases,1):
  c['case_id']=f'M{n:04d}';folder=OUT/'cases'/c['case_id'];c['output']=str(folder/'main');c['task_path']=str(folder/'main/task.json')
  immutable(Path(c['task_path']),c['task']);c['task_sha256']=sha(c['task_path'])
 for path in [SNAP,ROWS,PILOT/'plan.json',PILOT/'comparison.json']:bindings[str(path)]=sha(path)
 sources=[Path(__file__),ROOT/'pretest/qwen140_pilot_video.py',ROOT/'pretest/run_lightx2v_a14b_4step_qwen140.py',OFFICIAL,BASE/'runtime/evaluate_benchmark_partial.py',BASE/'manifest.json']
 p=dict(created_utc=now(),scope='All1333 frozen-current dynamic0 shots across140 episodes; one fixed-input prompt-only candidate per shot, reuse existing6 candidates. Quality-selected supplement, not unconditional benchmark.',
  snapshot_sha256=sha(SNAP),cases=cases,total=1333,new_generation=1327,reuse=6,selection_policy=POLICY,
  category_counts=dict(Counter(c['category'] for c in cases)),route_counts=dict(Counter(c['route'] for c in cases)),
  prompt_compiler='Preserve existing video prompt, remove unsupported image-number references, append concise grounded action-process cue selected from official action paragraph. No new events/camera motion for static shots. Previous6 hand-authored candidates retained.',
  chunk_size=32,background_api_calls=0,code_sources=[dict(path=str(f),sha256=sha(f)) for f in sources],
  cache_bindings=[dict(path=k,sha256=v) for k,v in bindings.items()],disk_quota_gib=None,disk_reserve_gib=0,automatic_deletion=False,
  B_policy='Keep existing transition video untouched; re-generate only81-frame main segment from its exact original actual input, then stitch as before. No first-last-frame API or transition regeneration.',
  original_policy='For702 original shots use original input_image and seed from generation receipt; never use a newly decoded/reconstructed first frame.',
  runtime_validation='Persistent model worker must reproduce two existing pilot clips frame-for-frame before bulk generation.',
  estimated_generation_hours=[30,40],estimated_automatic_evaluation_hours=[3,10],visual_review_pending=True)
 immutable(OUT/'plan.json',p);immutable(OUT/'selection_policy.json',POLICY)
 with (OUT/'prompts.jsonl').open('w') as file:
  for c in cases:file.write(json.dumps({k:c[k] for k in ['case_id','episode_id','shot','category','old_prompt','new_prompt']},ensure_ascii=False)+'\n')
 status('prepared');document(p);return p

def document(p):
 text=['# 全部动态0镜头：固定首帧重跑与质量筛选','',p['scope'],'',
  '全部1333个目标各一个候选：复用已完成6个，新增1327个。原实际I2V输入PNG、原种子、模型、4步、81帧、画风与3D首帧结果保留，只改视频提示词。702个保留原版的shot也从原generation.json读取实际input_image，不替换首帧。15个B保留既有过渡，只重跑主段并按原方式拼接，无API。其他shot/人物素材/整个episode均不生成。','',
  '## 已确认门槛','',
  '- 动态必须0→1。','- 人物/物体/场景出现率不下降。','- 项目默认人物DINO、主体一致性、时间稳定性、运动平滑度、美学各最多下降0.01；成像最多下降1分。','- 无新增明显漏人、身份漂移、分屏或动作错误，需要可见画面复核。','- 原本可计分的新结果缺失视为不通过；双方都不适用保留null，不填0。','',
  '## 顺序与状态','',
  '1. 先冻结名单、每条原/新提示词与上述门槛，验证持久模型与原worker输出相同。','2. 按位移/运镜、连续操作、其他动作的顺序，但最终覆盖全部；每32例释放生成模型后，原算法评6项VBench并记录初筛。','3. 只有通过初筛的候选进入完整受影响episode语境评分、默认单脸DINO定位复核及画面检查。新视频失败/指标下降均保留，不只展示正例。','4. 所有复核完成后另建筛选后的媒体清单，完整重算受影响episode中心和实际边界，再聚合140集。原benchmark不覆盖；筛选版需同时报告全部1333尝试及通过/替换数。','',
  '6项VBench通过不表示完整筛选通过。完整人物中心必须重算，不拿孤立帧/旧中心分数代替。若最终组合改变中心而不再满足门槛，不能直接发布该组合。','',
  '## 提示词规则','',p['prompt_compiler'],'',
  '静止/死者/细微动作等剧情不强加走路、摇镜或新事件，预计许多候选仍会是0。所有新旧提示词在`prompts.jsonl`，执行清单和源SHA在`plan.json`。','',
  '## 保留与运行','',
  '- 不删原媒体、不设存储上限或自动清理。候选保留于`cases/M####/`。','- 后台无API、无守护修复Agent，无语义重试；技术异常保留日志和已完成结果。','- 初评在`cases/M####/vbench.json`；`vbench_pass`只代表初筛，不能当最终替换资格。','- `status.json`记录真实完成数；需要人工复核的阶段明确标pending，不冒充全部自动完成。','',
  '## 先导6例','',
  '按当前已确认的严格门槛，先导6例目前均未通过VBench初筛：3个动态提升样本的主体一致性下降分别约.0293、.0191、.0393，均超过.01。不会因动态涨分就直接替换，也不会事后放宽门槛。','']
 (OUT/'README.md').write_text('\n'.join(text))

def finish_case(c,main,validation=False):
 folder=Path(c['output']).parent;video=main/'video.mp4';last=main/'last.png'
 if c['transition'] and not validation:
  assert sha(c['transition'])==c['transition_sha256'];dest=folder/'combined';dest.mkdir(exist_ok=True);video=dest/'video.mp4';last=dest/'last.png'
  if not video.exists():subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',c['transition'],'-i',str(main/'video.mp4'),'-filter_complex',
   '[0:v]fps=16,scale=832:464,setsar=1,setpts=PTS-STARTPTS[a];[1:v]trim=start_frame=1,scale=832:464,setsar=1,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[v]',
   '-map','[v]','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
  import imageio.v2 as iio
  from PIL import Image
  reader=iio.get_reader(video,'ffmpeg',ffmpeg_params=['-threads','1']);count=0
  try:
   for frame in reader:tail=frame;count+=1
  finally:reader.close()
  Image.fromarray(tail).save(last)
 record=dict(case_id=c['case_id'],episode_id=c['episode_id'],shot_key=c['shot'],video=str(video),video_sha256=sha(video),last_frame=str(last),last_sha256=sha(last),
  input_sha256=c['input_sha256'],baseline_video_sha256=c['baseline_video_sha256'],plan_sha256=sha(OUT/'plan.json'),route=c['route'],transition_reused=bool(c['transition']))
 if not validation:write(folder/'result.json',record)
 return record

def worker(case_ids,validation=False):
 import torch
 from qwen140_pilot_video import load_pipeline
 from run_lightx2v_a14b_4step_qwen140 import decode_video
 p=prepare();lookup={c['case_id']:c for c in p['cases']};cases=[lookup[i] for i in case_ids];start=time.monotonic()
 pipeline,_=load_pipeline('i2v',81);load_seconds=time.monotonic()-start;results=[]
 for c in cases:
  main=OUT/'validation'/c['case_id'] if validation else Path(c['output']);main.mkdir(parents=True,exist_ok=True)
  if not validation and (main.parent/'result.json').exists():continue
  task=c['task'];assert sha(c['input_path'])==c['input_sha256'] and sha(c['baseline_video'])==c['baseline_video_sha256']
  if not (main/'generation.json').exists():
   tmp=main/'video.tmp.mp4';assert not tmp.exists();beg=time.monotonic();torch.cuda.reset_peak_memory_stats()
   if not validation:status('generating_persistent_chunk',case_id=c['case_id'],run=c['run'],shot=c['shot'],model_load_seconds=load_seconds)
   pipeline.generate(seed=task['seed'],prompt=task['prompt'],image_path=task['input_path'],save_result_path=str(tmp));torch.cuda.synchronize()
   decoded=decode_video(tmp,main/'first.png',main/'last.png',81,16);tmp.replace(main/'video.mp4')
   write(main/'generation.json',dict(condition_sha256=digest(task),inputs=task,decoded=decoded,video_sha256=sha(main/'video.mp4'),last_sha256=sha(main/'last.png'),
    elapsed_seconds=time.monotonic()-beg,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),persistent_model=True,model_load_seconds=load_seconds))
  g=read(main/'generation.json');assert g['condition_sha256']==digest(task) and sha(main/'video.mp4')==g['video_sha256']
  results.append(finish_case(c,main,validation))
 if validation:
  import imageio.v2 as iio
  import numpy as np
  checked=[]
  for c,record in zip(cases,results):
   expected=read(c['reuse_result']);a=iio.get_reader(expected['video'],'ffmpeg',ffmpeg_params=['-threads','1']);b=iio.get_reader(record['video'],'ffmpeg',ffmpeg_params=['-threads','1']);n=0
   try:
    for u,v in zip(a,b):assert np.array_equal(u,v),(c['case_id'],n);n+=1
   finally:a.close();b.close()
   assert n==81;checked.append(dict(case_id=c['case_id'],frames_exact=81,video_sha_identical=record['video_sha256']==expected['video_sha256'],reference=expected['video'],candidate=record['video']))
  immutable(OUT/'validation/receipt.json',dict(plan_sha256=sha(OUT/'plan.json'),checks=checked,persistent_seed_reset_verified=True,model_load_seconds=load_seconds,elapsed_seconds=time.monotonic()-start))

def vbench_pass(before,after):
 reasons=[]
 if after['dynamic_degree']!=1:reasons.append('dynamic_still_zero')
 for k,drop in POLICY['max_drop'].items():
  if k not in DIMS:continue
  if after[k]<before[k]-drop-POLICY['numerical_tolerance']:reasons.append(k+'_drop')
 return not reasons,reasons

def evaluate(case_ids,chunk):
 p=prepare();lookup={c['case_id']:c for c in p['cases']};cases=[lookup[k] for k in case_ids];directory=OUT/'vbench'/chunk
 spec=importlib.util.spec_from_file_location('all_zero_original_vbench',OFFICIAL);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 import torch
 torch.set_num_threads(4);os.environ['VBENCH_CACHE_DIR']='/root/autodl-tmp/models/entitybench_vbench';videos={}
 for n,c in enumerate(cases,1):
  r=read(Path(c['output']).parent/'result.json');assert sha(r['video'])==r['video_sha256'];videos[f'{n}:1']=r['video']
 receipt=directory/'receipt.json'
 if receipt.exists():
  proof=read(receipt);assert proof['plan_sha256']==sha(OUT/'plan.json') and proof['case_ids']==case_ids;scores=proof['scores']
 else:
  mscore=m.VBenchEvaluator('cuda',os.environ['VBENCH_CACHE_DIR'],str(directory),True);scores,failed=mscore.evaluate_episode_shots(videos,chunk,str(directory));assert not failed
  immutable(receipt,dict(plan_sha256=sha(OUT/'plan.json'),case_ids=case_ids,scores=scores,video_sha256={k:sha(v) for k,v in videos.items()},official_sha256=sha(OFFICIAL)))
 for n,c in enumerate(cases,1):
  after=scores[f'{n}:1'];ok,reasons=vbench_pass(c['baseline_scores'],after)
  immutable(Path(c['output']).parent/'vbench.json',dict(case_id=c['case_id'],before=c['baseline_scores'],after=after,vbench_pass=ok,reasons=reasons,
   final_eligible=False,pending=['entity_presence','complete_episode_dino','visual_review','joint_sequence_validation'],receipt=str(receipt)))

def run():
 p=prepare();beg=time.monotonic()
 with (OUT/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   for c in p['cases']:
    if not c['reuse_result']:continue
    original=read(c['reuse_result']);assert sha(original['video'])==original['video_sha256']
    folder=Path(c['output']).parent;immutable(folder/'result.json',{**original,'case_id':c['case_id'],'reused_pilot':True,'reuse_source':c['reuse_result']})
    assert all(abs(c['baseline_scores'][k]-next(x for x in read(PILOT/'comparison.json')['rows'] if x['after_video']==original['video'])['before'][k])<1e-5 for k in DIMS)
    ok,reasons=vbench_pass(c['baseline_scores'],c['reuse_scores']);immutable(folder/'vbench.json',dict(case_id=c['case_id'],before=c['baseline_scores'],after=c['reuse_scores'],vbench_pass=ok,reasons=reasons,
     final_eligible=False,pending=['entity_presence','complete_episode_dino','visual_review','joint_sequence_validation'],reuse_source=str(PILOT/'comparison.json')))
   probe=[c['case_id'] for c in p['cases'] if c['reuse_result'] and c['run'] in ['run103','run221']]
   assert len(probe)==2
   if not (OUT/'validation/receipt.json').exists():
    status('validating_persistent_model',case_ids=probe)
    with (OUT/'validation.log').open('a') as log:subprocess.run([PY,'-u',str(Path(__file__)),'worker','--ids',','.join(probe),'--validation'],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
   proof=read(OUT/'validation/receipt.json');assert proof['plan_sha256']==sha(OUT/'plan.json') and proof['persistent_seed_reset_verified']
   pending=[c for c in p['cases'] if not c['reuse_result']]
   for offset in range(0,len(pending),p['chunk_size']):
    batch=pending[offset:offset+p['chunk_size']];ids=[c['case_id'] for c in batch];tag=f'chunk_{offset//p["chunk_size"]+1:03d}'
    if any(not (Path(c['output']).parent/'result.json').exists() for c in batch):
     status('loading_next_generation_chunk',chunk=tag)
     with (OUT/(tag+'_generation.log')).open('a') as log:subprocess.run([PY,'-u',str(Path(__file__)),'worker','--ids',','.join(ids)],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    if any(not (Path(c['output']).parent/'vbench.json').exists() for c in batch):
     status('evaluating_original_six_vbench',chunk=tag)
     with (OUT/(tag+'_evaluation.log')).open('a') as log:subprocess.run([EVAL,'-u',str(Path(__file__)),'evaluate','--ids',','.join(ids),'--chunk',tag],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    status('chunk_complete',chunk=tag,elapsed_seconds=time.monotonic()-beg)
   status('all_candidates_and_vbench_complete_context_review_pending',elapsed_seconds=time.monotonic()-beg,
    next_steps=['complete_affected_episode_presence_centroids_boundaries','visible_single_face_localization_review','visual_action_review','joint_selected_sequence_recompute','publish_separate_selected_full140'])
  except Exception as error:
   status('stopped_technical_error_preserve_outputs',error=repr(error),elapsed_seconds=time.monotonic()-beg);raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker','evaluate']);parser.add_argument('--ids');parser.add_argument('--validation',action='store_true');parser.add_argument('--chunk');a=parser.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='run':run()
 elif a.action=='worker':worker(a.ids.split(','),a.validation)
 else:evaluate(a.ids.split(','),a.chunk)
