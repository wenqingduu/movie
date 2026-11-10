"""Explicit Torch RoPE repair for original40 trial; large queue remains user-paused.
Preserve failed trial and all original generation files; no altered model/sampler/CFG.
"""
from __future__ import annotations
import copy,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import qwen140_original40_vs_distill4_20261108 as original
PREVIOUS=original.OUT
OUT=original.ROOT/'outputs/entitybench_original40_vs_distill4_torch_rope_20261108'
PARENT=45449
WORKER=93303
read,write,sha=original.read,original.write,original.sha


def prepare():
 if (OUT/'plan.json').exists():
  p=read(OUT/'plan.json')
  for b in p['bindings']:assert sha(b['path'])==b['sha256'],b['path']
  return p
 old=read(PREVIOUS/'plan.json');p=copy.deepcopy(old);cfg=read(old['config'])
 cfg.update(rope_type='torch_complex_rope',double_precision_rope=True)
 original.immutable(OUT/'original40_config.json',cfg)
 p.update(created_utc=original.now(),config=str(OUT/'original40_config.json'),previous_failed_plan=str(PREVIOUS/'plan.json'),
  previous_failed_plan_sha256=sha(PREVIOUS/'plan.json'),repair='Explicit rope_type=torch_complex_rope and double_precision_rope=True, matching existing4-step API defaults. Prior customJSON omitted these and fell through to unavailable FlashInfer. No dependency install or mathematical backend recipe downgrade.',
  scheduling=dict(parent=PARENT,worker=WORKER,method='User explicitly paused large queue. Stop new dispatch; let last active batch finish, end idle scheduler, keep all outputs and do not resume at test end.'))
 p['bindings'] += [dict(path=str(f),sha256=sha(f)) for f in [Path(__file__),OUT/'original40_config.json',PREVIOUS/'plan.json']]
 original.immutable(OUT/'plan.json',p);original.status('prepared_original40_torch_rope',large_queue_auto_resume=False)
 return p


def configure():
 original.OUT=OUT;original.prepare=prepare;original.__file__=str(Path(__file__).resolve())


def freeze_large_queue_pause():
 source=original.SOURCE
 p=read(source/'plan.json');expected=[f'M{i:04d}' for i in range(199,231)]
 assert all((source/'cases'/k/'result.json').exists() for k in expected),'Active batch incomplete; preserve error and do not start another GPU model'
 cmd=original.active(PARENT)
 if cmd:
  assert 'qwen140_all_zero_background_20261108.py handoff-run' in cmd
  assert Path(f'/proc/{PARENT}/stat').read_text().split()[2]=='T'
  os.kill(PARENT,signal.SIGKILL)  # Idle scheduler only, after its GPU worker exits.
 generated=list((source/'cases').glob('*/result.json'));scored=list((source/'cases').glob('*/vbench.json'))
 write(source/'status.json',dict(stage='paused_by_user_prioritize_original40_trial',utc=original.now(),expected=1333,
  generated=len(generated),vbench_scored=len(scored),vbench_preliminary_pass=sum(read(x)['vbench_pass'] for x in scored),
  final_replacements=0,final_selection_pending=True,user_requested_pause=True,automatic_resume=False,
  active_generation_worker=None,next_experiment=str(OUT),all_existing_outputs_preserved=True,api_calls=0))
 write(OUT/'large_queue_paused.json',dict(utc=original.now(),generated=len(generated),vbench_scored=len(scored),
  all_existing_outputs_preserved=True,automatic_resume=False,paused_at_active_batch_end=True))


def schedule():
 configure();p=prepare()
 with (OUT/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   original.status('waiting_for_last_active_large_queue_batch',large_queue_auto_resume=False,worker=WORKER)
   while True:
    cmd=original.active(WORKER)
    if not cmd:break
    assert 'qwen140_all_zero_background_20261108.py worker' in cmd
    time.sleep(10)
   freeze_large_queue_pause()
   for action,python in [('generate',original.PY),('evaluate',original.EVAL)]:
    with (OUT/(action+'.log')).open('a') as log:subprocess.run([python,'-u',str(Path(__file__)),action],env=original.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
  except Exception as error:
   original.status('stopped_original40_error_large_queue_remains_paused',error=repr(error),large_queue_auto_resume=False);raise


if __name__=='__main__':
 configure();action=sys.argv[1]
 if action=='prepare':print(json.dumps(dict(output=str(OUT),cases=prepare()['total'],large_queue_auto_resume=False)))
 elif action=='schedule':schedule()
 elif action=='generate':original.generate()
 elif action=='evaluate':original.evaluate()
 else:raise ValueError(action)
