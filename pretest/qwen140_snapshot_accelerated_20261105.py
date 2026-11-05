"""Verified per-video VBench caching and measured whole-episode concurrency.

Resumes the existing108-shot full140 snapshot in place without changing frozen
scores, inputs, scorer, generation, or review protocol. New neural workers use
independent directories; only verified complete results are linked for resume.
"""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os,signal,subprocess,sys,time,hashlib,math
import qwen140_completed_v2_snapshot_20261105 as snapshot
full=snapshot.full
ROOT,OUT=snapshot.ROOT,snapshot.OUT
A=OUT/'acceleration';A.mkdir(exist_ok=True)
ADAPTER=ROOT/'pretest/qwen140_vbench_verified_cache_20261105.py'
EVAL=full.EVAL
read,write,sha=full.read,full.write,full.sha
TIERS=['easy','medium','hard']

def status(stage,**kw):write(A/'status.json',{'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),**kw})
def proc(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
  return {'state':s[0],'start_ticks':s[19],'cmd':Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()}
 except FileNotFoundError:return None

def command(split,dst,cached=True):
 return [EVAL,'-u',str(ADAPTER if cached else full.EVALUATOR),'--results_dir',str(OUT/'candidate_media'),
   '--scripts_dir',str(ROOT/'benchmarks/entitybench/data/scripts'),'--split_json',str(split),
   '--out_dir',str(dst),'--method_name','qwen140_prompt_v2_badshot_replacement','--pillars','1,2,3',
   '--skip_vlm','--resume','--device','cuda:0','--vbench_cache_dir','/root/autodl-tmp/models/entitybench_vbench']

def shard_split(eids):
 original=read(OUT/'affected/split.json')
 return {tier:[eid for eid in original[tier] if eid in eids] for tier in TIERS}

def validate_result(source,eids):
 m=read(source/'run_manifest.json');original=read(full.BASE/'evaluation/official_parallel_v2/run_manifest.json')
 for key in ['config','python_version','libraries','model_checkpoints','pillars','evaluation_scope']:assert m[key]==original[key],key
 assert m['strict'] and m['skip_vlm'] and not m['allow_partial'] and not m['allow_config_override']
 assert m['vlm_call_stats']['total_calls']==0
 for eid in eids:
  raw=read(source/(eid+'.json'));assert raw['episode_id']==eid
  for dim in full.DIMENSIONS:
   metric=raw['vbench_means'][dim];assert metric['n_evaluated']==raw['num_shots'] and metric['n_failed']==0 and math.isfinite(metric['value'])
  for name in full.agg.LABELS:assert full.agg.metric(raw,name)['n_failed']==0,(eid,name)
 return m

def run_group(tag,shards):
 """Parallel workers with disjoint complete episodes and separate work dirs."""
 root=A/tag;root.mkdir(parents=True,exist_ok=True)
 plan={'shards':shards,'adapter_sha256':sha(ADAPTER),'cache_manifest_sha256':sha(A/'vbench_cache_manifest.json'),'threads':4,'gpu':0}
 full.immutable(root/'plan.json',plan)
 env=full.environment();jobs=[];started=time.monotonic();samples=[]
 try:
  for i,eids in enumerate(shards):
   directory=root/('worker_'+str(i));directory.mkdir(exist_ok=True);split=directory/'split.json';full.immutable(split,shard_split(set(eids)))
   log=(directory/'worker.log').open('a');p=subprocess.Popen(command(split,directory/'results'),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();jobs.append((p,directory,eids))
  while any(p.poll() is None for p,_,_ in jobs):
   bad=[p.returncode for p,_,_ in jobs if p.poll() not in (None,0)]
   assert not bad,('Worker failed',bad,tag)
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
   rss=0
   for p,_,_ in jobs:
    try:
     for line in Path(f'/proc/{p.pid}/status').read_text().splitlines():
      if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
    except FileNotFoundError:pass
   samples.append({'elapsed':time.monotonic()-started,'gpu_used_mib':int(gpu[0]),'gpu_utilization':int(gpu[1]),'worker_rss_bytes':rss})
   write(root/'progress.json',{'elapsed_seconds':time.monotonic()-started,'jobs':[{'pid':p.pid,'exit':p.poll(),'episodes':len(eids),'complete':sum((d/'results'/(e+'.json')).exists() for e in eids)} for p,d,eids in jobs],'latest_resources':samples[-1]})
   time.sleep(2)
  for p,d,eids in jobs:
   assert p.returncode==0,(tag,p.returncode)
   validate_result(d/'results',eids)
  result={'wall_seconds':time.monotonic()-started,'workers':len(shards),'gpu_peak_mib':max((s['gpu_used_mib'] for s in samples),default=0),'worker_rss_peak_bytes':max((s['worker_rss_bytes'] for s in samples),default=0),'samples':samples}
  write(root/'execution.json',result)
  return [(d/'results',eids) for p,d,eids in jobs],result
 except BaseException:
  for p,_,_ in jobs:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  for p,_,_ in jobs:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  raise

def compare(a,b,eids):
 checks=[]
 for eid in eids:
  x,y=read(a/(eid+'.json')),read(b/(eid+'.json'));diffs={}
  for name in full.agg.LABELS:
   xx,yy=full.agg.metric(x,name),full.agg.metric(y,name)
   for n in ['n_evaluated','n_failed','n_skipped']:assert xx.get(n,0)==yy.get(n,0),(eid,name,n)
   if xx['value'] is None or yy['value'] is None:assert xx['value']==yy['value']
   else:
    delta=abs(xx['value']-yy['value']);diffs[name]=delta;assert delta<=1e-5,(eid,name,xx,yy)
  # Ensure cached execution did not change grounding/crop choice.
  af=a/'.work/eval_artifacts'/eid/'intra_shot';bf=b/'.work/eval_artifacts'/eid/'intra_shot';crop_checks=0
  for f in af.glob('*/_eval.json'):
   aa,bb=read(f),read(bf/f.parent.name/'_eval.json')
   for kind in ['character','object','location']:
    left,right=aa['presence'][kind],bb['presence'][kind]
    assert {k:v for k,v in left.items() if k!='items'}=={k:v for k,v in right.items() if k!='items'}
    for ar,br in zip(left['items'],right['items']):
     assert ar['name']==br['name'] and ar['status']==br['status']
     assert ar.get('frame_idx')==br.get('frame_idx')
     if ar.get('canonical_path'):
      assert sha(ar['canonical_path'])==sha(br['canonical_path']),(eid,f.parent.name,ar['name'],'canonical pixel change')
      crop_checks+=1
  checks.append({'episode_id':eid,'absolute_metric_differences':diffs,'identical_canonical_regions':crop_checks})
 return checks

def handoff():
 """Stop only at an already persisted episode boundary; keep every cache."""
 ids={'supervisor':113871,'snapshot':113886,'scorer':113895}
 before={k:proc(v) for k,v in ids.items()}
 assert before['supervisor'] and 'run_snapshot_serial.py' in before['supervisor']['cmd']
 assert before['snapshot'] and 'qwen140_completed_v2_snapshot_20261105.py run' in before['snapshot']['cmd']
 assert before['scorer'] and 'evaluate_benchmark_partial.py' in before['scorer']['cmd']
 for k in ['supervisor','snapshot']:os.kill(ids[k],signal.SIGSTOP)
 complete={p.stem for p in (OUT/'affected/results').glob('*__run*.json')}
 status('waiting_for_current_episode_checkpoint',completed=len(complete))
 while True:
  now={p.stem for p in (OUT/'affected/results').glob('*__run*.json')}
  if now-complete:break
  info=proc(ids['scorer']);assert info and info['state']!='Z','Original scorer exited before checkpoint'
  time.sleep(.2)
 os.kill(ids['scorer'],signal.SIGSTOP)
 # Every completed JSON and .work cache stays in place. The next episode may
 # have begun CPU setup; it has no complete JSON and will resume independently.
 for k in ['scorer','snapshot','supervisor']:
  info=proc(ids[k])
  if info and info['start_ticks']==before[k]['start_ticks']:os.kill(ids[k],signal.SIGKILL)
 for _ in range(60):
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if not apps:break
  time.sleep(1)
 assert not apps,apps
 receipt={'utc':datetime.now(timezone.utc).isoformat(),'old_processes':before,'checkpoint_new_episodes':sorted(now-complete),'complete_preserved':sorted(now),'generation_hold_unchanged':read(ROOT/'outputs/entitybench_prompt_v2_badshots_20261104/user_hold_after_snapshot_20261105.json')}
 write(A/'handoff.json',receipt)

def accelerated_evaluate(which,data):
 assert which=='affected'
 before_files={p.stem:p for p in (OUT/'affected/results').glob('*__run*.json')}
 serial_dir=A/'probe_serial/worker_0/results'
 probe=[e for e in before_files if '__run560__i85_j105__' in e or '__run654__i20_j41__' in e]
 assert len(probe)==2
 status('validating_serial_cache_reuse',preserved_episodes=len(before_files),probe=probe)
 serial,st=run_group('probe_serial',[probe])
 checks_cached=compare(OUT/'affected/results',serial[0][0],probe)
 status('validating_two_workers',serial_seconds=st['wall_seconds'])
 parallel,pt=run_group('probe_parallel',[[e] for e in probe])
 checks_parallel=[]
 for source,eids in parallel:checks_parallel+=compare(serial[0][0],source,eids)
 workers=2 if pt['wall_seconds']<st['wall_seconds'] else 1
 validation={'passed':True,'tolerance':1e-5,'serial_vs_original':checks_cached,'serial_vs_parallel':checks_parallel,
   'serial_seconds':st['wall_seconds'],'parallel_seconds':pt['wall_seconds'],'measured_speedup':st['wall_seconds']/pt['wall_seconds'],
   'serial_peak_gpu_mib':st['gpu_peak_mib'],'parallel_peak_gpu_mib':pt['gpu_peak_mib'],
   'serial_peak_worker_rss_bytes':st['worker_rss_peak_bytes'],'parallel_peak_worker_rss_bytes':pt['worker_rss_peak_bytes'],
   'chosen_workers':workers,'scope':'Two representative complete episodes: multi-person transformation and B transition. Six cached VBench dimensions, full grounding and centroids. Speedup is probe-only, not a full-cohort guarantee.'}
 write(A/'validation.json',validation)
 split=read(OUT/'affected/split.json');allids=[e for t in TIERS for e in split[t]]
 pending=[e for e in allids if not (OUT/'affected/results'/(e+'.json')).exists()]
 manifest={e['episode_id']:e for e in read(full.BASE/'manifest.json')['episodes']}
 shards=[[] for _ in range(min(workers,len(pending)))];loads=[0]*len(shards)
 for eid in sorted(pending,key=lambda e:(-len(manifest[e]['shots']),e)):
  i=min(range(len(shards)),key=lambda n:(loads[n],n));shards[i].append(eid);loads[i]+=len(manifest[eid]['shots'])
 status('accelerated_workers_running',workers=len(shards),preserved_episodes=len(before_files),pending_episodes=len(pending),worker_shots=loads)
 if shards:
  assignments,execution=run_group('remaining',shards)
  for source,eids in assignments:
   for eid in eids:
    destination=OUT/'affected/results'/(eid+'.json');assert not destination.exists()
    destination.symlink_to(source/(eid+'.json'))
    work=OUT/'affected/results/.work';sw=source/'.work'
    for src in sw.glob(eid+'*'):
     dst=work/src.name
     # A partial next-episode artifact from boundary switching must not
     # override the newly completed worker's complete diagnostic artifacts.
     if dst.exists() or dst.is_symlink():
      archive=A/'pre_switch_partial_work'/dst.name;archive.parent.mkdir(exist_ok=True);assert not archive.exists();dst.rename(archive)
     dst.symlink_to(src)
    src=sw/'eval_artifacts'/eid;dst=work/'eval_artifacts'/eid
    if dst.exists() or dst.is_symlink():
     archive=A/'pre_switch_partial_artifacts'/eid;archive.parent.mkdir(exist_ok=True);assert not archive.exists();dst.rename(archive)
    dst.parent.mkdir(exist_ok=True);dst.symlink_to(src,target_is_directory=True)
  write(A/'remaining_execution.json',execution)
 # Run official --resume once to produce one compatible complete manifest and
 # validate all41 cached results. It performs no new video inference here.
 status('validating_merged_official_resume')
 ORIGINAL_EVALUATE(which,data)
 status('official_complete_continuing_snapshot_diagnostics',episodes=41)

ORIGINAL_EVALUATE=full.evaluate
if __name__=='__main__':
 try:
  snapshot.configure()
  contract=read(A/'adapter_contract.json');assert sha(ADAPTER)==contract['adapter_sha256'];assert sha(A/'vbench_cache_manifest.json')==contract['cache_manifest_sha256']
  assert read(ROOT/'outputs/entitybench_prompt_v2_badshots_20261104/user_hold_after_snapshot_20261105.json')['generation_authorized_to_resume'] is False
  write(A/'orchestrator_source.json',{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)})
  handoff()
  full.evaluate=accelerated_evaluate
  snapshot.run()
  status('snapshot_official_and_automatic_dino_complete_pending_localization_review')
 except BaseException as exc:
  status('stopped_with_recorded_error',error=repr(exc))
  raise
