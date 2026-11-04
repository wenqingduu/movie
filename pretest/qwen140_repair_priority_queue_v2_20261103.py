"""Resume the16 technical failures first, then continue the existing55 cohort.

Not an Agent/watchdog. Frozen generators and experiments are never edited.
All existing results/identity assets/tasks retain their original fingerprints.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

import qwen140_csface_priority_queue as prior

ROOT=Path('/root/autodl-tmp/movie')
OUT=ROOT/'outputs/entitybench_episode_repairs_20261103'
ADAPTER=ROOT/'pretest/qwen140_technical_resume_v2_20261103.py'
B=prior.B


def status(state,episode=None,**extra):
    B.write(OUT/'status.json',{'updated_utc':B.now(),'status':state,'active_episode':episode,
        'repair_agent_enabled':False,'technical_adapter':str(ADAPTER),**extra})


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'queue_v2.json').exists():
        plan=B.read(OUT/'queue_v2.json');validate(plan);return plan
    base_plan=B.read(prior.OUT/'queue.json')
    source=prior.validate(base_plan);by_id={e['episode_id']:e for e in source['episodes']}
    previous=B.read(OUT/'queue.json')
    errors=B.read(prior.OUT/'technical_exceptions.json')
    failed=list(dict.fromkeys(r['episode_id'] for r in errors if not (Path(by_id[r['episode_id']]['output_root'])/'complete.json').exists()))
    assert len(failed)==16
    # The previously SIGKILL'd12-shot episode is also the measured video test.
    first=next(eid for eid in failed if '__run1172__' in eid)
    order=[first]+[eid for eid in failed if eid!=first]
    order += [e['episode_id'] for e in base_plan['episodes'] if e['episode_id'] not in order]
    prefix=[];invalid=[]
    for eid in order:
        root=Path(by_id[eid]['output_root'])
        for path in (root/'episodes'/eid).glob('shot_*/result.json'):
            row=B.read(path)
            for p in [path,Path(row['video']),Path(row['last_frame'])]:
                prefix.append({'path':str(p),'sha256':B.sha(p)})
        # Keep the failed tool reply as evidence; rerun only the invalid plan.
        for folder in (root/'episodes'/eid).glob('shot_*/agent'):
            if not (folder/'plan.json').exists() and (folder/'planning_inputs.json').exists():
                target=OUT/'archived_schema400_agent'/eid/folder.parent.name
                assert not target.exists()
                snapshot=[{'name':p.name,'sha256':B.sha(p)} for p in folder.iterdir() if p.is_file()]
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(folder),str(target))
                invalid.append({'episode_id':eid,'shot':folder.parent.name,'archived':str(target),'files':snapshot})
    proof=B.read(OUT/'t5_mmap_validation.json');assert proof['all_tensors_bit_identical']
    first_test=B.read(OUT/'first_video_validation.json')
    first_task=Path(first_test['args'][first_test['args'].index('--task')+1])
    assert B.sha(first_task)==first_test['condition_sha256']
    assert (first_task.parent/'generation.json').exists(), 'First repaired GPU video must pass before queue launch'
    plan={'created_utc':B.now(),'cohort_episodes':55,'priority_repair_episodes':failed,
        'order':order,'source_queue':str(prior.OUT/'queue.json'),
        'source_queue_sha256':B.sha(prior.OUT/'queue.json'),
        'adapter_sha256':B.sha(ADAPTER),'queue_script_sha256':B.sha(Path(__file__)),
        'previous_plan_sha256':B.sha(OUT/'queue.json'),
        'previous_archived_invalid_plans':previous['invalid_planning_caches_archived'],
        'frozen_code':source['frozen_code'],'original_completed_prefix':prefix,
        'invalid_planning_caches_archived':invalid,
        'changes':['Exact-catalog lookup IDs; anonymous people described without invented asset IDs.',
                   'Only T5 checkpoint torch.load uses mmap=True; all242 tensors validated bit-identical.',
                   'TLS GET polling may retry same saved task ID; POST creation never retried.'],
        'unchanged':'Models/BF16/LoRAs/seeds/samplers/v7 gates/episode style; completed frames/videos/plans/2D and3D assets.',
        'failure_policy':'Record unknown technical failures and continue other episodes; stop on auth/quota or repeat SIGKILL after the validated loading fix.',
        'disk_policy':source['disk_policy'],'repair_agent_enabled':False}
    B.write(OUT/'queue_v2.json',plan,frozen=True)
    B.write(prior.SOURCE/'active_queue.json',{'queue_root':str(OUT),'status':str(OUT/'status.json'),
        'selection':'Technical failures first; continue the same original55 below-average episodes.'})
    old=B.read(prior.OUT/'status.json');old.update(status='superseded_by_technical_repair_queue',
        updated_utc=B.now(),active_queue=str(OUT),active_episode=None)
    B.write(prior.OUT/'status.json',old)
    refresh(plan);status('prepared')
    return plan


def validate(plan):
    assert B.sha(prior.OUT/'queue.json')==plan['source_queue_sha256']
    assert B.sha(ADAPTER)==plan['adapter_sha256']
    assert B.sha(OUT/'queue.json')==plan['previous_plan_sha256']
    assert B.sha(Path(__file__))==plan['queue_script_sha256']
    source=prior.validate(B.read(prior.OUT/'queue.json'))
    for item in plan['original_completed_prefix']:
        assert B.sha(item['path'])==item['sha256'],item['path']
    return source


def refresh(plan):
    prior.refresh(B.read(prior.OUT/'queue.json'))
    progress=B.read(prior.OUT/'progress.json')
    repairs=set(plan['priority_repair_episodes'])
    B.write(OUT/'progress.json',{'completed_episodes':progress['completed_episodes'],
        'expected_episodes':55,'repaired_complete_episodes':sum(e['complete_official_non_vlm'] for e in progress['episodes'] if e['episode_id'] in repairs),
        'expected_repair_episodes':len(repairs),'episodes':progress['episodes']})


def run(plan):
    with (prior.SOURCE/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=validate(plan);by_id={i['episode_id']:i for i in source['episodes']}
        errors=B.read(OUT/'technical_exceptions.json') if (OUT/'technical_exceptions.json').exists() else []
        for eid in plan['order']:
            item=by_id[eid];root=Path(item['output_root'])
            if (root/'complete.json').exists():continue
            validate(plan)
            needed=prior.original.disk_required(item)
            while shutil.disk_usage(OUT).free<needed:
                status('waiting_for_disk_space',eid,free_gib=shutil.disk_usage(OUT).free/1024**3,
                       needed_gib=needed/1024**3,output_root=str(root));time.sleep(60)
            try:
                env=B.assets(item,source)
                if not (root/'rerun_manifest.json').exists():
                    for action in ['prepare','refs']:
                        B.command(item,'resume_'+action,[B.WAN,'-u','pretest/qwen140_episode_single_pass.py',action,'--run',item['episode']['run']],env)
                if not (root/'episodes'/eid/'episode_result.json').exists():
                    B.command(item,'technical_resume_generation',[B.WAN,'-u',str(ADAPTER),'episode','--run',item['episode']['run']],env)
                for action in ['prepare','run','baseline-aggregate','compare']:
                    B.command(item,'technical_resume_evaluation_'+action,[B.EVAL,'-u','pretest/qwen140_integrated_episode_eval.py',action,'--name','weak4_v1','--runs',item['episode']['run']],env)
                prior.original.summarize(item)
                B.write(root/'technical_resume_complete.json',{'episode_id':eid,'completed_utc':B.now(),
                    'adapter_sha256':plan['adapter_sha256'],'queue':str(OUT),'original_prefix_preserved':True})
                print(B.now(),'COMPLETE',eid,flush=True)
            except Exception as exc:
                record={'episode_id':eid,'recorded_utc':B.now(),'error':repr(exc),'prefix_retained':True}
                errors.append(record);B.write(OUT/'technical_exceptions.json',errors)
                logs='\n'.join(p.read_text(errors='replace')[-7000:] for p in (root/'logs').glob('technical_resume*.log'))
                if any(k in logs for k in ['SIGKILL','invalid_api_key','insufficient_quota','Arrearage','HTTP 401','insufficient balance']):
                    status('stopped_for_systemic_error',eid,error=record,output_root=str(root));refresh(plan);return
            refresh(plan)
        progress=B.read(OUT/'progress.json')
        status('complete_official_pending_new_dino_review' if progress['completed_episodes']==55 else 'finished_with_recorded_errors',
               completed_episodes=progress['completed_episodes'],technical_exceptions=len(errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prior.configure();B.status=status
    plan=prepare()
    if args.action=='run':run(plan)
