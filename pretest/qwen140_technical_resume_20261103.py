"""Technical-only resumption adapter; preserves all frozen generation sources.

Repairs character lookup identifiers, GET-only SSL polling and T5 loading's
anonymous-memory peak. Does not change weights, quantization, prompts, routes,
seeds, frame/video samplers, or v7 injection gates for valid cached plans.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/entitybench_episode_repairs_20261103'
T5=ROOT/'models/video/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth'


def bounded_lookup_tool(template,names):
    tool=copy.deepcopy(template)
    spec=tool['function']['parameters']['properties']['names']
    spec['uniqueItems']=True
    if names:
        spec['items']['enum']=list(names)
    else:
        spec['maxItems']=0
    tool['function']['description']+=' names只能使用本次catalog中的精确人物名；不可查询道具、人物外貌句子或发明角色名。名单为空时names必须为[]。'
    return tool


def install_planner_adapter(episode,agent):
    original_plan=agent.plan_shot
    original_tool=copy.deepcopy(agent.LOOKUP_TOOL)

    def plan(context,tail,catalog,folder):
        # A valid cached plan retains its exact original context/fingerprint.
        if (folder/'plan.json').exists():
            return original_plan(context,tail,catalog,folder)
        context=copy.deepcopy(context)
        current=list(catalog)
        ep,_,_=episode.load(next(iter(episode.core.TAGS)),None)
        index=next(i for i,s in enumerate(ep['shots']) if s['shot_key']==context['shot_key'])
        inherited={}
        if tail is not None and index:
            previous=ep['shots'][index-1]
            parent=folder.parent.parent/f"shot_{previous['shot_key'].replace(':','_')}"/'result.json'
            names=previous['entity_schedule'].get('characters',[])
            if parent.exists():names=names+json.loads(parent.read_text())['route'].get('frame_characters',[])
            desc=episode.core.descriptions(ep)
            inherited={n:desc[n] for n in dict.fromkeys(names) if n in desc}
        context['valid_lookup_character_names']=current
        context['valid_frame_character_ids']=list(dict.fromkeys(current+list(inherited)))
        context['inherited_character_descriptions']=inherited
        context['identifier_contract']='lookup_characters仅查当前catalog精确人物名，道具和外貌描述不是人物ID。frame_characters只列有身份的可见角色ID；匿名/未调度人物仅在frame_prompt和shot_prompt描述，不发明名字；无可确认身份时frame_characters=[]。不据此省略shot原文要求的人物。'
        agent.LOOKUP_TOOL=bounded_lookup_tool(original_tool,current)
        try:
            result=original_plan(context,tail,catalog,folder)
            unknown=set(result['decision']['frame_characters'])-set(context['valid_frame_character_ids'])
            if unknown:raise ValueError(f'Planner invented frame character IDs: {sorted(unknown)}')
            return result
        finally:
            agent.LOOKUP_TOOL=original_tool
    episode.plan_shot=plan


def install_get_poll_resume(wan):
    original=wan.api_request
    def request(url,key,folder,payload=None):
        # Never resubmit a POST creation request. GETs retain the saved task ID.
        if payload is not None:return original(url,key,folder,payload)
        for attempt in range(3):
            try:return original(url,key,folder)
            except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
                if isinstance(exc,urllib.error.HTTPError) and exc.code not in {429,502,503,504}:raise
                if attempt==2:raise
                path=folder/'technical_get_resume_events.json'
                events=json.loads(path.read_text()) if path.exists() else []
                events.append({'attempt':attempt+1,'error_type':type(exc).__name__,
                    'method':'GET','same_saved_task_id':True,'replacement_task_submitted':False})
                path.write_text(json.dumps(events,indent=2)+'\n')
                time.sleep([5,15][attempt])
    wan.api_request=request


def install_t5_mmap():
    import torch
    original=torch.load
    def load(source,*args,**kwargs):
        if isinstance(source,(str,os.PathLike)) and Path(source).resolve()==T5.resolve():
            kwargs['mmap']=True
            print('TECHNICAL_RESUME: T5 torch.load(mmap=True); identical BF16 checkpoint tensors',flush=True)
        return original(source,*args,**kwargs)
    torch.load=load


def install_worker_dispatch(core):
    original=core.run_worker
    def run(python,action,*args):
        if action!='worker-video':return original(python,action,*args)
        env=dict(os.environ)
        env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
                   DTYPE='BF16',SENSITIVE_LAYER_DTYPE='None')
        subprocess.run([python,str(Path(__file__).resolve()),'worker-video',*args],env=env,check=True)
    core.run_worker=run


def install_episode():
    import qwen140_episode_single_pass as episode
    import qwen140_shot_agent as agent
    import qwen140_wan_api as wan
    install_planner_adapter(episode,agent)
    install_get_poll_resume(wan)
    install_worker_dispatch(episode.core)
    return episode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['episode','plan','worker-video'])
    parser.add_argument('--run');parser.add_argument('--shot');parser.add_argument('--task',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='worker-video':
        install_t5_mmap()
        import qwen140_integrated_episodes as core
        core.worker_video(args.task,args.output)
        core.write_once(args.output/'technical_resume_runtime.json',{
            'adapter':str(Path(__file__).resolve()),'adapter_sha256':core.sha(Path(__file__)),
            't5_load':'torch.load mmap=True, no weight/dtype/model/seed/sampler changes',
            'checkpoint_validation':str(OUT/'t5_mmap_validation.json')})
    else:
        episode=install_episode()
        if args.action=='plan':episode.stage('plan',args.run,args.shot)
        else:episode.run_remaining(args.run)


if __name__=='__main__':main()
