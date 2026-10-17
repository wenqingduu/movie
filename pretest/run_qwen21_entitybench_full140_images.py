"""Full140 Qwen adapter: same reference and injection algorithms; only new episodes."""
from __future__ import annotations
import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from prepare_entitybench_qwen21_20ep import MODEL, REVISION, sha256, write_json, digest, frame_fingerprint, shot_prompt
from prepare_entitybench_qwen21_full140 import OUTPUT


def guard(root):
    assert shutil.disk_usage(root).free >= 10*1024**3, 'Less than 10GiB free: preserve outputs, expand storage and resume'


def face_score(app,image):
    import numpy as np
    import cv2
    faces=app.get(cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2BGR))
    if len(faces)!=1:return -1000-abs(len(faces)-1),None
    face=faces[0];pose=[float(x) for x in face.pose];box=[float(x) for x in face.bbox]
    score=float(face.det_score)+(box[2]-box[0])*(box[3]-box[1])/(image.width*image.height)-sum(abs(x) for x in pose)/180
    return score,dict(bbox=box,pose=pose,confidence=float(face.det_score))


def references(root,plan,pipe):
    import torch
    from multishot.ip_adapter_experiment_utils import _face_app
    app=_face_app();rows=[]
    impl=json.loads((root/'runtime/qwen_implementation.json').read_text())
    for role in plan['roles']:
        guard(root)
        expected=digest(dict(role=role,settings=plan['reference_settings'],model_revision=REVISION,implementation=impl))
        path,meta_path=Path(role['reference']),Path(role['metadata'])
        if path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text());assert meta['dependency_fingerprint']==expected
            assert sha256(path)==meta['selected_sha256']
            rows.append(dict(episode_id=role['episode_id'],character=role['character'],status='resumed_verified'))
            continue
        candidates=[];count=3 if role['needs_face_asset'] else 1
        print(f"Reference {role['episode_id']} {role['character']} ({len(rows)+1}/{len(plan['roles'])})",flush=True)
        for index in range(count):
            guard(root)
            seed=role['seed']+index
            candidate=path.parent/'candidates'/path.stem/f'seed_{seed}.png';record_path=candidate.with_suffix('.json')
            if candidate.exists() and record_path.exists():
                row=json.loads(record_path.read_text());assert row['dependency_fingerprint']==expected and row['seed']==seed
                assert sha256(candidate)==row['sha256']
            else:
                begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
                image=pipe(prompt=role['prompt'],width=1024,height=1024,num_inference_steps=40,
                    generator=torch.Generator('cuda').manual_seed(seed),true_cfg_scale=1.0,
                    output_resolution=1024,use_kv_cache=True).images[0].convert('RGB')
                candidate.parent.mkdir(parents=True,exist_ok=True);image.save(candidate)
                score,face=face_score(app,image) if role['needs_face_asset'] else (0,None)
                row=dict(seed=seed,path=str(candidate),sha256=sha256(candidate),selection_score=score,selected_face=face,
                    dependency_fingerprint=expected,elapsed_seconds=time.monotonic()-begin,
                    peak_cuda_memory_mib=round(torch.cuda.max_memory_allocated()/1024**2,2))
                write_json(record_path,row)
                print(f"Saved reference candidate {role['character']} seed={seed} {row['elapsed_seconds']:.1f}s face={bool(face)}",flush=True)
            candidates.append(row)
        selected=max(candidates,key=lambda r:r['selection_score'])
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(selected['path'],path)
        meta=dict(episode_id=role['episode_id'],character=role['character'],description=role['description'],
            model='Qwen-Image-2.1',model_revision=REVISION,model_path=str(MODEL),prompt=role['prompt'],
            width=1024,height=1024,steps=40,selected_seed=selected['seed'],selected_sha256=sha256(path),
            face_eligible=bool(role['needs_face_asset'] and selected['selected_face']),selected_face=selected['selected_face'],
            face_skip_reason=None if selected['selected_face'] else 'no_reliable_single_reference_face' if role['needs_face_asset'] else '3D_not_required_for_this_role',
            candidates=candidates,dependency_fingerprint=expected)
        write_json(meta_path,meta)
        rows.append(dict(episode_id=role['episode_id'],character=role['character'],status='generated',face_eligible=meta['face_eligible']))
        write_json(root/'references_status.json',dict(status='running',complete_roles=len(rows),total_roles=len(plan['roles']),last_role=rows[-1]))
        gc.collect();torch.cuda.empty_cache()
    write_json(root/'references_status.json',dict(status='complete',complete_roles=len(rows),total_roles=len(plan['roles']),roles=rows))


def frames(root,plan,pipe):
    import torch
    from PIL import Image
    from facexlib.parsing import init_parsing_model
    from multishot.ip_adapter_experiment_utils import _face_app
    import run_qwen_image21_shared_prefix_adaptive as injection
    app=_face_app();parser=init_parsing_model(model_name='bisenet',device=torch.device('cuda'))
    args=SimpleNamespace(**plan['firstframe_settings'],overwrite=False)
    rows=[]
    for ep in plan['episodes']:
        if ep.get('reuse_generation_root'):continue
        for shot in ep['shots']:
            if not shot['cut']:continue
            guard(root);directory=Path(ep['firstframe_root'])/('shot_'+shot['shot_key'].replace(':','_'))
            path=directory/'frame_record.json';expected=frame_fingerprint(root,plan,ep,shot)
            if path.exists():
                row=json.loads(path.read_text());assert row['dependency_fingerprint']==expected
                assert sha256(row['final_image'])==row['final_sha256'];rows.append(row);continue
            names=shot['entity_schedule'].get('characters',[]);assert len(names)<=10
            roles=[next(r for r in plan['roles'] if r['episode_id']==ep['episode_id'] and r['character']==n) for n in names]
            characters=[dict(name=r['character'],reference_image=r['reference'],facelift_result=r['result']) for r in roles]
            prompt=shot_prompt(shot['prompt'],names)
            record=dict(run=ep['run'],episode_id=ep['episode_id'],shot_key=shot['shot_key'],seed=shot['seed'],
                official_prompt=shot['prompt'],qwen_prompt=prompt,characters=characters)
            eligible=(len(roles)==1 and roles[0]['needs_face_asset'] and json.loads(Path(roles[0]['metadata']).read_text())['face_eligible'])
            directory.mkdir(parents=True,exist_ok=True);begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
            print(f"Hardcut {ep['episode_id']} {shot['shot_key']} ({len(rows)+1}/{plan['counts']['new_hardcuts']}) eligible={eligible}",flush=True)
            if eligible:
                result=injection.run_shot(pipe,app,parser,record,directory,args)
                failures=[str(result.get('error',''))]+[str(r.get('error','')) for r in result.get('target_failures',[])]
                assert not any(any(term in error.lower() for term in ['out of memory','no space left','filenotfounderror','cuda error','modulenotfounderror']) for error in failures), failures
                final=Path(result['treatment']);plugin=result['injection_applied'];reason=result.get('reason')
            else:
                images=[Image.open(r['reference']).convert('RGB') for r in roles]
                output=pipe(prompt=prompt,image=images or None,width=args.width,height=args.height,
                    num_inference_steps=args.steps,output_resolution=args.output_resolution,true_cfg_scale=1.0,
                    generator=torch.Generator('cuda').manual_seed(shot['seed']),use_kv_cache=True).images[0]
                final=directory/'final.png';output.convert('RGB').save(final);plugin=False
                reason='multi_reference_without_multi_face_injection' if len(names)>1 else 'reference_face_gate' if names else 'no_character'
            row=dict(episode_id=ep['episode_id'],shot_key=shot['shot_key'],seed=shot['seed'],official_prompt=shot['prompt'],
                qwen_prompt=prompt,characters=names,plugin_applied=plugin,skip_reason=reason,
                final_image=str(final.resolve()),final_sha256=sha256(final),dependency_fingerprint=expected,
                elapsed_seconds=time.monotonic()-begin,peak_cuda_memory_mib=round(torch.cuda.max_memory_allocated()/1024**2,2))
            write_json(path,row);rows.append(row)
            write_json(root/'firstframes_status.json',dict(status='running',complete_hardcuts=len(rows),total_hardcuts=plan['counts']['new_hardcuts'],last_shot=row))
            gc.collect();torch.cuda.empty_cache()
    assert len(rows)==plan['counts']['new_hardcuts']
    write_json(root/'firstframes_status.json',dict(status='complete',complete_hardcuts=len(rows),total_hardcuts=plan['counts']['new_hardcuts'],shots=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=OUTPUT);p.add_argument('--stage',choices=['references','frames'],required=True)
    args=p.parse_args();root=args.output_root.resolve();plan=json.loads((root/'preparation_plan.json').read_text())
    from audit_entitybench_qwen21_full140 import verify_adapters
    verify_adapters(root)
    import torch
    from diffusers import QwenImage21Pipeline
    torch.set_num_threads(4)
    pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),dtype=torch.bfloat16,local_files_only=True)
    if args.stage=='references':pipe.to('cuda')
    else:pipe.enable_sequential_cpu_offload()
    (references if args.stage=='references' else frames)(root,plan,pipe)
