"""Existing FLUX sampler for new references and existing v7 single-character hardcuts."""
from __future__ import annotations
import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from prepare_entitybench_flux_native_20ep import OUTPUT, MODEL_ROOT, read, sha256, write_json, digest, frame_fingerprint, reusable_frame, validate_implementation


def guard(root):
    assert shutil.disk_usage(root).free>=5*1024**3, 'Less than 5GiB free: preserve all artifacts, expand storage and resume'


def reference_fingerprint(role,plan,implementation):
    return digest(dict(role=role,settings=plan['reference_settings'],implementation=implementation))


def load_models():
    import torch
    from multishot import pulid_flux_inner_face_experiment as core
    args=SimpleNamespace(fp8=True,max_sequence_length=128,onnx_provider='cpu',build_facelift=False,facelift_result=None)
    core._preflight_model_files(args)
    models=core._load_models(args,torch.device('cuda'))
    return core,models


def prepare_text(core,models,items,width,height):
    import torch
    _,_,t5,clip,_=models;t5.to('cuda');clip.to('cuda');result={}
    with torch.inference_mode():
        for key,prompt,seed in items:
            noise=core.get_noise(1,height,width,device=torch.device('cuda'),dtype=torch.bfloat16,seed=seed)
            result[key]=core.prepare(t5=t5,clip=clip,img=noise,prompt=prompt)
    t5.cpu();clip.cpu();gc.collect();torch.cuda.empty_cache()
    return result


def text_image(core,models,conditioning,seed,settings):
    import torch
    from einops import rearrange
    model,ae,_,_,_=models;device=torch.device('cuda')
    model.to(device);ae.to(device)
    width,height=settings['width'],settings['height']
    noise=core.get_noise(1,height,width,device=device,dtype=torch.bfloat16,seed=seed)
    state=rearrange(noise,'b c (h ph) (w pw) -> b (h w) (c ph pw)',ph=2,pw=2)
    schedule=core.get_schedule(settings['steps'],noise.shape[-1]*noise.shape[-2]//4,shift=True)
    with torch.inference_mode():
        for step in range(settings['steps']):
            pred=core._velocity(model,state,conditioning,schedule[step],settings['guidance'],None,0.0)
            state=state+(schedule[step+1]-schedule[step])*pred
        assert torch.isfinite(state).all(), 'Non-finite FLUX reference/basic latent'
        return core._decode(ae,state,height,width,device)


def references(root,plan,implementation,max_candidates=None):
    from PIL import Image
    from multishot.ip_adapter_experiment_utils import _face_app
    from run_qwen21_entitybench_20ep_images import face_score
    rows=[];pending=[]
    for role in plan['roles']:
        expected=reference_fingerprint(role,plan,implementation)
        if Path(role['reference']).exists() and Path(role['metadata']).exists():
            meta=read(role['metadata']);assert meta['dependency_fingerprint']==expected
            assert sha256(role['reference'])==meta['selected_sha256']
            rows.append(dict(episode_id=role['episode_id'],character=role['character'],status='resumed_verified'))
        else:pending.append(role)
    made=0
    if pending:
        core,models=load_models();app=_face_app()
        conditioning=prepare_text(core,models,[(r['reference'],r['prompt'],r['seed']) for r in pending],1024,1024)
        import torch
        for role in pending:
            guard(root);expected=reference_fingerprint(role,plan,implementation)
            path,metadata=Path(role['reference']),Path(role['metadata']);count=3 if role['needs_face_asset'] else 1;candidates=[]
            print(f"Reference {role['episode_id']} {role['character']} ({len(rows)+1}/112)",flush=True)
            for index in range(count):
                seed=role['seed']+index;candidate=path.parent/'candidates'/path.stem/f'seed_{seed}.png';record=candidate.with_suffix('.json')
                if candidate.exists() and record.exists():
                    row=read(record);assert row['dependency_fingerprint']==expected and row['seed']==seed and sha256(candidate)==row['sha256']
                else:
                    guard(root);begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
                    image=text_image(core,models,conditioning[role['reference']],seed,plan['reference_settings'])
                    candidate.parent.mkdir(parents=True,exist_ok=True);image.save(candidate)
                    score,face=face_score(app,image) if role['needs_face_asset'] else (0,None)
                    row=dict(seed=seed,path=str(candidate),sha256=sha256(candidate),selection_score=score,selected_face=face,
                        dependency_fingerprint=expected,elapsed_seconds=time.monotonic()-begin,
                        peak_cuda_memory_mib=round(torch.cuda.max_memory_allocated()/1024**2,2),
                        generator='FLUX.1-dev local FP8; id=None, id_weight=0; no SDXL/foreign identity carrier')
                    write_json(record,row);made+=1
                    print(f"Saved FLUX reference {role['character']} seed={seed} {row['elapsed_seconds']:.1f}s face={bool(face)}",flush=True)
                    if max_candidates is not None and made>=max_candidates:
                        write_json(root/'references_status.json',dict(status='limited_preflight',complete_roles=len(rows),total_roles=112,
                            candidate_count=made,last_candidate=row));return
                candidates.append(row)
            selected=max(candidates,key=lambda x:x['selection_score']);path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(selected['path'],path)
            meta=dict(episode_id=role['episode_id'],character=role['character'],description=role['description'],
                model='FLUX.1-dev local FP8',model_path=str(MODEL_ROOT/'flux-dev-fp8.safetensors'),
                prompt=role['prompt'],width=1024,height=1024,steps=40,guidance=4.0,
                selected_seed=selected['seed'],selected_sha256=sha256(path),
                face_eligible=bool(role['needs_face_asset'] and selected['selected_face']),selected_face=selected['selected_face'],
                face_skip_reason=None if selected['selected_face'] else 'no_reliable_single_reference_face' if role['needs_face_asset'] else '3D_not_required_for_this_role',
                candidates=candidates,dependency_fingerprint=expected)
            write_json(metadata,meta);rows.append(dict(episode_id=role['episode_id'],character=role['character'],status='generated',face_eligible=meta['face_eligible']))
            write_json(root/'references_status.json',dict(status='running',complete_roles=len(rows),total_roles=112,last_role=rows[-1]))
            gc.collect();torch.cuda.empty_cache()
    assert len(rows)==112
    write_json(root/'references_status.json',dict(status='complete',complete_roles=112,total_roles=112,roles=rows))


def single_arguments(core,role,shot,directory,settings):
    command=['flux_native_single','--reference-image',role['reference'],'--reference-origin',
        'EntityBench description; new FLUX-native reference; independent own-source FaceLift asset',
        '--reference-generated','--output-dir',str(directory),'--prompt',shot['prompt'],
        '--facelift-result',role['result'],'--harmonize-reference','--reference-conditioning','target',
        '--skip-unreliable-face','--onnx-provider','cpu']
    for argument,key in [('width','width'),('height','height'),('steps','steps'),('inject-start','inject_start'),
        ('guidance','guidance'),('pulid-id-weight','pulid_id_weight'),('injection-strength','injection_strength'),
        ('max-active-injection-steps','max_active_injection_steps'),('min-injection-face-height','min_injection_face_height'),
        ('max-face-detection-retries','max_face_detection_retries'),('max-sequence-length','max_sequence_length')]:
        command.extend(['--'+argument,str(settings[key])])
    command.extend(['--seed',str(shot['seed'])]);previous=sys.argv
    try:sys.argv=command;return core.parse_args()
    finally:sys.argv=previous


def frames(root,plan,implementation):
    rows=[];pending=[];roles={(r['episode_id'],r['character']):r for r in plan['roles']}
    for e in plan['episodes']:
        for s in e['shots']:
            if not s['cut']:continue
            directory=Path(e['firstframe_root'])/('shot_'+s['shot_key'].replace(':','_'));path=directory/'frame_record.json'
            expected=frame_fingerprint(root,plan,e,s)
            if path.exists():
                row=read(path);assert reusable_frame(row,expected), 'Changed frozen firstframe cache';rows.append(row);continue
            names=s['entity_schedule'].get('characters',[])
            role=roles[(e['episode_id'],names[0])] if len(names)==1 else None
            eligible=bool(role and role['needs_face_asset'] and read(role['metadata'])['face_eligible'])
            pending.append((e,s,directory,expected,role,eligible))
    if pending:
        core,models=load_models();settings=plan['firstframe_settings'];original_loader=core._load_models;original_record=core._model_record
        # Keep the validated model objects across shots, just as existing shared-model execution does.
        core._load_models=lambda args,device:models
        records={r['path']:r for r in implementation['weights']}
        def cached_model_record(path):
            path=str(path);record=records.get(path)
            if record is None:return original_record(Path(path))
            assert Path(path).stat().st_size==record['size']
            return dict(path=path,bytes=record['size'],sha256=record['sha256'])
        core._model_record=cached_model_record
        basics=[x for x in pending if not x[-1]]
        conditioning=prepare_text(core,models,[(e['episode_id']+'/'+s['shot_key'],s['prompt'],s['seed']) for e,s,*_ in basics],settings['width'],settings['height'])
        import torch
        try:
            for e,s,directory,expected,role,eligible in pending:
                guard(root);directory.mkdir(parents=True,exist_ok=True);begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
                names=s['entity_schedule'].get('characters',[])
                print(f"Hardcut {e['episode_id']} {s['shot_key']} ({len(rows)+1}/191) single_reference={eligible}",flush=True)
                if eligible:
                    core.run(single_arguments(core,role,s,directory,settings))
                    config=read(directory/'config.json');assert config['seed']==s['seed'] and config['prompt']==s['prompt']
                    final=directory/'treatment/final.png';applied=not config.get('plugin_skipped',False);reason=config.get('skip_reason')
                    source='flux_native_pulid_v7_hardcut' if applied else 'flux_native_single_control_reuse'
                else:
                    final=directory/'final.png'
                    image=text_image(core,models,conditioning[e['episode_id']+'/'+s['shot_key']],s['seed'],settings);image.save(final)
                    applied=False;reason='multi_character_text_only_no_identity_binding' if len(names)>1 else 'reference_face_gate' if names else 'no_character'
                    source='flux_native_basic_hardcut'
                row=dict(episode_id=e['episode_id'],shot_key=s['shot_key'],seed=s['seed'],official_prompt=s['prompt'],characters=names,
                    plugin_applied=applied,skip_reason=reason,source_kind=source,final_image=str(final.resolve()),
                    final_sha256=sha256(final),dependency_fingerprint=expected,elapsed_seconds=time.monotonic()-begin,
                    peak_cuda_memory_mib=round(torch.cuda.max_memory_allocated()/1024**2,2),
                    reference_conditioning='PuLID single reference' if eligible else 'none; text-only FLUX',
                    internal_control='shared-latent generation branch only; no Control ablation video')
                write_json(directory/'frame_record.json',row);rows.append(row)
                write_json(root/'firstframes_status.json',dict(status='running',complete_hardcuts=len(rows),total_hardcuts=191,last_shot=row))
                gc.collect();torch.cuda.empty_cache()
        finally:core._load_models=original_loader;core._model_record=original_record
    assert len(rows)==191
    write_json(root/'firstframes_status.json',dict(status='complete',complete_hardcuts=191,total_hardcuts=191,shots=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUTPUT)
    p.add_argument('--stage',choices=['references','frames'],required=True)
    p.add_argument('--max-candidates',type=int,help='Stop after this many new formal reference candidates for admission; do not score/select by benchmark results')
    args=p.parse_args();root=args.output_root.resolve();plan=read(root/'preparation_plan.json');guard(root)
    implementation=validate_implementation(root)
    os.chdir(MODEL_ROOT.parent)
    if args.stage=='references':references(root,plan,implementation,args.max_candidates)
    else:
        assert args.max_candidates is None
        frames(root,plan,implementation)
