"""Freeze a new FLUX-reference -> PuLID-FLUX -> own-tail A14B route on the existing cohort."""
from __future__ import annotations
import copy
import json
import shutil
from collections import Counter
from pathlib import Path
from prepare_entitybench_qwen21_20ep import PROJECT, BASELINE, sha256, write_json, immutable, digest, slug
from prepare_entitybench_a14b_4step import OFFICIAL_SHA, PARTIAL_SHA

QWEN = PROJECT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep'
OUTPUT = PROJECT/'outputs/entitybench_flux_portraitagent_i2v_a14b_4step_continuation_20ep'
MODEL_ROOT = PROJECT/'third_party/PuLID/models'
METHOD = 'flux_portraitagent_reference_pulid_a14b_4step_continuation_20ep_no_vlm'


def read(path):return json.loads(Path(path).read_text())

def assert_new_root(root):
    root=Path(root).resolve()
    for old in (QWEN, BASELINE, PROJECT/'outputs/entitybench_pulid_i2v_a14b_4step_full140',
                PROJECT/'outputs/entitybench_flux_native_i2v_a14b_4step_continuation_20ep'):
        assert root!=old and old not in root.parents, 'Preserve every existing run; choose an independent output root'
    return root


def portrait_pack(root, plan):
    from portrait_agent_interactive import episode_input, validate, sha
    records=[];prompts={}
    for ep in plan['episodes']:
        path=Path(root)/'portrait_agent'/(ep['episode_id']+'.json');record=read(path)
        assert record['source_plan_sha256']==sha(QWEN/'preparation_plan.json')
        output=validate(record['output'],episode_input(ep['episode_id']))
        assert record['requested_model']=='qwen3.7-plus'
        records.append(dict(episode_id=ep['episode_id'],path=str(path),sha256=sha(path),style=output['style'],
            requested_model=record['requested_model'],returned_model=record['returned_model'],usage=record['usage']))
        for item in output['characters']:
            prompts[(ep['episode_id'],item['name'])]=item['prompt']
    assert len(prompts)==112
    return records,prompts


def prepare(root=OUTPUT):
    root=assert_new_root(root)
    old=read(QWEN/'preparation_plan.json');source=read(BASELINE/'manifest.json')
    assert read(QWEN/'generation_report.json')['status']=='complete'
    assert read(QWEN/'split_20ep.json')==read(BASELINE/'split_20ep.json')
    plan=copy.deepcopy(old)
    records,prompts=portrait_pack(root,old)
    plan.update(kind='flux_native_reference_20ep_plan',model=str(MODEL_ROOT/'flux-dev-fp8.safetensors'),
        model_revision='local_FLUX.1-dev_FP8_checkpoint_SHA256_in_runtime',
        baseline=str(BASELINE),qwen_baseline=str(QWEN),qwen_plan_sha256=sha256(QWEN/'preparation_plan.json'),
        baseline_manifest_sha256=sha256(BASELINE/'manifest.json'))
    plan['kind']='flux_portraitagent_reference_20ep_plan'
    plan['portrait_agent']=dict(model='qwen3.7-plus',mode='text-only interactive prompt generation, frozen before local background inference',
        records=records,style_policy='preserve explicit animation; otherwise default live-action photorealism',
        paper_scope='adapted Portrait Agent text task; existing FaceLift framing/face selection, no SAM2/visual verification agent')
    plan['reference_settings']=dict(width=1024,height=1024,steps=40,face_candidates=3,other_candidates=1,
        guidance=4.0,true_cfg_scale=1.0,max_sequence_length=128,fp8=True,
        pulid_identity_conditioning=False,reference_selection='same Qwen20 unique-face/area/pose policy; no official-score selection')
    # User chose the existing FLUX route, preserving its 50-step/step30 protocol and resolution.
    plan['firstframe_settings']=dict(width=640,height=352,steps=50,inject_start=30,injection_strength=0.4,
        max_active_injection_steps=12,pulid_id_weight=1.0,guidance=4.0,min_face_confidence=0.5,
        min_injection_face_height=24,max_face_detection_retries=4,max_sequence_length=128,
        fp8=True,harmonize_reference=True,reference_conditioning='target',multi_face_injection=False,
        multi_reference=False,evaluate_identity=False,
        multi_character_policy='text_only_FLUX_without_identity_binding_or_3D',
        no_reliable_reference_policy='text_only_FLUX_without_identity_binding_or_3D')
    for ep in plan['episodes']:
        eid=ep['episode_id'];assert sha256(ep['episode_path'])==ep['episode_sha256']
        ep['asset_root']=str(root/'assets'/eid);ep['firstframe_root']=str(root/'firstframes'/eid)
        ep.pop('reuse_generation_root',None)
        for s in ep['shots']:
            if s['cut']:s['firstframe_source']=dict(source_kind='pending_flux_native_hardcut')
    for role in plan['roles']:
        role['prompt']=prompts[(role['episode_id'],role['character'])]
        directory=root/'assets'/role['episode_id'];name=slug(role['character'])
        role.update(asset_root=str(directory),reference=str(directory/'characters'/f'{name}_reference.png'),
            metadata=str(directory/'characters'/f'{name}_reference.metadata.json'),
            result=str(directory/'faces_3d'/name/'facelift_result.json'))
    assert plan['counts']==dict(episodes=20,shots=363,hardcuts=191,continuations=172,references=112,face_assets=40)
    assert sum(r['needs_face_asset'] for r in plan['roles'])==40
    assert sum(len(e['shots']) for e in plan['episodes'])==363
    assert plan['video_config']==source['video_config'] and plan['generation_backend']==source['generation_backend']
    immutable(root/'preparation_plan.json',plan)
    runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    for name in ['evaluate_benchmark_official.py','evaluate_benchmark_partial.py','skip_vlm.patch','evaluation_provenance.json']:
        src=BASELINE/'runtime'/name;dst=runtime/name
        if dst.exists():assert sha256(dst)==sha256(src)
        else:shutil.copy2(src,dst)
    assert sha256(runtime/'evaluate_benchmark_official.py')==OFFICIAL_SHA
    assert sha256(runtime/'evaluate_benchmark_partial.py')==PARTIAL_SHA
    immutable(root/'split_20ep.json',read(QWEN/'split_20ep.json'))
    return plan


def freeze_implementation(root):
    root=Path(root)
    names=['flux-dev-fp8.safetensors','flux_dev_quantization_map.json','ae.safetensors',
        'xflux_text_encoders/model-00001-of-00002.safetensors','xflux_text_encoders/model-00002-of-00002.safetensors',
        'clip-vit-large-patch14/model.safetensors','pulid_flux_v0.9.1.safetensors','eva_clip/EVA02_CLIP_L_336_psz14_s6B.pt']
    names += [str(p.relative_to(MODEL_ROOT)) for p in sorted((MODEL_ROOT/'antelopev2').glob('*.onnx'))]
    weights=[dict(path=str(MODEL_ROOT/n),size=(MODEL_ROOT/n).stat().st_size,sha256=sha256(MODEL_ROOT/n)) for n in names]
    files=[PROJECT/n for n in ['pretest/prepare_entitybench_flux_portraitagent_20ep.py','pretest/run_flux_portraitagent_20ep_images.py',
        'pretest/run_entitybench_flux_portraitagent_20ep.py','pretest/archive_entitybench_portraitflux_preliminary_results.py',
        'multishot/pulid_flux_inner_face_experiment.py','multishot/prompt_injection_safety.py',
        'pretest/run_qwen21_entitybench_20ep_images.py','pretest/prepare_entitybench_qwen21_20ep.py',
        'pretest/prepare_entitybench_character_assets.py','multishot/facelift_backend.py','multishot/facelift_pose_calibration.py',
        'pretest/run_entitybench_a14b_full140.py','pretest/run_lightx2v_a14b_4step_continuation.py']]
    files += sorted((PROJECT/'third_party/PuLID/flux').rglob('*.py'))
    files += [PROJECT/'third_party/PuLID/pulid/pipeline_flux.py']
    files += [PROJECT/'pretest/portrait_agent_interactive.py']
    value=dict(model='FLUX.1-dev local FP8',weights=weights,sources={str(p):sha256(p) for p in files},
        policy='Reference images: unconditioned FLUX; single hardcuts: existing PuLID-FLUX v7 step30; multi/no-reference-face: unconditioned FLUX; no SDXL references or foreign Gaussian assets')
    immutable(root/'runtime/flux_implementation.json',value)
    return value


def validate_implementation(root):
    implementation=read(Path(root)/'runtime/flux_implementation.json')
    for name,h in implementation['sources'].items():assert sha256(name)==h, f'Changed frozen source: {name}'
    for row in implementation['weights']:
        assert Path(row['path']).stat().st_size==row['size'] and sha256(row['path'])==row['sha256'], row['path']
    return implementation


def frame_fingerprint(root,plan,episode,shot):
    names=shot['entity_schedule'].get('characters',[]);references=[]
    # Only single-character reference binding affects this chosen FLUX route.
    for r in plan['roles']:
        if r['episode_id']!=episode['episode_id'] or r['character'] not in names:continue
        meta=read(r['metadata']);h=sha256(r['reference']);assert h==meta['selected_sha256']
        item=dict(name=r['character'],reference_sha256=h,metadata_sha256=sha256(r['metadata']))
        if len(names)==1 and r['needs_face_asset'] and meta['face_eligible']:
            result=read(r['result']);model=Path(result['model_path']);cal=model.with_suffix('.pose_calibration.json')
            assert result['source_reference_sha256']==h and result['facelift_status']=='success'
            assert read(cal)['gaussian_model_sha256']==sha256(model)
            item.update(result_sha256=sha256(r['result']),model_sha256=sha256(model),calibration_sha256=sha256(cal))
        references.append(item)
    return digest(dict(script_sha256=episode['episode_sha256'],shot=shot['shot_key'],seed=shot['seed'],
        prompt=shot['prompt'],settings=plan['firstframe_settings'],references=references,
        implementation=sha256(Path(root)/'runtime/flux_implementation.json')))


def reusable_frame(record,expected):
    return record.get('dependency_fingerprint')==expected and Path(record.get('final_image','')).is_file() and sha256(record['final_image'])==record.get('final_sha256')


def freeze_manifest(root):
    root=Path(root);plan=read(root/'preparation_plan.json');manifest=copy.deepcopy(read(BASELINE/'manifest.json'))
    manifest.update(kind='flux_native_reference_pulid_a14b_4step_20ep',output_root=str(root),
        route='flux_native_reference_single_pulid_v7_multi_text_only_own_tail_a14b_4step',
        episodes=copy.deepcopy(plan['episodes']),flux_settings=plan['firstframe_settings'],
        preparation_plan_sha256=sha256(root/'preparation_plan.json'),
        source_firstframe_manifest=str(root/'preparation_plan.json'),source_firstframe_manifest_sha256=sha256(root/'preparation_plan.json'),
        qwen_baseline_plan_sha256=plan['qwen_plan_sha256'])
    manifest['portrait_agent']=plan['portrait_agent']
    assets=[]
    for role in plan['roles']:
        meta=read(role['metadata']);assert sha256(role['reference'])==meta['selected_sha256']
        item=dict(role,reference_sha256=meta['selected_sha256'],face_eligible=meta['face_eligible'])
        if role['needs_face_asset'] and meta['face_eligible']:
            result=read(role['result']);model=Path(result['model_path']);cal=model.with_suffix('.pose_calibration.json')
            assert result['source_reference_sha256']==meta['selected_sha256'] and result['facelift_status']=='success'
            assert read(cal)['gaussian_model_sha256']==sha256(model)
            item.update(model_sha256=sha256(model),calibration_sha256=sha256(cal))
        assets.append(item)
    inventory=[]
    for e in manifest['episodes']:
        for shot in e['shots']:
            if not shot['cut']:continue
            path=Path(e['firstframe_root'])/('shot_'+shot['shot_key'].replace(':','_'))/'frame_record.json'
            row=read(path);assert row['seed']==shot['seed'] and row['official_prompt']==shot['prompt']
            assert reusable_frame(row,frame_fingerprint(root,plan,e,shot))
            source=dict(input_image=row['final_image'],input_sha256=row['final_sha256'],source_kind=row['source_kind'],
                source_config=str(path),source_config_sha256=sha256(path),firstframe_seed=row['seed'],skip_reason=row['skip_reason'])
            shot['firstframe_source']=source;inventory.append(dict(episode_id=e['episode_id'],shot_key=shot['shot_key'],**source))
    assert len(inventory)==191
    manifest['totals']=dict(episode_count=20,shot_count=363,hardcut=191,continuation=172,
        firstframe_source_counts=dict(Counter(x['source_kind'] for x in inventory)))
    immutable(root/'manifest.json',manifest)
    immutable(root/'firstframe_inventory.json',dict(items=inventory));immutable(root/'asset_manifest.json',dict(items=assets))
    return manifest
