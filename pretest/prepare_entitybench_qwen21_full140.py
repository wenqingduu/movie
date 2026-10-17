"""Independent Qwen full-cohort plan; original twenty episodes remain read-only."""
from __future__ import annotations
import copy
import json
import re
from collections import Counter
from pathlib import Path

from prepare_entitybench_qwen21_20ep import (PROJECT, OUTPUT as SOURCE, MODEL, REVISION,
    slug, reference_prompt, immutable, sha256, write_json, frame_fingerprint, digest)
from prepare_entitybench_a14b_continuation import ordered_shots
from prepare_entitybench_a14b_4step import OFFICIAL_SHA, PARTIAL_SHA

OUTPUT = PROJECT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140'
SPLIT = PROJECT/'benchmarks/entitybench/data/splits/final_split_validated_ids.json'

def read(path):
    return json.loads(Path(path).read_text())

def prepare(root=OUTPUT):
    root=Path(root).resolve()
    assert root not in {SOURCE, SOURCE.parent/'entitybench_pulid_i2v_a14b_4step_full140'}
    original=read(SOURCE/'manifest.json'); old=read(SOURCE/'preparation_plan.json')
    prior={e['episode_id']:e for e in original['episodes']}
    split=read(SPLIT); ids=[eid for tier in ['easy','medium','hard'] for eid in split[tier]]
    assert len(ids)==len(set(ids))==140 and set(prior).issubset(ids)
    episodes=[]; roles=[]; counts=Counter()
    for ordinal,eid in enumerate(ids):
        tier=next(t for t in ['easy','medium','hard'] if eid in split[t])
        path=PROJECT/'benchmarks/entitybench/data/scripts'/f'{eid}.json'
        script=read(path); shots=ordered_shots(script)
        if eid in prior:
            ep=copy.deepcopy(prior[eid]); assert sha256(path)==ep['episode_sha256']
            assert ep['difficulty']==tier
            for a,b in zip(ep['shots'],shots):
                assert all(a[k]==b[k] for k in ['shot_key','shot_index','prompt','cut','entity_schedule','action_description'])
            assert len(ep['shots'])==len(shots)
            ep['reuse_generation_root']=str(SOURCE);counts['reused_episodes']+=1
        else:
            directory=root/'assets'/eid; base_seed=30_000_000+ordinal*1000
            ep=dict(episode_id=eid,run='run'+re.search(r'__run(\d+)__',eid).group(1),difficulty=tier,
                episode_path=str(path),episode_sha256=sha256(path),entity_descriptions=script['entity_descriptions'],
                asset_root=str(directory),firstframe_root=str(root/'firstframes'/eid),base_seed=base_seed,shots=shots)
            names=sorted({n for s in shots if s['cut'] for n in s['entity_schedule'].get('characters',[])})
            singles={s['entity_schedule']['characters'][0] for s in shots if s['cut'] and len(s['entity_schedule'].get('characters',[]))==1}
            assert len({slug(n) for n in names})==len(names)
            for index,name in enumerate(names):
                description=next(d['description'] for d in script['entity_descriptions'].values() if d['name']==name)
                context=str(script.get('story_overview','')) or next(s['prompt'] for s in shots if name in s['entity_schedule'].get('characters',[]))
                roles.append(dict(episode_id=eid,episode_path=str(path),character=name,description=description,
                    needs_face_asset=name in singles,asset_root=str(directory),seed=40_000_000+ordinal*10000+index*100,
                    reference=str(directory/'characters'/f'{slug(name)}_reference.png'),
                    metadata=str(directory/'characters'/f'{slug(name)}_reference.metadata.json'),
                    result=str(directory/'faces_3d'/slug(name)/'facelift_result.json'),
                    prompt=reference_prompt(name,description,name in singles,context)))
            previous=None
            for s in shots:
                assert len(s['entity_schedule'].get('characters',[]))<=10, (eid,s['shot_key'])
                s.update(episode_id=eid,seed=base_seed+s['shot_index'],previous_shot_key=previous)
                previous=s['shot_key'];counts['new_shots']+=1
                counts['new_hardcuts' if s['cut'] else 'new_continuations']+=1
                if s['cut']:s['firstframe_source']=dict(source_kind='pending_qwen_hardcut')
            counts['new_episodes']+=1
        for s in ep['shots']:
            counts['shots']+=1;counts['hardcuts' if s['cut'] else 'continuations']+=1
        episodes.append(ep)
    counts.update(episodes=len(episodes),references=len(roles),face_assets=sum(r['needs_face_asset'] for r in roles),
        candidate_images=sum(3 if r['needs_face_asset'] else 1 for r in roles))
    assert dict(counts)==dict(reused_episodes=20,new_episodes=120,new_shots=2128,new_hardcuts=945,new_continuations=1183,
        episodes=140,shots=2491,hardcuts=1136,continuations=1355,references=711,face_assets=226,candidate_images=1163),dict(counts)
    plan=dict(kind='qwen21_native_reference_full140_plan',source_root=str(SOURCE),
        source_manifest_sha256=sha256(SOURCE/'manifest.json'),source_plan_sha256=sha256(SOURCE/'preparation_plan.json'),
        source_generation_manifest_sha256=sha256(SOURCE/'generation_manifest.json'),full_split_sha256=sha256(SPLIT),
        model=str(MODEL),model_revision=REVISION,episodes=episodes,roles=roles,counts=dict(counts),
        reference_settings=old['reference_settings'],firstframe_settings=old['firstframe_settings'],
        video_config=original['video_config'],generation_backend=original['generation_backend'],
        model_path=original['model_path'],video_model_revision=original['model_revision'],evaluation=original['evaluation'],
        seed_policy='old20 unchanged; new video 30M+full-split-ordinal*1000+shot_index; new reference 40M+ordinal*10000+role_index*100')
    immutable(root/'preparation_plan.json',plan)
    immutable(root/'split_20ep.json',split) # evaluator contract filename; contains all 140 IDs
    immutable(root/'split_full140.json',split)
    for name in ['evaluate_benchmark_official.py','evaluate_benchmark_partial.py','skip_vlm.patch','evaluation_provenance.json','qwen_implementation.json']:
        target=root/'runtime'/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha256(target)==sha256(SOURCE/'runtime'/name)
        else:target.write_bytes((SOURCE/'runtime'/name).read_bytes())
    assert sha256(root/'runtime/evaluate_benchmark_official.py')==OFFICIAL_SHA
    assert sha256(root/'runtime/evaluate_benchmark_partial.py')==PARTIAL_SHA
    return plan

def freeze_manifest(root=OUTPUT):
    root=Path(root);plan=read(root/'preparation_plan.json'); original=read(SOURCE/'manifest.json')
    manifest=copy.deepcopy(original)
    manifest.update(kind='qwen21_native_reference_a14b_4step_full140',output_root=str(root),episodes=copy.deepcopy(plan['episodes']),
        cohort_manifest=str(SPLIT),cohort_sha256=sha256(SPLIT),preparation_plan_sha256=sha256(root/'preparation_plan.json'),
        source_firstframe_manifest=str(root/'preparation_plan.json'),source_firstframe_manifest_sha256=sha256(root/'preparation_plan.json'))
    assets=[]; inventory=[]
    for role in plan['roles']:
        meta=read(role['metadata']);assert sha256(role['reference'])==meta['selected_sha256']
        impl=read(root/'runtime/qwen_implementation.json')
        assert meta['dependency_fingerprint']==digest(dict(role=role,settings=plan['reference_settings'],model_revision=REVISION,implementation=impl))
        row=dict(role,reference_sha256=meta['selected_sha256'],face_eligible=meta['face_eligible'])
        if role['needs_face_asset'] and meta['face_eligible']:
            result=read(role['result']);model=Path(result['model_path']);cal=model.with_suffix('.pose_calibration.json')
            assert result['facelift_status']=='success' and result['source_reference_sha256']==meta['selected_sha256']
            assert read(cal)['gaussian_model_sha256']==sha256(model) and read(cal)['profiles']
            row.update(model_sha256=sha256(model),calibration_sha256=sha256(cal))
        assets.append(row)
    for ep in manifest['episodes']:
        for shot in ep['shots']:
            if not shot['cut']:continue
            if ep.get('reuse_generation_root'):
                record=shot['firstframe_source']
                assert sha256(record['input_image'])==record['input_sha256']
                assert sha256(record['source_config'])==record['source_config_sha256']
            else:
                path=Path(ep['firstframe_root'])/('shot_'+shot['shot_key'].replace(':','_'))/'frame_record.json'
                row=read(path);assert row['seed']==shot['seed'] and row['official_prompt']==shot['prompt']
                assert sha256(row['final_image'])==row['final_sha256']
                assert row['dependency_fingerprint']==frame_fingerprint(root,plan,ep,shot)
                record=dict(input_image=row['final_image'],input_sha256=row['final_sha256'],source_kind='qwen_v7_hardcut' if row['plugin_applied'] else 'qwen_reference_hardcut_no_injection',
                    source_config=str(path),source_config_sha256=sha256(path),firstframe_seed=row['seed'])
                shot['firstframe_source']=record
            inventory.append(dict(episode_id=ep['episode_id'],shot_key=shot['shot_key'],**record))
    assert len(inventory)==1136
    manifest['totals']=dict(episode_count=140,shot_count=2491,hardcut=1136,continuation=1355,
        firstframe_source_counts=dict(Counter(r['source_kind'] for r in inventory)))
    immutable(root/'manifest.json',manifest)
    write_json(root/'asset_manifest.json',dict(items=assets,reused_inventory=str(SOURCE/'asset_manifest.json'),reused_inventory_sha256=sha256(SOURCE/'asset_manifest.json')))
    write_json(root/'firstframe_inventory.json',dict(items=inventory))
    return manifest
