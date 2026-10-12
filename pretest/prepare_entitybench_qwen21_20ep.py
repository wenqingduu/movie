"""Freeze Qwen-native references and hardcuts for the completed official 20-episode cohort."""
from __future__ import annotations
import copy
import json
import shutil
from collections import Counter
from pathlib import Path

from prepare_entitybench_a14b_4step import PROJECT, OUTPUT as BASELINE, OFFICIAL_SHA, PARTIAL_SHA
from prepare_entitybench_a14b_continuation import sha256, write_json

OUTPUT = PROJECT / 'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep'
MODEL = PROJECT / 'models/diffusion/qwen-image-2.1'
REVISION = '790c92633540aa0cb11d9abf19eb46d861714758'


def slug(name):
    return '_'.join(name.lower().replace('-', ' ').split())


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def immutable(path, value):
    path = Path(path)
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Frozen inputs changed: {path}'
    else:
        write_json(path, value)


def reference_prompt(name, description, needs_face, context):
    framing = ('Centered frontal head-and-shoulders character reference portrait, face large and fully visible, '
               'looking straight at camera, neutral expression, both eyes open, plain neutral background. '
               if needs_face else 'Single character reference, clearly showing the character and their clothing, plain neutral background. ')
    return (framing + f'Character {name}: {description}. '
            'Preserve the described species, age, appearance, hairstyle and clothing. '
            'Use the visual style appropriate to this story; do not turn an animal into a human. '
            'One character only, no captions, no text. Story context: ' + context[:1200])


def shot_prompt(prompt, names):
    if not names:
        return prompt
    bindings = ' '.join(f'Image {i} is the fixed appearance reference for {name}.' for i, name in enumerate(names, 1))
    return (bindings + f" Create a cinematic frame with the scheduled characters: {', '.join(names)}. "
            'Preserve each referenced character\'s identity, age, species, hairstyle and clothing; '
            'keep characters distinct, without swapping or duplicating them. Shot description: ' + prompt)


def prepare(root=OUTPUT):
    root = Path(root).resolve()
    assert root != BASELINE
    source = json.loads((BASELINE / 'manifest.json').read_text())
    status = json.loads((BASELINE / 'status.json').read_text())
    assert status['complete_episodes'] == 20 and status['generated_shots'] == 363
    episodes, roles = copy.deepcopy(source['episodes']), []
    for ep_index, ep in enumerate(episodes):
        assert sha256(ep['episode_path']) == ep['episode_sha256']
        script = json.loads(Path(ep['episode_path']).read_text())
        names = sorted({n for s in ep['shots'] if s['cut'] for n in s['entity_schedule'].get('characters', [])})
        singles = {s['entity_schedule']['characters'][0] for s in ep['shots']
                   if s['cut'] and len(s['entity_schedule'].get('characters', [])) == 1}
        assert len({slug(n) for n in names}) == len(names)
        directory = root / 'assets' / ep['episode_id']
        ep['asset_root'], ep['firstframe_root'] = str(directory), str(root / 'firstframes' / ep['episode_id'])
        ep.pop('source_report', None)
        ep.pop('source_report_sha256', None)
        for index, name in enumerate(names):
            description = next(d['description'] for d in script['entity_descriptions'].values() if d['name'] == name)
            context = str(script.get('story_overview', ''))
            if not context:
                context = next(s['prompt'] for s in ep['shots'] if name in s['entity_schedule'].get('characters', []))
            roles.append(dict(episode_id=ep['episode_id'], episode_path=ep['episode_path'], character=name,
                description=description, needs_face_asset=name in singles, asset_root=str(directory),
                seed=40_000_000+ep_index*10000+index*100,
                reference=str(directory / 'characters' / f'{slug(name)}_reference.png'),
                metadata=str(directory / 'characters' / f'{slug(name)}_reference.metadata.json'),
                result=str(directory / 'faces_3d' / slug(name) / 'facelift_result.json'),
                prompt=reference_prompt(name, description, name in singles, context)))
        for shot in ep['shots']:
            if shot['cut']:
                shot['firstframe_source'] = dict(source_kind='pending_qwen_hardcut')
    assert len(roles) == 112 and sum(r['needs_face_asset'] for r in roles) == 40
    assert sum(len(e['shots']) for e in episodes) == 363
    plan = dict(kind='qwen21_native_reference_20ep_plan', baseline=str(BASELINE),
        baseline_manifest_sha256=sha256(BASELINE/'manifest.json'), split_sha256=sha256(BASELINE/'split_20ep.json'),
        model=str(MODEL), model_revision=REVISION, episodes=episodes, roles=roles,
        counts=dict(episodes=20, shots=363, hardcuts=191, continuations=172, references=112, face_assets=40),
        reference_settings=dict(width=1024, height=1024, steps=40, face_candidates=3, other_candidates=1, true_cfg_scale=1.0),
        firstframe_settings=dict(width=1024, height=576, output_resolution=1024, steps=20, inject_start=12,
            injection_strength=0.4, identity_step_threshold=0.40, low_identity_active_steps=6,
            high_identity_active_steps=1, min_face_confidence=0.5, use_kv_cache=True,
            multi_reference=True, multi_face_injection=False, offload='sequential', evaluate_identity=False),
        video_config=source['video_config'], generation_backend=source['generation_backend'],
        model_path=source['model_path'], video_model_revision=source['model_revision'], evaluation=source['evaluation'])
    immutable(root/'preparation_plan.json', plan)
    runtime = root/'runtime';runtime.mkdir(exist_ok=True)
    shutil.copy2(BASELINE/'split_20ep.json', root/'split_20ep.json')
    for name in ['evaluate_benchmark_official.py','evaluate_benchmark_partial.py','skip_vlm.patch','evaluation_provenance.json']:
        shutil.copy2(BASELINE/'runtime'/name, runtime/name)
    assert sha256(runtime/'evaluate_benchmark_official.py') == OFFICIAL_SHA
    assert sha256(runtime/'evaluate_benchmark_partial.py') == PARTIAL_SHA
    return plan


def freeze_manifest(root=OUTPUT):
    root=Path(root);plan=json.loads((root/'preparation_plan.json').read_text())
    source=json.loads((BASELINE/'manifest.json').read_text());manifest=copy.deepcopy(source)
    manifest.update(kind='qwen21_native_reference_a14b_4step_20ep', output_root=str(root),
        route='qwen21_native_references_single_face_v7_own_tail_a14b_4step', episodes=copy.deepcopy(plan['episodes']),
        baseline_manifest_sha256=plan['baseline_manifest_sha256'], qwen_settings=plan['firstframe_settings'],
        preparation_plan_sha256=sha256(root/'preparation_plan.json'),
        source_firstframe_manifest=str(root/'preparation_plan.json'), source_firstframe_manifest_sha256=sha256(root/'preparation_plan.json'))
    assets=[]
    for role in plan['roles']:
        meta=json.loads(Path(role['metadata']).read_text());assert sha256(role['reference'])==meta['selected_sha256']
        row=dict(role,reference_sha256=meta['selected_sha256'],face_eligible=meta['face_eligible'])
        if role['needs_face_asset'] and meta['face_eligible']:
            result=json.loads(Path(role['result']).read_text());model=Path(result['model_path']);cal=model.with_suffix('.pose_calibration.json')
            assert result['facelift_status']=='success' and result['source_reference_sha256']==meta['selected_sha256']
            assert json.loads(cal.read_text())['gaussian_model_sha256']==sha256(model)
            row.update(model_sha256=sha256(model),calibration_sha256=sha256(cal))
        assets.append(row)
    inventory=[]
    for ep in manifest['episodes']:
        for shot in ep['shots']:
            if not shot['cut']:continue
            path=Path(ep['firstframe_root'])/('shot_'+shot['shot_key'].replace(':','_'))/'frame_record.json'
            row=json.loads(path.read_text());assert row['seed']==shot['seed'] and row['official_prompt']==shot['prompt']
            assert sha256(row['final_image'])==row['final_sha256']
            assert row['dependency_fingerprint']==frame_fingerprint(root,plan,ep,shot)
            record=dict(input_image=row['final_image'],input_sha256=row['final_sha256'],
                source_kind='qwen_v7_hardcut' if row['plugin_applied'] else 'qwen_reference_hardcut_no_injection',
                source_config=str(path),source_config_sha256=sha256(path),firstframe_seed=row['seed'])
            shot['firstframe_source']=record;inventory.append(dict(episode_id=ep['episode_id'],shot_key=shot['shot_key'],**record))
    assert len(inventory)==191
    manifest['totals']=dict(episode_count=20,shot_count=363,hardcut=191,continuation=172,
        firstframe_source_counts=dict(Counter(r['source_kind'] for r in inventory)))
    immutable(root/'manifest.json',manifest)
    write_json(root/'asset_manifest.json',dict(items=assets,scope='Qwen-native references; single-hardcut reliable reference faces only receive 3D'))
    write_json(root/'firstframe_inventory.json',dict(items=inventory))
    return manifest


def frame_fingerprint(root,plan,ep,shot):
    names=shot['entity_schedule'].get('characters',[])
    references=[]
    for role in plan['roles']:
        if role['episode_id']==ep['episode_id'] and role['character'] in names:
            meta=json.loads(Path(role['metadata']).read_text())
            row=dict(name=role['character'],reference_sha256=sha256(role['reference']),metadata_sha256=sha256(role['metadata']))
            if len(names)==1 and role['needs_face_asset'] and meta['face_eligible']:
                result=json.loads(Path(role['result']).read_text());model=Path(result['model_path'])
                row.update(result_sha256=sha256(role['result']),model_sha256=sha256(model),calibration_sha256=sha256(model.with_suffix('.pose_calibration.json')))
            references.append(row)
    return digest(dict(script_sha256=ep['episode_sha256'],shot=shot['shot_key'],seed=shot['seed'],prompt=shot['prompt'],
        settings=plan['firstframe_settings'],references=references,implementation=json.loads((Path(root)/'runtime/qwen_implementation.json').read_text())))
