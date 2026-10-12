"""Freeze full-cohort expansion inputs and reuse only audited completed episodes."""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from prepare_entitybench_a14b_4step import OUTPUT as SOURCE, PROJECT, OFFICIAL_SHA, PARTIAL_SHA
from prepare_entitybench_a14b_continuation import ordered_shots, sha256, write_json, single_source

OUTPUT = PROJECT / 'outputs/entitybench_pulid_i2v_a14b_4step_full140'
SPLIT = PROJECT / 'benchmarks/entitybench/data/splits/final_split_validated_ids.json'

# Reviewed against original entity descriptions, before reading output scores.
# Keys are episode-local: same-named people in other stories are unaffected.
NON_FACE_ROLES = {
    ('00045fec-573d-3dec-8d53-e2c18aa4e9fa__run582__i0_j47__T600', 'Sunny'): 'A stylized sun',
    ('00045fec-573d-3dec-8d53-e2c18aa4e9fa__run582__i0_j47__T600', 'Tingo'): 'full-body purple suit',
    ('00119050-8401-38c2-b85b-b2349a4b503a__run2183__i75_j93__T240', 'Goro'): 'monstrous creature',
    ('0000747b-5b4b-3c58-8983-eedc33a29970__run55__i60_j83__T300', 'Blaze'): 'Dalmatian pup',
    ('0000747b-5b4b-3c58-8983-eedc33a29970__run55__i60_j83__T300', 'Gizmo'): 'mixed-breed dog',
    ('00009a6a-988e-34a0-8dd1-ea557fc5338e__run69__i25_j44__T240', 'Crag'): 'Rocky Monster costume',
    ('00009a6a-988e-34a0-8dd1-ea557fc5338e__run69__i25_j44__T240', 'Dizzy'): 'doll or action figure',
    ('00009a6a-988e-34a0-8dd1-ea557fc5338e__run69__i25_j44__T240', 'Poppy'): 'Female doll',
    ('000064ca-3557-3599-a263-7cc848de075b__run44__i15_j47__T420', 'The Shadow'): 'featureless figure',
}


def freeze_face_routing(root, plan):
    """Keep the original expansion/seed plan; freeze an explicit face eligibility audit."""
    root = Path(root)
    episodes = {e['episode_id']: e for e in plan['episodes']}
    roles, found = [], set()
    for role in plan['roles']:
        key = (role['episode_id'], role['character'])
        episode = episodes[key[0]]
        description = next(d['description'] for d in episode['entity_descriptions'].values() if d['name'] == key[1])
        evidence = NON_FACE_ROLES.get(key)
        if evidence:
            assert evidence in description, f'Non-face description changed: {key}'
            found.add(key)
        roles.append(dict(episode_id=key[0], character=key[1], description=description,
            episode_sha256=episode['episode_sha256'], eligible=evidence is None,
            route='basic_no_face_injection' if evidence else 'single_character_face_injection',
            reason='non_human_or_non_human_face_entity' if evidence else 'existing_face_gates_apply',
            description_evidence=evidence,
            hardcut_shots=[s['shot_key'] for s in episode['shots'] if s['cut'] and s['entity_schedule'].get('characters') == [key[1]]]))
    assert found == set(NON_FACE_ROLES), 'Incomplete reviewed non-face role inventory'
    excluded = [r for r in roles if not r['eligible']]
    routing = dict(kind='full140_face_eligibility_v1', expansion_plan_sha256=sha256(root / 'expansion_plan.json'),
        policy='Original scripts/seeds unchanged; non-face entities use zero-identity basic hardcuts; every shot remains in evaluation; old20 episodes read-only',
        roles=roles, counts=dict(original_roles=len(roles), injection_roles=len(roles)-len(excluded),
            basic_only_roles=len(excluded), basic_only_single_hardcuts=sum(len(r['hardcut_shots']) for r in excluded)))
    path = root / 'face_injection_routing.json'
    if path.exists():
        assert json.loads(path.read_text()) == routing, 'Frozen face eligibility changed'
    else:
        assert not (root / 'manifest.json').exists(), 'Cannot reroute frozen generation inputs'
        write_json(path, routing)
    return routing


def injectable_hardcut(episode, shot, routing):
    names = shot['entity_schedule'].get('characters', [])
    if not shot['cut'] or len(names) != 1:
        return False
    role = next(r for r in routing['roles'] if r['episode_id'] == episode['episode_id'] and r['character'] == names[0])
    return role['eligible']


def slug(name):
    return '_'.join(name.lower().replace('-', ' ').split())


def prepare(root=OUTPUT):
    root = Path(root).resolve()
    source = json.loads((SOURCE / 'manifest.json').read_text())
    generation = json.loads((SOURCE / 'generation_report.json').read_text())
    assert generation['status'] == 'complete' and len(generation['jobs']) == 363
    prior = {e['episode_id']: e for e in source['episodes']}
    split = json.loads(SPLIT.read_text())
    episodes, roles, stats = [], [], Counter()
    for tier in ['easy', 'medium', 'hard']:
        for index, eid in enumerate(split[tier]):
            path = PROJECT / 'benchmarks/entitybench/data/scripts' / f'{eid}.json'
            script = json.loads(path.read_text())
            shots = ordered_shots(script)
            if eid in prior:
                ep = copy.deepcopy(prior[eid])
                assert sha256(path) == ep['episode_sha256']
                ep['reuse_generation_root'] = str(SOURCE)
                stats['reused_episodes'] += 1
            else:
                ordinal = len(episodes)
                base_seed = 30_000_000 + ordinal * 1000
                run = re.search(r'__run(\d+)__', eid).group(1)
                asset_root = root / 'assets' / eid
                ep = dict(episode_id=eid, episode_path=str(path), episode_sha256=sha256(path),
                    run=f'run{run}', difficulty=tier, base_seed=base_seed, asset_root=str(asset_root),
                    firstframe_root=str(root / 'firstframes' / eid), shots=shots,
                    entity_descriptions=script['entity_descriptions'])
                names = sorted({s['entity_schedule']['characters'][0] for s in shots
                    if s['cut'] and len(s['entity_schedule'].get('characters', [])) == 1})
                assert len({slug(n) for n in names}) == len(names), f'Character slug collision: {eid}'
                for role_index, name in enumerate(names):
                    roles.append(dict(episode_id=eid, episode_path=str(path), character=name,
                        asset_root=str(asset_root), seed=base_seed + role_index * 100,
                        reference=str(asset_root / 'characters' / f'{slug(name)}_reference.png'),
                        result=str(asset_root / 'faces_3d' / slug(name) / 'facelift_result.json')))
                previous = None
                for shot in shots:
                    shot.update(episode_id=eid, seed=base_seed + shot['shot_index'], previous_shot_key=previous)
                    previous = shot['shot_key']
                    stats['new_shots'] += 1
                    if shot['cut']:
                        count = len(shot['entity_schedule'].get('characters', []))
                        shot['firstframe_source'] = dict(source_kind='pending_single_hardcut' if count == 1 else 'pending_basic_hardcut')
                        stats['new_single_hardcut' if count == 1 else 'new_basic_hardcut'] += 1
            for shot in ep['shots']:
                stats['shots'] += 1
                stats['hardcut' if shot['cut'] else 'continuation'] += 1
            episodes.append(ep)
    assert len(episodes) == 140 and stats['shots'] == 2491
    assert stats['reused_episodes'] == 20 and stats['new_shots'] == 2128 and len(roles) == 226
    plan = dict(kind='entitybench_full140_expansion_plan', source_root=str(SOURCE), source_manifest_sha256=sha256(SOURCE / 'manifest.json'),
        source_generation_manifest_sha256=sha256(SOURCE / 'generation_manifest.json'), full_split_sha256=sha256(SPLIT),
        settings=source['video_config'], episodes=episodes, roles=roles, counts=dict(stats))
    if (root / 'expansion_plan.json').exists():
        assert json.loads((root / 'expansion_plan.json').read_text()) == plan, 'Expansion inputs changed; use a new root'
    write_json(root / 'expansion_plan.json', plan)
    write_json(root / 'split_20ep.json', split)  # Stable evaluator/runner contract; contains all 140 IDs.
    runtime = root / 'runtime'
    runtime.mkdir(exist_ok=True)
    for name in ['evaluate_benchmark_official.py', 'evaluate_benchmark_partial.py', 'skip_vlm.patch', 'evaluation_provenance.json']:
        shutil.copy2(SOURCE / 'runtime' / name, runtime / name)
    assert sha256(runtime / 'evaluate_benchmark_official.py') == OFFICIAL_SHA
    assert sha256(runtime / 'evaluate_benchmark_partial.py') == PARTIAL_SHA
    print(json.dumps(dict(plan['counts'], roles_to_prepare=len(roles)), indent=2), flush=True)
    return plan


def freeze_ready_manifest(root):
    root = Path(root)
    plan = json.loads((root / 'expansion_plan.json').read_text())
    routing = freeze_face_routing(root, plan)
    eligible = {(r['episode_id'], r['character']) for r in routing['roles'] if r['eligible']}
    original = json.loads((SOURCE / 'manifest.json').read_text())
    manifest = copy.deepcopy(original)
    manifest.update(kind='entitybench_pulid_i2v_a14b_4step_full140', output_root=str(root), episodes=copy.deepcopy(plan['episodes']))
    inventory, assets = [], []
    for role in plan['roles']:
        if (role['episode_id'], role['character']) not in eligible:
            continue
        result = json.loads(Path(role['result']).read_text())
        model = Path(result['model_path'])
        assert result['facelift_status'] == 'success' and model.is_file()
        assert result['source_reference_sha256'] == sha256(role['reference'])
        calibration = model.with_suffix('.pose_calibration.json')
        data = json.loads(calibration.read_text())
        assert data['gaussian_model_sha256'] == sha256(model) and data['profiles']
        assets.append(dict(role, reference_sha256=sha256(role['reference']), model_sha256=sha256(model),
            calibration_sha256=sha256(calibration), status='available'))
    for episode in manifest['episodes']:
        if episode.get('reuse_generation_root'):
            for shot in episode['shots']:
                if shot['cut']:
                    source = shot['firstframe_source']
                    assert sha256(source['input_image']) == source['input_sha256']
                    assert sha256(source['source_config']) == source['source_config_sha256']
                    inventory.append(dict(episode_id=episode['episode_id'], shot_key=shot['shot_key'], **source))
            continue
        directory = Path(episode['firstframe_root'])
        singles_path = directory / 'pulid_first_frame_report.json'
        singles = {r['shot_key']: r for r in json.loads(singles_path.read_text())['shots']} if singles_path.exists() else {}
        basics_path = directory / 'basic/flux_control_first_frame_report.json'
        basics = {r['shot_key']: r for r in json.loads(basics_path.read_text())['shots']} if basics_path.exists() else {}
        for shot in episode['shots']:
            if not shot['cut']:
                continue
            if injectable_hardcut(episode, shot, routing):
                row = singles[shot['shot_key']]
                assert row['status'] in {'generated', 'skipped_existing'}
                assert row['seed'] == shot['seed']
                source = single_source(row, shot)
            else:
                row = basics[shot['shot_key']]
                path = Path(row['control_image'])
                config = path.parent.parent / 'config.json'
                assert row['seed'] == shot['seed'] and row['prompt'] == shot['prompt']
                assert row['pulid_id_weight'] == 0 and not row['plugin_applied']
                assert sha256(path) == row['control_sha256']
                source = dict(input_image=str(path), input_sha256=sha256(path), source_kind='multi_character_no_injection' if row['characters'] else 'no_character_no_injection',
                    source_config=str(config), source_config_sha256=sha256(config), firstframe_seed=row['seed'])
                if len(row['characters']) == 1:
                    source.update(source_kind='non_face_character_no_injection', injection_skip_reason='non_human_or_non_human_face_entity')
            shot['firstframe_source'] = source
            inventory.append(dict(episode_id=episode['episode_id'], shot_key=shot['shot_key'], **source))
    assert len(inventory) == 1136
    manifest['totals'] = dict(episode_count=140, shot_count=2491, hardcut=1136, continuation=1355,
        firstframe_source_counts=dict(Counter(r['source_kind'] for r in inventory)))
    manifest['expansion_plan_sha256'] = sha256(root / 'expansion_plan.json')
    manifest['face_routing_sha256'] = sha256(root / 'face_injection_routing.json')
    manifest['face_routing_counts'] = routing['counts']
    if (root / 'manifest.json').exists():
        assert json.loads((root / 'manifest.json').read_text()) == manifest, 'Frozen generation manifest changed'
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'firstframe_inventory.json', dict(items=inventory))
    write_json(root / 'asset_manifest.json', dict(items=assets,
        basic_only_roles=[r for r in routing['roles'] if not r['eligible']],
        scope='New roles requiring single-character hard-cut injection; other new roles have no 3D injection',
        reused_20episode_inventory=str(SOURCE / 'asset_manifest.json'),
        reused_20episode_inventory_sha256=sha256(SOURCE / 'asset_manifest.json')))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output_root)
