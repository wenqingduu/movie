"""Audited semantic face gates and budget accounting for the interrupted Qwen run."""
from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

from prepare_entitybench_a14b_full140 import NON_FACE_ROLES
from prepare_entitybench_qwen21_full140 import PROJECT, read
from prepare_entitybench_qwen21_20ep import immutable, sha256, write_json

REASON = 'non_human_or_non_human_face_entity'
CONTRACT = 'runtime/face_routing_recovery_20261016.json'
BACKUP = 'runtime/face_routing_before_20261016'


def gated_metadata(original, role, evidence, script_sha):
    """Preserve selection, candidates and dependency identity; correct the semantic gate."""
    assert evidence in role['description'], (role['episode_id'], role['character'])
    result = copy.deepcopy(original)
    result.update(face_eligible=False, face_skip_reason=REASON,
                  face_eligibility_override=dict(reason=REASON, description_evidence=evidence,
                    episode_sha256=script_sha, original_face_eligible=original['face_eligible'],
                    route='qwen_native_reference_without_face_injection'))
    return result


def verify_contract(root):
    root = Path(root)
    contract = read(root / CONTRACT)
    assert sha256(root / 'preparation_plan.json') == contract['plan_sha256']
    for path, expected in contract['sources'].items():
        assert sha256(path) == expected, f'Changed routing recovery source: {path}'
    return contract


def prepare_recovery(root):
    """Freeze before editing; interrupted metadata application is safely repeatable."""
    root = Path(root).resolve()
    plan = read(root / 'preparation_plan.json')
    backup = root / BACKUP
    path = root / CONTRACT
    if not path.exists():
        assert not (root / 'manifest.json').exists(), 'Cannot reroute frozen generation'
        assert not any((root / 'firstframes').glob('**/frame_record.json')), 'Hardcuts already generated'
        episodes = {e['episode_id']: e for e in plan['episodes']}
        rows = []
        for role in plan['roles']:
            evidence = NON_FACE_ROLES.get((role['episode_id'], role['character']))
            if not evidence:
                continue
            episode = episodes[role['episode_id']]
            assert not episode.get('reuse_generation_root'), 'Old20 must remain untouched'
            assert sha256(role['episode_path']) == episode['episode_sha256']
            metadata = Path(role['metadata'])
            original = read(metadata)
            assert sha256(role['reference']) == original['selected_sha256']
            saved = backup / 'metadata' / role['episode_id'] / metadata.name
            saved.parent.mkdir(parents=True, exist_ok=True)
            if saved.exists():
                assert saved.read_bytes() == metadata.read_bytes()
            else:
                shutil.copy2(metadata, saved)
            expected = gated_metadata(original, role, evidence, episode['episode_sha256'])
            rows.append(dict(episode_id=role['episode_id'], character=role['character'],
                metadata=str(metadata), original_metadata=str(saved), original_sha256=sha256(saved),
                expected=expected, reference=role['reference'], reference_sha256=original['selected_sha256'],
                single_hardcuts=[s['shot_key'] for s in episode['shots'] if s['cut'] and
                    s['entity_schedule'].get('characters') == [role['character']]]))
        assert {(r['episode_id'], r['character']) for r in rows} == set(NON_FACE_ROLES)
        changed = {r['metadata'] for r in rows}
        # Snapshot all existing generated assets. Only the nine gate metadata files may change.
        protected = {str(p): sha256(p) for p in sorted((root / 'assets').rglob('*'))
                     if p.is_file() and str(p) not in changed}
        immutable(backup / 'protected_files.json', dict(files=protected))
        for name in ['status.json', 'asset_preparation_report.json']:
            source = root / name
            target = backup / name
            if not target.exists():
                shutil.copy2(source, target)
        sources = [Path(__file__), PROJECT / 'pretest/run_entitybench_qwen21_full140_resume_routing.py',
                   PROJECT / 'tests/test_qwen140_routing_recovery.py',
                   PROJECT / 'pretest/prepare_entitybench_a14b_full140.py']
        immutable(path, dict(kind='qwen140_semantic_face_gate_recovery_v1',
            plan_sha256=sha256(root / 'preparation_plan.json'), rows=rows,
            sources={str(p.resolve()): sha256(p) for p in sources},
            protected_files_manifest=str(backup / 'protected_files.json'),
            protected_files_manifest_sha256=sha256(backup / 'protected_files.json'),
            policy='Qwen references/candidates/prompts/seeds and original frozen code preserved; '
                   'nine reviewed episode-local non-face gates corrected; every shot retained'))
    contract = verify_contract(root)
    protected_path = Path(contract['protected_files_manifest'])
    assert sha256(protected_path) == contract['protected_files_manifest_sha256']
    for file, expected in read(protected_path)['files'].items():
        assert sha256(file) == expected, f'Existing asset changed: {file}'
    for row in contract['rows']:
        assert sha256(row['original_metadata']) == row['original_sha256']
        current = read(row['metadata'])
        assert current in [read(row['original_metadata']), row['expected']], row['metadata']
        assert sha256(row['reference']) == row['reference_sha256']
        if current != row['expected']:
            write_json(row['metadata'], row['expected'])
    print(f"Semantic face routing verified: {len(contract['rows'])} roles, "
          f"{sum(len(r['single_hardcuts']) for r in contract['rows'])} hardcuts; "
          f"{len(read(protected_path)['files'])} pre-existing files unchanged", flush=True)
    return contract


def allocated_asset_bytes(root):
    """Credit existing products once; symlinks and hardlink duplicates cannot inflate it."""
    total, seen = 0, set()
    for directory, dirs, files in os.walk(Path(root) / 'assets', followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                total += stat.st_blocks * 512
                seen.add(key)
    return total


def resume_budget(admission, approval, existing_bytes):
    """Apply the approved total product budget once, including products already on disk."""
    assert approval['startup_min_free_gib'] == 50
    assert approval['product_budget_gib'] == 40 and approval['reserve_gib'] == 10
    assert admission['startup_min_free_gib'] == approval['startup_min_free_gib']
    result = copy.deepcopy(admission)
    credit = min(existing_bytes / 1024**3, approval['product_budget_gib'])
    required = approval['product_budget_gib'] - credit + approval['reserve_gib']
    disk_reason = 'startup_free_space_below_50GiB'
    if result['free_gib'] >= required:
        result['reasons'] = [reason for reason in result['reasons'] if reason != disk_reason]
    elif disk_reason not in result['reasons']:
        result['reasons'].append(disk_reason)
    result.update(ready=not result['reasons'], resume_accounting=dict(
        approved_total_product_budget_gib=40, reserve_gib=10,
        existing_product_credit_gib=credit, remaining_product_budget_gib=40-credit,
        required_free_gib=required, credited_scope='unique non-symlink allocated files under this run assets/'))
    return result
