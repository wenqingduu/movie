#!/usr/bin/env python3
"""Build an independent, input-pinned full140 snapshot and publish after validation.

Does not change the running queue, its frozen helper, scores, or media. Each
invocation selects complete episodes once; later queue completions go in the
next snapshot. The full140_published symlink is the sole publication pointer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = ROOT / 'outputs/entitybench_qwen140_multiface_remaining126_20261030/cohort.json'
HELPER = ROOT / 'pretest/qwen140_reaggregate_partial_reruns.py'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def capture_inputs(cohort, selected):
    paths = {cohort, HELPER, Path(__file__).resolve()}
    for value in selected.values():
        directory, evaluation = value['root'], value['evaluation']
        # Pilot and legacy cohorts have different completion/audit contracts;
        # the unchanged helper validates their generation and scoring inputs.
        for name in ('complete.json', 'evaluation_validation.json', 'rerun_manifest.json'):
            path = directory / name
            if path.is_file():
                paths.add(path)
        paths.add(Path(value['cohort']))
        paths.add(evaluation / 'input_manifest.json')
        paths.add(evaluation / 'results/run_manifest.json')
        paths.update((evaluation / 'results').glob('*.json'))
        paths.update(directory.glob('episodes/*/episode_result.json'))
    return {str(p): sha(p) for p in sorted(paths)}


def check_inputs(fingerprints):
    for path, expected in fingerprints.items():
        if sha(path) != expected:
            raise ValueError(f'Snapshot input changed during aggregation: {path}')


def publish(snapshot, pointer):
    """Only expose a completely validated snapshot; preserve prior publication."""
    state = read(snapshot / 'status.json')
    validation = read(snapshot / 'validation.json')
    selection = read(snapshot / 'selection_manifest.json')
    assert state['status'] == 'complete_reaggregation_only'
    assert validation['episode_count'] == selection['num_episodes'] == 140
    assert validation['shot_count'] == selection['num_shots'] == 2491
    assert len(selection['episodes']) == len({r['episode_id'] for r in selection['episodes']}) == 140
    assert len(selection['videos']) == 2491
    assert state['replaced_episodes'] == validation['replaced_episode_count'] == selection['rerun_episodes']
    assert state['replaced_shots'] == validation['replaced_shot_count'] == selection['rerun_shots']
    for key in ('baseline_aggregate_reproduced_exactly_for_all_tiers',
                'no_scoring_configuration_or_model_differences',
                'all_original_and_selected_video_hashes_verified',
                'canonical_links_verified', 'original_and_rerun_parent_chains_verified',
                'no_neural_inference'):
        assert validation[key], key
    assert validation['vlm_calls'] == selection['vlm_calls'] == 0
    check_inputs(read(snapshot / 'snapshot_inputs.json')['sha256'])
    if os.path.lexists(pointer) and not pointer.is_symlink():
        raise ValueError(f'Refusing to replace a real file/directory: {pointer}')
    entry = pointer.with_suffix('.html')
    if os.path.lexists(entry) and (entry.is_symlink() or not entry.is_file()):
        raise ValueError(f'Refusing to replace a non-regular publication entry: {entry}')
    # Keep the final version directory in place: canonical paths recorded by the
    # original helper must remain valid. Only the pointer changes atomically.
    temporary = pointer.with_name('.' + pointer.name + '.' + uuid.uuid4().hex)
    temporary_entry = temporary.with_suffix('.html')
    try:
        # A browser does not resolve a directory symlink before resolving URLs.
        # Redirect to the canonical version URL so the helper's relative links
        # to episode review pages work at their original directory depth.
        target = html.escape(os.path.relpath(snapshot / 'index.html', pointer.parent), quote=True)
        temporary_entry.write_text('<!doctype html><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="0;url={target}">'
            f'<title>已验证的140集快照</title><a href="{target}">打开最近发布的已验证140集快照</a>')
        temporary.symlink_to(os.path.relpath(snapshot, pointer.parent), target_is_directory=True)
        os.replace(temporary, pointer)
        os.replace(temporary_entry, entry)
    finally:
        if temporary.is_symlink():
            temporary.unlink()
        if temporary_entry.is_file():
            temporary_entry.unlink()


def build(cohort):
    cohort = cohort.resolve()
    parent = cohort.parent
    with (parent / '.full140_publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(cohort)
        for item in plan['frozen_code']:
            assert sha(item['path']) == item['sha256'], item['path']
        helper_expected = next(x['sha256'] for x in plan['frozen_code']
                               if Path(x['path']) == HELPER)
        spec = importlib.util.spec_from_file_location('full140_frozen_helper', HELPER)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        assert sha(HELPER) == helper_expected
        selected = helper.replacements(cohort)
        fingerprints = capture_inputs(cohort, selected)
        # Pin once before any media hashing: a concurrently completed episode
        # must not change this invocation's cohort half way through the build.
        helper.replacements = lambda additional_cohort=None: dict(selected)
        version = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        snapshot = parent / 'full140_versions' / version
        snapshot.mkdir(parents=True, exist_ok=False)
        write(snapshot / 'snapshot_inputs.json', {
            'version': 'independent_snapshot_publication_v1',
            'selection_captured_utc': datetime.now(timezone.utc).isoformat(),
            'cohort': str(cohort), 'rerun_episode_ids': sorted(selected),
            'sha256': fingerprints, 'frozen_helper_sha256': helper_expected,
            'publisher_sha256': sha(__file__),
            'selection_policy': 'All complete episodes at capture time, regardless of score; later completions await next snapshot',
            'neural_inference': False, 'vlm_calls': 0})
        try:
            helper.main(snapshot, cohort)
            check_inputs(fingerprints)
            shutil.copy2(__file__, snapshot / 'runtime' / Path(__file__).name)
            write(snapshot / 'publication_ready.json', {
                'validated_utc': datetime.now(timezone.utc).isoformat(),
                'snapshot': str(snapshot), 'pointer': str(parent / 'full140_published'),
                'selection_manifest_sha256': sha(snapshot / 'selection_manifest.json'),
                'validation_sha256': sha(snapshot / 'validation.json'),
                'policy': 'Published only after full input, media, provenance and official aggregation validation'})
            publish(snapshot, parent / 'full140_published')
        except Exception as exc:
            write(snapshot / 'publication_failure.json', {'error': repr(exc),
                'action': 'Preserve failed version and prior publication; do not alter queue or original scores/media'})
            raise
        print('Published validated snapshot: ' + str(snapshot), flush=True)
        return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=DEFAULT_COHORT)
    build(parser.parse_args().cohort)
