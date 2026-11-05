"""Freeze completed v2 shots and score a mixed full140 snapshot now."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import qwen140_badshot_full_evaluation_20261104 as full
import qwen140_badshot_reviewed_dino_20261104 as diagnostic

ROOT = full.ROOT
OUT = ROOT / 'outputs/entitybench_v2_completed_snapshot_20261105'
OLD = full.SNAPSHOT
read, write, sha = full.read, full.write, full.sha


def validate():
    plan = read(OUT / 'frozen_sources.json')
    for item in plan:
        assert sha(item['path']) == item['sha256'], item['path']
    return read(OUT / 'snapshot.json')


def configure():
    full.OUT = OUT
    full.BATCH = OUT / 'input_batch'
    full.prepare = validate
    diagnostic.OUT = OUT / 'visible_region'
    diagnostic.OLD_REGIONS = [p for p in diagnostic.OLD_REGIONS]
    diagnostic.prepare = prepare_diagnostic
    return validate()


def prepare_diagnostic():
    directory = OUT / 'visible_region'
    directory.mkdir(exist_ok=True)
    data = read(OUT / 'replacement_manifest.json')
    changed = {k.split('/')[0] for k in data['targets']}
    before = {e['episode_id']: e for e in read(OUT / 'baseline_selection.json')['episodes']}
    oldmap, _, _ = diagnostic.before_state()
    items, changed_items = [], []
    for ep in read(full.BASE / 'manifest.json')['episodes']:
        eid = ep['episode_id']
        path = OUT / 'affected/results' / (eid + '.json') if eid in changed else Path(before[eid]['selected_result'])
        item = {'episode_id': eid, 'episode': ep, 'output_root': str(OUT / 'episode_media'),
                'result': str(path), 'result_sha256': sha(path),
                'artifacts': str(path.parent / '.work/eval_artifacts' / eid)}
        items.append(item)
        if eid in changed:
            changed_items.append(item)
        else:
            # Same video/claims/canonical pixels and reviewed input protocol.
            # Reuse exact prior rows rather than rerunning unchanged detections.
            for shot in ep['shots']:
                rows = [r for k, r in oldmap.items() if k[0] == eid and k[2] == shot['shot_key']]
                diagnostic.diag.immutable(directory / 'extraction' / eid / (shot['shot_key'].replace(':', '_') + '.json'),
                    {'episode_id': eid, 'shot': shot['shot_key'], 'rows': rows,
                     'reuse': 'Exact prior canonical regions and gate rows; input video is unchanged and SHA-validated in replacement manifest.'})
    diagnostic.diag.OUT = directory
    diagnostic.diag.immutable(directory / 'input_manifest.json', {'episodes': changed_items,
        'selected_episode_count': len(changed_items),
        'selected_shot_count': sum(len(i['episode']['shots']) for i in changed_items),
        'scope': 'Only affected complete episodes need new canonical face detection; other full140 rows reuse the exact previous inputs.'})
    shutil.copyfile(diagnostic.OLD_REGIONS[0] / 'required_span_validation.json', directory / 'required_span_validation.json')
    return items, data


def run():
    snap = configure()
    if not (OUT / 'baseline_selection.json').exists():
        shutil.copyfile(OLD / 'official/selection_manifest.json', OUT / 'baseline_selection.json')
    data = full.mapping()
    assert len(data['targets']) == snap['total_replaced_shots'] and not data['missing_targets_retained_baseline']
    full.evaluate('affected', data)
    full.status('comparing_completed_target_shots')
    full.compare_shots(data)
    changed = {k.split('/')[0] for k in data['targets']}
    for episode in read(OUT / 'baseline_selection.json')['episodes']:
        eid = episode['episode_id']
        source = OUT / 'affected/results' / (eid + '.json') if eid in changed else Path(episode['selected_result'])
        dest = OUT / 'full140/results' / (eid + '.json')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert sha(dest) == sha(source)
        else:
            shutil.copyfile(source, dest)
    full.aggregate_full(data)
    prepare_diagnostic()
    diagnostic.diag.extract('cpu')
    diagnostic.score(False)
    full.status('official_full140_complete_reviewed_dino_pending_localization_review',
                episodes=140, newly_replaced_shots=snap['total_replaced_shots'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'final'])
    args = parser.parse_args()
    configure()
    if args.action == 'run':
        try:
            run()
        except Exception as exc:
            full.status('stopped_with_recorded_error', error=repr(exc))
            raise
    else:
        diagnostic.score(True)
