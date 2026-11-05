"""Resume unfinished benchmark reruns after the active v2 shot batch.

Scheduling only. Preserve successful prefixes and in-flight frames; all newly
planned shots use the unchanged, validated 20261104 v2 installer.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import qwen140_multiface_remaining_episode_batch as original

ROOT = Path(__file__).resolve().parents[1]
SOURCE = original.OUT
OUT = ROOT / 'outputs/entitybench_v2_remaining_episodes_20261105'
SHOTS = ROOT / 'outputs/entitybench_prompt_v2_badshots_20261104'
B = original.batch


def status(state, episode=None, **extra):
    B.write(OUT / 'status.json', dict(updated_utc=B.now(), state=state,
        episode_id=episode, repair_agent_enabled=False, **extra))


def prepare():
    if (OUT / 'plan.json').exists():
        return B.read(OUT / 'plan.json')
    source = B.read(SOURCE / 'cohort.json')
    priority = B.read(ROOT / 'outputs/entitybench_episode_repairs_20261103/queue_v2.json')['order']
    by_id = {r['episode_id']: r for r in source['episodes']}
    order = priority + [e for e in by_id if e not in priority]
    pending, prefix, inflight = [], [], []
    for eid in order:
        item = by_id[eid]; directory = Path(item['output_root'])
        if (directory / 'complete.json').exists():
            continue
        missing = False; completed = 0
        for shot in item['episode']['shots']:
            folder = directory / 'episodes' / eid / ('shot_' + shot['shot_key'].replace(':', '_'))
            result = folder / 'result.json'
            if result.exists():
                assert not missing, f'Non-contiguous prefix: {result}'
                row = B.read(result)
                assert B.sha(row['video']) == row['video_sha256']
                assert B.sha(row['last_frame']) == row['last_sha256']
                for path in [result, Path(row['video']), Path(row['last_frame'])]:
                    prefix.append(dict(path=str(path), sha256=B.sha(path)))
                completed += 1
            else:
                missing = True
                if (folder / 'frame/target.png').exists():
                    # Finish these exact cached inputs once, rather than discard
                    # an expensive frame or silently rebind its prompt.
                    for name in ['agent/plan.json', 'compiled_prompts.json', 'frame/target.png',
                                 'frame/target.json', 'frame/target.task.json']:
                        path = folder / name
                        prefix.append(dict(path=str(path), sha256=B.sha(path)))
                    inflight.append(dict(episode_id=eid, shot=shot['shot_key'],
                        policy='Finish preserved frame with its original compiled plan; subsequent new plans use v2.'))
        pending.append(dict(episode_id=eid, output_root=str(directory),
            completed_shots=completed, total_shots=len(item['episode']['shots']),
            remaining_shots=len(item['episode']['shots']) - completed,
            priority='below_original_reviewed_dino_mean' if eid in priority else 'remaining_benchmark'))
    files = [Path(f['path']) for f in source['frozen_code']]
    for f in source['frozen_code']:
        assert B.sha(f['path']) == f['sha256'], f['path']
    files += [Path(__file__), ROOT / 'pretest/qwen140_prompt_v2_20261104.py',
        ROOT / 'pretest/qwen140_technical_resume_v2_20261103.py',
        ROOT / 'docs/entitybench_qwen140_route_agent_prompt_v2_20261104.md',
        SOURCE / 'cohort.json']
    plan = dict(created_utc=B.now(), prompt_version='v2_20261104_unchanged',
        episodes=pending, remaining_episodes=len(pending),
        remaining_shots=sum(e['remaining_shots'] for e in pending),
        completed_prefix=prefix, preserved_inflight_frames=inflight,
        frozen_files=[dict(path=str(p), sha256=B.sha(p)) for p in dict.fromkeys(files)],
        disk_policy=dict(reserve_bytes=0, quota_bytes=None, automatic_deletion=False),
        order='Active motion shot batch and its scoring first; unfinished low-DINO episodes next; other unfinished episodes last.',
        evaluation='Official strict/skip_vlm 12 metrics per complete episode; default reviewed DINO remains pending localization review.',
        repair_agent_enabled=False)
    B.write(OUT / 'plan.json', plan, frozen=True)
    status('queued_after_active_v2_shot_batch', remaining_episodes=len(pending),
           remaining_shots=plan['remaining_shots'])
    return plan


def validate(plan):
    for row in plan['frozen_files'] + plan['completed_prefix']:
        assert B.sha(row['path']) == row['sha256'], row['path']


def episode_worker(run, probe=False):
    from qwen140_prompt_v2_20261104 import install, SYSTEM
    episode = install()
    import qwen140_shot_agent as agent
    assert agent.SYSTEM == SYSTEM
    ep, directory, _ = episode.load(run, None)
    if probe:
        print(json.dumps(dict(episode_id=ep['episode_id'], prompt_version='v2_20261104',
            compiled_function=episode.compile_prompts.__module__, api_calls=0)))
        return
    for shot in ep['shots']:
        folder = directory / ('shot_' + shot['shot_key'].replace(':', '_'))
        if (folder / 'result.json').exists() or (folder / 'frame/target.png').exists():
            continue
        # Incomplete old API failures have no valid frame to preserve. Archive
        # them so the next single-pass v2 call has a fresh fingerprint.
        if (folder / 'agent').exists():
            archive = folder / 'before_v2_20261105'
            archive.mkdir(exist_ok=True)
            for name in ['agent', 'compiled_prompts.json', 'inherited_characters.json',
                         'frame_asset_projection.json', 'missing_setting_descriptions.json', 'frame']:
                path = folder / name
                if path.exists():
                    assert not (archive / name).exists()
                    shutil.move(str(path), str(archive / name))
    episode.run_remaining(run)


def run(plan):
    OUT.mkdir(parents=True, exist_ok=True)
    # The current shot coordinator holds this through its scoring follow-up.
    # Never run a second heavy GPU job alongside it.
    with (OUT / '.lock').open('a') as own, (SHOTS / '.lock').open('a') as shot_lock:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                fcntl.flock(shot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status('waiting_for_active_v2_shots_and_scoring',
                       upstream_status=B.read(SHOTS / 'status.json'))
                time.sleep(15)
        upstream = B.read(SHOTS / 'status.json')
        if upstream['state'] in {'running', 'stopped_systemic_error'}:
            status('blocked_by_upstream_failure', upstream_status=upstream)
            return
        with (SOURCE / '.queue.lock').open('a') as cohort_lock:
            fcntl.flock(cohort_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            validate(plan)
            original.configure()
            B.status = lambda state, episode=None, **kw: status(state, episode, **kw)
            source = B.read(SOURCE / 'cohort.json')
            by_id = {e['episode_id']: e for e in source['episodes']}
            errors = B.read(OUT / 'errors.json') if (OUT / 'errors.json').exists() else []
            for row in plan['episodes']:
                item = by_id[row['episode_id']]; directory = Path(item['output_root'])
                if (directory / 'complete.json').exists():
                    continue
                try:
                    for contract in plan['frozen_files']:
                        assert B.sha(contract['path']) == contract['sha256']
                    assert B.sha(directory / 'cohort_config.json') == item['cohort_config_sha256']
                    assert B.sha(directory / 'style_specs.json') == item['style_specs_sha256']
                    env = B.assets(item, source)
                    if not (directory / 'rerun_manifest.json').exists():
                        for action in ['prepare', 'refs']:
                            B.command(item, 'v2_' + action, [B.WAN, '-u',
                                'pretest/qwen140_episode_single_pass.py', action,
                                '--run', item['episode']['run']], env)
                    if not (directory / 'episodes' / row['episode_id'] / 'episode_result.json').exists():
                        B.command(item, 'v2_generation', [B.WAN, '-u', str(Path(__file__)),
                            'episode', '--run', item['episode']['run']], env)
                    for action in ['prepare', 'run', 'baseline-aggregate', 'compare']:
                        B.command(item, 'v2_evaluation_' + action, [B.EVAL, '-u',
                            'pretest/qwen140_integrated_episode_eval.py', action,
                            '--name', 'weak4_v1', '--runs', item['episode']['run']], env)
                    original.summarize(item)
                    B.write(directory / 'v2_resume_complete_20261105.json', dict(
                        completed_utc=B.now(), queue=str(OUT), original_prefix_preserved=True,
                        default_reviewed_dino='pending_localization_review'))
                    status('episode_complete_official_non_vlm', row['episode_id'])
                except Exception as exc:
                    error = dict(utc=B.now(), episode_id=row['episode_id'], error=repr(exc))
                    errors.append(error); B.write(OUT / 'errors.json', errors)
                    logs = '\n'.join(p.read_text(errors='replace')[-10000:]
                        for p in (directory / 'logs').glob('v2_*.log'))
                    if any(x in logs for x in ['SIGKILL', 'invalid_api_key', 'insufficient_quota',
                            'Arrearage', 'HTTP 401', 'insufficient balance', 'No space left on device']):
                        status('stopped_systemic_error', row['episode_id'], error=error)
                        return
                    status('episode_error_recorded_continue', row['episode_id'], error=error)
            done = sum((Path(e['output_root']) / 'complete.json').exists() for e in plan['episodes'])
            status('complete_official_pending_dino_review' if done == len(plan['episodes'])
                   else 'finished_with_recorded_errors', complete_episodes=done,
                   expected_episodes=len(plan['episodes']), errors=len(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'episode', 'probe'])
    parser.add_argument('--run')
    args = parser.parse_args()
    if args.action in {'episode', 'probe'}:
        episode_worker(args.run, probe=args.action == 'probe')
    else:
        plan = prepare()
        if args.action == 'run':
            try:
                run(plan)
            except Exception as exc:
                status('queue_error', error=repr(exc))
                raise
