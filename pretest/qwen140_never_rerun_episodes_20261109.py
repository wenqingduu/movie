"""Schedule only never-started whole episodes with the existing v2/4-step pipeline.

No motion-only jobs, model changes, semantic retries, disk quota or repair agent.
The original cohort owns assets/media; this directory owns the new queue receipts.
"""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess

import qwen140_multiface_remaining_episode_batch as original
import qwen140_v2_remaining_episodes_20261105 as v2

ROOT, B, SOURCE = original.ROOT, original.batch, original.OUT
OUT = ROOT / 'outputs/entitybench_never_rerun_episodes_v2_20261109'


def status(state, episode=None, **extra):
    B.write(OUT / 'status.json', dict(updated_utc=B.now(), state=state,
        episode_id=episode, main_model='Wan2.2-I2V-A14B LightX2V 4-step',
        prompt_version='v2_20261104_unchanged', repair_agent_enabled=False,
        motion_enhancement_queue_resumed=False, **extra))


def validate(plan):
    for row in plan['frozen_files']:
        assert B.sha(row['path']) == row['sha256'], row['path']


def prepare():
    if (OUT / 'plan.json').exists():
        plan = B.read(OUT / 'plan.json')
        validate(plan)
        return plan
    source = B.read(SOURCE / 'cohort.json')
    snapshot = ROOT / 'outputs/entitybench_full140_latest57_v2_141_20261106/snapshot.json'
    editions = {e['episode_id']: e['edition'] for e in B.read(snapshot)['episodes']}
    chosen, inventory = [], []
    for item in source['episodes']:
        directory = Path(item['output_root'])
        folders = list(directory.glob('episodes/*/shot_*'))
        results = list(directory.glob('episodes/*/shot_*/result.json'))
        complete = (directory / 'complete.json').exists()
        eligible = not complete and not folders and not (directory / 'rerun_manifest.json').exists()
        inventory.append(dict(episode_id=item['episode_id'], output_root=str(directory),
            total_shots=len(item['episode']['shots']), generated_shots=len(results),
            shot_folders=len(folders), complete=complete, selected=eligible,
            existing_assets=(directory / 'assets').exists(),
            policy='No whole-episode shot has started; independent shot experiments do not count as a whole-episode rerun.'))
        if eligible:
            assert editions[item['episode_id']] == 'original'
            chosen.append(item)
    assert len(chosen) == 78 and sum(len(e['episode']['shots']) for e in chosen) == 1218
    frozen = {r['path']: r for r in source['frozen_code']}
    for path in [Path(__file__), Path(v2.__file__), SOURCE / 'cohort.json', snapshot,
                 ROOT / 'pretest/qwen140_prompt_v2_20261104.py',
                 ROOT / 'pretest/qwen140_technical_resume_v2_20261103.py',
                 ROOT / 'docs/entitybench_qwen140_route_agent_prompt_v2_20261104.md']:
        frozen[str(path)] = dict(path=str(path), sha256=B.sha(path))
    plan = dict(created_utc=B.now(), episodes=chosen, num_episodes=len(chosen),
        num_shots=sum(len(e['episode']['shots']) for e in chosen),
        source_cohort=str(SOURCE / 'cohort.json'), frozen_files=list(frozen.values()),
        policy='Only whole-episode never-started shots. Preserve completed/partial episodes and independent experiments. Source cohort order; reuse exact existing assets.',
        main_model='Wan2.2-I2V-A14B LightX2V 4-step; 81 frames, 16 fps',
        transition_model='Existing wan3.0-video API first/last-frame B transition, unchanged',
        agent_model=source['agent_model'], style=source['style'],
        prompt_version='v2_20261104_unchanged; no new motion/background enhancement',
        injection='Existing all-generated B/C keyframes multi-face v7; original per-role gates, A inherits actual own tail',
        evaluation='Complete-episode official strict/skip_vlm 12 metrics. Default visible-single-face reviewed DINO requires later localization review; raw cs_face is not its substitute.',
        repair_agent_enabled=False, disk_policy=dict(reserve_bytes=0, quota_bytes=None, automatic_deletion=False),
        errors='Record isolated technical errors and continue other episodes. Stop systemic API/quota/disk/OOM failures; do not lower parameters or change semantics.')
    validate(plan)
    B.write(OUT / 'inventory_at_launch.json', inventory, frozen=True)
    B.write(OUT / 'plan.json', plan, frozen=True)
    status('prepared', expected_episodes=78, expected_shots=1218)
    return plan


def run(plan):
    with (OUT / '.lock').open('a') as own, (SOURCE / '.queue.lock').open('a') as shared:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader'], text=True).strip(), 'Another GPU job is still active'
        validate(plan)
        original.configure()
        B.OUT, B.status = OUT, status
        source = B.read(SOURCE / 'cohort.json')
        errors = B.read(OUT / 'errors.json') if (OUT / 'errors.json').exists() else []
        for item in plan['episodes']:
            directory = Path(item['output_root'])
            if (directory / 'complete.json').exists():
                continue
            try:
                validate(plan)
                for name in ['cohort_config', 'style_specs']:
                    assert B.sha(directory / (name + '.json')) == item[name + '_sha256']
                env = B.assets(item, source)
                if not (directory / 'rerun_manifest.json').exists():
                    for action in ['prepare', 'refs']:
                        B.command(item, 'never_' + action, [B.WAN, '-u',
                            'pretest/qwen140_episode_single_pass.py', action, '--run', item['episode']['run']], env)
                if not (directory / 'episodes' / item['episode_id'] / 'episode_result.json').exists():
                    B.command(item, 'never_v2_generation', [B.WAN, '-u', str(Path(__file__)),
                        'episode', '--run', item['episode']['run']], env)
                for action in ['prepare', 'run', 'baseline-aggregate', 'compare']:
                    B.command(item, 'never_evaluation_' + action, [B.EVAL, '-u',
                        'pretest/qwen140_integrated_episode_eval.py', action,
                        '--name', 'weak4_v1', '--runs', item['episode']['run']], env)
                original.summarize(item)
                B.write(directory / 'never_rerun_v2_complete_20261109.json', dict(
                    completed_utc=B.now(), queue=str(OUT), plan_sha256=B.sha(OUT / 'plan.json'),
                    default_reviewed_dino='pending_localization_review'))
                done = sum((Path(e['output_root']) / 'complete.json').exists() for e in plan['episodes'])
                status('episode_complete_official_non_vlm', item['episode_id'],
                    completed_episodes=done, expected_episodes=plan['num_episodes'])
            except Exception as exc:
                entry = dict(utc=B.now(), episode_id=item['episode_id'], error=repr(exc))
                errors.append(entry)
                B.write(OUT / 'errors.json', errors)
                logs = '\n'.join(p.read_text(errors='replace')[-12000:]
                    for p in (directory / 'logs').glob('*.log'))
                systemic = ['SIGKILL', 'invalid_api_key', 'insufficient_quota', 'Arrearage',
                    'HTTP 401', 'insufficient balance', 'No space left on device',
                    'CUDA out of memory', 'Allocation on device', 'out of memory']
                if any(token.lower() in logs.lower() for token in systemic):
                    status('stopped_systemic_error', item['episode_id'], error=entry)
                    return
                status('episode_error_recorded_continue', item['episode_id'], error=entry)
        done = sum((Path(e['output_root']) / 'complete.json').exists() for e in plan['episodes'])
        status('complete_official_pending_dino_review' if done == plan['num_episodes']
            else 'finished_with_recorded_errors', complete_episodes=done,
            expected_episodes=plan['num_episodes'], errors=len(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'episode', 'probe'])
    parser.add_argument('--run')
    args = parser.parse_args()
    if args.action in {'episode', 'probe'}:
        v2.episode_worker(args.run, probe=args.action == 'probe')
    else:
        plan = prepare()
        if args.action == 'run':
            try:
                run(plan)
            except Exception as exc:
                status('queue_error', error=repr(exc))
                raise
        else:
            print(OUT / 'plan.json')
