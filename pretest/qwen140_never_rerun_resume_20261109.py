"""Resume the frozen whole-episode queue after restart, preserving v2 plans.

Wait for an already-running scoring job; do not archive/replan an interrupted
v2 frame with the historical v1-to-v2 migration helper.
"""
import argparse
import fcntl
from pathlib import Path
import time

import qwen140_never_rerun_episodes_20261109 as queue

RECEIPT = queue.OUT / 'resume_after_expansion_20261109.json'


def validate_resume():
    record = queue.B.read(RECEIPT)
    for row in record['preserved_inputs'] + record['resume_code']:
        assert queue.B.sha(row['path']) == row['sha256'], row['path']
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'episode', 'probe'])
    parser.add_argument('--run')
    args = parser.parse_args()
    record = validate_resume()
    if args.action in {'episode', 'probe'}:
        from qwen140_prompt_v2_20261104 import install, SYSTEM
        import qwen140_shot_agent as agent
        episode = install()
        assert agent.SYSTEM == SYSTEM
        if args.action == 'probe':
            ep, _, _ = episode.load(args.run, None)
            print(ep['episode_id'], 'v2 cached-plan preservation, no API calls')
            return
        episode.run_remaining(args.run)
        return
    with (queue.OUT / '.resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for dependency in record['wait_processes']:
            stat = Path('/proc') / str(dependency['pid']) / 'stat'
            while stat.exists():
                try:
                    fields = stat.read_text().split()
                except FileNotFoundError:
                    break
                if fields[21] != dependency['start_ticks'] or fields[2] == 'Z':
                    break
                queue.status('waiting_for_existing_evaluation',
                    dependency_pid=dependency['pid'], completed_shots_before_resume=1)
                time.sleep(10)
        validate_resume()
        # Subprocesses must use this entry point, including the plan-preserving
        # episode branch above. Original frozen queue/scorer code stays intact.
        queue.__file__ = str(Path(__file__).resolve())
        try:
            queue.run(queue.prepare())
        except Exception as exc:
            queue.status('queue_error', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
