"""User-scoped P0/P1/P2 only, then hand GPU to unfinished episodes."""
from pathlib import Path
import fcntl
import os
import subprocess

import qwen140_prompt_v2_selective_only_20261104 as queue

SCOPE = queue.OUT / 'scope33_20261105.json'


def status(state, cases, **extra):
    complete = [c for c in cases if (queue.folder(c) / 'result.json').exists()]
    queue.write(queue.OUT / 'status.json', dict(updated_utc=queue.now(), state=state,
        candidate_completed=len(complete), candidate_expected=len(cases),
        repair_agent_enabled=False, active_scope=str(SCOPE),
        followup='Unfinished episodes immediately; no P3/P4 or intervening full-benchmark evaluation.', **extra))


def run():
    scope = queue.read(SCOPE)
    assert queue.sha(Path(__file__)) == scope['entry_sha256']
    queue.prepare()  # Validate the unchanged generation source graph.
    original = {(c['episode_id'], c['shot']): c for c in queue.read(queue.OUT / 'cases.json')}
    cases = [original[(r['episode_id'], r['shot'])] for r in scope['selected']]
    assert len(cases) == 33
    errors = []
    with (queue.OUT / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in cases:
            folder = queue.folder(case)
            log_path = Path(case['output_root']) / 'batch.log'
            start = log_path.stat().st_size if log_path.exists() else 0
            try:
                with log_path.open('a') as log:
                    def stage(action):
                        status('running', cases, run=case['run'], shot=case['shot'], stage=action)
                        env = dict(os.environ, QWEN140_INTEGRATED_OUTPUT_ROOT=case['output_root'],
                            QWEN140_COHORT_CONFIG=str(Path(case['output_root']) / 'cohort_config.json'),
                            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                        subprocess.run([queue.trial.PYTHON, '-u', queue.trial.__file__,
                            'case', '--run', case['run'], '--shot', case['shot'], '--stage', action],
                            env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    if not (folder / 'result.json').exists():
                        if not (folder / 'compiled_prompts.json').exists():
                            stage('plan')
                        route = queue.read(folder / 'agent/plan.json')['decision']['route']
                        if route != 'A' and not (folder / 'frame/target.png').exists():
                            stage('frame')
                        if route == 'B' and not (folder / 'transition/decoded.json').exists():
                            stage('transition')
                        if not (folder / 'main_media.json').exists():
                            stage('main')
                        stage('finish')
                    record = queue.read(folder / 'result.json')
                    assert queue.sha(record['video']) == record['video_sha256']
                    assert queue.sha(record['last_frame']) == record['last_sha256']
                    if not (folder / 'dynamic_v2.json').exists():
                        status('running', cases, run=case['run'], shot=case['shot'], stage='dynamic_evaluation')
                        subprocess.run([queue.EVAL, '-u', str(queue.ROOT /
                            'pretest/qwen140_prompt_v2_badshot_dynamic_20261104.py'),
                            '--folder', str(folder), '--before', case['video']],
                            env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                                MKL_NUM_THREADS='4'), stdout=log, stderr=subprocess.STDOUT, check=True)
                queue.refresh()
            except Exception as exc:
                errors.append(dict(utc=queue.now(), episode_id=case['episode_id'],
                    shot=case['shot'], error=repr(exc)))
                queue.write(queue.OUT / 'scope33_errors_20261105.json', errors)
                with log_path.open('rb') as stream:
                    stream.seek(start)
                    text = stream.read().decode(errors='replace')[-12000:]
                if any(s in text for s in ['SIGKILL', 'invalid_api_key', 'insufficient_quota',
                        'Arrearage', 'HTTP 401', 'insufficient balance', 'No space left on device']):
                    status('stopped_systemic_error', cases, error=errors[-1])
                    return
        queue.refresh()
        status('scope33_complete_handoff_to_episodes' if not errors else
               'scope33_attempted_errors_recorded_handoff_to_episodes', cases, errors=errors,
               full_evaluation='Deferred; next job is unfinished episode generation.')


if __name__ == '__main__':
    run()
