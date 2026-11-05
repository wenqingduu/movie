"""Selective-shot-only continuation; legacy whole-episode generation stays stopped."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import qwen140_prompt_trial_20261104 as trial

ROOT = trial.ROOT
OUT = ROOT / 'outputs/entitybench_prompt_v2_badshots_20261104'
PRIOR = ROOT / 'outputs/entitybench_prompt_v2_trial_20261104'
AUDIT = trial.AUDIT.parent
EVAL = '/root/autodl-tmp/envs/entitybench-eval/bin/python'
QUEUE_PID = 446545
EPISODE_PID = 756528


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def folder(case):
    return Path(case['output_root']) / 'episodes' / case['episode_id'] / ('shot_' + case['shot'].replace(':', '_'))


def prepare():
    if (OUT / 'plan.json').exists():
        plan = read(OUT / 'plan.json')
        for item in plan['sources']:
            assert sha(item['path']) == item['sha256'], item['path']
        return plan
    rows = read(trial.AUDIT)['shots']
    refs = read(AUDIT / 'video_prompt_image_reference_audit.json')['affected']
    affected = {(r['episode_id'], r['shot']) for r in refs}
    bad = [r for r in rows if r['current'] == 0 and
           (r['baseline'] == 1 or (r['episode_id'], r['shot']) in affected)]
    reuse = {(r['episode_id'], r['shot']): r for r in read(PRIOR / 'cases.json')}
    pending = [r for r in bad if (r['episode_id'], r['shot']) not in reuse]
    first = [r for pair in [('run406', '2:3'), ('run406', '2:5')]
             for r in pending if (r['run'], r['shot']) == pair]
    # First20 cover C14/B4/A2, rather than selecting only likely improvements.
    for route, quota in [('C', 14), ('B', 4), ('A', 2)]:
        pool = sorted((r for r in pending if r['route'] == route and r not in first),
                      key=lambda r: ((r['episode_id'], r['shot']) not in affected, r['episode_id'], r['shot']))
        first += pool[:quota - sum(r['route'] == route for r in first)]
    pending = first + [r for r in pending if r not in first]
    assert len(bad) == 218 and len(pending) == 215 and len(first) == 20
    cases = []
    for row in pending:
        # A run number can contain several windows. Preserve the full episode ID.
        source = Path(row['folder']).parent.parent.parent
        generation = source
        if not (generation / 'face_injection_policy.json').exists():
            # Latest run594 is an evaluation view of the accepted media repair.
            generation = ROOT / 'outputs/entitybench_qwen140_multiface_remaining126_20261030/experiments' / row['episode_id']
        tag = row['episode_id'].split('__')[2]
        root = OUT / f"{row['run']}__{tag}__shot_{row['shot'].replace(':', '_')}"
        root.mkdir(parents=True, exist_ok=True)
        for name in ['face_injection_policy.json', 'style_specs.json']:
            shutil.copyfile(generation / name, root / name)
        write(root / 'cohort_config.json', {'run_tags': {row['run']: tag}})
        shots = read(generation / 'rerun_manifest.json')['shots']
        index = next(i for i, s in enumerate(shots) if s['shot_key'] == row['shot'])
        if index:
            key = shots[index - 1]['shot_key']
            parent_path = source / 'episodes' / row['episode_id'] / ('shot_' + key.replace(':', '_')) / 'result.json'
            if parent_path.exists():
                parent = read(parent_path)
            else:
                parent = next(s for s in read(source / 'episodes' / row['episode_id'] / 'episode_result.json')['shots'] if s['shot_key'] == key)
            assert sha(parent['last_frame']) == parent['last_sha256']
            write(root / 'episodes' / row['episode_id'] / ('shot_' + key.replace(':', '_')) / 'result.json', parent)
        cases.append({**row, 'output_root': str(root), 'previous_output_root': str(source),
                      'identity_asset_source_root': str(generation)})
    write(OUT / 'cases.json', cases)
    assert len({(c['episode_id'], c['shot']) for c in cases}) == len(cases)
    reused = [reuse[r['episode_id'], r['shot']] for r in bad if (r['episode_id'], r['shot']) in reuse]
    assert len(reused) == 3
    for case in reused:
        record = read(folder(case) / 'result.json')
        assert sha(record['video']) == record['video_sha256']
    sources = [trial.AUDIT, AUDIT / 'video_prompt_image_reference_audit.json',
               PRIOR / 'cases.json', PRIOR / 'report.json', OUT / 'cases.json',
               Path(__file__).resolve(), ROOT / 'pretest/qwen140_prompt_v2_20261104.py',
               Path(trial.__file__), ROOT / 'pretest/qwen140_prompt_v2_badshot_dynamic_20261104.py',
               ROOT / 'pretest/qwen140_technical_resume_v2_20261103.py',
               ROOT / 'docs/entitybench_qwen140_route_agent_prompt_v2_20261104.md']
    frozen = read(ROOT / 'outputs/entitybench_episode_repairs_20261103/queue_v2.json')['frozen_code']
    for item in frozen:
        assert sha(item['path']) == item['sha256']
    plan = {'created_utc': now(), 'selected': 218, 'new_shots': len(cases), 'reused': reused,
            'selection': '194 original-pass/current-fail plus24 other current-fail shots with actual video image-number references; candidates, not proof of wrong motion.',
            'scope': 'Independent shot reruns using the same original integrated parent tail, not a regenerated episode chain or official benchmark replacement.',
            'first20_original_routes': dict(Counter(c['route'] for c in cases[:20])),
            'remaining_original_routes': dict(Counter(c['route'] for c in cases)),
            'sources': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in sources],
            'frozen_queue_sources': frozen, 'generation': 'Prompt v2; existing identity assets/styles/models/seeds/v7 gates; single pass, no semantic retry.',
            'eta_continuous_hours': [24, 32], 'first20_hours': [2, 3],
            'failure_policy': 'Record per-shot errors and continue; stop for API auth/quota or SIGKILL. Normal gate skips are accepted. No repair Agent.',
            'disk_reserve_gib': 0, 'disk_quota_gib': None, 'automatic_deletion': False}
    write(OUT / 'plan.json', plan)
    return plan


def status(state, **extra):
    cases = read(OUT / 'cases.json')
    complete = [c for c in cases if (folder(c) / 'result.json').exists()]
    scored = [c for c in complete if (folder(c) / 'dynamic_v2.json').exists()]
    write(OUT / 'status.json', {'updated_utc': now(), 'state': state,
          'new_completed': len(complete), 'new_expected': len(cases), 'reused_completed': 3,
          'dynamic_scored': len(scored), 'repair_agent_enabled': False, **extra})


def process_info(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'state': stat[0], 'start_ticks': stat[19],
                'cmd': Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except FileNotFoundError:
        return None


def run(plan):
    paused = []
    errors = read(OUT / 'errors.json') if (OUT / 'errors.json').exists() else []
    with (OUT / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if not (OUT / 'legacy_episode_queue_stopped.json').exists():
                queue, episode = process_info(QUEUE_PID), process_info(EPISODE_PID)
                assert queue and 'qwen140_repair_priority_queue_v2_20261103.py run' in queue['cmd']
                assert episode and 'technical_resume_v2_20261103.py episode --run run4' in episode['cmd']
                for row in [queue, episode]:
                    os.kill(row['pid'], signal.SIGSTOP)
                    paused.append(row)
                write(OUT / 'maintenance.json', {'state': 'paused_at_worker_boundary', 'utc': now(), 'paused': paused})
                status('waiting_for_existing_worker')
                while True:
                    children = Path(f'/proc/{EPISODE_PID}/task/{EPISODE_PID}/children').read_text().split()
                    if not any((row := process_info(int(pid))) and row['state'] != 'Z' for pid in children):
                        break
                    time.sleep(5)
            # A reaped/zombie CUDA worker may remain in NVML briefly on exit.
            # Wait for its context release before starting the next model.
            for _ in range(30):
                gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                if not gpu:
                    break
                time.sleep(2)
            assert not gpu, f'Unexpected GPU worker: {gpu}'
            for position, case in enumerate(read(OUT / 'cases.json'), 1):
                f = folder(case)
                try:
                    with (Path(case['output_root']) / 'batch.log').open('a') as log:
                        # Subprocess stage API preserves per-case env and cached inputs.
                        def stage(action):
                            status('running', run=case['run'], shot=case['shot'], stage=action)
                            env = dict(os.environ, QWEN140_INTEGRATED_OUTPUT_ROOT=case['output_root'],
                                       QWEN140_COHORT_CONFIG=str(Path(case['output_root']) / 'cohort_config.json'),
                                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                            subprocess.run([trial.PYTHON, '-u', trial.__file__, 'case', '--run', case['run'],
                                            '--shot', case['shot'], '--stage', action], env=env,
                                           stdout=log, stderr=subprocess.STDOUT, check=True)
                        if not (f / 'result.json').exists():
                            if not (f / 'compiled_prompts.json').exists():
                                stage('plan')
                            route = read(f / 'agent/plan.json')['decision']['route']
                            if route != 'A' and not (f / 'frame/target.png').exists():
                                stage('frame')
                            if route == 'B' and not (f / 'transition/decoded.json').exists():
                                stage('transition')
                            if not (f / 'main_media.json').exists():
                                stage('main')
                            stage('finish')
                        record = read(f / 'result.json')
                        assert sha(record['video']) == record['video_sha256']
                        if not (f / 'dynamic_v2.json').exists():
                            status('running', run=case['run'], shot=case['shot'], stage='dynamic_evaluation')
                            subprocess.run([EVAL, '-u', str(ROOT / 'pretest/qwen140_prompt_v2_badshot_dynamic_20261104.py'),
                                            '--folder', str(f), '--before', case['video']],
                                           env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                                           stdout=log, stderr=subprocess.STDOUT, check=True)
                    refresh()
                    if position == 20:
                        write(OUT / 'first20_checkpoint.json', {'utc': now(), 'comparison': str(OUT / 'comparison.json'),
                              'selection': read(OUT / 'cases.json')[:20],
                              'scope': 'First20 attempted shots; preserve failures and missing scores, not an episode aggregate.'})
                except Exception as exc:
                    text = (Path(case['output_root']) / 'batch.log').read_text(errors='replace')[-10000:]
                    errors.append({'utc': now(), 'episode_id': case['episode_id'], 'shot': case['shot'], 'error': repr(exc)})
                    write(OUT / 'errors.json', errors)
                    if any(x in text for x in ['SIGKILL', 'invalid_api_key', 'insufficient_quota', 'Arrearage', 'HTTP 401', 'insufficient balance']):
                        status('stopped_systemic_error', error=errors[-1])
                        return
                    refresh()
            refresh()
            status('generation_complete_pending_full_evaluation', errors=len(errors))
            from qwen140_badshot_full_evaluation_20261104 import OUT as followup_root
            with (followup_root / 'followup.log').open('a') as log:
                subprocess.run([EVAL, '-u', str(ROOT / 'pretest/qwen140_badshot_full_evaluation_20261104.py'), 'run'],
                               env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            status(read(followup_root / 'status.json')['stage'], followup_status=str(followup_root / 'status.json'))
        finally:
            # Let any already launched worker finish before handing the GPU back.
            while True:
                children = Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read_text().split()
                if not any((item := process_info(int(pid))) and item['state'] != 'Z' for pid in children):
                    break
                time.sleep(5)
            for row in reversed(paused):
                current = process_info(row['pid'])
                if current and current['start_ticks'] == row['start_ticks']:
                    os.kill(row['pid'], signal.SIGCONT)
            write(OUT / 'old_queue_resumed.json', {'utc': now(), 'parents': [r['pid'] for r in paused]})


def refresh():
    import html
    items = []
    for case in read(OUT / 'cases.json') + read(OUT / 'plan.json')['reused']:
        f = folder(case)
        if not (f / 'result.json').exists():
            continue
        rec = read(f / 'result.json')
        metric = read(f / 'dynamic_v2.json') if (f / 'dynamic_v2.json').exists() else None
        if metric is None and Path(case['output_root']).parent == PRIOR:
            metric = read(PRIOR / (case['run'] + '_dynamic.json'))
        items.append({'episode_id': case['episode_id'], 'run': case['run'], 'shot': case['shot'],
                      'route': rec['route']['route'], 'before_video': case['video'], 'after_video': rec['video'],
                      'dynamic': metric, 'injection_applied': rec['injection_applied']})
    write(OUT / 'comparison.json', items)
    page = ['<meta charset="utf-8"><title>V2 bad-shot reruns</title><style>body{font:16px sans-serif;margin:30px}video{width:46%;max-width:700px}</style><h1>V2 problem-shot reruns</h1><p>Before: previous integrated run. Independent original parent tails; not whole-episode replacements.</p>']
    for row in items:
        page.append('<h2>' + html.escape(row['run'] + ' ' + row['shot'] + ' / ' + row['route']) + '</h2>')
        for key in ['before_video', 'after_video']:
            page.append('<video controls preload="none" src="' + html.escape(os.path.relpath(row[key], OUT), quote=True) + '"></video>')
    (OUT / 'index.html').write_text('\n'.join(page))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'status'])
    args = parser.parse_args()
    resume_contract = read(OUT / 'selective_only_resume.json')
    assert sha(Path(__file__)) == resume_contract['entry_sha256']
    assert sha(OUT / 'full_evaluation/extension_plan.json') == resume_contract['extension_plan_sha256']
    plan = prepare()
    from qwen140_badshot_full_evaluation_20261104 import prepare as prepare_followup
    prepare_followup()
    if args.action == 'prepare':
        status('prepared')
        refresh()
        print(json.dumps({k: plan[k] for k in ['selected', 'new_shots', 'first20_original_routes', 'remaining_original_routes', 'eta_continuous_hours']}))
    elif args.action == 'run':
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'Batch interrupted by signal {signum}')
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, interrupt)
        try:
            run(plan)
        except BaseException as exc:
            status('queue_stopped', error=repr(exc))
            raise
    else:
        refresh()
        print((OUT / 'status.json').read_text())


if __name__ == '__main__':
    main()
