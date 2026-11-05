"""Three single-pass shot trials with borrowed, hash-verified parent tails."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/entitybench_prompt_v2_trial_20261104'
AUDIT = ROOT / 'outputs/entitybench_full140_completed51_snapshot_20261104/dynamic_degree_audit/analysis.json'
CASES = [('run719', '3:1'), ('run1', '2:1'), ('run227', '10:2')]
PYTHON = '/root/autodl-tmp/envs/lightx2v-trial/bin/python'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def initialize():
    if (OUT / 'cases.json').exists():
        return json.loads((OUT / 'cases.json').read_text())
    rows = json.loads(AUDIT.read_text())['shots']
    cases = []
    for run, key in CASES:
        row = next(r for r in rows if (r['run'], r['shot']) == (run, key))
        source = Path(row['folder']).parent.parent.parent
        root = OUT / f"{run}_{key.replace(':', '_')}"
        root.mkdir(parents=True, exist_ok=True)
        for name in ['face_injection_policy.json', 'style_specs.json']:
            shutil.copyfile(source / name, root / name)
        tag = row['episode_id'].split('__')[2]
        write(root / 'cohort_config.json', {'run_tags': {run: tag}})
        manifest = json.loads((source / 'rerun_manifest.json').read_text())
        index = next(i for i, s in enumerate(manifest['shots']) if s['shot_key'] == key)
        if index:
            previous = 'shot_' + manifest['shots'][index - 1]['shot_key'].replace(':', '_')
            dest = root / 'episodes' / row['episode_id'] / previous / 'result.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / 'episodes' / row['episode_id'] / previous / 'result.json', dest)
        cases.append({**row, 'output_root': str(root), 'previous_output_root': str(source)})
    write(OUT / 'cases.json', cases)
    return cases


def stage(case, action):
    env = dict(os.environ)
    env.update(QWEN140_INTEGRATED_OUTPUT_ROOT=case['output_root'],
               QWEN140_COHORT_CONFIG=str(Path(case['output_root']) / 'cohort_config.json'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    subprocess.run([PYTHON, '-u', str(Path(__file__).resolve()), 'case', '--run', case['run'],
                    '--shot', case['shot'], '--stage', action], env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'generate', 'case'])
    parser.add_argument('--run'); parser.add_argument('--shot'); parser.add_argument('--stage')
    args = parser.parse_args()
    if args.action == 'case':
        from qwen140_prompt_v2_20261104 import install
        episode = install()
        if args.stage == 'plan':
            episode.stage('prepare', args.run, None)
            episode.stage('plan', args.run, args.shot)
        else:
            episode.stage(args.stage, args.run, args.shot)
        return
    for case in initialize():
        folder = Path(case['output_root']) / 'episodes' / case['episode_id'] / ('shot_' + case['shot'].replace(':', '_'))
        if args.action == 'plan':
            stage(case, 'plan')
        else:
            if (folder / 'result.json').exists():
                continue
            route = json.loads((folder / 'agent/plan.json').read_text())['decision']['route']
            actions = ([] if route == 'A' else ['frame']) + (['transition'] if route == 'B' else []) + ['main', 'finish']
            for action in actions:
                write(OUT / 'status.json', {'state': 'running', 'run': case['run'], 'shot': case['shot'], 'stage': action})
                stage(case, action)
    write(OUT / (args.action + '_complete.json'), {'cases': len(CASES)})


if __name__ == '__main__':
    main()
