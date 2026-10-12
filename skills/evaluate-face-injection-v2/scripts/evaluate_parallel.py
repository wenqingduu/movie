#!/usr/bin/env python3
"""Parallel scheduling around a frozen official non-VLM scorer; no score changes."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/movie')
PYTHON = '/root/autodl-tmp/envs/entitybench-eval/bin/python'
DEFAULT_ROOT = PROJECT / 'outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep'
OFFICIAL_SHA = '2334aad4b61cae2a0ea38cc2ee9ae85125b6df4e128610903654fd2b49607bb4'
ADAPTER_SHA = '79caf1327e7f1a63527c0e18bc9c936b8618b59080197d2f50677ee2ba5cb5a0'
DIMENSIONS = ['subject_consistency', 'temporal_flickering', 'motion_smoothness',
              'dynamic_degree', 'aesthetic_quality', 'imaging_quality']
TIERS = ['easy', 'medium', 'hard']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def freeze_inputs(root):
    manifest, generation, split = [read(root / p) for p in ['manifest.json', 'generation_report.json', 'split_20ep.json']]
    assert generation['status'] == 'complete', 'Finish generation before this evaluation'
    assert sha(root / 'runtime/evaluate_benchmark_official.py') == OFFICIAL_SHA
    assert sha(root / 'runtime/evaluate_benchmark_partial.py') == ADAPTER_SHA
    ids = [eid for tier in TIERS for eid in split.get(tier, [])]
    assert ids and len(ids) == len(set(ids)), 'Duplicate or empty episode split'
    if split.get('all'):
        assert len(split['all']) == len(ids) and set(split['all']) == set(ids)
    episodes = {e['episode_id']: e for e in manifest['episodes']}
    assert len(episodes) == len(manifest['episodes']) and set(ids) == set(episodes)
    jobs = {(r['episode_id'], r['shot_key']): r for r in generation['jobs']}
    assert len(jobs) == len(generation['jobs']) == sum(len(e['shots']) for e in episodes.values())
    inputs = dict(manifest_sha256=sha(root / 'manifest.json'),
        evaluator_sha256=ADAPTER_SHA, split_sha256=sha(root / 'split_20ep.json'), scripts={}, videos={})
    for eid in ids:
        episode = episodes[eid]
        script_sha = sha(episode['episode_path'])
        assert script_sha == episode['episode_sha256'], f'Changed script: {eid}'
        # The official CLI takes a directory, so validate its canonical script too.
        assert sha(PROJECT / 'benchmarks/entitybench/data/scripts' / f'{eid}.json') == script_sha
        inputs['scripts'][eid] = script_sha
        for shot in episode['shots']:
            row = jobs[(eid, shot['shot_key'])]
            assert row['status'] in {'generated', 'resumed_verified'}
            scene, local = map(int, shot['shot_key'].split(':'))
            canonical = root / 'official_results' / eid / f'{scene:03d}_{local:03d}.mp4'
            assert canonical.resolve() == Path(row['output_video']).resolve(), f'Wrong video mapping: {canonical}'
            assert sha(canonical) == row['video_sha256'], f'Changed video: {canonical}'
            inputs['videos'][f"{eid}/{shot['shot_key']}"] = row['video_sha256']
    return episodes, split, inputs


def partition(episodes, split, workers):
    assert workers > 0
    items = [(eid, tier) for tier in TIERS for eid in split.get(tier, [])]
    shards = [[] for _ in range(min(workers, len(items)))]
    loads = [0] * len(shards)
    for eid, tier in sorted(items, key=lambda item: (-len(episodes[item[0]]['shots']), item[0])):
        target = min(range(len(shards)), key=lambda i: (loads[i], i))
        shards[target].append((eid, tier))
        loads[target] += len(episodes[eid]['shots'])
    assert sorted(eid for shard in shards for eid, _ in shard) == sorted(eid for eid, _ in items)
    return shards, loads


def environment(threads, gpu):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
        NUMEXPR_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(PROJECT),
        DINO_MODEL_PATH='/root/autodl-tmp/models/dinov2-base',
        CLIP_MODEL_PATH='/root/autodl-tmp/models/clip-vit-base-patch32',
        GDINO_CONFIG='/root/autodl-tmp/models/entitybench_groundingdino/GroundingDINO_SwinT_OGC.py',
        GDINO_WEIGHTS='/root/autodl-tmp/models/entitybench_groundingdino/groundingdino_swint_ogc.pth',
        VBENCH_CACHE_DIR='/root/autodl-tmp/models/entitybench_vbench',
        TORCH_HOME='/root/autodl-tmp/models/entitybench_torch_cache',
        HF_HOME='/root/autodl-tmp/models/entitybench_hf_cache')
    for key in list(env):
        if any(x in key.upper() for x in ['OPENAI', 'AZURE', 'GEMINI', 'GOOGLE_API_KEY']):
            env.pop(key)
    return env


def command(root, worker_dir, method):
    return [PYTHON, '-u', str(root / 'runtime/evaluate_benchmark_partial.py'),
        '--results_dir', str(root / 'official_results'), '--scripts_dir', str(PROJECT / 'benchmarks/entitybench/data/scripts'),
        '--split_json', str(worker_dir / 'split.json'), '--out_dir', str(worker_dir / 'results'),
        '--method_name', method, '--pillars', '1,2,3', '--skip_vlm', '--resume', '--device', 'cuda:0',
        '--vbench_cache_dir', '/root/autodl-tmp/models/entitybench_vbench']


def resource_snapshot(gpu):
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    row = next(line for line in text.splitlines() if int(line.split(',')[0]) == gpu)
    _, free, utilization = [int(x.strip()) for x in row.split(',')]
    snapshot = dict(gpu=gpu, free_vram_mib=free, gpu_utilization_percent=utilization, time_utc=datetime.now(timezone.utc).isoformat())
    for name in ['memory.current', 'memory.max', 'cpu.max']:
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            snapshot[name] = path.read_text().strip()
    return snapshot


def admit(args, workers):
    snapshot = resource_snapshot(args.gpu_id)
    assert snapshot['free_vram_mib'] >= workers * args.vram_per_worker_mib + 2048, 'Not enough free GPU headroom; defer or use a measured worker budget'
    quota = snapshot.get('cpu.max', 'max 100000').split()
    capacity = os.cpu_count() or 1
    if quota[0] != 'max':
        capacity = min(capacity, max(1, math.floor(int(quota[0]) / int(quota[1]))))
    assert workers * args.threads <= capacity, 'Workers × threads exceeds container CPU quota'
    if snapshot.get('memory.max', 'max') != 'max':
        free = int(snapshot['memory.max']) - int(snapshot['memory.current'])
        assert free >= (workers * 4 + 2) * 1024**3, 'Insufficient initial host-memory headroom'
    return snapshot


def import_scorer(root):
    path = root / 'runtime/evaluate_benchmark_partial.py'
    spec = importlib.util.spec_from_file_location('_skill_v2_entitybench', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.VBENCH_DIMENSIONS == DIMENSIONS
    return module


def validate_manifest(manifest):
    assert manifest['strict'] and not manifest['allow_partial'] and not manifest['allow_config_override']
    assert manifest['skip_vlm'] and manifest['evaluation_scope'] == 'partial_without_vlm'
    assert manifest['pillars'] == [1, 2, 3]
    assert manifest['vlm_call_stats']['total_calls'] == 0, 'VLM calls detected'


def aggregate(root, out, episodes, split, inputs, assignments, method):
    scorer = import_scorer(root)
    manifests, results, seen = [], [], set()
    for source, items in assignments:
        manifest = read(source / 'run_manifest.json')
        validate_manifest(manifest)
        if manifests:
            ok, differences = scorer.manifest_comparable(manifests[0], manifest)
            assert ok and manifests[0]['libraries'] == manifest['libraries'], differences
        manifests.append(manifest)
        for eid, tier in items:
            assert eid not in seen, f'Episode assigned twice: {eid}'
            cached = read(source / f'{eid}.json')
            assert cached['episode_id'] == eid and cached['tier'] == tier
            assert cached['num_shots'] == len(episodes[eid]['shots'])
            for dimension in DIMENSIONS:
                metric = cached['vbench_means'][dimension]
                assert metric['value'] is not None and math.isfinite(metric['value'])
                assert metric['n_evaluated'] == cached['num_shots'] and metric['n_failed'] == 0
            results.append(scorer._episode_result_from_json(eid, tier, cached))
            if source.resolve() != out.resolve():
                out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / f'{eid}.json', out / f'{eid}.json')
            seen.add(eid)
    assert seen == set(episodes), 'Incomplete episode coverage'
    report = scorer.aggregate_results(results, split)
    base = copy.deepcopy(manifests[0])
    base['method_name'] = method
    report['_manifest'] = base
    write(out / 'run_manifest.json', base)
    write(out / 'input_fingerprint.json', dict(fingerprint=fingerprint(inputs), inputs=inputs))
    write(out / f'report_{method}.json', report)
    scorer.write_report(report, method, str(out / f'report_{method}.md'))
    return report


def stop_children(children):
    for _, process in children:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while any(p.poll() is None for _, p in children) and time.monotonic() < deadline:
        time.sleep(.2)
    for _, process in children:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(args):
    assert args.workers > 0 and args.threads > 0 and args.gpu_id >= 0 and args.vram_per_worker_mib > 0
    root = args.output_root.resolve()
    out = (args.out_dir or root / 'evaluation/official_parallel_v2').resolve()
    assert out != (root / 'evaluation/official').resolve(), 'Preserve historical serial scores'
    assert out != root, 'Never use the run root as score directory'
    episodes, split, inputs = freeze_inputs(root)
    shards, loads = partition(episodes, split, args.workers)
    plan = dict(inputs=inputs, helper_sha256=sha(__file__), workers=len(shards), gpu=args.gpu_id,
        threads=args.threads, vram_per_worker_mib=args.vram_per_worker_mib, method=args.method_name,
        shards=shards, source_evaluation_dir=str(args.source_evaluation_dir.resolve()) if args.source_evaluation_dir else None,
        aggregate_only=args.aggregate_only)
    if args.dry_run:
        print(json.dumps(dict(worker_shot_counts=loads, gpu=args.gpu_id, threads_per_worker=args.threads,
            commands=[command(root, out / '.shards' / f'worker_{i:02d}', args.method_name) for i in range(len(shards))]), indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.parallel.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'parallel_plan.json').exists():
        assert read(out / 'parallel_plan.json') == json.loads(json.dumps(plan)), 'Changed input/worker plan: choose a fresh score directory'
    else:
        assert not (out / 'run_manifest.json').exists(), 'Scores already exist without this plan: choose a fresh directory'
        write(out / 'parallel_plan.json', plan)
    started, children = time.monotonic(), []
    def status(phase, **extra):
        write(out / 'parallel_status.json', dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - started, **extra))
    try:
        if args.aggregate_only:
            assert args.source_evaluation_dir, '--aggregate-only requires --source-evaluation-dir'
            source = args.source_evaluation_dir.resolve()
            assert source != out
            assert read(source / 'input_fingerprint.json')['inputs'] == inputs, 'Source scores have different media/config inputs'
            assignments = [(source, [(eid, tier) for tier in TIERS for eid in split.get(tier, [])])]
        else:
            assert not args.source_evaluation_dir, 'Use --aggregate-only with cached source results'
            status('admission')
            write(out / 'initial_resources.json', admit(args, len(shards)))
            env = environment(args.threads, args.gpu_id)
            for i, items in enumerate(shards):
                worker = out / '.shards' / f'worker_{i:02d}'
                write(worker / 'split.json', {tier: [eid for eid, t in items if t == tier] for tier in TIERS})
                write(worker / 'input_fingerprint.json', dict(full_inputs_fingerprint=fingerprint(inputs), episodes=[eid for eid, _ in items]))
                with (worker / 'worker.log').open('a') as log:
                    process = subprocess.Popen(command(root, worker, args.method_name), cwd=PROJECT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((i, process))
            status('running', workers=len(children))
            while any(p.poll() is None for _, p in children):
                failures = [(i, p.returncode) for i, p in children if p.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(f'Worker failure: {failures}; inspect preserved worker logs')
                write(out / 'latest_resources.json', resource_snapshot(args.gpu_id))
                time.sleep(5)
            assert all(p.returncode == 0 for _, p in children), 'A worker failed'
            assignments = [(out / '.shards' / f'worker_{i:02d}' / 'results', items) for i, items in enumerate(shards)]
        status('aggregating')
        assert freeze_inputs(root)[2] == inputs, 'Inputs changed during evaluation: refuse stale aggregation'
        aggregate(root, out, episodes, split, inputs, assignments, args.method_name)
        write(out / 'parallel_execution.json', dict(mode='cached_reaggregation_validation' if args.aggregate_only else 'parallel_inference',
            workers=0 if args.aggregate_only else len(shards), worker_shot_counts=loads, gpu=args.gpu_id,
            elapsed_seconds=time.monotonic() - started, helper_sha256=sha(__file__), zero_vlm_calls=True))
        status('official_non_vlm_complete_pending_visual_review', complete_episodes=len(episodes), complete_shots=sum(loads))
        print(f'Complete: {len(episodes)} episodes / {sum(loads)} shots; {out}', flush=True)
    except BaseException as error:
        stop_children(children)
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--vram-per-worker-mib', type=int, default=8192)
    parser.add_argument('--method-name', default='entitybench_official_non_vlm')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--source-evaluation-dir', type=Path)
    run(parser.parse_args())
