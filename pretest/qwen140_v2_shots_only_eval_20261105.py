"""All completed independent v2-v1 shot pairs, with full episode context.

No full-benchmark aggregate or media generation. Frozen108-score snapshot supplies
unchanged episode caches; only11 contexts changed by the next33 need inference.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import qwen140_badshot_full_evaluation_20261104 as full

ROOT, BASE = full.ROOT, full.BASE
OUT = ROOT / 'outputs/entitybench_v2_shots_only_comparison_20261105'
PRIOR = ROOT / 'outputs/entitybench_v2_completed_snapshot_20261105'
BATCH = ROOT / 'outputs/entitybench_prompt_v2_badshots_20261104'
read, write, sha, immutable = full.read, full.write, full.sha, full.immutable


def status(stage, **kw):
    from datetime import datetime, timezone
    write(OUT / 'status.json', dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **kw))


def link(src, dst):
    src, dst = Path(src).resolve(), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert dst.resolve() == src, dst
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())


def attach(source, eid):
    source = Path(source)
    link(source / (eid + '.json'), OUT / 'affected/results' / (eid + '.json'))
    for p in (source / '.work').glob(eid + '*'):
        link(p, OUT / 'affected/results/.work' / p.name)
    link(source / '.work/eval_artifacts' / eid, OUT / 'affected/results/.work/eval_artifacts' / eid)


def prepare():
    if (OUT / 'plan.json').exists():
        p = read(OUT / 'plan.json')
        for b in p['bindings']:
            assert sha(b['path']) == b['sha256'], b['path']
        return p
    OUT.mkdir(exist_ok=True)
    baseline = read(PRIOR / 'baseline_selection.json')
    immutable(OUT / 'baseline_selection.json', baseline)
    selected = {e['episode_id']: e for e in baseline['episodes']}
    previous = read(PRIOR / 'replacement_manifest.json')
    candidates = read(BATCH / 'cases.json') + read(BATCH / 'plan.json')['reused']
    targets = {}; cases = []
    for c in candidates:
        f = Path(c['output_root']) / 'episodes' / c['episode_id'] / ('shot_' + c['shot'].replace(':', '_')) / 'result.json'
        if not f.exists():
            continue
        r = read(f); key = c['episode_id'] + '/' + c['shot']
        assert sha(r['video']) == r['video_sha256'] and sha(r['last_frame']) == r['last_sha256']
        assert sha(c['video']) == baseline['videos'][key]['sha256']
        targets[key] = dict(record=r, result_path=str(f), result_sha256=sha(f))
        cases.append(c)
    assert len(targets) == 141
    episodes = [e for e in read(BASE / 'manifest.json')['episodes'] if e['episode_id'] in {k.split('/')[0] for k in targets}]
    data = dict(scope='Only141 completed independent shot pairs; full43-episode context, no benchmark aggregation.', videos={}, targets=targets)
    delta = []
    bindings = {}
    def bind(path):
        path = str(Path(path).resolve()); bindings[path] = dict(path=path, sha256=sha(path))
    for ep in episodes:
        eid = ep['episode_id']; changed = False; records = []
        bind(selected[eid]['selected_result'])
        for shot in ep['shots']:
            key = eid + '/' + shot['shot_key']; old = baseline['videos'][key]; t = targets.get(key)
            source = t['record']['video'] if t else old['source']; digest = t['record']['video_sha256'] if t else old['sha256']
            assert sha(source) == digest and sha(old['source']) == old['sha256']
            scene, number = map(int, shot['shot_key'].split(':'))
            canonical = OUT / 'candidate_media' / eid / f'{scene:03d}_{number:03d}.mp4'
            link(source, canonical)
            data['videos'][key] = dict(source=source, sha256=digest, canonical=str(canonical),
                before_source=old['source'], before_sha256=old['sha256'], replaced=bool(t))
            records.append(dict(shot_key=shot['shot_key'], cut=shot['cut'], video=source, video_sha256=digest))
            if key not in previous['videos'] or previous['videos'][key]['sha256'] != digest:
                changed = True
        immutable(OUT / 'episode_media/episodes' / eid / 'episode_result.json', dict(episode_id=eid, shots=records))
        if changed:
            delta.append(eid)
        else:
            attach(PRIOR / 'affected/results', eid)
    immutable(OUT / 'replacement_manifest.json', data)
    immutable(OUT / 'cases.json', cases)
    cache = dict(scope='Exact per-video six-dimension VBench reuse; all other scoring unchanged.',
        evaluator_sha256=sha(full.EVALUATOR), dimensions=full.DIMENSIONS, bindings=[], shots={})
    expected_manifest = read(BASE / 'evaluation/official_parallel_v2/run_manifest.json')
    source_manifests = set()
    for eid in delta:
        for shot in next(e for e in episodes if e['episode_id'] == eid)['shots']:
            key = eid + '/' + shot['shot_key']; video = data['videos'][key]
            if (PRIOR / 'affected/results' / (eid + '.json')).exists() and key in previous['videos'] and previous['videos'][key]['sha256'] == video['sha256']:
                source = PRIOR / 'affected/results' / (eid + '.json')
            elif baseline['videos'][key]['sha256'] == video['sha256']:
                source = Path(selected[eid]['selected_result'])
            else:
                cache['shots'][key] = dict(reuse=False, video_sha256=video['sha256'], scores={}, score_sources=[])
                continue
            manifest = source.parent / 'run_manifest.json'
            m = read(manifest)
            for field in ['config', 'python_version', 'libraries', 'model_checkpoints', 'pillars', 'evaluation_scope']:
                assert m[field] == expected_manifest[field], (eid, field)
            assert m['strict'] and m['skip_vlm'] and m['vlm_call_stats']['total_calls'] == 0
            source_manifests.add(str(manifest)); scores = {}; sources = []
            for dim in full.DIMENSIONS:
                f = source.parent / '.work' / (eid + '_' + dim + '_eval_results.json')
                name = Path(video['canonical']).stem
                values = [float(x['video_results']) for x in read(f)[dim][1] if Path(x['video_path']).stem == name]
                assert len(values) == 1 and math.isfinite(values[0])
                scores[dim] = values[0]; sources.append(dict(path=str(f), sha256=sha(f))); bind(f)
            bind(source)
            cache['shots'][key] = dict(reuse=True, video_sha256=video['sha256'], scores=scores, score_sources=sources)
    for f in source_manifests:
        bind(f)
    for f in [Path(__file__), full.EVALUATOR, PRIOR / 'replacement_manifest.json', PRIOR / 'baseline_selection.json',
              PRIOR / 'acceleration/validation.json', ROOT / 'pretest/qwen140_vbench_verified_cache_20261105.py',
              ROOT / 'pretest/qwen140_badshot_full_evaluation_20261104.py',
              ROOT / 'pretest/qwen140_badshot_reviewed_dino_20261104.py',
              ROOT / 'pretest/qwen140_rerun_region_diagnostic.py', OUT / 'replacement_manifest.json']:
        bind(f)
    cache['bindings'] = list(bindings.values())
    immutable(OUT / 'vbench_cache_manifest.json', cache)
    plan = dict(target_shots=len(targets), context_episodes=len(episodes),
        context_shots=sum(len(e['shots']) for e in episodes), episodes=episodes, recompute_episode_ids=delta,
        full_episode_caches=len(episodes) - len(delta), new_vbench_shots=sum(not x['reuse'] for x in cache['shots'].values()),
        reused_vbench_shots=sum(x['reuse'] for x in cache['shots'].values()), bindings=list(bindings.values()),
        scope='Target-shot means only, not an episode/full-benchmark aggregate; fixed v1 inputs, all completed v2 targets included.',
        paired_cache_validation=str(PRIOR / 'acceleration/validation.json'), workers=2, threads_per_worker=4)
    immutable(OUT / 'plan.json', plan)
    status('prepared', **{k:plan[k] for k in ['target_shots','context_episodes','context_shots','full_episode_caches','new_vbench_shots','reused_vbench_shots']})
    return plan


def restore_generation():
    proof = read(OUT / 'generation_pause.json'); restored = []
    for row in reversed(proof['parents']):
        p = Path(f"/proc/{row['pid']}/stat")
        if p.exists() and p.read_text().split(') ', 1)[1].split()[19] == row['start_ticks']:
            os.kill(row['pid'], signal.SIGCONT); restored.append(row['pid'])
    write(OUT / 'generation_resumed.json', dict(pids=restored, media_generation_requested_by_eval=False))


def wait_for_generation_boundary():
    proof = read(OUT / 'generation_pause.json')
    for row in proof['parents']:
        if ' episode ' not in row['cmd']:
            continue
        while True:
            children = Path(f"/proc/{row['pid']}/task/{row['pid']}/children").read_text().split()
            live = [p for p in children if Path(f'/proc/{p}/stat').exists() and
                    Path(f'/proc/{p}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z']
            if not live:
                break
            status('waiting_for_generation_worker_boundary', children=live); time.sleep(5)
    for _ in range(30):
        if not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            return
        time.sleep(2)
    raise RuntimeError('Existing GPU worker has not released its context')


def evaluate(plan):
    split = read(BASE / 'split_full140.json'); by = {e['episode_id']:e for e in plan['episodes']}
    pending = [eid for eid in plan['recompute_episode_ids'] if not (OUT / 'affected/results' / (eid + '.json')).exists()]
    shards = [[], []]; loads = [0, 0]
    for eid in sorted(pending, key=lambda e: (-len(by[e]['shots']), e)):
        i = min(range(2), key=lambda n: (loads[n], n)); shards[i].append(eid); loads[i] += len(by[eid]['shots'])
    env = full.environment(); jobs = []; started = time.monotonic()
    try:
        for i, eids in enumerate(shards):
            if not eids:
                continue
            d = OUT / 'workers' / str(i); d.mkdir(parents=True, exist_ok=True)
            immutable(d / 'split.json', {t:[e for e in split[t] if e in eids] for t in ['easy','medium','hard']})
            cmd = [full.EVAL, '-u', str(Path(__file__)), 'scorer', '--results_dir', str(OUT / 'candidate_media'),
                '--scripts_dir', str(ROOT / 'benchmarks/entitybench/data/scripts'), '--split_json', str(d / 'split.json'),
                '--out_dir', str(d / 'results'), '--method_name', 'v2_independent_shots_only_141',
                '--pillars', '1,2,3', '--skip_vlm', '--resume', '--device', 'cuda:0',
                '--vbench_cache_dir', '/root/autodl-tmp/models/entitybench_vbench']
            with (d / 'worker.log').open('a') as log:
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((p,d,eids))
        while any(p.poll() is None for p,_,_ in jobs):
            assert all(p.poll() in (None,0) for p,_,_ in jobs), 'Scoring worker failed'
            status('parallel_official_evaluation', elapsed_seconds=time.monotonic()-started,
                jobs=[dict(pid=p.pid, exit=p.poll(), episodes=len(es),
                    complete=sum((d/'results'/(e+'.json')).exists() for e in es)) for p,d,es in jobs])
            time.sleep(10)
        for p,d,eids in jobs:
            assert p.returncode == 0
            m = read(d / 'results/run_manifest.json'); base = read(BASE / 'evaluation/official_parallel_v2/run_manifest.json')
            for field in ['config','python_version','libraries','model_checkpoints','pillars','evaluation_scope']:
                assert m[field] == base[field]
            assert m['strict'] and m['skip_vlm'] and m['vlm_call_stats']['total_calls'] == 0
            for eid in eids:
                raw = read(d / 'results' / (eid + '.json'))
                assert raw['num_shots'] == len(by[eid]['shots'])
                for name in full.agg.LABELS:
                    assert full.agg.metric(raw,name)['n_failed'] == 0
                attach(d / 'results', eid)
    finally:
        for p,_,_ in jobs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM); p.wait()


def dino_prepare(plan):
    import qwen140_rerun_region_diagnostic as diag
    directory = OUT / 'visible_region'; items = []
    for ep in plan['episodes']:
        eid=ep['episode_id']
        if eid not in plan['recompute_episode_ids']:
            continue
        result = OUT / 'affected/results' / (eid + '.json')
        items.append(dict(episode_id=eid, episode=ep, output_root=str(OUT/'episode_media'),
            result=str(result), result_sha256=sha(result),
            artifacts=str(OUT/'affected/results/.work/eval_artifacts'/eid)))
    diag.immutable(directory/'input_manifest.json',dict(episodes=items, selected_episode_count=len(items),
        selected_shot_count=sum(len(i['episode']['shots']) for i in items), scope='Only changed episode contexts; exact previous extraction caches supply the other contexts.'))
    shutil.copyfile(PRIOR/'visible_region/required_span_validation.json', directory/'required_span_validation.json')
    diag.OUT=directory
    diag.extract('cuda')


def compare(plan):
    full.OUT=OUT
    status('comparing_target_shots')
    full.compare_shots(read(OUT/'replacement_manifest.json'))


def run():
    try:
        plan=prepare(); wait_for_generation_boundary(); evaluate(plan); compare(plan); dino_prepare(plan)
        status('official_complete_dino_extracted_pending_review', target_shots=141)
    except BaseException as exc:
        status('stopped_with_error', error=repr(exc)); raise
    finally:
        restore_generation()


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='scorer':
        import qwen140_vbench_verified_cache_20261105 as adapter
        adapter.CACHE=OUT/'vbench_cache_manifest.json'; sys.argv.pop(1); adapter.main()
    else:
        parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
        if args.action=='prepare':
            print(json.dumps({k:v for k,v in prepare().items() if k not in ['bindings','episodes']},indent=2))
        else:
            run()
