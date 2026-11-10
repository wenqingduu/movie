"""Versioned handoff: preserve active/completed motion cases; add short scene cues to remaining low-scene cases."""
from __future__ import annotations
import argparse, copy, fcntl, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path
import qwen140_all_zero_motion_20261108 as base

SOURCE_OUT = base.OUT
OUT = base.ROOT/'outputs/entitybench_all_zero_motion_background_20261108'
AUDIT = base.ROOT/'outputs/entitybench_cached_matching_audit_20261107/inventory.json'
OLD_PARENT = 29497
OLD_WORKER = 40556
INHERIT_THROUGH = 70
read, write, sha = base.read, base.write, base.sha


def scene_cue(case, episode, weak):
    if (case['episode_id'], case['shot']) not in weak:
        return '', [], 'baseline_scene_already_present_or_not_scheduled'
    shot = next(s for s in episode['shots'] if s['shot_key'] == case['shot'])
    descriptions = {v['name']: v['description'] for v in episode['entity_descriptions'].values()}
    context = []
    for name in shot['entity_schedule'].get('places', []):
        if name not in weak[case['episode_id'], case['shot']]:
            continue
        desc = descriptions.get(name, '').strip()
        # Featureless rooms / unspecified spaces offer no grounded visual anchors.
        if len(desc.split()) < 5 or re.search(r'featureless|non.?descript|no discernible|unspecified', desc, re.I):
            continue
        # Exact source excerpt, bounded length; this is context, not a request to add objects.
        clauses = re.split(r',\s*', desc)
        excerpt = clauses[0]
        for clause in clauses[1:]:
            if len((excerpt+', '+clause).split()) > 30:
                break
            excerpt += ', '+clause
        if len(excerpt.split()) > 36:
            continue
        context.append(dict(name=name, description=desc, excerpt=excerpt.rstrip('.')))
        if len(context) == 1:
            break
    if not context:
        return '', [], 'no_specific_compact_scene_description'
    item = context[0]
    cue = (f"Setting context: {item['name']} — {item['excerpt']}. "
           'Keep its already-visible background features recognizable within the specified camera movement and focus; do not reveal off-screen features.')
    return cue, context, 'low_scene_background_continuity'


def prepare():
    path = OUT/'plan.json'
    if path.exists():
        p = read(path)
        assert p['source_plan_sha256'] == sha(SOURCE_OUT/'plan.json')
        for source in p['code_sources']:
            assert sha(source['path']) == source['sha256'], source['path']
        return p
    previous = read(SOURCE_OUT/'plan.json')
    snapshot = read(base.SNAP)
    episodes = {e['episode_id']: e['episode'] for e in snapshot['episodes']}
    weak = {}
    for row in read(AUDIT)['rows']:
        if row['kind'] == 'location' and any(x['edition']=='current' and x['status']=='weak' for x in row['references']):
            weak.setdefault((row['episode_id'], row['shot']), []).append(row['name'])
    p = copy.deepcopy(previous)
    changes = []
    for index, case in enumerate(p['cases'], 1):
        original = previous['cases'][index-1]
        case['source_candidate_output'] = original['output']
        case['inherited_motion_candidate'] = index <= INHERIT_THROUGH
        cue, context, reason = ('', [], 'completed_or_active_chunk_preserved') if case['inherited_motion_candidate'] else scene_cue(case, episodes[case['episode_id']], weak)
        case['background_cue'], case['background_sources'], case['background_reason'] = cue, context, reason
        case['motion_only_prompt'] = original['new_prompt']
        if cue:
            case['new_prompt'] += '\n'+cue
            case['task']['prompt'] = case['new_prompt']
            assert [k for k in case['task'] if case['task'][k] != original['task'][k]] == ['prompt']
        else:
            assert case['task'] == original['task']
        case['output'] = str(OUT/'cases'/case['case_id']/'main')
        case['task_path'] = str(Path(case['output'])/'task.json')
        base.immutable(case['task_path'], case['task'])
        case['task_sha256'] = sha(case['task_path'])
        changes.append({k:case[k] for k in ['case_id','episode_id','shot','inherited_motion_candidate','background_reason','background_sources','motion_only_prompt','new_prompt']})
    p.update(created_utc=base.now(), source_plan=str(SOURCE_OUT/'plan.json'), source_plan_sha256=sha(SOURCE_OUT/'plan.json'),
        scope='Same1333 dynamic0 targets. Preserve70 pilot/completed/active-chunk cases; remaining1263 get the existing motion cue, with compact dataset scene context only for low-scene cases with specific descriptions. One candidate per target, no new frame/seed/model/route.',
        inherited_through=INHERIT_THROUGH, background_enhanced=sum(bool(c['background_cue']) for c in p['cases']),
        prompt_compiler='Original motion compiler unchanged. Append one <=36-word exact scene-description excerpt plus an already-visible background-continuity instruction for pending baseline-weak locations. No forced zoom-out, lighting change, new props, or off-screen reveal.',
        code_sources=previous['code_sources']+[dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))],
        handoff=dict(old_parent=OLD_PARENT,old_worker=OLD_WORKER,old_parent_paused=True,policy='Let current worker finish; preserve every completed case. No overlapping GPU generation.'))
    base.immutable(OUT/'plan.json', p)
    base.immutable(OUT/'selection_policy.json', previous['selection_policy'])
    base.immutable(OUT/'prompt_changes.json', changes)
    with (OUT/'prompts.jsonl').open('w') as file:
        for c in changes:
            file.write(base.json.dumps(c,ensure_ascii=False)+'\n')
    return p


def configure():
    base.OUT = OUT
    base.prepare = prepare
    base.__file__ = str(Path(__file__).resolve())


def inherit(p):
    source = read(SOURCE_OUT/'plan.json')
    for case, original in zip(p['cases'][:INHERIT_THROUGH], source['cases'][:INHERIT_THROUGH]):
        if case['reuse_result']:
            continue  # Original base.run materializes the six frozen pilot receipts.
        old = Path(original['output']).parent
        record = read(old/'result.json')
        assert sha(record['video']) == record['video_sha256']
        generation = read(old/'main/generation.json')
        assert generation['inputs'] == case['task']
        assert generation['condition_sha256'] == base.digest(case['task'])
        dest = Path(case['output']).parent
        base.immutable(dest/'result.json', {**record, 'case_id':case['case_id'],
            'plan_sha256':sha(OUT/'plan.json'), 'reused_motion_candidate':True,
            'reuse_source':str(old/'result.json'), 'reuse_source_sha256':sha(old/'result.json')})
        if (old/'vbench.json').exists():
            scores = read(old/'vbench.json')
            assert scores['before'] == case['baseline_scores']
            base.immutable(dest/'vbench.json', {**scores,'reuse_motion_score_source':str(old/'vbench.json'),
                'reuse_motion_score_sha256':sha(old/'vbench.json')})
    proof = read(SOURCE_OUT/'validation/receipt.json')
    assert proof['plan_sha256'] == sha(SOURCE_OUT/'plan.json') and proof['persistent_seed_reset_verified']
    base.immutable(OUT/'validation/receipt.json', {**proof,'plan_sha256':sha(OUT/'plan.json'),
        'validation_reused_from':str(SOURCE_OUT/'validation/receipt.json'),
        'validation_source_sha256':sha(SOURCE_OUT/'validation/receipt.json'),
        'reason':'Same persistent Wan code, inputs, model/settings. Prompt-only version does not change pipeline.'})


def process(pid):
    path = Path(f'/proc/{pid}')
    if not path.exists():
        return None
    stat = (path/'stat').read_text().split()
    if stat[2] == 'Z':
        return None
    return (path/'cmdline').read_bytes().replace(b'\0',b' ').decode()


def handoff_run():
    configure(); p = prepare()
    write(OUT/'status.json',dict(stage='waiting_for_current_motion_chunk_boundary',utc=base.now(),
        expected=1333,background_enhanced=p['background_enhanced'],inherited_through=INHERIT_THROUGH,
        source_worker=OLD_WORKER,api_calls=0,whole_episode_generation=False))
    while True:
        active = process(OLD_WORKER)
        if not active:
            break
        assert 'qwen140_all_zero_motion_20261108.py worker' in active
        time.sleep(15)
    # Require every output of the active batch before changing the scheduler.
    for case in p['cases'][6:INHERIT_THROUGH]:
        assert (Path(case['source_candidate_output']).parent/'result.json').exists(), case['case_id']
    old = process(OLD_PARENT)
    if old:
        assert 'qwen140_all_zero_motion_20261108.py run' in old
        assert Path(f'/proc/{OLD_PARENT}/stat').read_text().split()[2] == 'T'
        os.kill(OLD_PARENT,signal.SIGKILL)  # Idle stopped scheduler only; its GPU worker has exited.
    inherit(p)
    write(SOURCE_OUT/'handoff.json',dict(utc=base.now(),superseded_by=str(OUT),
        new_plan_sha256=sha(OUT/'plan.json'),inherited_through=INHERIT_THROUGH,
        original_plan_preserved=True,reason='User requested background cues together with motion improvement.'))
    # Ordinary dependent scoring job; no repair agent.
    cmd = (f'cd {base.ROOT} && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 '
           f'{base.EVAL} -u {Path(__file__).resolve()} wait-context >> {OUT}/context_queue.log 2>&1')
    subprocess.run(['tmux','new-session','-d','-s','qwen140_allzero_background_context_20261108',cmd],check=True)
    base.run()


def context(action):
    configure()
    import qwen140_all_zero_context_20261108 as ctx
    ctx.__file__ = str(Path(__file__).resolve())
    if action == 'scorer':
        import qwen140_vbench_verified_cache_20261105 as adapter
        adapter.CACHE = ctx.OUT/'vbench_cache_manifest.json'
        sys.argv.pop(1)
        if '--method_name' in sys.argv:
            sys.argv[sys.argv.index('--method_name')+1]='all_zero_motion_background_context_20261108'
        adapter.main()
    else:
        ctx.wait_run()

if __name__ == '__main__':
    action = sys.argv[1]
    if action in {'wait-context','scorer'}:
        context(action)
    else:
        parser=argparse.ArgumentParser()
        parser.add_argument('action',choices=['prepare','handoff-run','run','worker','evaluate'])
        parser.add_argument('--ids');parser.add_argument('--chunk');parser.add_argument('--validation',action='store_true')
        args=parser.parse_args()
        configure()
        if args.action=='prepare':
            p=prepare();print(base.json.dumps(dict(total=p['total'],inherited=p['inherited_through'],background_enhanced=p['background_enhanced'],output=str(OUT)),ensure_ascii=False))
        elif args.action=='handoff-run':handoff_run()
        elif args.action=='run':base.run()
        elif args.action=='worker':base.worker(args.ids.split(','),args.validation)
        else:base.evaluate(args.ids.split(','),args.chunk)
