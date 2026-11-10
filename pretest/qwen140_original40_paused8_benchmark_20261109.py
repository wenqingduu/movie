"""Score only the eight completed priority videos while generation stays paused.

Reuse the frozen full-context evaluator unchanged in an isolated output.  This
entry point has no generation action and cannot resume the remaining queue.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import qwen140_original40_priority_context_20261109 as context

batch = context.batch
SOURCE = batch.OUT
OUT = SOURCE / 'paused8_benchmark_20261109'
IDS = ('M0010', 'M0012', 'M0069', 'M0074', 'M0189', 'M0022', 'M0045', 'M0079')
original_prepare = batch.prepare


def completed_plan():
    plan = copy.deepcopy(original_prepare())
    assert {p.parent.name for p in (SOURCE / 'cases').glob('*/generation.json')} == set(IDS)
    plan['cases'] = [c for c in plan['cases'] if c['case_id'] in IDS]
    assert len(plan['cases']) == len(IDS)
    for case in plan['cases']:
        folder = SOURCE / 'cases' / case['case_id']
        assert (folder / 'generation.json').exists() and (folder / 'scores.json').exists()
    plan.update(total=8, new_cases=8, scope='Only eight completed priority videos at user pause; '
                'remaining1269 ungenerated and paused. No generation or VLM. '
                'Keep prior28 and append only new dynamic0_to1; retain all declines.')
    return plan


def paused_status(stage, **kwargs):
    batch.write(SOURCE / 'benchmark_update_status.json', dict(stage=stage,
        utc=batch.core.now(), generation_paused=True, automatic_resume=False,
        generated=8, scored=8, **kwargs))


context.OUT = OUT
context.REGION = OUT / 'visible_region'
context.__file__ = str(Path(__file__).resolve())
batch.prepare = completed_plan
batch.status = paused_status


def rewrite_reports():
    for name in ('official_report.md', 'complete_report.md'):
        path = OUT / name
        if path.exists():
            text = path.read_text().replace('新增1277条尝试', '暂停时已完成8条尝试')
            text = text.replace('所有24尝试', '所有8条尝试')
            path.write_text(text)
    if (OUT / 'plan.json').exists():
        p = batch.read(OUT / 'plan.json')
        batch.write(OUT / 'paused_scope.json', dict(attempted=8, full_queue=1277,
            ungenerated=1269, generation_paused=True, automatic_resume=False,
            selected=p['target_shots'], cumulative_selected=28+p['target_shots'],
            source_plan_sha256=batch.sha(SOURCE / 'plan.json'),
            adapter_sha256=batch.sha(__file__), vlm_calls=0))
    if (OUT / 'complete_report.md').exists():
        receipt = batch.read(OUT / 'completion_validation.json')
        receipt['report_sha256'] = batch.sha(OUT / 'complete_report.md')
        batch.write(OUT / 'completion_validation.json', receipt)


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'scorer':
        import qwen140_vbench_verified_cache_20261105 as adapter
        adapter.CACHE = OUT / 'vbench_cache_manifest.json'
        sys.argv.pop(1)
        adapter.main()
    elif action == 'prepare':
        p = context.prepare()
        rewrite_reports()
        print({k:p[k] for k in ('affected_episodes', 'complete_context_shots', 'target_shots')})
    elif action == 'run':
        context.run()
        rewrite_reports()
    elif action == 'final':
        context.dino(True)
        rewrite_reports()
    else:
        raise ValueError(action)
