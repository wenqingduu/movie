"""Scheduling-only adapter: freeze case order, preserve the existing v2 pipeline."""
from pathlib import Path
import json
import qwen140_prompt_v2_selective_only_20261104 as queue

PLAN=queue.OUT/'priority_plan_20261105.json'
CONTRACT=queue.OUT/'priority_contract_20261105.json'

def configure():
    hold=queue.OUT/'user_hold_after_snapshot_20261105.json'
    if hold.exists() and not queue.read(hold).get('generation_authorized_to_resume',False):
        raise RuntimeError('User hold: finish evaluation and wait for user decision before new generation.')
    contract=queue.read(CONTRACT)
    assert queue.sha(Path(__file__))==contract['entry_sha256']
    assert queue.sha(PLAN)==contract['plan_sha256']
    assert queue.sha(queue.OUT/'cases.json')==contract['original_cases_sha256']
    plan=queue.read(PLAN);original=queue.read(queue.OUT/'cases.json')
    def key(row):return row['episode_id'],row['shot']
    original_map={key(c):c for c in original}
    assert len(plan['cases'])==len(original_map)==215
    assert {key(c) for c in plan['cases']}==set(original_map)
    assert all(c==original_map[key(c)] for c in plan['cases'])
    original_read=queue.read
    def read(path):
        if Path(path).resolve()==(queue.OUT/'cases.json').resolve():return plan['cases']
        return original_read(path)
    queue.read=read
    original_status=queue.status
    def status(state,**extra):
        errors=original_read(queue.OUT/'errors.json') if (queue.OUT/'errors.json').exists() else []
        missing={(r['episode_id'],r['shot']) for r in errors if not (queue.folder(original_map[(r['episode_id'],r['shot'])])/'result.json').exists()}
        original_status(state,priority_plan=str(PLAN),historical_error_records=len(errors),unresolved_errored_shots=len(missing),**extra)
    queue.status=status
    original_write=queue.write
    def write(path,value):
        # Original first20 checkpoint describes the original order, not this one.
        if Path(path)==queue.OUT/'first20_checkpoint.json' and Path(path).exists():return
        original_write(path,value)
    queue.write=write

if __name__=='__main__':
    configure()
    queue.main()
