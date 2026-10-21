"""Resume only final auditing, face extraction/review/scoring and archive."""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import time
from pathlib import Path

from prepare_entitybench_qwen21_full140 import OUTPUT, PROJECT, read
from prepare_entitybench_qwen21_20ep import immutable, sha256
import run_entitybench_qwen21_full140 as route

CONTRACT = 'runtime/posteval_recovery_20261019.json'


def freeze_recovery(root):
    path = root / CONTRACT
    sources = [Path(__file__).resolve(), PROJECT / 'pretest/audit_entitybench_qwen140_final.py',
               PROJECT / 'tests/test_qwen140_final_audit.py']
    if not path.exists():
        assert read(root / 'generation_report.json')['status'] == 'complete'
        backup = root / 'runtime/posteval_before_20261019'
        backup.mkdir(parents=True, exist_ok=True)
        for name in ['status.json', 'logs/official_audit.log']:
            target = backup / Path(name).name
            if not target.exists():
                shutil.copy2(root / name, target)
        preserved = [root / name for name in ['manifest.json', 'generation_report.json',
            'generation_manifest.json', 'audit/final_media.json', 'audit/validation.json',
            'runtime/full140_adapters.json', 'runtime/rgba_recovery_implementation.json',
            'runtime/face_routing_recovery_20261016.json']]
        out = root / 'evaluation/official_parallel_v2'
        preserved += sorted(out.glob('*.json')) + sorted(out.glob('*.md'))
        preserved += sorted((out / '.shards').glob('*/results/*.json'))
        immutable(path, dict(kind='qwen140_posteval_recovery_v1',
            sources={str(p): sha256(p) for p in sources},
            preserved={str(p): sha256(p) for p in preserved},
            policy='Read official value:null schema; resume after completed scoring; '
                   'no media generation or score rewriting; actual visual review before face scoring'))
    contract = read(path)
    assert set(contract['sources']) == {str(p) for p in sources}
    for group in ['sources', 'preserved']:
        for name, expected in contract[group].items():
            assert sha256(name) == expected, f'Changed post-evaluation dependency: {name}'


def main(root):
    root = root.resolve()
    lock = (PROJECT / 'outputs/.entitybench_qwen140_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    route.verify_adapters(root)
    route.guard(root)
    freeze_recovery(root)
    env = route.environment()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(PROJECT))
    for key in list(env):
        if any(token in key.upper() for token in ['API_KEY', 'OPENAI', 'AZURE', 'GEMINI', 'TOKEN_PLAN']):
            env.pop(key)
    env = route.shared.asset_environment(root, env)
    try:
        route.state(root, 'auditing_completed_official_non_vlm', complete_episodes=140, complete_shots=2491)
        route.run_logged(root, 'official_audit_schema_recovery', [route.EVAL, '-u',
            'pretest/audit_entitybench_qwen140_final.py', '--output-root', str(root)], env)
        route.run_logged(root, 'cuda_face_extract_recovery', [route.ASSET, '-c',
            'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'], env)
        route.state(root, 'extracting_face_only_crops', official_complete=True)
        route.run_logged(root, 'face_extract', [route.ASSET, '-u',
            'pretest/evaluate_entitybench_face_only_dino_qwen140.py', 'extract', '--output-root', str(root)], env)
        decisions = root / 'evaluation/face_only_dino/review/decisions.json'
        route.state(root, 'waiting_face_crop_visual_review', decisions=str(decisions), official_complete=True)
        print('Face extraction complete. Waiting for actual manifest-bound visual crop review: '
              + str(decisions), flush=True)
        while not decisions.exists():
            route.guard(root)
            time.sleep(30)
        route.run_logged(root, 'face_score', [route.EVAL, '-u',
            'pretest/evaluate_entitybench_face_only_dino_qwen140.py', 'score', '--output-root', str(root)], env)
        freeze_recovery(root)
        route.archive(root)
    except BaseException as error:
        previous = read(root / 'status.json')
        route.state(root, 'failed', failed_phase=previous.get('phase'), error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    main(parser.parse_args().output_root)
