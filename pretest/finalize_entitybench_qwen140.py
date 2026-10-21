"""Validate saved face scores after the publication-precision assertion and archive."""
import argparse
import fcntl
import shutil
from pathlib import Path

from prepare_entitybench_qwen21_full140 import OUTPUT, PROJECT, read
from prepare_entitybench_qwen21_20ep import immutable, sha256
from resume_entitybench_qwen140_posteval import freeze_recovery
import run_entitybench_qwen21_full140 as route


def main(root):
    root = root.resolve()
    lock = (PROJECT / 'outputs/.entitybench_qwen140_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    route.guard(root)
    route.verify_adapters(root)
    freeze_recovery(root)
    contract = root / 'runtime/face_audit_precision_recovery_20261019.json'
    if not contract.exists():
        backup = root / 'runtime/face_audit_before_20261019'
        backup.mkdir(exist_ok=True)
        for name in ['status.json', 'logs/face_score.log']:
            shutil.copy2(root / name, backup / Path(name).name)
        sources = [Path(__file__).resolve(), PROJECT / 'pretest/audit_entitybench_qwen140_face_final.py',
                   PROJECT / 'tests/test_qwen140_face_final_audit.py']
        preserved = [root / 'evaluation/face_only_dino' / name for name in
            ['report.json', 'plan.json', 'face_crop_manifest.json', 'frozen_review.json', 'review/decisions.json']]
        immutable(contract, dict(sources={str(p): sha256(p) for p in sources},
            preserved={str(p): sha256(p) for p in preserved},
            policy='Validate raw per-episode centroids and official rounded aggregates; no recomputation of embeddings or report scores'))
    for group in ['sources', 'preserved']:
        for path, expected in read(contract)[group].items():
            assert sha256(path) == expected, path
    try:
        route.state(root, 'auditing_face_only_publication_precision', official_complete=True)
        route.run_logged(root, 'face_final_audit_precision', [route.EVAL, '-u',
            'pretest/audit_entitybench_qwen140_face_final.py', '--output-root', str(root)], route.environment())
        route.state(root, 'archiving_completed_non_vlm', official_complete=True, face_only_complete=True)
        route.archive(root)
    except BaseException as error:
        previous = read(root / 'status.json')
        route.state(root, 'failed', failed_phase=previous.get('phase'), error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    main(parser.parse_args().output_root)
