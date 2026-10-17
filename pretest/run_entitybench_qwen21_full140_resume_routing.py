"""Resume original frozen Qwen pipeline with audited semantic gates and existing-product credit."""
import argparse
import fcntl
from pathlib import Path

import run_entitybench_qwen21_full140_resume_rgba as rgba
from prepare_entitybench_qwen21_full140 import OUTPUT, PROJECT, read
from prepare_entitybench_qwen21_20ep import write_json
from qwen140_routing_recovery import prepare_recovery, verify_contract, allocated_asset_bytes, resume_budget


def main(args):
    root = args.output_root.resolve()
    # No edits while any generator owns the existing repository-wide GPU lock.
    with (PROJECT / 'outputs/.entitybench_qwen140_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_recovery(root)
    original = rgba.route.resources

    def resources(output, startup_min_free_gib=50):
        verify_contract(output)
        result = original(output, startup_min_free_gib)
        result = resume_budget(result, read(output / 'runtime/storage_budget_approval.json'),
                               allocated_asset_bytes(output))
        write_json(output / 'admission.json', result)
        write_json(output / 'runtime/resume_storage_admission_20261016.json', result)
        return result

    rgba.route.resources = resources
    rgba.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--startup-min-free-gib', type=float, default=50)
    parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args())
