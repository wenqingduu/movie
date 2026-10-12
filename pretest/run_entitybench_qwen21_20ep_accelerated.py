"""Resume the original frozen route; redirect only reference model placement after exact-output validation."""
import argparse
import json
from pathlib import Path

from prepare_entitybench_qwen21_20ep import OUTPUT,PROJECT,sha256,immutable

PROBE=PROJECT/'outputs/entitybench_qwen21_reference_acceleration_probe/report.json'
WORKER=PROJECT/'pretest/run_qwen21_gpu_resident_references.py'


def validate_probe(report):
    rows=report['results']['cuda']['rows']
    assert len(rows)==2 and len({r['seed'] for r in rows})==2
    assert all(r['exact_png_match'] and r['image_sha256']==r['expected_sha256'] for r in rows)
    assert report['results']['cuda']['inference_seconds']<report['baseline_inference_seconds']
    for case in report['baseline']:
        assert sha256(case['source'])==case['expected_sha256']


def reference_command(command):
    result=list(command)
    if 'pretest/run_qwen21_entitybench_20ep_images.py' in result and '--stage' in result and result[result.index('--stage')+1]=='references':
        result[result.index('pretest/run_qwen21_entitybench_20ep_images.py')]=str(WORKER)
    return result


def main(root):
    report=json.loads(PROBE.read_text());validate_probe(report)
    immutable(root/'runtime/reference_acceleration.json',dict(kind='placement_only_reference_acceleration_v1',
        mode='cuda',phase='references_only',probe_report=str(PROBE),probe_report_sha256=sha256(PROBE),
        worker_sha256=sha256(WORKER),entry_sha256=sha256(Path(__file__)),
        original_implementation_sha256=sha256(root/'runtime/qwen_implementation.json'),
        policy='Same weights, seeds, prompts, 40 steps, KV-cache flag and original references function; keep verified existing candidates; hardcut frames retain original sequential offload',
        validation='Two distinct formal candidate PNGs match SHA256 exactly under CUDA-resident placement'))
    import run_entitybench_qwen21_20ep as route
    original=route.logged
    def redirected(directory,label,command,env):
        return original(directory,label,reference_command(command),env)
    route.logged=redirected
    route.main(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUTPUT)
    main(p.parse_args().output_root.resolve())
