"""Placement-only acceleration: reuse the frozen reference algorithm with GPU-resident weights."""
import argparse
import json
from pathlib import Path

from prepare_entitybench_qwen21_20ep import OUTPUT, MODEL, sha256


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUTPUT);p.add_argument('--stage',choices=['references'],required=True)
    args=p.parse_args();root=args.output_root.resolve()
    execution=json.loads((root/'runtime/reference_acceleration.json').read_text())
    assert execution['mode']=='cuda' and execution['phase']=='references_only'
    assert execution['worker_sha256']==sha256(Path(__file__))
    assert execution['original_implementation_sha256']==sha256(root/'runtime/qwen_implementation.json')
    plan=json.loads((root/'preparation_plan.json').read_text())
    import torch
    from diffusers import QwenImage21Pipeline
    from run_qwen21_entitybench_20ep_images import references
    torch.set_num_threads(4)
    pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),dtype=torch.bfloat16,local_files_only=True)
    pipe.to('cuda')
    references(root,plan,pipe)
