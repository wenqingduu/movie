"""Isolated placement-only Qwen reference benchmark; never overwrite formal media."""
import argparse
import json
import time
from pathlib import Path

from prepare_entitybench_qwen21_20ep import MODEL, sha256, write_json


def main(args):
    import torch
    from diffusers import QwenImage21Pipeline
    torch.set_num_threads(4)
    cases=json.loads(args.cases.read_text())
    begin=time.monotonic()
    pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),dtype=torch.bfloat16,local_files_only=True)
    if args.mode=='cuda':pipe.to('cuda')
    elif args.mode=='model':pipe.enable_model_cpu_offload()
    else:pipe.enable_sequential_cpu_offload()
    rows=[];loaded=time.monotonic()
    for case in cases:
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        image=pipe(prompt=case['prompt'],width=1024,height=1024,num_inference_steps=40,
            generator=torch.Generator('cuda').manual_seed(case['seed']),true_cfg_scale=1.0,
            output_resolution=1024,use_kv_cache=True).images[0].convert('RGB')
        torch.cuda.synchronize()
        path=args.out/case['name'];path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        import hashlib
        rows.append(dict(name=case['name'],seed=case['seed'],elapsed_seconds=time.monotonic()-started,
            image_sha256=sha256(path),pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            expected_sha256=case['expected_sha256'],exact_png_match=sha256(path)==case['expected_sha256'],
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))
        write_json(args.out/'results.json',dict(mode=args.mode,model_load_seconds=loaded-begin,
            wall_seconds=time.monotonic()-begin,inference_seconds=sum(r['elapsed_seconds'] for r in rows),rows=rows))
        print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['sequential','model','cuda'],required=True)
    main(p.parse_args())
