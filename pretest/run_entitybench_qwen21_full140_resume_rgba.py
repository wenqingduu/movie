"""Resume unchanged Qwen140 with a separately frozen FaceLift RGBA compatibility shim."""
import argparse
import json
import os
from pathlib import Path
import run_entitybench_qwen21_full140 as route
from prepare_entitybench_qwen21_full140 import OUTPUT,SOURCE,PROJECT,read
from facelift_rgba_compat import verify_contract

def reclaim_verified_weight_cache():
    # SHA auditing reads >150GiB. Clean file cache is reclaimable, not model RSS.
    # Drop only read-only verified weight pages before the strict scorer admission.
    paths=[Path(r['path']) for r in read(SOURCE/'runtime/qwen_implementation.json')['weights']]
    model=Path(read(SOURCE/'manifest.json')['model_path'])
    paths += [model/r['path'] for r in read(PROJECT/'outputs/model_downloads/wan22_i2v_a14b/manifest.json')['files']]
    lora=read(SOURCE/'generation_manifest.json')['lora_inventory']
    paths += [Path(lora['destination'])/r['path'] for r in lora['files']]
    for path in paths:
        with path.open('rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)

def main(args):
    os.environ['ENTITYBENCH_RGBA_RECOVERY_ROOT']=str(args.output_root.resolve())
    verify_contract()
    original_logged=route.run_logged
    def logged(root,label,command,env):
        command=list(command);env=dict(env)
        env['ENTITYBENCH_RGBA_RECOVERY_ROOT']=str(root)
        target='pretest/prepare_entitybench_character_assets.py'
        if target in command:
            command[command.index(target)]='pretest/prepare_entitybench_character_assets_rgba_compat.py'
        return original_logged(root,label,command,env)
    route.run_logged=logged
    original_audit=route.audit_reuse
    def audit(*pos,**kw):
        result=original_audit(*pos,**kw)
        assert result['weights_verified']
        reclaim_verified_weight_cache()
        return result
    route.audit_reuse=audit
    route.main(args)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=OUTPUT)
    p.add_argument('--startup-min-free-gib',type=float,default=50)
    p.add_argument('--prepare-only',action='store_true')
    main(p.parse_args())

