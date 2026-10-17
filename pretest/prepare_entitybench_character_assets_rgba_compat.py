"""Delegate to the frozen asset helper; inject format repair only in FaceLift children."""
import os
import subprocess
from pathlib import Path
import prepare_entitybench_character_assets as core
from facelift_rgba_compat import sha,verify_contract

def main():
    verify_contract()
    original_build=core.build_facelift_asset
    def build(reference,output):
        original_run=subprocess.run
        def run(command,*args,**kwargs):
            if isinstance(command,list) and len(command)==3 and command[1]=='-c' and 'from inference import main;' in command[2] and 'main(input_dir=' in command[2]:
                command=list(command)
                command[2]=command[2].replace('from inference import main;',
                    'from pretest.facelift_rgba_compat import install; install(); from inference import main;',1)
                env=dict(kwargs['env'])
                env.update(ENTITYBENCH_RGBA_AUDIT_DIR=str(Path(output)/'rgba_compat'),
                    ENTITYBENCH_RGBA_SOURCE_REFERENCE=str(Path(reference).resolve()),
                    ENTITYBENCH_RGBA_SOURCE_SHA256=sha(reference))
                kwargs['env']=env
            return original_run(command,*args,**kwargs)
        subprocess.run=run
        try:
            result=original_build(reference,output)
            records=sorted((Path(output)/'rgba_compat').glob('*.json'))
            if records:result['rgba_format_compatibility']=[dict(path=str(p),sha256=sha(p)) for p in records]
            return result
        finally:subprocess.run=original_run
    core.build_facelift_asset=build
    core.main()

if __name__=='__main__':main()

