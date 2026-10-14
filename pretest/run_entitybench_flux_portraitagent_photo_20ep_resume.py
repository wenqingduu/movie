"""Resume the existing cohort with an explicit reference-alignment fallback audit."""
import argparse
import copy
from pathlib import Path
import run_entitybench_flux_portraitagent_photo_20ep as pipeline
import prepare_entitybench_flux_portraitagent_photo_20ep as preparation
from flux_reference_gate_recovery import freeze_recovery,validate_recovery_record

def main(root):
    root=Path(root).resolve();freeze_recovery(root)
    original_logged=pipeline.logged
    original_immutable=preparation.immutable

    def logged(run_root,label,command,env):
        if label=='flux_hardcuts':
            command=list(command)
            index=command.index('pretest/run_flux_portraitagent_photo_20ep_images.py')
            command[index]='pretest/run_flux_portraitagent_photo_20ep_images_resume.py'
        return original_logged(run_root,label,command,env)

    def immutable(path,value):
        if Path(path).name=='manifest.json':
            value=copy.deepcopy(value)
            value['reference_alignment_recovery']=dict(protocol_path=str(root/'runtime/reference_gate_recovery.json'),
                protocol_sha256=preparation.sha256(root/'runtime/reference_gate_recovery.json'),
                policy='Reference-incompatible shots use the declared unconditioned FLUX fallback, with explicit per-shot provenance')
            for ep in value['episodes']:
                for shot in ep['shots']:
                    if not shot['cut']:continue
                    path_record=shot['firstframe_source']['source_config']
                    validate_recovery_record(root,preparation.read(path_record))
        return original_immutable(path,value)

    pipeline.logged=logged;preparation.immutable=immutable
    try:
        pipeline.main(root)
    finally:
        pipeline.logged=original_logged;preparation.immutable=original_immutable

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,default=preparation.OUTPUT)
    args=parser.parse_args();main(args.output_root)
