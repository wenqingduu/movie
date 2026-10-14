"""Audited reference-alignment fallback; never alter frozen sampler or thresholds."""
from pathlib import Path
import gc
from prepare_entitybench_flux_portraitagent_photo_20ep import PROJECT,read,sha256,immutable

ALIGN_FAILURE='facexlib align face fail'

def call_single_with_reference_gate(core,args):
    try:
        core.run(args)
    except RuntimeError as error:
        if str(error)!=ALIGN_FAILURE:
            raise
        return dict(error_type=type(error).__name__,error=str(error),stage='PuLID reference identity embedding before image denoising')
    return None

def release_pulid_to_cpu(pulid):
    import torch
    device=torch.device('cpu')
    pulid.components_to_device(device)
    pulid.device=device;pulid.face_helper.device=device;pulid.face_helper.face_det.device=device
    pulid.face_helper.face_det.mean_tensor=pulid.face_helper.face_det.mean_tensor.to(device)
    pulid.face_helper.clean_all()
    gc.collect();torch.cuda.empty_cache()

def freeze_recovery(root):
    root=Path(root)
    files=['pretest/flux_reference_gate_recovery.py','pretest/run_flux_portraitagent_photo_20ep_images_resume.py',
        'pretest/run_entitybench_flux_portraitagent_photo_20ep_resume.py']
    protocol=dict(kind='reference_gate_recovery_v1',
        preparation_plan_sha256=sha256(root/'preparation_plan.json'),
        frozen_implementation_sha256=sha256(root/'runtime/flux_implementation.json'),
        sources={str(PROJECT/p):sha256(PROJECT/p) for p in files},
        trigger='RuntimeError with exact message facexlib align face fail from reference identity extraction',
        action='Use existing unconditioned FLUX fallback sampler; no identity binding/3D injection for this shot; keep all shots',
        unchanged='All frozen weights, source files, prompts, seeds, numerical settings, 3D assets and official evaluator',
        reuse='Verify and reuse existing completed frames without rewriting media or records',
        evaluation_scope='partial_without_vlm')
    immutable(root/'runtime/reference_gate_recovery.json',protocol)
    return protocol

def validate_recovery_record(root,record):
    recovery=record.get('reference_alignment_recovery')
    if recovery is None:
        return
    assert recovery['error']==ALIGN_FAILURE
    assert record['plugin_applied'] is False and record['source_kind']=='flux_native_reference_alignment_fallback_hardcut'
    assert recovery['policy_sha256']==sha256(Path(root)/'runtime/reference_gate_recovery.json')
    assert recovery['reference_sha256']==sha256(recovery['reference'])
