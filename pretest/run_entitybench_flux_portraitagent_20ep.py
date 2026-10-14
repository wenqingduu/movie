"""Background new-FLUX-reference 20episode pipeline; preserve old runs and frozen algorithms."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path
from prepare_entitybench_flux_portraitagent_20ep import OUTPUT,PROJECT,METHOD,prepare,freeze_implementation,freeze_manifest,read,sha256,write_json
from run_entitybench_a14b_full140 import asset_environment,prepare_assets,audit_and_concat,disk_guard,logged,state
from run_entitybench_a14b_4step_20ep import environment,verify_evaluation
from archive_entitybench_portraitflux_preliminary_results import render

IMAGE=str(PROJECT/'.venv/bin/python')
GEN='/root/autodl-tmp/envs/lightx2v-trial/bin/python'
EVAL='/root/autodl-tmp/envs/entitybench-eval/bin/python'
HELPER=Path('/root/.codex/skills/evaluate-face-injection-v2/scripts/evaluate_parallel.py')


def expected_budget(root):
    return dict(measured_qwen20_root=str(PROJECT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep'),
        new_artifacts_gib_range=[5,7],minimum_free_on_start_gib=12,reserve_gib=5,
        policy='No downloads, no cleanup; keep all historic media and unfinished FLUX full140 assets')


def main(root,max_reference_candidates=None):
    root=Path(root).resolve();plan=prepare(root)
    lock=(root/'.pipeline.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=environment();assets_env=asset_environment(root,env)
    # The background pipeline never loads .env.portrait or calls a cloud model.
    for mapping in (env,assets_env):
        for key in list(mapping):
            if any(word in key.upper() for word in ('DASHSCOPE','PORTRAIT_QWEN','ANTHROPIC','API_KEY','AUTH_TOKEN')):
                mapping.pop(key)
    os.environ.update({k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    try:
        state(root,'freeze_flux_native_20ep_inputs',total_episodes=20,total_shots=363)
        budget=expected_budget(root)
        if not (root/'references_status.json').exists():
            assert shutil.disk_usage(root).free>=budget['minimum_free_on_start_gib']*1024**3, 'Need 12GiB free at admission; retain artifacts, expand disk first'
        write_json(root/'runtime/storage_budget.json',dict(budget,observed_free_gib=shutil.disk_usage(root).free/1024**3))
        state(root,'verify_existing_flux_weights');freeze_implementation(root);render(root)
        if max_reference_candidates is None:
            admission=read(root/'review/style_admission.json')
            assert admission['status']=='accepted' and admission['preparation_plan_sha256']==sha256(root/'preparation_plan.json')
        disk_guard(root);state(root,'generating_flux_character_references',total_roles=112,candidate_images=192,face_asset_roles=40)
        command=[IMAGE,'-u','pretest/run_flux_portraitagent_20ep_images.py','--output-root',str(root),'--stage','references']
        if max_reference_candidates is not None:command.extend(['--max-candidates',str(max_reference_candidates)])
        logged(root,'flux_references',command,assets_env)
        if max_reference_candidates is not None:
            state(root,'reference_admission_sample_complete',sample_candidates=max_reference_candidates,total_roles=112)
            return
        eligible=[role for role in plan['roles'] if role['needs_face_asset'] and read(role['metadata'])['face_eligible']]
        prepare_assets(root,dict(plan,roles=eligible),assets_env,stages=('facelift','calibrate'))
        disk_guard(root);state(root,'generating_flux_hardcut_frames',total_hardcuts=191,face_asset_roles=len(eligible))
        logged(root,'flux_hardcuts',[IMAGE,'-u','pretest/run_flux_portraitagent_20ep_images.py','--output-root',str(root),'--stage','frames'],assets_env)
        state(root,'freeze_flux_native_generation_manifest');freeze_manifest(root)
        disk_guard(root);state(root,'generating_flux_native_a14b_videos',total_shots=363)
        logged(root,'generate_formal',[GEN,'-u','pretest/run_lightx2v_a14b_4step_continuation.py','--output-root',str(root)],env)
        disk_guard(root);state(root,'audit_and_concat');audit_and_concat(root)
        disk_guard(root);state(root,'evaluate_official_parallel_non_vlm',workers=2)
        logged(root,'official_evaluation',[EVAL,'-u',str(HELPER),'--output-root',str(root),'--workers','2','--gpu-id','0','--method-name',METHOD],env)
        out=root/'evaluation/official_parallel_v2';status=read(out/'parallel_status.json')
        assert status['complete_episodes']==20 and status['complete_shots']==363
        verify_evaluation(out,[e['episode_id'] for e in plan['episodes']],METHOD)
        assert render(root)
        state(root,'official_non_vlm_complete_pending_visual_review',complete_episodes=20,complete_shots=363)
        (root/'report.md').write_text('# FLUX原生素材＋PuLID-FLUX/v7＋4步A14B，20episode\n\n363镜头及官方非VLM评分完成；VLM调用0。\n\n'
            f'[官方报告](evaluation/official_parallel_v2/report_{METHOD}.md)；[三路线比较](official_comparison.md)。\n\n'
            '单人PuLID参考绑定；多人FLUX文生图，无多人身份绑定。新FLUX参考对应新FaceLift/标定。\n'
            '无VLM/fidelity gate，跨镜头分数暂定；人工视觉审查仍待完成。\n')
    except Exception as error:
        previous=read(root/'status.json') if (root/'status.json').exists() else dict(phase='preparing')
        state(root,'failed',failed_phase=previous['phase'],error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUTPUT)
    p.add_argument('--max-reference-candidates',type=int,help='Admission sample of new formal candidates; resume full pipeline without this option')
    args=p.parse_args();main(args.output_root,args.max_reference_candidates)
