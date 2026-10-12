"""Durable Qwen reference/3D/hardcut/own-tail A14B route and official non-VLM comparison."""
from __future__ import annotations
import argparse
import copy
import json
import os
import subprocess
from pathlib import Path

from prepare_entitybench_qwen21_20ep import OUTPUT,MODEL,REVISION,BASELINE,PROJECT,prepare,freeze_manifest,immutable,sha256,write_json
from run_entitybench_a14b_full140 import asset_environment,prepare_assets,audit_and_concat,disk_guard,logged,state
from run_entitybench_a14b_4step_20ep import environment

QWEN='/root/autodl-tmp/qwen-image21-venv/bin/python'
GEN='/root/autodl-tmp/envs/lightx2v-trial/bin/python'
EVAL='/root/autodl-tmp/envs/entitybench-eval/bin/python'
HELPER=Path('/root/.codex/skills/evaluate-face-injection-v2/scripts/evaluate_parallel.py')
METHOD='qwen21_native_reference_a14b_4step_continuation_20ep_no_vlm'


def freeze_implementation(root):
    weights=[]
    for p in sorted(MODEL.rglob('*.safetensors')):
        meta=MODEL/'.cache/huggingface/download'/p.relative_to(MODEL).with_suffix(p.suffix+'.metadata')
        lines=meta.read_text().splitlines();assert lines[0]==REVISION
        actual=sha256(p);assert actual==lines[1],f'Changed model weight: {p}'
        weights.append(dict(path=str(p),sha256=actual,size=p.stat().st_size))
    assert len(weights)==7
    scripts=['pretest/prepare_entitybench_qwen21_20ep.py','pretest/run_qwen21_entitybench_20ep_images.py',
        'pretest/run_entitybench_qwen21_20ep.py','pretest/run_qwen_image21_shared_prefix_adaptive.py',
        'pretest/prepare_qwen_multiface_residual.py','multishot/multi_face_reference.py',
        'multishot/facelift_backend.py','multishot/facelift_pose_calibration.py',
        'pretest/run_entitybench_a14b_full140.py','pretest/prepare_entitybench_character_assets.py']
    libraries=Path('/root/autodl-tmp/qwen-image21-venv/lib/python3.10/site-packages/diffusers')
    files=[PROJECT/s for s in scripts]+[libraries/'pipelines/qwenimage21/pipeline_qwenimage21.py',
        libraries/'models/transformers/transformer_qwenimage21.py']
    immutable(root/'runtime/qwen_implementation.json',dict(model_revision=REVISION,weights=weights,
        sources={str(p):sha256(p) for p in files}))


def compare(root):
    source=BASELINE/'evaluation/official/report_pulid_a14b_4step_continuation_20ep_no_vlm.json'
    target=root/'evaluation/official_parallel_v2'/f'report_{METHOD}.json'
    # Full raw reports are retained: both are produced by the same pinned official aggregator.
    write_json(root/'official_comparison_sources.json',dict(baseline_path=str(source),baseline_sha256=sha256(source),
        qwen_path=str(target),qwen_sha256=sha256(target),cohort_sha256=sha256(root/'split_20ep.json'),
        scope='Same frozen episodes and A14B; different reference/firstframe systems; no plugin causal delta',
        baseline=json.loads(source.read_text()),qwen=json.loads(target.read_text())))
    first,second=json.loads(source.read_text()),json.loads(target.read_text())
    assert first['all']['num_episodes']==second['all']['num_episodes']==20
    assert first['all']['num_shots']==second['all']['num_shots']==363
    lines=['# 相同20episode的系统路线比较','',
        '两条路线使用相同A14B视频配置和官方非VLM评分；素材与首帧系统不同。',
        '无VLM/fidelity gate，跨镜头分数暂定；差值不表示插件因果收益。','',
        '| 范围 | 指标 | FLUX | Qwen | Qwen − FLUX | FLUX/Qwen有效episode |',
        '|---|---|---:|---:|---:|---|']
    for tier in ['all','easy','medium','hard']:
        for key in ['intra_character_presence','intra_object_presence','intra_location_presence',
                    'cs_face','cs_object','cs_transition_boundary',*first[tier]['vbench']]:
            a=first[tier]['vbench'].get(key,first[tier].get(key))
            b=second[tier]['vbench'].get(key,second[tier].get(key))
            if a is None or b is None:continue
            av,bv=a.get('mean'),b.get('mean')
            if av is None or bv is None:continue
            lines.append(f"| {tier} | {key} | {av:.4f} | {bv:.4f} | {bv-av:+.4f} | {a['n_episodes_evaluated']}/{b['n_episodes_evaluated']} |")
    (root/'official_comparison.md').write_text('\n'.join(lines)+'\n')


def main(root):
    root=Path(root).resolve();env=environment()
    os.environ.update({k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    try:
        state(root,'freeze_qwen_20ep_inputs');plan=prepare(root);disk_guard(root)
        state(root,'verify_existing_qwen_weights');freeze_implementation(root)
        assets_env=asset_environment(root,env)
        state(root,'generating_qwen_character_references',total_roles=112,candidate_images=192,face_asset_roles=40)
        logged(root,'qwen_references',[QWEN,'-u','pretest/run_qwen21_entitybench_20ep_images.py','--output-root',str(root),'--stage','references'],assets_env)
        eligible=[]
        for role in plan['roles']:
            if role['needs_face_asset'] and json.loads(Path(role['metadata']).read_text())['face_eligible']:
                eligible.append(role)
        # Never invoke the helper's SDXL reference generation on Qwen-native roles.
        prepare_assets(root,dict(plan,roles=eligible),assets_env,stages=('facelift','calibrate'))
        state(root,'generating_qwen_hardcut_frames',total_hardcuts=191)
        logged(root,'qwen_hardcuts',[QWEN,'-u','pretest/run_qwen21_entitybench_20ep_images.py','--output-root',str(root),'--stage','frames'],assets_env)
        state(root,'freeze_qwen_generation_manifest');freeze_manifest(root)
        state(root,'generating_qwen_a14b_videos',total_shots=363)
        logged(root,'generate_formal',[GEN,'-u','pretest/run_lightx2v_a14b_4step_continuation.py','--output-root',str(root)],env)
        state(root,'audit_and_concat');audit_and_concat(root)
        state(root,'evaluate_official_parallel_non_vlm',workers=2)
        logged(root,'official_evaluation',[EVAL,'-u',str(HELPER),'--output-root',str(root),'--workers','2','--gpu-id','0','--method-name',METHOD],env)
        status=json.loads((root/'evaluation/official_parallel_v2/parallel_status.json').read_text())
        assert status['complete_episodes']==20 and status['complete_shots']==363
        compare(root)
        state(root,'official_non_vlm_complete_pending_visual_review',complete_episodes=20,complete_shots=363)
        (root/'report.md').write_text('# Qwen原生素材＋首帧＋A14B，20episode\n\n363镜头及官方非VLM评分完成。\n\n'
            f'[官方报告](evaluation/official_parallel_v2/report_{METHOD}.md)；[比较来源](official_comparison_sources.json)。\n\n'
            '无VLM/fidelity gate，跨镜头分数暂定；单组系统路线，人工视觉审查另记。\n')
    except Exception as error:
        previous=json.loads((root/'status.json').read_text())
        state(root,'failed',failed_phase=previous['phase'],error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=OUTPUT)
    main(p.parse_args().output_root)
