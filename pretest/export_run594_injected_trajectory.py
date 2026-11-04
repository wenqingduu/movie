"""Faithful replay of the NEW run594 3:4 and reused 3:5 injected keyframes.

Only Treatment trajectories are exported. The unchanged v7 worker internally
generates its official Control to obtain the identical shared-prefix fork.
No new detection, 3D rendering, planner, prompt change or video/API generation.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path('/root/autodl-tmp/movie')
OUT = ROOT/'outputs/run594_new_injected_denoising_replay_20261101'
DELIVERY = ROOT/'outputs/run594_new_injected_denoising_images_20261101'
NEW = ROOT/'outputs/run594_i75_j98_shot34_35_prompt_fix_20261101'
PYTHON = '/root/autodl-tmp/qwen-image21-venv/bin/python'


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(p)


def status(state, **kw):
    write(OUT/'status.json', {'status': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **kw})


def validate(plan):
    for item in plan['frozen_inputs']:
        assert sha(item['copy']) == item['sha256'], item['copy']
        assert sha(item['source']) == item['sha256'], item['source']
    for item in plan['code']:
        assert sha(item['path']) == item['sha256'], item['path']


def prepare():
    if (OUT/'plan.json').exists():
        plan = read(OUT/'plan.json'); validate(plan); return plan
    OUT.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(OUT).free > 8*1024**3
    frozen = []

    def freeze(source, dest):
        source, dest = Path(source).resolve(), Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): assert sha(source) == sha(dest)
        else: shutil.copy2(source, dest)
        frozen.append({'source': str(source), 'copy': str(dest), 'sha256': sha(dest)})
        return str(dest)

    reused_result = Path(read(NEW/'shot_3_5/frame/target.json')['injection_result'])
    specs = [('3:4', NEW/'shot_3_4/frame'), ('3:5', reused_result.parent.parent)]
    shots = []
    for key, frame in specs:
        tag = 'shot_'+key.replace(':', '_'); inputs = OUT/'inputs'/tag
        task = read(frame/'target.task.json'); original = read(frame/'injection/result.json')
        spec = task['face_injection']; settings = spec['settings']
        assert task['prompt'] == original['prompt'] and task['seed'] == original['seed']
        assert settings['steps'] == 20 and settings['inject_start'] == 12
        assert original['injection_applied'] and spec['separate_conditioning_images']
        for item in spec['frozen_files']: assert sha(item['path']) == item['sha256'], item['path']
        for branch in ['control', 'treatment']:
            assert sha(original[branch]) == original[branch+'_sha256']
            freeze(original[branch], inputs/(branch+'.png'))
        assert sha(NEW/tag/'frame/target.png') == original['treatment_sha256']
        freeze(frame/'target.task.json', inputs/'original_task.json')
        freeze(frame/'injection/result.json', inputs/'original_result.json')
        prepared_source = Path(original['prepared']).parent
        for file in sorted(prepared_source.rglob('*')):
            if file.is_file(): freeze(file, inputs/'prepared'/file.relative_to(prepared_source))
        task = copy.deepcopy(task)
        for index, image in enumerate(task['images']):
            assert sha(image['path']) == image['sha256']
            image['path'] = freeze(image['path'], inputs/'conditioning'/f'image_{index+1}.png')
        for character in task['face_injection']['characters']:
            name = character['name']
            assert read(character['facelift_result'])['source_reference_sha256'] == sha(character['reference_image'])
            character['reference_image'] = freeze(character['reference_image'], inputs/'identities'/(name+'.png'))
            freeze(character['facelift_result'], inputs/'identities'/(name+'_facelift_result.json'))
        prepared = read(original['prepared'])
        for target in prepared['targets']:
            name = target['name']
            target['reference_image'] = str(inputs/'identities'/(name+'.png'))
            for field in ['harmonized_reference', 'mask']:
                target[field] = str(inputs/'prepared'/Path(target[field]).relative_to(prepared_source))
        write(inputs/'replay_prepared.json', prepared)
        write(inputs/'replay_task.json', task)
        (inputs/'qwen_prompt.txt').write_text(task['prompt'])
        (inputs/'official_prompt.txt').write_text(spec['official_prompt'])
        shots.append({'key': key, 'tag': tag, 'task': str(inputs/'replay_task.json'),
                      'prepared': str(inputs/'replay_prepared.json'), 'inputs': str(inputs),
                      'original_result': original, 'settings': settings})
    code_paths = [Path(__file__).resolve(), ROOT/'pretest/qwen140_export_clara_trajectory.py',
        ROOT/'pretest/run_qwen_image21_shared_prefix_adaptive.py',
        ROOT/'pretest/run_qwen_image21_multiface_residual.py', ROOT/'pretest/qwen140_v7_frame.py',
        ROOT/'pretest/qwen140_pilot_qwen.py',
        Path('/root/autodl-tmp/qwen-image21-venv/lib/python3.10/site-packages/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py'),
        Path('/root/autodl-tmp/qwen-image21-venv/lib/python3.10/site-packages/diffusers/models/transformers/transformer_qwenimage21.py')]
    code = []
    for i, file in enumerate(code_paths):
        freeze(file, OUT/'runtime/frozen_source'/f'{i}_{file.name}')
        code.append({'path': str(file), 'sha256': sha(file)})
    plan = {'shots': shots, 'frozen_inputs': frozen, 'code': code,
        'model': str(ROOT/'models/diffusion/qwen-image-2.1'),
        'model_revision': '790c92633540aa0cb11d9abf19eb46d861714758',
        'scope': 'Injected trajectory only. v7 internal official Control for exact shared fork is not delivered.',
        'shot35_provenance': 'Exact reused target generation uses its original historical conditioning tail, not the new 3:4 video tail. The new tail conditioned only the new video transition.',
        'preparation_mode': 'Reuse saved role matching, pose, aligned 3D guides, masks and 1/6-step schedules; no detection or rendering.'}
    write(OUT/'plan.json', plan); status('prepared', shots=[s['key'] for s in shots])
    return plan


def worker(key):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    import run_qwen_image21_shared_prefix_adaptive as v7
    from qwen140_pilot_qwen import make_pipe
    from qwen140_export_clara_trajectory import TrajectoryRecorder
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    plan = read(OUT/'plan.json'); validate(plan)
    item = next(s for s in plan['shots'] if s['key'] == key)
    dest = OUT/item['tag']; runtime = OUT/'runtime'/item['tag']
    if (dest/'complete.json').exists(): return
    assert not (runtime/'result.json').exists(), 'Partial replay exists; inspect it, do not overwrite.'
    dest.mkdir(parents=True, exist_ok=True)
    task = read(item['task']); spec = task['face_injection']; stored = read(item['prepared'])
    pipe = make_pipe('frames')
    args = SimpleNamespace(**item['settings'], overwrite=False)
    record = {'run': spec['run'], 'episode_id': task['episode_id'], 'shot_key': key,
              'seed': task['seed'], 'official_prompt': spec['official_prompt'],
              'qwen_prompt': task['prompt'], 'characters': spec['characters']}
    original_control, original_conditioning = v7._official_control_and_fork_latent, v7._prepare_conditioning
    original_prepare = v7.prepare_from_image

    def conditioning_record(row):
        return {**row, 'characters': [{'reference_image': image['path']} for image in task['images']]}

    def control(pipe, row, args): return original_control(pipe, conditioning_record(row), args)
    def conditioning(pipe, row, args): return original_conditioning(pipe, conditioning_record(row), args)

    def reuse_prepared(*, preview, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True); preview.save(output_dir/'pred_x0.png')
        prepared = copy.deepcopy(stored); prepared['preview'] = str(output_dir/'pred_x0.png')
        prepared['preview_source'] = 'Replay prefix; saved original role/pose/guide/mask reused unchanged'
        write(output_dir/'prepared.json', prepared); return prepared

    v7._official_control_and_fork_latent, v7._prepare_conditioning = control, conditioning
    v7.prepare_from_image = reuse_prepared
    observer = TrajectoryRecorder(pipe, v7, args.inject_start); observer.install()
    try:
        result = v7.run_shot(pipe, None, None, record, runtime, args)
    finally:
        observer.restore()
        v7._official_control_and_fork_latent, v7._prepare_conditioning = original_control, original_conditioning
        v7.prepare_from_image = original_prepare
    assert result['injection_applied'], result
    expected = {t['name']: t['active_injection_steps'] for t in item['original_result']['targets']}
    actual = {t['name']: t['active_injection_steps'] for t in result['targets']}
    assert actual == expected, (actual, expected)
    for index in range(args.inject_start):
        for suffix in ['z', 'x0', 'velocity']:
            observer.data['treatment_'+suffix][index] = observer.data['control_'+suffix][index]
    for group, count in [('treatment_z', 21), ('treatment_x0', 20), ('treatment_velocity', 20)]:
        assert set(observer.data[group]) == set(range(count)), group
    assert torch.equal(observer.data['control_z'][12], observer.data['treatment_z'][12])
    states = {k: v for k, v in observer.data.items() if k.startswith('treatment_')}
    states['injection_residual'] = {i: states['treatment_z'][i+1]-before
        for i, before in states['treatment_euler_before'].items()}
    tensor_path = runtime/'trajectory_states.pt'
    torch.save({'states': states, 'sigmas': observer.sigmas, 'rebuilt_noise': observer.initial}, tensor_path)
    images = []

    def decoded(tensor, relative, kind, **kw):
        p = dest/relative; p.parent.mkdir(parents=True, exist_ok=True)
        v7._decode(pipe, tensor.to(pipe._execution_device), args.width, args.height).save(p)
        images.append({'file': str(p.relative_to(dest)), 'kind': kind, 'sha256': sha(p), **kw})

    for group, kind in [('treatment_z', 'z'), ('treatment_x0', 'pred_x0')]:
        for index, tensor in sorted(states[group].items()):
            decoded(tensor, f'treatment/{kind}_{index:02d}.png', kind, index=index)
    for step in result['step_log']:
        if not step['injected']: continue
        i = step['step']
        decoded(states['treatment_euler_before'][i], f'injection/step_{i:02d}_euler_before.png',
                'Euler output before residual', step=i, active_characters=step['active_characters'])
        decoded(states['treatment_z'][i+1], f'injection/step_{i:02d}_actual_after.png',
                'Actual next latent after residual', step=i, active_characters=step['active_characters'])
    shutil.copy2(runtime/'treatment.png', dest/'treatment.png')
    for target in stored['targets']:
        role = dest/'guidance'/target['name']; role.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(target['harmonized_reference']).parent, role, dirs_exist_ok=True)
        shutil.copy2(target['reference_image'], role/'identity_reference.png')
    for kind, indices in [('z', [0,1,4,5,8,9,12,13,14,17,19,20]), ('pred_x0', [0,4,8,11,12,13,14,15,16,17,18,19])]:
        sheet = Image.new('RGB', (4*320, 3*204), '#151515'); draw = ImageDraw.Draw(sheet)
        for j, i in enumerate(indices):
            im = Image.open(dest/'treatment'/f'{kind}_{i:02d}.png')
            sheet.paste(ImageOps.contain(im, (320,180)), ((j%4)*320,(j//4)*204+24))
            draw.text(((j%4)*320+5,(j//4)*204+4), f'{key} injected / {kind}_{i}', fill='white')
        sheet.save(dest/f'treatment_{kind}_overview.jpg', quality=95)
    comparison = {}
    for branch in ['control', 'treatment']:
        a = np.asarray(Image.open(runtime/(branch+'.png')).convert('RGB')).astype(np.float32)
        b = np.asarray(Image.open(Path(item['inputs'])/(branch+'.png')).convert('RGB')).astype(np.float32)
        comparison[branch] = {'pixel_exact': bool(np.array_equal(a,b)), 'pixel_mae_0_255': float(np.abs(a-b).mean()),
            'original_sha256': item['original_result'][branch+'_sha256'], 'replay_sha256': sha(runtime/(branch+'.png'))}
    assert np.array_equal(np.asarray(Image.open(dest/'treatment.png')), np.asarray(Image.open(dest/'treatment/z_20.png')))
    a = np.asarray(Image.open(runtime/'prepared/pred_x0.png')).astype(np.float32)
    b = np.asarray(Image.open(Path(item['inputs'])/'prepared/pred_x0.png')).astype(np.float32)
    meta = {'shot_key': key, 'steps': 20, 'inject_start_zero_based': 12, 'injected_role_steps': actual,
        'z_definition': 'z_0 initial noise; z_n after n updates, including actual identity residual.',
        'pred_x0_definition': 'z_i-sigma_i*velocity_i; clean prediction BEFORE update i; not actual z_i.',
        'injection_definition': 'step k Euler-before -> residual -> actual-after z_(k+1).',
        'shared_fork_latent_exact': True, 'final_z20_equals_replay_final': True,
        'original_comparison': comparison, 'step12_preview_pixel_exact': bool(np.array_equal(a,b)),
        'step12_preview_pixel_mae': float(np.abs(a-b).mean()),
        'replay_not_recorded_during_original': True, 'control_trajectory_delivered': False,
        'conditioning_images': task['images'], 'source_task': item['task'], 'images': images,
        'states_file': str(tensor_path), 'states_sha256': sha(tensor_path),
        'step_log': result['step_log'], 'note': 'Use final and intermediate states from this same replay together if original pixel equality differs.'}
    write(dest/'trajectory_metadata.json', meta)
    write(dest/'complete.json', {'shot_key': key, 'main_trajectory_images': 41, 'injection_images': 12,
        'injected_role_steps': actual, 'original_comparison': comparison})
    print('Completed', key, json.dumps(comparison), flush=True)


def run():
    plan = prepare(); status('running', expected_shots=2)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    for item in plan['shots']:
        status('running', active_shot=item['key'], expected_shots=2)
        with (OUT/(item['tag']+'.log')).open('a') as f:
            subprocess.run([PYTHON,'-u',str(Path(__file__).resolve()),'worker','--shot',item['key']],
                cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    subprocess.run([PYTHON, '/root/.codex/skills/movie-firstframe-figure-assets/scripts/pack_trajectory.py',
                    '--source',str(OUT),'--out',str(DELIVERY)], check=True)
    for item in plan['shots']:
        for filename in ['trajectory_metadata.json','complete.json']:
            shutil.copy2(OUT/item['tag']/filename, DELIVERY/item['tag']/filename)
    text = '''# run594 新版3:4、3:5：含3D注入的真实去噪过程图

仅导出注入分支，不包含Control图片或Control轨迹。本次按原提示词、输入图顺序、seed、模型、20步采样与保存的3D引导/mask重放，不是当时已经记录的过程，也没有重新规划或重新渲染身份素材。

- shot_3_4：新版单Leo关键帧，Leo注入6步（零基step12–17）。
- shot_3_5：复用的原过渡目标帧，Leo6步、Marcus1步（Marcus仅step12）。为重放这张图，保持当时的历史条件尾帧；新的3:4尾帧只用于已完成的视频过渡，不能替换为图像重放的输入。
- 各shot的treatment/：21张z_00..z_20含噪状态解码图、20张pred_x0_00..pred_x0_19清晰预测图，均为原尺寸PNG。
- injection/：step12–17每步Euler输出（注入前）与实际下一状态（注入后），每shot12张。step k注入后的图是z_(k+1)。
- guidance/：各角色原参考、3D渲染、姿态对齐、颜色协调、mask及准备图。
- treatment_z_overview.jpg及treatment_pred_x0_overview.jpg：预览拼图；正式画图用对应独立PNG。
- prompts/：原始帧提示词、数据集描述。trajectory_metadata.json记录实际角色步数、SHA、原版与重放最终图像素差异。

清晰预测pred_x0不是实际含噪z状态，画图时需分开标注。重放最终图与z_20逐像素相同；是否与原图完全相同见下表。若存在差异，请将本次过程图与本次最终图配套使用。原结果未覆盖，也未调用视频API或重新评测。

内部原v7需生成官方共享前缀并捕获分叉状态，这不作为Control输出交付。原始CPU latent/velocity/sigma和残差张量保留在源目录runtime/shot_*/trajectory_states.pt，不放入图片包。

'''
    text += '| 镜头 | 原注入最终图像素完全复现 | MAE（0–255） |\n|---|---|---:|\n'
    for item in plan['shots']:
        m = read(OUT/item['tag']/'trajectory_metadata.json')['original_comparison']['treatment']
        text += f"| {item['key']} | {m['pixel_exact']} | {m['pixel_mae_0_255']:.8f} |\n"
    text += '\n流程遵循movie-firstframe-figure-assets skill（/root/.codex/skills/movie-firstframe-figure-assets/SKILL.md）。\n'
    (DELIVERY/'README.md').write_text(text); (OUT/'README.md').write_text(text)
    import zipfile
    archive = DELIVERY.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for p in sorted(DELIVERY.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(DELIVERY.parent))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    status('complete', completed_shots=[s['key'] for s in plan['shots']],
           delivery=str(DELIVERY), archive=str(archive))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare','run','worker'])
    parser.add_argument('--shot', choices=['3:4','3:5']); args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run':
        try: run()
        except Exception as e:
            status('stopped_with_exception', error=repr(e)); raise
    else:
        assert args.shot; worker(args.shot)
