#!/usr/bin/env python3
"""Replay the two original Clara figure frames and export true denoising states."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/model_figure_clara_denoising_replay_20261029'
OLD = ROOT / 'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep'
PACK = ROOT / 'outputs/model_figure_handoff_clara_20261022'
EID = '00031e93-cf1a-30ed-97d1-fed61261cb9f__run354__i90_j105__T180'
FOCUS = ROOT / 'outputs/entitybench_qwen140_multiface_focus4_20261029'
PYTHON = '/root/autodl-tmp/qwen-image21-venv/bin/python'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def status(state, **kw):
    write(OUT / 'status.json', {'updated_utc': datetime.now(timezone.utc).isoformat(), 'status': state, **kw})


class TrajectoryRecorder:
    """Observe outputs without changing the solver or the injected updates."""
    def __init__(self, pipe, module, inject_start=12):
        self.pipe, self.module, self.inject_start = pipe, module, inject_start
        self.data = {name: {} for name in ['control_z', 'control_x0', 'control_velocity',
                                          'treatment_z', 'treatment_x0', 'treatment_velocity',
                                          'treatment_euler_before']}
        self.last_euler = None
        self.initial = None
        self.sigmas = None
        self.originals = {}

    def save(self, group, index, tensor):
        self.data[group][int(index)] = tensor.detach().cpu().clone()

    def install(self):
        original_step = self.pipe.scheduler.step
        original_predict = self.module._predict
        original_euler = self.module._euler_step
        original_decode = self.module._decode
        self.originals = dict(step=original_step, predict=original_predict,
                              euler=original_euler, decode=original_decode)

        def step(model_output, timestep, sample, *args, **kwargs):
            result = original_step(model_output, timestep, sample, *args, **kwargs)
            index = int(self.pipe.scheduler.step_index) - 1
            sigmas = self.pipe.scheduler.sigmas
            self.sigmas = sigmas.detach().cpu().clone()
            sigma = sigmas[index].to(device=sample.device, dtype=sample.dtype)
            self.save('control_z', index, sample)
            self.save('control_x0', index, sample - sigma * model_output)
            self.save('control_velocity', index, model_output)
            self.save('control_z', index + 1, result[0] if isinstance(result, tuple) else result.prev_sample)
            return result

        def predict(pipe, state, latent, index):
            prediction = original_predict(pipe, state, latent, index)
            if index >= self.inject_start:
                self.initial = state['initial'].detach().cpu().clone()
                self.sigmas = state['sigmas'].detach().cpu().clone()
                sigma = state['sigmas'][index].to(device=latent.device, dtype=latent.dtype)
                self.save('treatment_z', index, latent)
                self.save('treatment_x0', index, latent - sigma * prediction)
                self.save('treatment_velocity', index, prediction)
            return prediction

        def euler(sample, velocity, sigmas, index):
            result = original_euler(sample, velocity, sigmas, index)
            self.last_euler = index
            self.save('treatment_euler_before', index, result)
            return result

        def decode(pipe, latent, width, height):
            if self.last_euler is not None:
                self.save('treatment_z', self.last_euler + 1, latent)
            return original_decode(pipe, latent, width, height)

        self.pipe.scheduler.step = step
        self.module._predict, self.module._euler_step, self.module._decode = predict, euler, decode

    def restore(self):
        self.pipe.scheduler.step = self.originals['step']
        self.module._predict = self.originals['predict']
        self.module._euler_step = self.originals['euler']
        self.module._decode = self.originals['decode']


def prepare():
    if (OUT / 'plan.json').exists():
        return read(OUT / 'plan.json')
    OUT.mkdir(parents=True, exist_ok=True)
    settings = read(OLD / 'preparation_plan.json')['firstframe_settings']
    assert settings['steps'] == 20 and settings['inject_start'] == 12
    assert settings['offload'] == 'sequential' and not settings['evaluate_identity']
    snapshots = []
    def freeze(source, dest):
        source, dest = Path(source), Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        snapshots.append({'source': str(source), 'copy': str(dest), 'sha256': sha(dest)})
        return str(dest)
    reference = OLD / 'assets' / EID / 'characters/clara_reference.png'
    assert sha(reference) == sha(PACK / '02_identity/clara_reference.png')
    local_ref = freeze(reference, OUT / 'inputs/clara_reference.png')
    shots = []
    for key in ['4:1', '5:1']:
        tag = key.replace(':', '_')
        directory = OLD / 'firstframes' / EID / ('shot_' + tag)
        original = read(directory / 'result.json')
        frame = read(directory / 'frame_record.json')
        assert original['injection_applied'] and original['prompt'] == frame['qwen_prompt']
        assert sha(original['treatment']) == original['treatment_sha256'] == sha(PACK / '03_shots' / ('shot_' + tag) / 'final_first_frame.png')
        assert sha(original['control']) == original['control_sha256']
        prepared = read(original['prepared'])
        assert len(prepared['targets']) == 1 and prepared['targets'][0]['name'] == 'Clara'
        inputs = OUT / 'inputs' / ('shot_' + tag)
        for name in ['frame_record.json', 'result.json', 'prepared/prepared.json', 'prepared/pred_x0.png', 'control.png', 'treatment.png']:
            freeze(directory / name, inputs / name)
        target = prepared['targets'][0]
        target['reference_image'] = local_ref
        for field in ['harmonized_reference', 'mask']:
            target[field] = freeze(target[field], inputs / Path(target[field]).name)
        record = {'run': 'run354', 'episode_id': EID, 'shot_key': key, 'seed': frame['seed'],
                  'official_prompt': frame['official_prompt'], 'qwen_prompt': frame['qwen_prompt'],
                  'characters': [{'name': 'Clara', 'reference_image': local_ref,
                                  'facelift_result': target['facelift_result']}]}
        write(inputs / 'replay_prepared.json', prepared)
        (inputs / 'qwen_prompt.txt').write_text(frame['qwen_prompt'])
        (inputs / 'official_prompt.txt').write_text(frame['official_prompt'])
        shots.append({'record': record, 'original_result': original, 'prepared': str(inputs / 'replay_prepared.json'),
                      'original_preview': str(inputs / 'prepared/pred_x0.png'), 'input_dir': str(inputs)})
    code = [Path(__file__).resolve(), ROOT / 'pretest/run_qwen_image21_shared_prefix_adaptive.py',
            ROOT / 'pretest/run_qwen_image21_multiface_residual.py',
            Path('/root/autodl-tmp/qwen-image21-venv/lib/python3.10/site-packages/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py'),
            Path('/root/autodl-tmp/qwen-image21-venv/lib/python3.10/site-packages/diffusers/models/transformers/transformer_qwenimage21.py')]
    for index, path in enumerate(code):
        freeze(path, OUT / 'runtime/frozen_source' / (str(index) + '_' + path.name))
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'episode_id': EID,
            'settings': settings, 'shots': shots, 'frozen_inputs': snapshots,
            'frozen_code': [{'path': str(p), 'sha256': sha(p)} for p in code],
            'model': str(ROOT / 'models/diffusion/qwen-image-2.1'),
            'model_revision': '790c92633540aa0cb11d9abf19eb46d861714758',
            'original_model_provenance': str(OLD / 'runtime/qwen_implementation.json'),
            'preparation_mode': 'Replay original saved pose-aligned 3D guide and mask; no new identity asset or Agent planning',
            'queue_after': str(FOCUS), 'export': 'Every z_0..z_20, pred_x0_0..pred_x0_19, raw CPU tensors, Euler-before and actual-after injection, original/replayed image comparisons'}
    write(OUT / 'plan.json', plan)
    status('queued_after_episode_batch', expected_shots=2, queue_after=str(FOCUS))
    (OUT / 'index.html').write_text('<meta charset="utf-8"><h1>Clara两首帧去噪重放</h1><p>4:1、5:1；原prompt、seed、参考图、3D对齐参考与mask。排在当前55shot队列后，结果完成后更新。</p><a href="status.json">状态</a>')
    return plan


def validate(plan):
    for item in plan['frozen_inputs']:
        assert sha(item['copy']) == item['sha256'], item['copy']
    for item in plan['frozen_code']:
        assert sha(item['path']) == item['sha256'], item['path']


def worker(key):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    from diffusers import QwenImage21Pipeline
    import run_qwen_image21_shared_prefix_adaptive as v7
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    plan = read(OUT / 'plan.json')
    validate(plan)
    item = next(s for s in plan['shots'] if s['record']['shot_key'] == key)
    dest = OUT / ('shot_' + key.replace(':', '_'))
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    assert not (dest / 'result.json').exists(), 'Partial result exists; preserve it and inspect before resume'
    pipe = QwenImage21Pipeline.from_pretrained(plan['model'], dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_sequential_cpu_offload()
    args = SimpleNamespace(**plan['settings'], overwrite=False)
    stored_prepared = read(item['prepared'])
    original_prepare = v7.prepare_from_image
    def reuse_prepared(*, preview, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        preview.save(output_dir / 'pred_x0.png')
        prepared = copy.deepcopy(stored_prepared)
        prepared['preview'] = str(output_dir / 'pred_x0.png')
        prepared['preview_source'] = 'Regenerated original-prefix x0; original saved 3D pose/guide/mask replayed'
        write(output_dir / 'prepared.json', prepared)
        return prepared
    v7.prepare_from_image = reuse_prepared
    observer = TrajectoryRecorder(pipe, v7, args.inject_start)
    observer.install()
    try:
        result = v7.run_shot(pipe, None, None, item['record'], dest, args)
    finally:
        observer.restore()
        v7.prepare_from_image = original_prepare
    assert result['injection_applied'], result
    for index in range(args.inject_start):
        observer.data['treatment_z'][index] = observer.data['control_z'][index]
        observer.data['treatment_x0'][index] = observer.data['control_x0'][index]
        observer.data['treatment_velocity'][index] = observer.data['control_velocity'][index]
    assert set(observer.data['control_z']) == set(range(21))
    assert set(observer.data['treatment_z']) == set(range(21))
    assert set(observer.data['treatment_x0']) == set(range(20))
    assert torch.equal(observer.data['control_z'][12], observer.data['treatment_z'][12])
    tensor_path = dest / 'trajectory_states.pt'
    torch.save({'states': observer.data, 'sigmas': observer.sigmas, 'rebuilt_noise': observer.initial}, tensor_path)
    metadata = []
    for branch in ['control', 'treatment']:
        for kind, group in [('z', branch + '_z'), ('pred_x0', branch + '_x0')]:
            for index, value in sorted(observer.data[group].items()):
                path = dest / branch / f'{kind}_{index:02d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                v7._decode(pipe, value.to(pipe._execution_device), args.width, args.height).save(path)
                metadata.append({'branch': branch, 'kind': kind, 'index': index, 'path': str(path), 'sha256': sha(path)})
    for index, before in observer.data['treatment_euler_before'].items():
        folder = dest / 'injection'
        folder.mkdir(exist_ok=True)
        if result['step_log'][index]['injected']:
            for label, value in [('euler_before', before), ('actual_after', observer.data['treatment_z'][index + 1])]:
                path = folder / f'step_{index:02d}_{label}.png'
                v7._decode(pipe, value.to(pipe._execution_device), args.width, args.height).save(path)
    # These are the exact raw z previews, not a different x0-based substitute.
    chosen = [0, 1, 4, 5, 8, 9, 12, 13, 14, 19, 20]
    for branch in ['control', 'treatment']:
        sheet = Image.new('RGB', (5 * 256, 3 * 166), '#151515')
        draw = ImageDraw.Draw(sheet)
        for j, index in enumerate(chosen):
            im = Image.open(dest / branch / f'z_{index:02d}.png')
            sheet.paste(ImageOps.contain(im, (256, 144)), ((j % 5) * 256, (j // 5) * 166 + 20))
            draw.text(((j % 5) * 256 + 4, (j // 5) * 166 + 3), f'{branch} z_{index}', fill='white')
        sheet.save(dest / f'{branch}_trajectory.jpg', quality=94)
    comparison = {}
    for name in ['control', 'treatment']:
        new = np.array(Image.open(dest / (name + '.png')).convert('RGB')).astype(np.float32)
        old = np.array(Image.open(Path(item['input_dir']) / (name + '.png')).convert('RGB')).astype(np.float32)
        comparison[name] = {'pixel_mae_0_255': float(np.abs(new - old).mean()), 'pixel_exact': bool(np.array_equal(new, old)),
                            'original_sha256': item['original_result'][name + '_sha256'], 'replay_sha256': sha(dest / (name + '.png'))}
    assert np.array_equal(np.array(Image.open(dest / 'treatment.png')), np.array(Image.open(dest / 'treatment/z_20.png')))
    write(dest / 'trajectory_metadata.json', {'steps': 20, 'inject_start_zero_based': 12, 'z_index_definition': 'z_0 initial noise; z_n after n Euler updates, including actual injection when active',
        'x0_definition': 'z_i - sigma_i * velocity_i before Euler update; clean prediction, not actual z_i',
        'states_file': str(tensor_path), 'states_sha256': sha(tensor_path), 'images': metadata,
        'shared_fork_latent_exact': True, 'original_comparison': comparison,
        'preparation_mode': plan['preparation_mode'], 'figure_reuse_note': 'If replay differs from original, use final and intermediate frames from this same replay together.'})
    write(dest / 'complete.json', {'shot_key': key, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'images': len(metadata), 'original_comparison': comparison})
    print('Completed trajectory', key, comparison, flush=True)


def busy_focus():
    if (FOCUS / 'status.json').exists():
        state = read(FOCUS / 'status.json')['status']
        if state not in {'complete_non_vlm_pending_video_visual_review', 'batch_finished_with_recorded_exceptions', 'batch_setup_exception_recorded'}:
            return True
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [v.decode(errors='replace') for v in (entry / 'cmdline').read_bytes().split(b'\0') if v]
            if (args and Path(args[0]).name.startswith('python')
                    and any(Path(v).name == 'qwen140_multiface_focus_episode_batch.py' for v in args[1:])
                    and 'run' in args[1:]):
                return True
        except (OSError, ProcessLookupError):
            continue
    return False


def queue():
    plan = read(OUT / 'plan.json')
    validate(plan)
    with (OUT / '.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while busy_focus():
            status('queued_after_episode_batch', expected_shots=2, queue_after=str(FOCUS))
            time.sleep(30)
        status('running', expected_shots=2)
        errors = []
        for item in plan['shots']:
            key = item['record']['shot_key']
            status('running', active_shot=key, expected_shots=2)
            env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
            log = OUT / ('shot_' + key.replace(':', '_') + '.log')
            with log.open('a') as f:
                done = subprocess.run([PYTHON, '-u', str(Path(__file__).resolve()), 'worker', '--shot', key], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if done.returncode:
                errors.append({'shot': key, 'returncode': done.returncode, 'log': str(log)})
                write(OUT / 'exceptions.json', errors)
        completed = [s['record']['shot_key'] for s in plan['shots'] if (OUT / ('shot_' + s['record']['shot_key'].replace(':', '_')) / 'complete.json').exists()]
        status('complete' if len(completed) == 2 else 'finished_with_recorded_exceptions', expected_shots=2, completed_shots=completed, exceptions=errors)
        panels = []
        for key in completed:
            tag = 'shot_' + key.replace(':', '_')
            panels.append(f'<h2>{html.escape(key)}</h2><p><a href="{tag}/treatment.png">新首帧</a> · <a href="{tag}/trajectory_metadata.json">逐步清单</a> · <a href="{tag}/trajectory_states.pt">原始latent</a></p><img src="{tag}/treatment_trajectory.jpg" style="max-width:100%">')
        (OUT / 'index.html').write_text('<meta charset="utf-8"><h1>Clara原始两首帧去噪重放</h1><p>真实z状态解码图和pred_x0预测分别保存。原版与重放是否像素一致见各镜头metadata；作图请配套使用同一次重放的中间状态与最终图。</p>' + ''.join(panels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'queue', 'worker'])
    parser.add_argument('--shot', choices=['4:1', '5:1'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'queue':
        queue()
    else:
        assert args.shot
        worker(args.shot)
