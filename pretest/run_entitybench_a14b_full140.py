"""Durable full-cohort assets -> hard cuts -> own-tail 4-step rollout -> parallel official scores."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from prepare_entitybench_a14b_full140 import OUTPUT, SOURCE, PROJECT, prepare, freeze_ready_manifest, freeze_face_routing, injectable_hardcut, slug, sha256, write_json
from run_entitybench_a14b_4step_20ep import environment as base_environment

HELPER = Path('/root/.codex/skills/evaluate-face-injection-v2/scripts/evaluate_parallel.py')
EVAL_PYTHON = '/root/autodl-tmp/envs/entitybench-eval/bin/python'
GEN_PYTHON = '/root/autodl-tmp/envs/lightx2v-trial/bin/python'
ASSET_PYTHON = str(PROJECT / '.venv/bin/python')
METHOD = 'pulid_a14b_4step_continuation_full140_no_vlm'


def state(root, phase, **extra):
    write_json(root / 'status.json', dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(),
        vlm_enabled=False, evaluation_scope='partial_without_vlm', **extra))


def disk_guard(root):
    assert shutil.disk_usage(root).free >= 5 * 1024**3, 'Less than 5GiB free; expand storage and resume, retaining all outputs'


def logged(root, label, command, env):
    path = root / 'logs' / f'{label}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Starting {label}', flush=True)
    with path.open('a') as stream:
        subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def asset_ready(role):
    result_path, reference = Path(role['result']), Path(role['reference'])
    if not reference.is_file() or not result_path.is_file():
        return False
    result = json.loads(result_path.read_text())
    model = Path(result['model_path'])
    calibration = model.with_suffix('.pose_calibration.json')
    if result.get('facelift_status') != 'success' or not model.is_file() or not calibration.is_file():
        return False
    assert result.get('source_reference_sha256') == sha256(reference), 'Changed identity reference after FaceLift'
    data = json.loads(calibration.read_text())
    assert data['gaussian_model_sha256'] == sha256(model), 'Calibration targets a different Gaussian'
    return bool(data.get('profiles'))


def asset_environment(root, env):
    # Bound InsightFace's actual ONNX session pools, including calibration children.
    code = '''import os
if os.environ.get("ENTITYBENCH_ORT_THREADS"):
    import onnxruntime as ort
    from insightface.model_zoo.model_zoo import PickableInferenceSession
    original = PickableInferenceSession.__init__
    def bounded(self, *args, **kwargs):
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ["ENTITYBENCH_ORT_THREADS"])
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        kwargs["sess_options"] = options
        original(self, *args, **kwargs)
    PickableInferenceSession.__init__ = bounded
'''
    path = root / 'runtime/sitecustomize.py'
    if path.exists():
        assert path.read_text() == code
    else:
        path.write_text(code)
    assets = dict(env)
    assets.update(PYTHONPATH=str(root / 'runtime') + os.pathsep + str(PROJECT), ENTITYBENCH_ORT_THREADS='2',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        MULTISHOT_ALLOW_FACELIFT_FALLBACK='0', MULTISHOT_DISABLE_FACELIFT='0', MULTISHOT_FACELIFT_STEP_2D='50')
    return assets


def prepare_assets(root, plan, env, stages=('all',)):
    stop = threading.Event()
    def task(role):
        label = role['episode_id'] + '__' + slug(role['character'])
        path = root / 'asset_status' / f'{label}.json'
        if stop.is_set():
            return dict(character=role['character'], episode_id=role['episode_id'], status='not_started')
        row = dict(character=role['character'], episode_id=role['episode_id'], status='preparing')
        write_json(path, row)
        try:
            if asset_ready(role):
                row['status'] = 'ready_existing'
            else:
                disk_guard(root)
                begin = time.monotonic()
                for stage in stages:
                    logged(root, 'assets/' + label + ('_' + stage if stage != 'all' else ''), [ASSET_PYTHON, '-u', 'pretest/prepare_entitybench_character_assets.py',
                        '--episode', role['episode_path'], '--asset-root', role['asset_root'], '--stage', stage,
                        '--character-scope', 'single-shot', '--only-character', role['character'],
                        '--base-seed', str(role['seed']), '--candidates', '3', '--size', '1024', '--steps', '30',
                        '--guidance', '5.0', '--calibration-size', '512', '--python', ASSET_PYTHON], env)
                assert asset_ready(role), f'Incomplete asset after preparation: {label}'
                row.update(status='ready', elapsed_seconds=time.monotonic() - begin)
        except Exception as error:
            row.update(status='failed', error=repr(error))
            stop.set()  # Avoid hundreds of repetitions of a shared runtime failure.
        write_json(path, row)
        return row
    rows = []
    state(root, 'preparing_character_assets', total_roles=len(plan['roles']), asset_workers=2, ready_roles=0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(task, role) for role in plan['roles']]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            ready = sum(r['status'] in {'ready', 'ready_existing'} for r in rows)
            state(root, 'preparing_character_assets', total_roles=len(plan['roles']), ready_roles=ready,
                last_role=row, failures=[r for r in rows if r['status'] == 'failed'])
    write_json(root / 'asset_preparation_report.json', dict(roles=rows))
    assert all(r['status'] in {'ready', 'ready_existing'} for r in rows), 'Asset preparation failed; inspect asset_status and logs, then resume'


def prepare_frames(root, plan, env, routing):
    neutral = json.loads((SOURCE / 'asset_manifest.json').read_text())['items']
    neutral_path = next(r['reference_image'] for r in neutral if r.get('reference_image'))
    # Zero-weight carrier only: it does not bind any new episode's identities.
    for episode in plan['episodes']:
        if episode.get('reuse_generation_root'):
            continue
        disk_guard(root)
        state(root, 'preparing_hardcut_frames', current_episode=episode['episode_id'])
        single = [s['shot_key'] for s in episode['shots'] if injectable_hardcut(episode, s, routing)]
        basic = [s['shot_key'] for s in episode['shots'] if s['cut'] and not injectable_hardcut(episode, s, routing)]
        if single:
            command = [ASSET_PYTHON, '-u', 'pretest/prepare_entitybench_pulid_pairs.py', '--episode', episode['episode_path'],
                '--output-dir', episode['firstframe_root'], '--asset-root', episode['asset_root'],
                '--base-seed', str(episode['base_seed']), '--width', '640', '--height', '352', '--guidance', '4.0',
                '--pulid-id-weight', '1.0', '--injection-strength', '0.4', '--max-active-injection-steps', '12',
                '--min-injection-face-height', '24', '--max-face-detection-retries', '4', '--onnx-provider', 'cpu',
                '--shared-model-process', '--allow-selected-only']
            for key in single:
                command.extend(['--shot', key])
            logged(root, 'frames/' + episode['episode_id'] + '_single', command, env)
        if basic:
            command = [ASSET_PYTHON, '-u', 'pretest/prepare_entitybench_flux_controls.py', '--episode', episode['episode_path'],
                '--output-dir', str(Path(episode['firstframe_root']) / 'basic'), '--base-seed', str(episode['base_seed']),
                '--neutral-reference', neutral_path, '--width', '640', '--height', '352', '--steps', '50', '--guidance', '4.0']
            if any(s['shot_key'] in basic and len(s['entity_schedule'].get('characters', [])) == 1 for s in episode['shots']):
                command.append('--allow-single-basic')
            for key in basic:
                command.extend(['--shot', key])
            logged(root, 'frames/' + episode['episode_id'] + '_basic', command, env)


def parallel_probe(root, env):
    pilot = root / 'parallel_probe'
    validation = pilot / 'validation.json'
    if validation.exists():
        checked = json.loads(validation.read_text())
        assert checked['status'] == 'passed' and checked['helper_sha256'] == sha256(HELPER)
        return
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    manifest['episodes'] = manifest['episodes'][:2]
    ids = {e['episode_id'] for e in manifest['episodes']}
    manifest['totals']['episode_count'], manifest['totals']['shot_count'] = 2, sum(len(e['shots']) for e in manifest['episodes'])
    write_json(pilot / 'manifest.json', manifest)
    report = json.loads((SOURCE / 'generation_report.json').read_text())
    report['jobs'] = [r for r in report['jobs'] if r['episode_id'] in ids]
    write_json(pilot / 'generation_report.json', report)
    write_json(pilot / 'split_20ep.json', {tier: [e['episode_id'] for e in manifest['episodes'] if e['difficulty'] == tier]
        for tier in ['easy', 'medium', 'hard']})
    for folder in ['runtime', 'official_results']:
        path = pilot / folder
        if not path.exists():
            path.symlink_to(SOURCE / folder, target_is_directory=True)
    timings = {}
    state(root, 'profiling_official_parallel_evaluation', sample_episodes=2, sample_shots=len(report['jobs']))
    for workers in [1, 2]:
        out = pilot / f'eval_workers{workers}'
        logged(root, f'parallel_probe_workers{workers}', [EVAL_PYTHON, '-u', str(HELPER), '--output-root', str(pilot),
            '--out-dir', str(out), '--workers', str(workers), '--gpu-id', '0', '--threads', '4', '--method-name', 'parallel_probe'], env)
        timings[str(workers)] = json.loads((out / 'parallel_execution.json').read_text())['elapsed_seconds']
    for eid in ids:
        first = json.loads((pilot / 'eval_workers1' / f'{eid}.json').read_text())
        second = json.loads((pilot / 'eval_workers2' / f'{eid}.json').read_text())
        for group in ['vbench_means', 'metrics']:
            assert first[group].keys() == second[group].keys()
            for key, a in first[group].items():
                b = second[group][key]
                assert (a['value'] is None) == (b['value'] is None)
                if a['value'] is not None:
                    assert abs(a['value'] - b['value']) <= 1e-5, (eid, key, a, b)
                for count in ['n_evaluated', 'n_failed', 'n_skipped']:
                    assert a[count] == b[count], (eid, key, count)
    write_json(validation, dict(status='passed', helper_sha256=sha256(HELPER), worker_seconds=timings,
        speedup=timings['1'] / timings['2'], score_tolerance=1e-5,
        scope='same existing media; integration/throughput only; no output selection'))


def audit_and_concat(root):
    from PIL import Image, ImageDraw
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "generation_report.json").read_text())
    assert report["status"] == "complete" and len(report["jobs"]) == manifest["totals"]["shot_count"]
    jobs = {(r["episode_id"], r["shot_key"]): r for r in report["jobs"]}
    audit, episode_rows = [], []
    for episode in manifest["episodes"]:
        previous, videos = None, []
        for shot in episode["shots"]:
            row = jobs[(episode["episode_id"], shot["shot_key"])]
            video = Path(row["output_video"])
            assert row["status"] in {"generated", "resumed_verified"}
            assert sha256(video) == row["video_sha256"]
            assert sha256(row["last_frame"]) == row["last_frame_sha256"]
            assert sha256(video.parent / "first.png") == row["first_frame_sha256"]
            assert sha256(row["input_image"]) == row["input_sha256"]
            assert row["decoded_frames"] == 81 and row["decoded_fps"] == 16
            if shot["cut"]:
                assert row["input_sha256"] == shot["firstframe_source"]["input_sha256"]
            else:
                assert previous is not None and row["previous_shot_key"] == previous["shot_key"]
                assert row["parent_video_sha256"] == previous["video_sha256"]
                assert row["input_sha256"] == previous["last_frame_sha256"]
            scene, local = map(int, shot["shot_key"].split(":"))
            link = root / "official_results" / episode["episode_id"] / f"{scene:03d}_{local:03d}.mp4"
            assert link.is_symlink() and link.resolve() == video.resolve()
            panels = [("hard-cut source" if shot["cut"] else "own previous tail", Path(row["source_input"])),
                      ("actual I2V input", Path(row["input_image"])),
                      ("actual video first", video.parent / "first.png"),
                      ("actual video last", video.parent / "last.png")]
            canvas = Image.new("RGB", (1600, 265), "#151515")
            draw = ImageDraw.Draw(canvas)
            for index, (label, path) in enumerate(panels):
                image = Image.open(path).convert("RGB")
                image.thumbnail((400, 220))
                canvas.paste(image, (index * 400, 35))
                draw.text((index * 400 + 8, 8), label, fill="white")
            contact = root / "audit" / episode["episode_id"] / ("shot_" + shot["shot_key"].replace(":", "_") + ".jpg")
            contact.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(contact, quality=90)
            audit.append({"episode_id": episode["episode_id"], "shot_key": shot["shot_key"],
                          "cut": shot["cut"], "source_type": row["input_source_type"],
                          "video_sha256": row["video_sha256"], "contact_sheet": str(contact), "verified": True})
            videos.append(video)
            previous = row
        destination = root / "episodes" / (episode["episode_id"] + ".mp4")
        destination.parent.mkdir(parents=True, exist_ok=True)
        concat_file = destination.with_suffix(".concat.txt")
        concat_file.write_text("".join(f"file '{video}'\n" for video in videos))
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                        str(concat_file), "-c", "copy", str(destination)], check=True)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", str(destination)], text=True))
        assert int(probe["streams"][0]["nb_read_frames"]) == len(videos) * 81
        episode_rows.append({"episode_id": episode["episode_id"], "video": str(destination),
                             "sha256": sha256(destination), "shot_count": len(videos), "frame_count": len(videos) * 81,
                             "concat_policy": "retain every canonical frame; no trimming; official evaluation uses original shots"})
    write_json(root / "audit/validation.json", {"verified_shots": len(audit), "verified_continuations": sum(not r["cut"] for r in audit),
               "shots": audit, "episodes": episode_rows})

def main(root):
    root = Path(root).resolve()
    env = base_environment()
    os.environ.update({k: env[k] for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
    try:
        state(root, 'prepare_expansion')
        plan = prepare(root)
        routing = freeze_face_routing(root, plan)
        eligible = {(r['episode_id'], r['character']) for r in routing['roles'] if r['eligible']}
        asset_plan = dict(plan, roles=[r for r in plan['roles'] if (r['episode_id'], r['character']) in eligible])
        disk_guard(root)
        parallel_probe(root, env)
        assets_env = asset_environment(root, env)
        prepare_assets(root, asset_plan, assets_env)
        prepare_frames(root, plan, assets_env, routing)
        state(root, 'freeze_generation_inputs')
        freeze_ready_manifest(root)
        state(root, 'generating_full140', new_video_count=2128, imported_video_count=363)
        logged(root, 'generate_formal', [GEN_PYTHON, '-u', 'pretest/run_lightx2v_a14b_4step_full140.py', '--output-root', str(root)], env)
        state(root, 'audit_and_concat')
        audit_and_concat(root)
        state(root, 'evaluate_official_parallel_non_vlm', workers=2)
        logged(root, 'evaluate_official_parallel', [EVAL_PYTHON, '-u', str(HELPER), '--output-root', str(root),
            '--workers', '2', '--gpu-id', '0', '--method-name', METHOD], env)
        destination = root / 'evaluation/official_parallel_v2'
        result = json.loads((destination / 'parallel_status.json').read_text())
        assert result['complete_episodes'] == 140 and result['complete_shots'] == 2491
        state(root, 'official_non_vlm_complete_pending_visual_review', complete_episodes=140, complete_shots=2491)
        (root / 'report.md').write_text('# EntityBench全140episode的4步接续路线\n\n'
            '2,491镜头生成、来源/尾帧审计及官方非VLM评价已完成。复用20个经兼容性核验的完整episode。\n\n'
            f'[官方报告](evaluation/official_parallel_v2/report_{METHOD}.md)；[镜头审计](audit/validation.json)。\n\n'
            '单组系统路线；VLM未测，保真度gate缺失，跨镜头分数暂定；人工效果审查仍待完成。\n')
    except Exception as error:
        previous = json.loads((root / 'status.json').read_text())
        state(root, 'failed', failed_phase=previous['phase'], error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    main(parser.parse_args().output_root)
