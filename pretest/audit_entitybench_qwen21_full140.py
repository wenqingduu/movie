"""Input SHA, full schedules and independently decoded own-tail audits; no inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_entitybench_qwen21_full140 import PROJECT, SOURCE, OUTPUT, SPLIT, read
from prepare_entitybench_qwen21_20ep import sha256, digest, frame_fingerprint, immutable, write_json, REVISION
from prepare_entitybench_a14b_continuation import ordered_shots

ADAPTERS=['prepare_entitybench_qwen21_full140.py','audit_entitybench_qwen21_full140.py',
    'run_qwen21_entitybench_full140_images.py','run_lightx2v_a14b_4step_qwen140.py',
    'run_entitybench_qwen21_full140.py','evaluate_entitybench_face_only_dino_qwen140.py']

def freeze_adapters(root):
    from prepare_entitybench_qwen21_20ep import MODEL
    paths=[PROJECT/'pretest'/n for n in ADAPTERS]
    paths.append(PROJECT/'tests/test_qwen21_full140.py')
    paths += [PROJECT/'pretest/run_entitybench_qwen21_20ep_accelerated.py',
        PROJECT/'pretest/run_qwen21_gpu_resident_references.py',
        PROJECT/'pretest/evaluate_entitybench_face_only_dino.py',
        PROJECT/'pretest/evaluate_entitybench_corrected_character_crops.py',
        Path('/root/.codex/skills/evaluate-face-injection-v2/scripts/evaluate_parallel.py')]
    paths += [p for p in MODEL.rglob('*') if p.is_file() and p.suffix in {'.json','.txt','.jinja'} and '.cache' not in p.parts]
    immutable(Path(root)/'runtime/full140_adapters.json',dict(sources={str(p):sha256(p) for p in paths},
        reference_placement='cuda',firstframe_placement='sequential_cpu_offload',disk_floor_gib=10,
        evaluation_workers=2,evaluation_threads=4,vlm_calls=0))

def verify_adapters(root):
    for p,h in read(Path(root)/'runtime/full140_adapters.json')['sources'].items():
        assert sha256(p)==h, f'Frozen full140 source changed: {p}'

def verify_models(source=SOURCE):
    impl=read(source/'runtime/qwen_implementation.json')
    assert impl['model_revision']==REVISION and len(impl['weights'])==7
    for p,h in impl['sources'].items():assert sha256(p)==h, p
    for row in impl['weights']:
        assert Path(row['path']).stat().st_size==row['size'] and sha256(row['path'])==row['sha256'],row['path']
    runtime=read(source/'generation_manifest.json')
    for p,h in runtime['implementation_sha256'].items():assert sha256(p)==h,p
    from prepare_entitybench_a14b_4step import LIGHT, LIGHT_REVISION
    assert subprocess.check_output(['git','-C',str(LIGHT),'rev-parse','HEAD'],text=True).strip()==LIGHT_REVISION==runtime['lightx2v_commit']
    for p,h in runtime['lightx2v_source_sha256'].items():assert sha256(LIGHT/p)==h,p
    inventory=read(PROJECT/'outputs/model_downloads/wan22_i2v_a14b/manifest.json')
    assert digest(inventory)==runtime['model_inventory_hash']
    model=Path(read(source/'manifest.json')['model_path'])
    for row in inventory['files']:
        path=model/row['path'];assert path.stat().st_size==row['size'],path
        expected=row.get('lfs',{}).get('oid')
        if expected:assert sha256(path)==expected,path
        else:
            payload=path.read_bytes()
            assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==row['oid'],path
    for row in runtime['lora_inventory']['files']:
        assert sha256(Path(runtime['lora_inventory']['destination'])/row['path'])==row['sha256']
    return dict(qwen_weights=7,wan_files=len(inventory['files']),lora_files=2,
        qwen_implementation_sha256=sha256(source/'runtime/qwen_implementation.json'),
        generation_manifest_sha256=sha256(source/'generation_manifest.json'))

def audit_media(root, expected_episodes, expected_shots, expected_continuations, decode=True):
    from PIL import Image
    import numpy as np
    root=Path(root);manifest=read(root/'manifest.json');report=read(root/'generation_report.json')
    assert report['status']=='complete'
    episodes=manifest['episodes']; assert len(episodes)==len({e['episode_id'] for e in episodes})==expected_episodes
    rows=report['jobs']; jobs={(r['episode_id'],r['shot_key']):r for r in rows}
    assert len(jobs)==len(rows)==expected_shots
    expected_keys={(e['episode_id'],s['shot_key']) for e in episodes for s in e['shots']}
    assert set(jobs)==expected_keys
    continuations=0; audit=[]
    for ep in episodes:
        assert sha256(ep['episode_path'])==ep['episode_sha256']
        original=ordered_shots(read(ep['episode_path'])); assert len(original)==len(ep['shots'])
        previous=None
        for shot,raw in zip(ep['shots'],original):
            assert all(shot[k]==raw[k] for k in ['shot_key','shot_index','prompt','cut','entity_schedule','action_description'])
            row=jobs[(ep['episode_id'],shot['shot_key'])];video=Path(row['output_video'])
            assert row['status'] in {'generated','resumed_verified'}
            assert all(row[k]==shot[k] for k in ['shot_key','shot_index','seed','prompt','cut','entity_schedule'])
            for p,h in [(video,row['video_sha256']),(row['input_image'],row['input_sha256']),
                (row['last_frame'],row['last_frame_sha256']),(video.parent/'first.png',row['first_frame_sha256'])]:
                assert sha256(p)==h,p
            assert row['decoded_frames']==81 and row['decoded_fps']==16
            scene,local=map(int,shot['shot_key'].split(':'))
            assert (root/'official_results'/ep['episode_id']/f'{scene:03d}_{local:03d}.mp4').resolve()==video.resolve()
            if shot['cut']:
                source=shot['firstframe_source'];assert row['input_sha256']==source['input_sha256']
                assert sha256(source['source_config'])==source['source_config_sha256']
                assert sha256(source['input_image'])==source['input_sha256']
            else:
                assert previous and row['previous_shot_key']==previous['shot_key']
                assert row['parent_video_sha256']==previous['video_sha256']
                assert row['input_sha256']==previous['last_frame_sha256']
                assert row['input_source_type']=='own_previous_video_last_frame'
                if decode:
                    payload=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','1','-i',previous['output_video'],
                        '-vf',r'select=eq(n\,80)','-vsync','0','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
                    actual=np.asarray(Image.open(row['input_image']).convert('RGB'))
                    assert len(payload)==actual.size and np.array_equal(np.frombuffer(payload,dtype=np.uint8).reshape(actual.shape),actual),row['input_image']
                continuations+=1
            audit.append(dict(episode_id=ep['episode_id'],shot_key=shot['shot_key'],video_sha256=row['video_sha256'],cut=shot['cut']))
            previous=row
    assert continuations==expected_continuations
    return dict(status='passed',episodes=expected_episodes,shots=expected_shots,continuations=continuations,
        independent_tail_decode=decode,failures=[],missing=[],shots_verified=audit,
        manifest_sha256=sha256(root/'manifest.json'),generation_report_sha256=sha256(root/'generation_report.json'))

def audit_reuse(root=OUTPUT, weights=True, decode=True):
    print('Read-only audit: original Qwen references, 3D assets and hardcuts',flush=True)
    plan=read(SOURCE/'preparation_plan.json');manifest=read(SOURCE/'manifest.json')
    assets=read(SOURCE/'asset_manifest.json')['items']; assert len(assets)==112
    for role in assets:
        meta=read(role['metadata']);assert sha256(role['reference'])==meta['selected_sha256']==role['reference_sha256']
        original=next(r for r in plan['roles'] if r['episode_id']==role['episode_id'] and r['character']==role['character'])
        assert meta['dependency_fingerprint']==digest(dict(role=original,settings=plan['reference_settings'],model_revision=REVISION,
            implementation=read(SOURCE/'runtime/qwen_implementation.json')))
        for candidate in meta['candidates']:assert sha256(candidate['path'])==candidate['sha256']
        if role['needs_face_asset'] and role['face_eligible']:
            result=read(role['result']);model=Path(result['model_path']);cal=model.with_suffix('.pose_calibration.json')
            assert result['facelift_status']=='success' and result['source_reference_sha256']==role['reference_sha256']
            assert sha256(model)==role['model_sha256']==read(cal)['gaussian_model_sha256']
            assert sha256(cal)==role['calibration_sha256'] and read(cal)['profiles']
    for ep in manifest['episodes']:
        for shot in ep['shots']:
            if shot['cut']:
                row=read(shot['firstframe_source']['source_config'])
                assert row['dependency_fingerprint']==frame_fingerprint(SOURCE,plan,ep,shot)
    print('Read-only audit: 363 videos and 172 independently decoded own-tail dependencies',flush=True)
    result=audit_media(SOURCE,20,363,172,decode)
    official=SOURCE/'evaluation/official_parallel_v2/report_qwen21_native_reference_a14b_4step_continuation_20ep_no_vlm.json'
    assert sha256(official)=='6ed4d81d7aa575830fd101bd9701f621d4dd2bad77cf6a808ee803df001a4e86'
    result.update(references=112,face_assets=sum(r['needs_face_asset'] and r['face_eligible'] for r in assets),
        official_report_sha256=sha256(official),weights_verified=weights)
    if weights:
        print('Read-only audit: existing Qwen, A14B and LoRA weights; no download or inference',flush=True)
        result['models']=verify_models()
    write_json(Path(root)/'audit/reuse20.json',result)
    # Bind every existing Qwen artifact, including diagnostic trajectories and old reports.
    paths=[p for p in SOURCE.rglob('*') if p.is_file()]
    immutable(Path(root)/'runtime/reuse20_files.json',dict(files={str(p):sha256(p) for p in paths}))
    print('Read-only reuse audit passed',flush=True)
    return result

def verify_preserved(root):
    for name in ['reuse20_files.json','before_sources.json']:
        data=read(Path(root)/'runtime'/name);files=data.get('files',data)
        for p,h in files.items():assert sha256(p)==h,f'Preserved source changed: {p}'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUTPUT)
    p.add_argument('--reuse',action='store_true');a=p.parse_args()
    if a.reuse:
        result=audit_reuse(a.output_root);print({k:v for k,v in result.items() if k!='shots_verified'})
    else:
        result=audit_media(a.output_root,140,2491,1355);verify_preserved(a.output_root)
        write_json(a.output_root/'audit/final_media.json',result)
