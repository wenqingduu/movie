"""Qwen full140 durable pipeline. --prepare-only is CPU/read-only-media admission work."""
from __future__ import annotations
import argparse
import copy
import fcntl
import html
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from prepare_entitybench_qwen21_full140 import OUTPUT, PROJECT, SOURCE, read, prepare, freeze_manifest
from prepare_entitybench_qwen21_20ep import immutable, sha256, write_json
from audit_entitybench_qwen21_full140 import (audit_reuse, audit_media, freeze_adapters,
    verify_adapters, verify_preserved, verify_models)
from run_entitybench_a14b_4step_20ep import environment
import run_entitybench_a14b_full140 as shared

QWEN='/root/autodl-tmp/qwen-image21-venv/bin/python'
GEN='/root/autodl-tmp/envs/lightx2v-trial/bin/python'
EVAL='/root/autodl-tmp/envs/entitybench-eval/bin/python'
ASSET=str(PROJECT/'.venv/bin/python')
HELPER=Path('/root/.codex/skills/evaluate-face-injection-v2/scripts/evaluate_parallel.py')
METHOD='qwen21_native_reference_a14b_4step_continuation_full140_no_vlm'

def state(root,phase,**extra):
    shared.state(root,phase,**extra)

def guard(root):
    assert shutil.disk_usage(root).free>=10*1024**3,'Less than 10GiB free; expand disk and resume; all outputs retained'

def resources(root,startup_min_free_gib=60):
    assert startup_min_free_gib>=50, 'Startup budget must cover 40GiB products plus the 10GiB reserve'
    free=shutil.disk_usage(root).free/1024**3
    try:
        query=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        gpus=[line for line in query.stdout.splitlines() if line.strip()] if query.returncode==0 else []
    except (OSError,subprocess.TimeoutExpired):gpus=[]
    cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    cores=(os.cpu_count() or 1) if cpu[0]=='max' else int(cpu[0])/int(cpu[1])
    memory=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    reasons=[]
    if free<startup_min_free_gib:reasons.append(f'startup_free_space_below_{startup_min_free_gib:g}GiB')
    # Container-visible CUDA index 0 can map to /dev/nvidiaN on the host.
    # NVML establishes visibility; each actual Python runtime checks CUDA before work.
    if not gpus:reasons.append('gpu_unavailable')
    else:
        fields=[s.strip() for s in gpus[0].split(',')]
        if int(fields[3])<45000:reasons.append('insufficient_free_vram_for_frozen_route')
        if int(fields[4])>10:reasons.append('gpu_busy')
    if cores<8:reasons.append('cpu_quota_below_8_threads')
    if memory!='max' and int(memory)<64*1024**3:reasons.append('host_memory_limit_below_64GiB')
    result=dict(free_gib=free,startup_min_free_gib=startup_min_free_gib,gpus=gpus,cpu_quota_cores=cores,memory_limit_bytes=memory,
        ready=not reasons,reasons=reasons)
    write_json(root/'admission.json',result);return result

def run_logged(root,label,command,env):
    guard(root)
    path=root/'logs'/f'{label}.log';path.parent.mkdir(parents=True,exist_ok=True)
    print('Starting',label,flush=True)
    with path.open('a') as stream:
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                guard(root)
                if label.startswith('parallel_probe_'):
                    probe=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                    sample=dict(time=time.time(),gpu=probe.stdout.strip(),
                        host_memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()))
                    with (root/'logs'/f'{label}.resources.jsonl').open('a') as resource_log:
                        resource_log.write(json.dumps(sample)+'\n')
                time.sleep(5)
            assert process.returncode==0,f'{label} exited {process.returncode}; see {path}'
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise

def profile(root,env):
    """A changed resource context is measured on two existing complete Qwen episodes."""
    pilot=root/'parallel_probe';pilot.mkdir(exist_ok=True)
    manifest=read(SOURCE/'manifest.json');manifest['episodes']=manifest['episodes'][:2]
    ids={e['episode_id'] for e in manifest['episodes']}
    manifest['totals']['episode_count']=2;manifest['totals']['shot_count']=sum(len(e['shots']) for e in manifest['episodes'])
    immutable(pilot/'manifest.json',manifest)
    generation=read(SOURCE/'generation_report.json');generation['jobs']=[r for r in generation['jobs'] if r['episode_id'] in ids]
    immutable(pilot/'generation_report.json',generation)
    immutable(pilot/'split_20ep.json',{t:[e['episode_id'] for e in manifest['episodes'] if e['difficulty']==t] for t in ['easy','medium','hard']})
    for folder in ['runtime','official_results']:
        path=pilot/folder
        if not path.exists():path.symlink_to(SOURCE/folder,target_is_directory=True)
    timings={};peaks={}
    for workers in [1,2]:
        out=pilot/f'eval_workers{workers}'
        run_logged(root,f'parallel_probe_{workers}',[EVAL,'-u',str(HELPER),'--output-root',str(pilot),'--out-dir',str(out),
            '--workers',str(workers),'--threads','4','--gpu-id','0','--method-name','qwen140_parallel_probe'],env)
        timings[str(workers)]=read(out/'parallel_execution.json')['elapsed_seconds']
        samples=[json.loads(line) for line in (root/'logs'/f'parallel_probe_{workers}.resources.jsonl').read_text().splitlines()]
        peaks[str(workers)]=dict(log=str(root/'logs'/f'parallel_probe_{workers}.resources.jsonl'),
            peak_host_memory_bytes=max(s['host_memory_bytes'] for s in samples),
            peak_gpu_used_mib=max(int(s['gpu'].split(',')[0]) for s in samples))
    for eid in ids:
        a=read(pilot/'eval_workers1'/f'{eid}.json');b=read(pilot/'eval_workers2'/f'{eid}.json')
        for group in ['vbench_means','metrics']:
            assert a[group].keys()==b[group].keys()
            for k,x in a[group].items():
                y=b[group][k];assert (x['value'] is None)==(y['value'] is None)
                if x['value'] is not None:assert abs(x['value']-y['value'])<=1e-5,(eid,k)
                assert all(x[c]==y[c] for c in ['n_evaluated','n_failed','n_skipped'])
    write_json(pilot/'validation.json',dict(status='passed',worker_seconds=timings,speedup=timings['1']/timings['2'],
        resource_logs=peaks,helper_sha256=sha256(HELPER),scope='existing Qwen media; no new generation or ablation'))

def audit_and_concat(root):
    from PIL import Image, ImageDraw
    guard(root)
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "generation_report.json").read_text())
    assert report["status"] == "complete" and len(report["jobs"]) == manifest["totals"]["shot_count"]
    jobs = {(r["episode_id"], r["shot_key"]): r for r in report["jobs"]}
    audit, episode_rows = [], []
    for episode in manifest["episodes"]:
        guard(root)
        previous, videos = None, []
        for shot in episode["shots"]:
            guard(root)
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
            if episode.get('reuse_generation_root'):
                original_contact=SOURCE/'audit'/episode['episode_id']/contact.name
                if not contact.exists():contact.symlink_to(original_contact)
                assert contact.resolve()==original_contact.resolve()
            elif not contact.exists():canvas.save(contact, quality=90)
            audit.append({"episode_id": episode["episode_id"], "shot_key": shot["shot_key"],
                          "cut": shot["cut"], "source_type": row["input_source_type"],
                          "video_sha256": row["video_sha256"], "contact_sheet": str(contact), "verified": True})
            videos.append(video)
            previous = row
        destination = root / "episodes" / (episode["episode_id"] + ".mp4")
        destination.parent.mkdir(parents=True, exist_ok=True)
        concat_file = destination.with_suffix(".concat.txt")
        concat_file.write_text("".join(f"file '{video}'\n" for video in videos))
        expected=dict(videos=[dict(path=str(v),sha256=sha256(v)) for v in videos])
        dependency_path=destination.with_suffix('.inputs.json')
        if dependency_path.exists():assert read(dependency_path)==expected
        if episode.get('reuse_generation_root'):
            source_episode=SOURCE/'episodes'/destination.name
            if not destination.exists():destination.symlink_to(source_episode)
            assert destination.resolve()==source_episode.resolve()
        elif not destination.exists():
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(concat_file),
                '-c','copy',str(destination)],check=True)
        immutable(dependency_path,expected)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", str(destination)], text=True))
        assert int(probe["streams"][0]["nb_read_frames"]) == len(videos) * 81
        episode_rows.append({"episode_id": episode["episode_id"], "video": str(destination),
                             "sha256": sha256(destination), "shot_count": len(videos), "frame_count": len(videos) * 81,
                             "concat_policy": "retain every canonical frame; no trimming; official evaluation uses original shots"})
    write_json(root / "audit/validation.json", {"verified_shots": len(audit), "verified_continuations": sum(not r["cut"] for r in audit),
               "shots": audit, "episodes": episode_rows})


def final_official_audit(root):
    import importlib.util
    spec=importlib.util.spec_from_file_location('full140_official_helper',HELPER)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    out=root/'evaluation/official_parallel_v2'
    manifest=read(root/'manifest.json');split=read(root/'split_full140.json');scorer=mod.import_scorer(root)
    mod.validate_manifest(read(out/'run_manifest.json'))
    assert read(out/'run_manifest.json')['n_vlm_keys']==0
    results=[];expected_schedule=[]
    for ep in manifest['episodes']:
        saved=read(out/(ep['episode_id']+'.json'))
        assert saved['num_shots']==len(ep['shots'])
        for dimension in mod.DIMENSIONS:
            metric=saved['vbench_means'][dimension]
            assert metric['value'] is not None and metric['n_evaluated']==len(ep['shots']) and metric['n_failed']==0
        results.append(scorer._episode_result_from_json(ep['episode_id'],ep['difficulty'],saved))
        artifact=list(out.glob('**/eval_artifacts/'+ep['episode_id']));assert len(artifact)==1
        for shot in ep['shots']:
            ev=read(artifact[0]/'intra_shot'/shot['shot_key'].replace(':','_')/'_eval.json')
            for entity_type,key in [('character','characters'),('object','objects'),('location','places')]:
                items=ev['presence'].get(entity_type,{}).get('items',[])
                assert {x['name'] for x in items}==set(shot['entity_schedule'].get(key,[])),(ep['episode_id'],shot['shot_key'],entity_type)
            expected_schedule.append((ep['episode_id'],shot['shot_key']))
    actual=read(out/f'report_{METHOD}.json');recomputed=scorer.aggregate_results(results,split)
    assert all(recomputed[t]==actual[t] for t in ['all','easy','medium','hard'])
    absent=[key for key in actual['all'] if key.startswith(('intra_face_','intra_object_','intra_location_','intra_action_','llm_'))
        and not key.endswith('_presence')]
    assert len(absent)==39
    for tier in ['all','easy','medium','hard']:
        assert all(actual[tier][key]['mean'] is None for key in absent)
    write_json(root/'audit/final_official.json',dict(status='passed',episodes=140,shots=2491,
        vbench_dimensions=mod.DIMENSIONS,vlm_calls=0,scope='partial_without_vlm',
        report_sha256=sha256(out/f'report_{METHOD}.json'),schedule_verified=len(expected_schedule),
        official_aggregation_verified=True,missing_vlm_metric_names=absent,
        metrics_counts={t:actual[t] for t in ['all','easy','medium','hard']}))

def archive(root):
    verify_preserved(root)
    reports=dict(full140_official=root/'evaluation/official_parallel_v2'/f'report_{METHOD}.json',
        full140_face_only=root/'evaluation/face_only_dino/report.json',
        qwen20_official=SOURCE/'evaluation/official_parallel_v2/report_qwen21_native_reference_a14b_4step_continuation_20ep_no_vlm.json',
        face20=PROJECT/'outputs/entitybench_face_only_dino_20ep_20261014/report.json',
        paper=PROJECT/'outputs/entitybench_paper_flux_qwen20_comparison/paper_reference.json')
    summary={}
    for name,path in reports.items():
        summary[name]=dict(path=str(path),sha256=sha256(path),report=read(path))
    write_json(root/'archive/comparison_sources.json',summary)
    episodes=read(root/'audit/validation.json')['episodes']
    options=''.join(f'<option value="../episodes/{html.escape(Path(e["video"]).name)}">{html.escape(e["episode_id"])}</option>' for e in episodes)
    viewer='<meta charset="utf-8"><title>Qwen EntityBench 140</title><h1>Qwen EntityBench 140</h1><select id="ep">'+options+'</select><br><video id="v" controls width="960"></video><script>const e=document.getElementById("ep"),v=document.getElementById("v");function show(){v.src=e.value}e.onchange=show;show();</script>'
    (root/'archive/viewer.html').write_text(viewer)
    (root/'report.md').write_text('# Qwen完整EntityBench\n\n140集／2491镜头；复用20集、补120集。1355个接续依赖已独立解码核验。\n\n'
        f'[官方人物区域报告](evaluation/official_parallel_v2/report_{METHOD}.md)；[脸部版](evaluation/face_only_dino/report.json)；[视频](archive/viewer.html)；[来源](archive/comparison_sources.json)。\n\n'
        '官方非VLM范围 partial_without_vlm，cs_face/cs_object 暂定、fidelity gate 缺失。脸部版单独保存。单路线绝对表现；全片人工审查未完成。\n')
    state(root,'complete_non_vlm_pending_full_video_visual_review',complete_episodes=140,complete_shots=2491,verified_continuations=1355)

def main(args):
    root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    assert root!=SOURCE
    # Repository-wide GPU lock also prevents different full140 roots from racing.
    lock=(PROJECT/'outputs/.entitybench_qwen140_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=prepare(root);freeze_adapters(root);verify_adapters(root)
    admission=resources(root,args.startup_min_free_gib)
    if args.prepare_only:
        state(root,'checking_reuse20',counts=plan['counts'])
        audit_reuse(root)
        state(root,'prepared_waiting_resources' if not admission['ready'] else 'prepared_ready_to_launch',counts=plan['counts'],admission=admission)
        print(json.dumps(dict(counts=plan['counts'],admission=admission),ensure_ascii=False,indent=2));return
    if not admission['ready']:
        state(root,'prepared_waiting_resources',counts=plan['counts'],admission=admission)
        raise RuntimeError('Resource admission failed: '+', '.join(admission['reasons']))
    immutable(root/'runtime/storage_admission.json',dict(startup_min_free_gib=args.startup_min_free_gib,
        new_products_budget_gib=40,reserve_gib=10,preparation_plan_sha256=sha256(root/'preparation_plan.json')))
    env=environment();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(PROJECT))
    for k in list(env):
        if any(x in k.upper() for x in ['API_KEY','OPENAI','AZURE','GEMINI','TOKEN_PLAN']):env.pop(k)
    try:
        verify_preserved(root);audit_reuse(root)
        # Admission succeeded: verify actual GPU/runtime access before any generation.
        for python in [QWEN,GEN,EVAL]:
            run_logged(root,'cuda_'+Path(python).parent.parent.name,[python,'-c','import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'],env)
        profile(root,env)
        shared.disk_guard=guard;shared.logged=run_logged
        assets_env=shared.asset_environment(root,env)
        state(root,'generating_qwen_references',total_roles=711,candidate_images=1163)
        run_logged(root,'qwen_references',[QWEN,'-u','pretest/run_qwen21_entitybench_full140_images.py','--output-root',str(root),'--stage','references'],assets_env)
        eligible=[r for r in plan['roles'] if r['needs_face_asset'] and read(r['metadata'])['face_eligible']]
        shared.prepare_assets(root,dict(plan,roles=eligible),assets_env,stages=('facelift','calibrate'))
        state(root,'generating_qwen_hardcuts',total_hardcuts=945)
        run_logged(root,'qwen_hardcuts',[QWEN,'-u','pretest/run_qwen21_entitybench_full140_images.py','--output-root',str(root),'--stage','frames'],assets_env)
        freeze_manifest(root);state(root,'generating_qwen_videos',new_shots=2128,reused_shots=363)
        run_logged(root,'generate_formal',[GEN,'-u','pretest/run_lightx2v_a14b_4step_qwen140.py','--output-root',str(root)],env)
        state(root,'audit_and_concat')
        run_logged(root,'media_audit',[ASSET,'-u','pretest/audit_entitybench_qwen21_full140.py','--output-root',str(root)],env)
        audit_and_concat(root)
        state(root,'evaluate_official_non_vlm')
        run_logged(root,'official_evaluation',[EVAL,'-u',str(HELPER),'--output-root',str(root),'--workers','2','--threads','4','--gpu-id','0','--method-name',METHOD],env)
        # Aggregation and all original entity schedules are independently verified.
        run_logged(root,'official_audit',[EVAL,'-u',str(Path(__file__).resolve()),'--output-root',str(root),'--official-audit'],env)
        state(root,'extracting_face_only_crops')
        run_logged(root,'face_extract',[ASSET,'-u','pretest/evaluate_entitybench_face_only_dino_qwen140.py','extract','--output-root',str(root)],env)
        decisions=root/'evaluation/face_only_dino/review/decisions.json'
        state(root,'waiting_face_crop_visual_review',decisions=str(decisions),official_complete=True)
        # No fabricated review approval. A reviewed, manifest-bound decision file resumes automatically.
        while not decisions.exists():guard(root);time.sleep(30)
        run_logged(root,'face_score',[EVAL,'-u','pretest/evaluate_entitybench_face_only_dino_qwen140.py','score','--output-root',str(root)],env)
        archive(root)
    except BaseException as error:
        previous=read(root/'status.json') if (root/'status.json').exists() else {}
        state(root,'failed',failed_phase=previous.get('phase'),error=repr(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=OUTPUT)
    p.add_argument('--startup-min-free-gib',type=float,default=60,
        help='Default 60; an explicitly approved alternative must still cover 40GiB products plus 10GiB reserve')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--official-audit',action='store_true');a=p.parse_args()
    if a.official_audit:final_official_audit(a.output_root)
    else:main(a)
