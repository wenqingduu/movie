"""All1277 fresh original40 attempts in recovery-evidence order, eight per batch.

Preserve current28; reuse28 already failed original40 conditions. Only dynamic
gains enter an independent cumulative mapping. No prompt rewriting or quality gate.
"""
from __future__ import annotations
from collections import Counter
import copy
import fcntl
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import time

import qwen140_original40_vs_distill4_20261108 as core
import qwen140_original40_recovery_priority_20261109 as priority

ROOT, BASE = core.ROOT, core.BASE
OUT = ROOT/'outputs/entitybench_original40_priority_remaining_20261109'
BASE28 = ROOT/'outputs/entitybench_original40_next24_20261109/replacement'
BASE20 = ROOT/'outputs/entitybench_original40_dynamic_gain_selected20_20261109'
CONTEXT_SCRIPT = ROOT/'pretest/qwen140_original40_priority_context_20261109.py'
read, write, sha, immutable = core.read, core.write, core.sha, core.immutable


def status(stage, **kw):
    write(OUT/'status.json',dict(stage=stage,utc=core.now(),expected_new_generation=1277,
        generated=len(list((OUT/'cases').glob('*/generation.json'))),
        scored=len(list((OUT/'cases').glob('*/scores.json'))),known_original40_zero_reused=28,
        baseline_selected=28,quality_gate_enabled=False,prompt_edits=False,api_calls=0,vlm_calls=0,
        old_media_overwritten=False,old4step_queue_resumed=False,**kw))


def prepare():
    if (OUT/'plan.json').exists():
        p=read(OUT/'plan.json')
        for b in p['bindings']:assert sha(b['path'])==b['sha256'],b['path']
        return p
    ranking=priority.build();pure=read(priority.PURE);lookup={c['case_id']:c for c in pure['cases']}
    assert read(BASE28/'status.json')['stage']=='complete_all_available_metrics_with_reviewed_dino'
    snapshot=copy.deepcopy(read(BASE28.parent/'baseline_snapshot.json'));media=read(BASE28/'replacement_manifest.json')['videos']
    bindings={}
    def bind(f):
        f=str(Path(f).resolve());bindings[f]=sha(f)
    for item in snapshot['episodes']:
        result=BASE28/'affected/results'/(item['episode_id']+'.json')
        if result.exists():item.update(result=str(result),result_sha256=sha(result),output_root=str(BASE28/'episode_media'),
            artifacts=str(result.parent/'.work/eval_artifacts'/item['episode_id']))
        assert sha(item['result'])==item['result_sha256'];bind(item['result'])
    for row in media.values():assert sha(row['canonical'])==row['sha256']
    snapshot.update(media={k:dict(source=r['source'],sha256=r['sha256']) for k,r in media.items()},
        scope='Frozen complete cumulative28 full140 baseline; priority remaining40 run.',parent_media_sha256=sha(BASE28/'replacement_manifest.json'))
    immutable(OUT/'baseline_snapshot.json',snapshot)
    cases=[]
    for row in ranking['cases']:
        c=copy.deepcopy(lookup[row['case_id']]);key=c['episode_id']+'/'+c['shot'];task=read(c['task_path'])
        assert task==c['task'] and task['prompt']==c['new_prompt'] and c['new_prompt']!=c['old_prompt']
        assert task['input_sha256']==c['input_sha256']==sha(c['input_path'])
        assert task['frames']==81 and task['fps']==16 and task['steps']==4 and task['last_frame_path'] is None
        assert media[key]['sha256']==c['baseline_video_sha256'] and not media[key]['cumulative_replaced']
        assert c['baseline_scores']['dynamic_degree']==0 and sha(c['source_task'])==c['source_task_sha256']
        if c['transition']:assert sha(c['transition'])==c['transition_sha256']
        c.update(source_case=c['case_id'],source_task_compiled=c['task_path'],source_task_compiled_sha256=sha(c['task_path']),
            prompt=c['new_prompt'],seed=task['seed'],frames=81,fps=16,size=task['size'],
            original_description=c['official_description'],frozen_benchmark_scores=c['baseline_scores'],
            output=str(OUT/'cases'/c['case_id']/'main'),priority=row,
            baseline_is_dynamic_zero=True,same_prompt4_reference=row['exact_same_prompt4'])
        cases.append(c)
    assert len(cases)==1277 and len({c['episode_id']+'/'+c['shot'] for c in cases})==1277
    earlier=read(ROOT/'outputs/entitybench_original40_scope32_20261108/plan.json')
    cfg=read(earlier['config'])
    assert cfg['infer_steps']==40 and cfg['sample_guide_scale']==[3.5,3.5] and cfg['enable_cfg']
    assert cfg['sample_shift']==5 and cfg['boundary']==.9 and cfg['rope_type']=='torch_complex_rope' and cfg['double_precision_rope']
    for f in [Path(__file__),CONTEXT_SCRIPT,Path(core.__file__),Path(priority.__file__),priority.PURE,
              priority.OUT/'priority_manifest_v2.json',OUT/'baseline_snapshot.json',earlier['config'],
              BASE28/'replacement_manifest.json',BASE28/'official_aggregates.json',BASE28/'visible_region/report.json',
              BASE28/'visible_region/review/decisions.json',BASE20/'visible_region/report.json',
              ROOT/'pretest/qwen140_all_zero_motion_20261108.py',ROOT/'pretest/qwen140_v2_shots_only_eval_20261105.py',
              ROOT/'pretest/qwen140_vbench_verified_cache_20261105.py',BASE/'runtime/evaluate_benchmark_official.py',
              BASE/'runtime/evaluate_benchmark_partial.py']:
        bind(f)
    p=dict(created_utc=core.now(),cases=cases,total=1277,new_cases=1277,baseline_selected=28,
        current_zero_scope=1305,known_original40_zero_reused=ranking['reused_zero_cases'],batch_size=8,
        model_path=earlier['model_path'],config=earlier['config'],negative_prompt='',
        scope='All1277 fresh original40 conditions, rank30 verified enhanced4 positives first, then description-based hypotheses. Baseline cumulative28; prompts may differ from benchmark, no causal step/LoRA claim. All negatives/declines retained; only dynamic0_to1 selects media.',
        quality_gate_enabled=False,category_counts=dict(Counter(c['category'] for c in cases)),
        route_counts=dict(Counter(c['route'] for c in cases)),
        B_policy='Reuse existing transition and exact original main input/seed. Original40 main81frames, stitch with unchanged filter, score complete combined video.',
        prompt_policy='Use frozen motion-only1333 plan byte-for-byte; do not append historical background cues or invent action/camera events.',
        prior_pairing='Existing same-prompt4 scores only when verified; no fabricated4-step control for ungenerated prompts.',
        model_validation='Same loaded original40 pipeline, explicit TorchRoPE/config as completed56; persistent seed use unchanged. B stitching verified independently before bulk.',
        bindings=[dict(path=k,sha256=v) for k,v in bindings.items()])
    immutable(OUT/'plan.json',p)
    write(OUT/'validation.json',dict(cases=1277,current_zero=1305,known_zero_reused=28,priority_verified4_positive=30,
        all_input_png_and_source_task_shas_verified=True,all_frozen_prompts_unchanged=True,baseline2491_shas_verified=True,
        B_routes=15,all_transitions_verified=True,config_40_cfg35_nolora_torchrope=True,quality_gate_enabled=False,
        plan_sha256=sha(OUT/'plan.json')))
    status('prepared_priority1277')
    return p


def worker(ids):
    import torch
    from run_lightx2v_a14b_4step_qwen140 import decode_video
    import qwen140_all_zero_motion_20261108 as stitch
    p=prepare();by={c['case_id']:c for c in p['cases']};core.OUT=OUT;stitch.OUT=OUT
    start=time.monotonic();status('loading_original40_pipeline',case_ids=ids);pipe=core.load(p);load_seconds=time.monotonic()-start
    for cid in ids:
        c=by[cid];folder=OUT/'cases'/cid;main=folder/'main';receipt=folder/'generation.json'
        if receipt.exists():
            r=read(receipt);assert r['plan_sha256']==sha(OUT/'plan.json') and sha(r['video'])==r['video_sha256'];continue
        assert sha(c['input_path'])==c['input_sha256'] and sha(c['source_task_compiled'])==c['source_task_compiled_sha256']
        assert sha(c['baseline_video'])==c['baseline_video_sha256']
        main.mkdir(parents=True,exist_ok=True);gpath=main/'generation.json'
        if not gpath.exists():
            tmp=main/'video.tmp.mp4';assert not tmp.exists(),'Preserve partial video for technical inspection'
            status('generating_original40',case_id=cid,run=c['run'],shot=c['shot'],priority_rank=c['priority']['priority_rank'],model_load_seconds=load_seconds)
            begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
            pipe.generate(seed=c['seed'],prompt=c['prompt'],negative_prompt='',image_path=c['input_path'],save_result_path=str(tmp));torch.cuda.synchronize()
            decoded=decode_video(tmp,main/'first.png',main/'last.png',81,16);tmp.replace(main/'video.mp4')
            immutable(gpath,dict(case_id=cid,episode_id=c['episode_id'],shot=c['shot'],prompt=c['prompt'],seed=c['seed'],
                input_sha256=c['input_sha256'],video=str(main/'video.mp4'),video_sha256=sha(main/'video.mp4'),decoded=decoded,
                config_sha256=sha(p['config']),plan_sha256=sha(OUT/'plan.json'),original40_no_lora=True,
                elapsed_seconds=time.monotonic()-begin,model_load_seconds=load_seconds,cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
        g=read(gpath);assert g['plan_sha256']==sha(OUT/'plan.json') and sha(g['video'])==g['video_sha256']
        assert g['prompt']==c['prompt'] and g['seed']==c['seed'] and g['input_sha256']==c['input_sha256']
        r=stitch.finish_case(c,main)
        immutable(receipt,{**g,'video':r['video'],'video_sha256':r['video_sha256'],'main_generation':str(gpath),
            'main_generation_sha256':sha(gpath),'main_video':g['video'],'main_video_sha256':g['video_sha256'],
            'route':c['route'],'transition_reused':bool(c['transition']),'transition_sha256':c['transition_sha256'],
            'scoring_video':'complete combined transition+main' if c['transition'] else 'main81frames'})
        print('COMPLETE',cid,'rank',c['priority']['priority_rank'],c['run'],c['shot'],flush=True)


def evaluate(ids,tag):
    p=prepare();by={c['case_id']:c for c in p['cases']};official=BASE/'runtime/evaluate_benchmark_official.py'
    spec=importlib.util.spec_from_file_location('_priority1277_original_vbench',official);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    import torch
    torch.set_num_threads(4);folder=OUT/'evaluation'/tag;videos={}
    for i,cid in enumerate(ids,1):
        r=read(OUT/'cases'/cid/'generation.json');assert sha(r['video'])==r['video_sha256'];videos[f'{i}:1']=r['video']
    receipt=folder/'receipt.json'
    if receipt.exists():
        saved=read(receipt);assert saved['plan_sha256']==sha(OUT/'plan.json') and saved['case_ids']==ids
        assert saved['video_sha256']=={k:sha(v) for k,v in videos.items()};scores=saved['scores']
    else:
        scorer=mod.VBenchEvaluator('cuda','/root/autodl-tmp/models/entitybench_vbench',str(folder),True)
        scores,failed=scorer.evaluate_episode_shots(videos,tag,str(folder));assert not failed
        immutable(receipt,dict(plan_sha256=sha(OUT/'plan.json'),case_ids=ids,scores=scores,
            video_sha256={k:sha(v) for k,v in videos.items()},official_sha256=sha(official),vlm_calls=0))
    for i,cid in enumerate(ids,1):
        after=scores[f'{i}:1'];assert all(math.isfinite(after[d]) for d in core.DIMS) and after['dynamic_degree'] in (0,1)
        immutable(OUT/'cases'/cid/'scores.json',dict(case_id=cid,before=by[cid]['baseline_scores'],after=after,
            video_sha256=sha(videos[f'{i}:1']),source_score=str(receipt),source_score_sha256=sha(receipt),
            quality_filtered=False,reference='Current cumulative28 benchmark, not necessarily same-prompt4.',vlm_calls=0))


def progress_overlay(p):
    selected={};decisions=[]
    for c in p['cases']:
        f=OUT/'cases'/c['case_id']/'scores.json'
        if not f.exists():continue
        s=read(f);r=read(f.parent/'generation.json');assert s['video_sha256']==r['video_sha256']
        chosen=s['after']['dynamic_degree']==1
        decisions.append(dict(case_id=c['case_id'],episode_id=c['episode_id'],shot=c['shot'],selected=chosen,
            before_dynamic=0,after_dynamic=s['after']['dynamic_degree'],scores=str(f),scores_sha256=sha(f)))
        if chosen:selected[c['episode_id']+'/'+c['shot']]=r
    baseline=read(BASE28/'replacement_manifest.json')['videos'];media={}
    for key,old in baseline.items():
        r=selected.get(key);eid,sk=key.split('/');a,b=map(int,sk.split(':'));link=OUT/'candidate_media'/eid/f'{a:03d}_{b:03d}.mp4'
        source=r['video'] if r else old['source'];digest=r['video_sha256'] if r else old['sha256']
        link.parent.mkdir(parents=True,exist_ok=True)
        if link.is_symlink():
            if link.resolve()!=Path(source).resolve():
                pending=link.with_suffix('.mp4.next');assert not pending.exists()
                pending.symlink_to(Path(source).resolve());pending.replace(link)
        else:
            assert not link.exists();link.symlink_to(Path(source).resolve())
        media[key]=dict(source=source,sha256=digest,canonical=str(link),replaced=bool(r),
            cumulative_replaced=bool(r) or old['cumulative_replaced'],before_source=old['source'],before_sha256=old['sha256'])
    counts={e['episode_id']:len(e['episode']['shots']) for e in read(OUT/'baseline_snapshot.json')['episodes']}
    before=read(BASE28/'dynamic_exact.json')['after'];delta=sum(1/counts[k.split('/')[0]]/140 for k in selected)
    version=f'scored_{len(decisions):04d}'
    report=dict(stage='dynamic_selection_complete_full_context_pending',scored=len(decisions),expected=1277,new_selected=len(selected),
        cumulative_selected=28+len(selected),videos=media,decisions=decisions,
        baseline_sha256=sha(BASE28/'replacement_manifest.json'),other_full140_metrics='pending complete new-context evaluation',
        dynamic_exact=dict(before=before,after=before+delta,delta_percentage_points=delta*100),quality_gate_enabled=False)
    checkpoint=copy.deepcopy(report)
    for row in checkpoint['videos'].values():row['canonical']=row['source']
    checkpoint['canonical_note']='Frozen checkpoints bind immutable source videos; live candidate_media links belong only to latest progressive manifest.'
    immutable(OUT/'selection_checkpoints'/f'{version}.json',checkpoint)
    write(OUT/'progressive_replacement_manifest.json',report)
    return report


def run():
    p=prepare()
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Existing GPU task'
        try:
            progress_overlay(p)
            for start in range(0,len(p['cases']),p['batch_size']):
                ids=[c['case_id'] for c in p['cases'][start:start+p['batch_size']]];tag=f'priority_batch_{start//p["batch_size"]+1:03d}'
                for action,python,needed in [('worker',core.PY,'generation.json'),('evaluate',core.EVAL,'scores.json')]:
                    if all((OUT/'cases'/cid/needed).exists() for cid in ids):continue
                    status('dispatching_'+action,batch=tag,case_ids=ids)
                    command=[python,'-u',str(Path(__file__)),action,','.join(ids)]+([tag] if action=='evaluate' else [])
                    with (OUT/(tag+'_'+action+'.log')).open('a') as log:
                        subprocess.run(command,env=core.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
                r=progress_overlay(p);status('batch_complete',batch=tag,new_selected=r['new_selected'],cumulative_selected=r['cumulative_selected'])
            status('all1277_generated_scored_starting_full_context')
            with (OUT/'context.log').open('a') as log:
                subprocess.run([core.EVAL,'-u',str(CONTEXT_SCRIPT),'run'],env=core.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
            status('official_complete_default_dino_localization_review_pending',report=str(OUT/'replacement/official_report.md'))
        except Exception as error:
            status('stopped_technical_error_preserve_all_outputs',error=repr(error));raise


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':print({k:prepare()[k] for k in ['total','baseline_selected','current_zero_scope']})
    elif action=='run':run()
    elif action=='worker':worker(sys.argv[2].split(','))
    elif action=='evaluate':evaluate(sys.argv[2].split(','),sys.argv[3])
    else:raise ValueError(action)
