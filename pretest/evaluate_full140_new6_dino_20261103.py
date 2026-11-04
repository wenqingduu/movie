"""Isolated default-DINO adapter for the newly completed6 episodes only.

Original unchanged30 retain their reviewed results. CPU RetinaFace preserves
GPU generation; all embeddings/centroids use the existing official crops and
pairwise region-DINO values. No new model/video generation or paid requests.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import random
from pathlib import Path
import shutil

import qwen140_reaggregate_partial_reruns as agg
import qwen140_rerun_region_diagnostic as diag

ROOT = Path('/root/autodl-tmp/movie')
PARENT = ROOT/'outputs/entitybench_full140_completed41_snapshot_20261103'
OUT = PARENT/'new6_visible_region'
OLD = ROOT/'outputs/entitybench_rerun30_latest_evaluation_20261101/visible_region'
read, write, sha = diag.read, diag.write, diag.sha


def prepare():
    OUT.mkdir(exist_ok=True)
    snapshot = read(PARENT/'cohort_snapshot.json')
    episodes = [e for e in snapshot['episodes'] if e['comparison_group']=='new6']
    for item in episodes:
        item['artifacts'] = str(Path(item['evaluation'])/'results/.work/eval_artifacts'/item['episode_id'])
    diag.immutable(OUT/'input_manifest.json', {'episodes':episodes,'selected_episode_count':6,
        'selected_shot_count':150,'scope':'Only the6 new complete episodes; same default visible-single-face/localization-review protocol.'})
    shutil.copy2(OLD/'required_span_validation.json',OUT/'required_span_validation.json')
    write(OUT/'adapter_provenance.json', {'adapter_sha256':sha(Path(__file__)),
        'unchanged_helper_sha256':sha(Path(diag.__file__)),
        'snapshot_sha256':sha(PARENT/'cohort_snapshot.json'), 'device':'cpu','threads':2,
        'no_gpu_generation_changes':True,'vlm_calls':0})
    diag.OUT = OUT
    return episodes


def score(final=False):
    episodes = prepare()
    old, historical = diag.historical()
    ids = {i['episode_id'] for i in episodes}
    oldmap = {diag.key(r):r for r in old if r['episode_id'] in ids}
    newmap = {}
    for item in episodes:
        eid = item['episode_id']
        files = list((OUT/'extraction'/eid).glob('*.json'))
        assert len(files)==len(item['episode']['shots']), (eid,len(files))
        for path in files:
            cache = read(path)
            for row in cache['rows']:
                if row['original_crop']:
                    assert sha(row['original_crop'])==row['original_crop_sha256']
                newmap[diag.key(row)] = row
    assert set(oldmap)==set(newmap)
    carry = {k:r for k,r in historical.items() if k in newmap and r['original_crop_sha256']==newmap[k]['original_crop_sha256']}
    decisions={}
    review=None
    if final:
        review=read(OUT/'review/decisions.json');selection=read(OUT/'review/manifest.json')
        assert review['manifest_sha256']==sha(OUT/'review/manifest.json')
        by_id={r['review_id']:r for r in selection['rows']}
        assert {r['review_id'] for r in review['decisions']}==set(by_id)
        assert len(review['decisions'])==len(by_id)
        for row in review['decisions']:
            source=by_id[row['review_id']]
            assert diag.key(row)==diag.key(source)
            assert row['original_crop_sha256']==source['original_crop_sha256']
            assert row['decision'] in ['keep','exclude'] and row['reason'] and row['evidence']
            if row['decision']=='exclude':decisions[diag.key(row)]=row
    rejected={**carry,**decisions}
    mod=diag.gate.load_official_pure()
    old_results=[];new_results=[];scored=[];details=[]
    for item in episodes:
        ep=item['episode'];eid=item['episode_id']
        a,ae=diag.compute(ep,read(agg.ORIGINAL/(eid+'.json')),oldmap,historical,mod)
        b,be=diag.compute(ep,read(item['result']),newmap,rejected,mod)
        published=read(diag.gate.OUT/'remaining_low_score_review/recomputed/results'/(eid+'.json'))['metrics']['cs_face']
        assert a.metrics['cs_face'].n_evaluated==published['n_evaluated']
        if published['value'] is not None:assert abs(a.metrics['cs_face'].value-published['value'])<1e-10
        old_results.append(a);new_results.append(b)
        for name,data in be.items():
            for app in data['per_shot_sim_to_centroid']:
                row=dict(newmap[eid,name,app['shot']]);row['similarity']=app['sim_to_centroid'];scored.append(row)
        details.append({'episode_id':eid,'baseline':a.metrics['cs_face'].to_json(),
            'current':b.metrics['cs_face'].to_json(),'current_entities':be,
            'official_result_sha256':sha(item['result'])})
    split={tier:[i['episode_id'] for i in episodes if i['episode']['difficulty']==tier] for tier in ['easy','medium','hard']}
    report={'metric':'cs_face_visible_single_face_reviewed' if final else 'cs_face_visible_single_face_auto',
        'status':'complete_with_localization_review' if final else 'automatic_pending_localization_review',
        'episodes':6,'shots':150,'baseline':mod.aggregate_results(old_results,split),
        'current':mod.aggregate_results(new_results,split),'per_episode':details,
        'coverage':{'scheduled_role_shots':len(newmap),
            'baseline_present':sum(r['original_status']=='present' for r in oldmap.values()),
            'current_present':sum(r['original_status']=='present' for r in newmap.values()),
            'baseline_auto_eligible':sum(r['kept'] for r in oldmap.values()),
            'current_auto_eligible':sum(r['kept'] for r in newmap.values()),
            'baseline_reviewed_retained':sum(r['kept'] and k not in historical for k,r in oldmap.items()),
            'current_reviewed_retained':sum(r['kept'] and k not in rejected for k,r in newmap.items()),
            'baseline_scored':sum(r.metrics['cs_face'].n_evaluated for r in old_results),
            'current_scored':len(scored),'current_gate_reasons':dict(Counter(r['reason'] for r in newmap.values()))},
        'review_items':len(review['decisions']) if review else 0,'new_exclusions':list(decisions.values()),
        'historical_exclusions_carried_by_same_crop_sha':len(carry),
        'canonical_region_pixels_and_dino_features_unchanged':True,'new_dino_inference_calls':0,'vlm_calls':0,
        'detector_device':'cpu','manifest_sha256':sha(OUT/'input_manifest.json')}
    write(OUT/('report.json' if final else 'automatic_report.json'),report)
    if not final:
        write(OUT/'scored_crops.json',scored)
        if len(scored)>=80:
            diag.prepare_review(scored)
        else:
            # Current protocol explicitly reviews all when <80 are available.
            raise RuntimeError('Fewer than80 scored crops: create all-crop review sheets before final, never weaken the face gate.')
        write(OUT/'review/role_descriptions.json',{i['episode_id']:i['episode']['entity_descriptions'] for i in episodes})
    print({'status':report['status'],'episodes':6,'shots':150,'review_items':report['review_items'],
           'coverage':report['coverage']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['extract','score','final'])
    args=parser.parse_args();prepare()
    if args.action=='extract':diag.extract('cpu')
    else:score(args.action=='final')
