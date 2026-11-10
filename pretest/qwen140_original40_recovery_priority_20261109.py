"""Evidence-based ordering only; no prompt edits or media-quality rejection."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re

import qwen140_original40_vs_distill4_20261108 as core

ROOT = core.ROOT
OUT = ROOT / 'outputs/entitybench_original40_recovery_priority_analysis_20261109'
PURE = ROOT / 'outputs/entitybench_all_zero_fixed_frame_motion_20261108/plan.json'
BACKGROUND = ROOT / 'outputs/entitybench_all_zero_motion_background_20261108/plan.json'
CURRENT = ROOT / 'outputs/entitybench_original40_next24_20261109/replacement'
read, write, sha, immutable = core.read, core.write, core.sha, core.immutable


def semantics(c):
    text = c['official_description'].split('\n\n')[-1].lower()
    features = []
    score = 0
    if re.search(r'\bcamera\s+(?:(?:slowly|gradually|smoothly|steadily)\s+)?(?:pans?|zooms?|tracks?|dollies|moves?|pushes?|pulls?)\b|\bpanning\b|\btracking shot\b|\bfollow shot\b|\bhandheld\b', text):
        score += 70; features.append('explicit_existing_camera_motion')
    if re.search(r'\b(danc(?:e|es|ing)|twirls?|swinging|strumming|runs?|running|climbs?|climbing|riding)\b', text):
        score += 55; features.append('sustained_or_large_body_action')
    if re.search(r'\bwalk(?:s|ing)?\b|\bstorms? in\b|\benters?\b|\bapproach(?:es|ing)?\b|\bheadlock\b', text):
        score += 35; features.append('explicit_body_travel')
    if re.search(r'walks? (?:from left to right|across|past|through)|across the|headlock|embrace|hugs?', text):
        score += 20; features.append('cross_frame_or_whole_body_interaction')
    if re.search(r'\bforeground\b|\bmedium shot\b|\bfull shot\b', text):
        score += 8; features.append('text_indicates_visible_subject')
    if c['category'] == 'prop_or_arm':
        score += 8; features.append('local_prop_action_uncertain')
    if re.search(r'\b(distant|distance|out of focus|faint silhouette|aerial|high.angle aerial)\b', text):
        score -= 25; features.append('small_or_distant_motion_risk')
    if re.search(r'\b(slightly|subtle|gently|small.*step|slowly.*head|eyes|smirk|gaze)\b', text):
        score -= 20; features.append('subtle_or_face_only_motion_risk')
    if re.search(r'\b(still|motionless|stationary|dead|lifeless)\b', text):
        score -= 10; features.append('explicit_static_constraint_preserved')
    if c['route'] == 'B':
        score -= 5; features.append('existing_transition_full_video_scoring')
    return score, features


def reference4(c, bg):
    """Use positive evidence only with exact prompt/input/seed/video/score bindings."""
    for candidate in [c, bg[c['case_id']]]:
        directory = Path(candidate['output']).parent
        if candidate['new_prompt'] != c['new_prompt'] or not (directory/'vbench.json').exists() or not (directory/'result.json').exists():
            continue
        record = read(directory/'result.json'); score = read(directory/'vbench.json')
        video = Path(record['video']); generation_path = video.parent/'generation.json'
        if not generation_path.exists():
            generation_path = video.parent.parent/'main/generation.json'
        if not generation_path.exists():
            continue
        generation = read(generation_path); actual = generation.get('inputs', {})
        if any(actual.get(k) != c['task'].get(k) for k in ['prompt','input_sha256','seed','frames','fps','size']):
            continue
        assert sha(video) == record['video_sha256']
        proof_path = Path(score['receipt']); proof = read(proof_path)
        slot = f"{proof['case_ids'].index(c['case_id'])+1}:1"
        assert proof['video_sha256'][slot] == record['video_sha256']
        assert proof['scores'][slot] == score['after']
        assert proof['official_sha256'] == sha(core.BASE/'runtime/evaluate_benchmark_official.py')
        return dict(dynamic=score['after']['dynamic_degree'], video=str(video), video_sha256=record['video_sha256'],
                    generation=str(generation_path), generation_sha256=sha(generation_path),
                    score=str(directory/'vbench.json'), score_sha256=sha(directory/'vbench.json'),
                    receipt=str(proof_path), receipt_sha256=sha(proof_path), scores=score['after'])
    return None


def build():
    if (OUT/'priority_manifest_v2.json').exists():
        p = read(OUT/'priority_manifest_v2.json')
        for b in p['bindings']:
            assert sha(b['path']) == b['sha256'], b['path']
        return p
    pure = read(PURE); bg = {c['case_id']: c for c in read(BACKGROUND)['cases']}
    current = read(CURRENT/'replacement_manifest.json')['videos']
    attempted = {}; stats = defaultdict(lambda: dict(attempted=0, dynamic1=0, same_prompt4_zero=0, zero_to_one_same_prompt4=0))
    bindings = {str(p): sha(p) for p in [Path(__file__), PURE, BACKGROUND, CURRENT/'replacement_manifest.json']}
    for directory in [ROOT/'outputs/entitybench_original40_scope32_20261108', ROOT/'outputs/entitybench_original40_next24_20261109']:
        bindings[str(directory/'plan.json')] = sha(directory/'plan.json')
        for c in read(directory/'plan.json')['cases']:
            gpath = directory/'cases'/c['case_id']/'generation.json'; spath = gpath.parent/'scores.json'
            g,s = read(gpath),read(spath)
            assert sha(g['video']) == s['video_sha256'] == g['video_sha256']
            key = c['episode_id']+'/'+c['shot']; attempted[key] = dict(case=c,generation=str(gpath),score=str(spath),
                generation_sha256=sha(gpath),score_sha256=sha(spath),dynamic=s['after']['dynamic_degree'],same_prompt4=s['before']['dynamic_degree'])
            bindings[str(gpath)] = sha(gpath); bindings[str(spath)] = sha(spath)
            st=stats[c['category']];st['attempted']+=1;st['dynamic1']+=s['after']['dynamic_degree']==1
            st['same_prompt4_zero']+=s['before']['dynamic_degree']==0
            st['zero_to_one_same_prompt4']+=s['before']['dynamic_degree']==0 and s['after']['dynamic_degree']==1
    ranked=[]; reuse=[]
    for c in pure['cases']:
        key=c['episode_id']+'/'+c['shot']
        if current[key]['cumulative_replaced']:
            continue
        if key in attempted:
            old=attempted[key];assert old['dynamic']==0
            assert old['case']['prompt']==c['new_prompt'] and old['case']['input_sha256']==c['input_sha256'] and old['case']['seed']==c['task']['seed']
            reuse.append(dict(case_id=c['case_id'],episode_id=c['episode_id'],shot=c['shot'],**{k:v for k,v in old.items() if k!='case'}))
            continue
        score,features=semantics(c);ref=reference4(c,bg)
        positive=ref is not None and ref['dynamic']==1
        ranked.append(dict(case_id=c['case_id'],episode_id=c['episode_id'],shot=c['shot'],category=c['category'],route=c['route'],
            evidence_tier=0 if positive else 1,semantic_priority_score=score,features=features,
            exact_same_prompt4=ref,priority_basis='verified_same_prompt4_dynamic1' if positive else 'description_based_hypothesis_not_probability'))
        if ref:
            for field in ['generation','score','receipt']:bindings[ref[field]]=ref[field+'_sha256']
    ranked.sort(key=lambda r:(r['evidence_tier'],-r['semantic_priority_score'],r['case_id']))
    for i,r in enumerate(ranked,1):r['priority_rank']=i
    assert len(ranked)==1277 and len(reuse)==28 and len(attempted)==56
    assert sum(r['evidence_tier']==0 for r in ranked)==30
    p=dict(scope='All remaining1305 current dynamic0 shots: reuse28 exact existing original40 negatives; new1277, ordered by verified same-prompt4 positives first, then explicit motion hypotheses. No candidates dropped.',
        baseline=str(CURRENT),prior_attempts=56,current_zero=1305,new_generation=1277,known_original40_zero_reused=28,
        existing_same_prompt4_positive=30,all1333_motion_prompts_compiled=True,prompt_edits=False,quality_gate_enabled=False,
        ordering='Evidence tiers are scheduling priorities, not calibrated probabilities. Historical56 are purposively sampled and coarse categories are noisy.',
        previous56_category_stats=dict(stats),cases=ranked,reused_zero_cases=reuse,
        bindings=[dict(path=k,sha256=v) for k,v in bindings.items()])
    immutable(OUT/'priority_manifest_v2.json',p)
    return p


if __name__=='__main__':
    p=build();print({k:p[k] for k in ['current_zero','new_generation','known_original40_zero_reused','existing_same_prompt4_positive']})
    print('First16:',[(r['case_id'],r['priority_basis'],r['semantic_priority_score']) for r in p['cases'][:16]])
