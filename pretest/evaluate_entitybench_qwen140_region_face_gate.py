"""Post-hoc visible-face gate on immutable official character-region crops.

This is a separate diagnostic, not a replacement for EntityBench scores.
No new localization, CLIP, DINO or VLM inference is performed. Saved pairwise
DINO dot products reconstruct an equivalent Gram representation; the frozen
official centroid function and aggregator compute the scores. Full baseline
reproduction is mandatory before computing the gated variant.
"""
from __future__ import annotations

import argparse
import ast
import collections
import dataclasses
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import types
import typing

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path('/root/autodl-tmp/movie')
RUN = PROJECT / 'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140'
EVAL = RUN / 'evaluation/official_parallel_v2'
FACES = RUN / 'evaluation/face_only_dino'
OUT = RUN / 'evaluation/character_region_visible_face_gate_20261025'
OFFICIAL = RUN / 'runtime/evaluate_benchmark_official.py'
CONFIG = dict(
    version=1,
    population='original canonical character crops with status=present only',
    face_detector='reuse frozen RetinaFace ResNet50 cache, confidence >=0.9 and native width/height >=24px',
    min_face_area_inside_region=0.80,
    unique_face='exactly one qualifying detected face inside the padded canonical crop at its original sample',
    role_conflict='reject all roles claiming the same cached face track within one shot, across sample times',
    unmodified='original GDINO query, CLIP, thresholds, canonical pixels, padding, sampling, DINO similarities and official centroid/aggregation functions',
    exclusions='automatic spatial rules only; no identity cosine, DINO-score filtering, manual score-based rejection or parameter tuning',
    missing='missing/ambiguous/back-view/nonhuman appearances are not assigned zero; report coverage separately',
    scoring='reconstruct saved unit-feature Gram matrices, validate original scores, then recompute centroids on retained crops',
    scope='non-VLM diagnostic on visible single-face character regions; not paper-comparable official cs_face',
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def immutable(path, obj):
    if path.exists():
        assert read(path) == obj, f'Frozen inputs/protocol changed: {path}'
    else:
        write(path, obj)


def containment(face, region):
    x1, y1, x2, y2 = face
    a, b, c, d = region
    inter = max(0., min(x2, c)-max(x1, a))*max(0., min(y2, d)-max(y1, b))
    return inter / max(1., (x2-x1)*(y2-y1))


def prepare():
    """Freeze all gate decisions without reading DINO scores or pairwise data."""
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(RUN/'manifest.json')
    plan = read(FACES/'plan.json')
    official_report = EVAL/'report_qwen21_native_reference_a14b_4step_continuation_full140_no_vlm.json'
    inputs = [RUN/'manifest.json', RUN/'generation_report.json', FACES/'plan.json',
              official_report, OFFICIAL, Path(__file__)]
    rows = []
    for ep in manifest['episodes']:
        eid = ep['episode_id']
        cache = FACES/'extraction/A'/f'{eid}.json'
        saved = read(cache)
        assert saved['plan_sha256'] == sha(FACES/'plan.json')
        assert saved['episode_id'] == eid
        inputs.extend([cache, EVAL/f'{eid}.json'])
        anchors = {s['shot_key']: s for e in plan['episodes'] if e['episode_id'] == eid for s in e['shots']}
        audits = {s['shot_key']: s for s in saved['shot_audits']}
        for r in saved['rows']:
            sk, name = r['shot_key'], r['entity']
            anchor = next(a for a in anchors[sk]['anchors'] if a['entity'] == name)
            assert anchor['original_status'] == r['original_character_status']
            row = dict(episode_id=eid, tier=ep['difficulty'], character=name, shot=sk,
                       original_status=r['original_character_status'],
                       original_crop=anchor['canonical_path'],
                       original_crop_sha256=anchor['canonical_sha256'],
                       sample_idx=anchor['original_sample_idx'],
                       region=r['original_region'], kept=False, reason='original_not_present',
                       candidates=[], face_track_id=None)
            if row['original_status'] == 'present':
                assert row['original_crop'] and row['region'] is not None
                assert sha(row['original_crop']) == row['original_crop_sha256']
                assert Image.open(row['original_crop']).size == (row['region'][2]-row['region'][0], row['region'][3]-row['region'][1])
                for track in audits[sk]['tracks']:
                    for det in track['items']:
                        if det['sample_idx'] != row['sample_idx']:
                            continue
                        b = det['bbox']
                        if det['confidence'] < .9 or min(b[2]-b[0], b[3]-b[1]) < 24:
                            continue
                        overlap = containment(b, row['region'])
                        if overlap >= CONFIG['min_face_area_inside_region']:
                            row['candidates'].append(dict(track_id=track['track_id'], bbox=b, face_coverage=overlap, confidence=det['confidence']))
                n = len(row['candidates'])
                row['reason'] = 'no_complete_face_inside_region' if n == 0 else 'multiple_faces_inside_region' if n > 1 else 'unique_visible_face'
                row['kept'] = n == 1
                if n == 1:
                    row['face_track_id'] = row['candidates'][0]['track_id']
            rows.append(row)
    claims = collections.defaultdict(list)
    for r in rows:
        if r['kept']:
            claims[(r['episode_id'], r['shot'], r['face_track_id'])].append(r)
    for claim in claims.values():
        if len(claim) > 1:
            for r in claim:
                r['kept'], r['reason'] = False, 'same_face_track_claimed_by_multiple_roles'
    assert len(rows) == 4991
    frozen = dict(config=CONFIG, inputs=[dict(path=str(p), sha256=sha(p)) for p in inputs],
                  episodes=len(manifest['episodes']), scheduled_slots=len(rows))
    immutable(OUT/'plan.json', frozen)
    immutable(OUT/'gate_decisions.json', dict(plan_sha256=sha(OUT/'plan.json'), rows=rows))
    print('Frozen gate decisions before scoring:', dict(collections.Counter(r['reason'] for r in rows)), flush=True)


def load_official_pure():
    """Load exact official arithmetic without importing CUDA/model libraries."""
    names = {'MetricValue', 'EntityCrop', 'ShotResult', 'EpisodeResult',
             'compute_cross_shot_consistency', 'aggregate_results',
             '_episode_result_to_json', '_episode_result_from_json'}
    constants = {'ALL_METRICS', 'VBENCH_DIMENSIONS', '_LEGACY_METRIC_NAMES'}
    nodes = []
    for node in ast.parse(OFFICIAL.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
    module = types.ModuleType('official_visible_face_gate_arithmetic')
    sys.modules[module.__name__] = module
    module.__dict__.update(vars(typing))
    module.__dict__.update(np=np, dataclass=dataclasses.dataclass, field=dataclasses.field,
                           defaultdict=collections.defaultdict, os=os, re=re, Image=Image,
                           logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(OFFICIAL), 'exec'), module.__dict__)
    return module


def gram_to_features(gram):
    values, vectors = np.linalg.eigh(gram)
    assert values.min() > -1e-5, f'Invalid saved Gram matrix: min eigenvalue {values.min()}'
    features = vectors * np.sqrt(np.maximum(values, 0))[None, :]
    assert np.max(np.abs(features@features.T-gram)) < 1e-5
    return features


def validate_subset_arithmetic():
    rng = np.random.default_rng(20261025)
    x = rng.normal(size=(11, 48))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    g = x@x.T
    y = gram_to_features(g)
    for ids in [list(range(11)), [0, 2, 5, 8], [1, 7]]:
        a, b = x[ids], y[ids]
        ac, bc = a.mean(0), b.mean(0)
        expected = a@(ac/np.linalg.norm(ac))
        actual = b@(bc/np.linalg.norm(bc))
        assert np.max(np.abs(expected-actual)) < 1e-12


def score():
    frozen = read(OUT/'plan.json')
    assert frozen['config'] == CONFIG
    for source in frozen['inputs']:
        assert sha(source['path']) == source['sha256'], f'Input changed: {source["path"]}'
    gate_hash = sha(OUT/'gate_decisions.json')
    rows = read(OUT/'gate_decisions.json')['rows']
    by_key = {(r['episode_id'], r['character'], r['shot']): r for r in rows}
    assert len(by_key) == 4991
    manifest = read(RUN/'manifest.json')
    mod = load_official_pure()
    validate_subset_arithmetic()
    baseline, gated, survivor_old_centroids = [], [], []
    recovered = []
    validation = dict(max_baseline_episode_error=0., max_baseline_appearance_error=0., checked_roles=0, checked_pairs=0)
    # First recover and validate EVERY role and episode before variant scores.
    for ep in manifest['episodes']:
        eid = ep['episode_id']
        result = read(EVAL/f'{eid}.json')
        order = [s['shot_key'] for s in ep['shots']]
        pos = {s: i for i, s in enumerate(order)}
        pools = {}
        baseline_sims = []
        for key, data in result['per_entity_scores'].items():
            if not key.startswith('face_'):
                continue
            name = key[5:]
            appearances = sorted(data['per_shot_sim_to_centroid'], key=lambda a: pos[a['shot']])
            n = len(appearances)
            assert n == data['n_appearances'] and n >= 2
            pairs = [p for p in result['gap_decay_data'] if p['entity_type'] == 'character' and p['entity_name'] == name]
            assert len(pairs) == n*(n-1)//2
            gram = np.eye(n, dtype=np.float64)
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    pair = pairs[k]
                    assert pair['gap'] == abs(pos[appearances[j]['shot']]-pos[appearances[i]['shot']])
                    gram[i, j] = gram[j, i] = pair['similarity']
                    k += 1
            features = gram_to_features(gram)
            unit_center = features.mean(0)
            unit_center /= np.linalg.norm(unit_center)
            similarities = features@unit_center
            error = float(np.max(np.abs(similarities-np.array([a['sim_to_centroid'] for a in appearances]))))
            assert error < 2e-6, (eid, name, error)
            validation['max_baseline_appearance_error'] = max(validation['max_baseline_appearance_error'], error)
            validation['checked_roles'] += 1
            validation['checked_pairs'] += len(pairs)
            crops = []
            for a, feature in zip(appearances, features):
                row = by_key[(eid, name, a['shot'])]
                assert row['original_status'] == 'present'
                crops.append(mod.EntityCrop(name, 'character', name, a['shot'], row['sample_idx'], tuple(row['region']), row['original_crop'], a['clip_text_sim'], feature))
            pools[name] = crops
            baseline_sims.extend(similarities.tolist())
        expected = result['metrics']['cs_face']['value']
        delta = abs(float(np.mean(baseline_sims))-expected)
        assert delta < 2e-6
        validation['max_baseline_episode_error'] = max(validation['max_baseline_episode_error'], delta)
        recovered.append((ep, result, pools))
        baseline.append(mod._episode_result_from_json(eid, ep['difficulty'], result))
    split = {tier: [e['episode_id'] for e in manifest['episodes'] if e['difficulty'] == tier] for tier in ('easy', 'medium', 'hard')}
    original_agg = mod.aggregate_results(baseline, split)
    published = read(EVAL/'report_qwen21_native_reference_a14b_4step_continuation_full140_no_vlm.json')
    for tier in split.keys() | {'all'}:
        assert original_agg[tier]['cs_face'] == published[tier]['cs_face']
    print('All baseline roles/episodes reproduced before variant scoring.', validation, flush=True)
    diagnostics = []
    for ep, original, pools in recovered:
        eid = ep['episode_id']
        subset = {name: [c for c in crops if by_key[(eid, name, c.shot_key)]['kept']] for name, crops in pools.items()}
        order = [s['shot_key'] for s in ep['shots']]
        value, entities, n_scored = mod.compute_cross_shot_consistency(subset, 'character', shot_order=order)
        independent_sims = []
        fixed_original_sims = []
        for name, crops in subset.items():
            if len(crops) < 2:
                continue
            original_map = {a['shot']: a['sim_to_centroid'] for a in original['per_entity_scores']['face_'+name]['per_shot_sim_to_centroid']}
            gram = np.array([[float(np.dot(a.dino_embedding, b.dino_embedding)) for b in crops] for a in crops])
            # Exact centroid similarities depend only on Gram row sums.
            sims = gram.sum(axis=1)/math.sqrt(float(gram.sum()))
            independent_sims.extend(sims.tolist())
            fixed_original_sims.extend(original_map[c.shot_key] for c in crops)
        assert n_scored == len(independent_sims)
        assert value is None if not independent_sims else abs(value-float(np.mean(independent_sims))) < 1e-10
        n_kept = sum(r['kept'] for r in rows if r['episode_id'] == eid)
        result = mod.EpisodeResult(eid, ep['difficulty'], len(ep['shots']))
        result.metrics['cs_face'] = mod.MetricValue(value=value, n_evaluated=n_scored)
        gated.append(result)
        fixed = mod.EpisodeResult(eid, ep['difficulty'], len(ep['shots']))
        fixed.metrics['cs_face'] = mod.MetricValue(value=float(np.mean(fixed_original_sims)) if fixed_original_sims else None, n_evaluated=n_scored)
        survivor_old_centroids.append(fixed)
        episode_data = dict(episode_id=eid, tier=ep['difficulty'], num_shots=len(ep['shots']),
                            metrics={'cs_face': result.metrics['cs_face'].to_json()},
                            n_crops_passing_gate=n_kept, per_entity_scores=entities)
        write(OUT/'results'/f'{eid}.json', episode_data)
        diagnostics.append(dict(episode_id=eid, original=original['metrics']['cs_face']['value'],
                                gated=value, n_scored=n_scored, n_crops_passing_gate=n_kept,
                                retained_original_centroid_mean=fixed.metrics['cs_face'].value))
    stats = collections.Counter(r['reason'] for r in rows)
    n_present = sum(r['original_status'] == 'present' for r in rows)
    kept = sum(r['kept'] for r in rows)
    n_scored = sum(r.metrics['cs_face'].n_evaluated for r in gated)
    agg = mod.aggregate_results(gated, split)
    fixed_agg = mod.aggregate_results(survivor_old_centroids, split)
    report = dict(status='complete', scope=CONFIG['scope'], config=CONFIG,
                  plan_sha256=sha(OUT/'plan.json'), gate_decisions_sha256=gate_hash,
                  official_source_sha256=sha(OFFICIAL), validation=validation,
                  baseline=original_agg, visible_face_gate=agg,
                  retained_crops_against_original_centroids=fixed_agg,
                  coverage=dict(scheduled_slots=len(rows), original_present_crops=n_present,
                                original_scored_appearances=sum(r.metrics['cs_face'].n_evaluated for r in baseline),
                                gated_crops=kept, gated_scored_appearances=n_scored,
                                gated_crop_coverage_of_scheduled=kept/len(rows),
                                gated_crop_retention_of_present=kept/n_present,
                                scored_episode_count=agg['all']['cs_face']['n_episodes_evaluated']),
                  rejection_counts=dict(stats), per_episode=diagnostics,
                  new_model_inference_calls=0, vlm_calls=0,
                  interpretation='Same original canonical pixels; score changes reflect eligibility and centroid population changes, not improved generated media or repaired crops. Gate does not verify identity, and missing back/profile/occluded/nonhuman appearances lower coverage.')
    write(OUT/'report.json', report)
    assert sha(OUT/'gate_decisions.json') == gate_hash
    for source in frozen['inputs']:
        assert sha(source['path']) == source['sha256']
    review(rows)
    lines = ['# 官方人物区域DINO：可见单脸门控变体', '',
             '这是用户授权的独立诊断，不修改官方0.7155，不是完整论文口径，也不是人脸身份准确率。', '',
             '## 结果', '', '| 版本 | 分数 | 可评分集 | 参与评分的角色—镜头数 |', '|---|---:|---:|---:|',
             f"| 原官方非VLM | {original_agg['all']['cs_face']['mean']:.4f} | 140 | {report['coverage']['original_scored_appearances']} |",
             f"| 原人物区域＋可见单脸门控 | {agg['all']['cs_face']['mean']:.4f} | {agg['all']['cs_face']['n_episodes_evaluated']} | {n_scored} |", '',
             f'4991个计划角色—镜头，原present裁剪{n_present}项；门控保留{kept}项，占计划集合{kept/len(rows):.2%}、原present集合{kept/n_present:.2%}。只保留同角色至少两个镜头的appearance参与质心评分。', '',
             '## 改动与边界', '',
             '沿用原GroundingDINO＋CLIP流程已选出的canonical裁剪，不换prompt、不调阈值、不重新生成或改裁剪像素。缓存未保存全部GroundingDINO候选框，因此本次不能在不新增推理的情况下重新选候选框。', '',
             '规则在算变体分数前写入plan.json和gate_decisions.json：复用全部RetinaFace检测；原代表框内恰有一张至少80%面积被包含的有效脸（置信度≥0.9、宽高≥24px）；不同角色争用同一镜头内同一人脸轨迹时全部排除。只从原present集合筛选，无脸／多人歧义／冲突记缺失，不填0；没有按DINO分数筛图或调规则。', '',
             '这些规则筛选可见且归属较明确的人物区域，仍不能验证是不是正确身份，也可能丢掉符合剧情的背影、遮挡或多人镜头。因此分数上升不能证明原视频质量改善，必须同时看覆盖率。', '',
             '## 计算核验', '',
             '复用官方保存的每角色全部两两DINO相似度，按官方固定的上三角顺序和shot gap逐对校验，构造Gram矩阵及等价特征。先重现全部原角色、原集分数，再调用冻结官方质心函数、反序列化器和聚合器计算变体。另用Gram行和公式独立验证门控后分数；无需GPU或重新提取DINO特征。', '',
             f"原始appearance分数最大误差{validation['max_baseline_appearance_error']:.3g}，原集分数最大误差{validation['max_baseline_episode_error']:.3g}；原四位小数及各难度聚合完全重现。", '',
             f"为区分样本选择和质心变化，保留集合仍对原质心计算时的集均值为{fixed_agg['all']['cs_face']['mean']:.4f}；重建质心后的{agg['all']['cs_face']['mean']:.4f}是另一个样本集合上的一致性，不是配对生成增益。", '',
             '## 门控原因计数', '', '| 原因 | 数量 |', '|---|---:|']
    lines += [f'| {key} | {value} |' for key, value in stats.items()]
    lines += ['', '代表框保留／排除示例见[门控检查页](gate_review.jpg)。完整数值见[report.json](report.json)，规则见[plan.json](plan.json)，逐项决策见[gate_decisions.json](gate_decisions.json)。检查页按原因选例，不代表各类的随机错误率。', '',
              '后续若要真正修复裁剪，应在所有GroundingDINO候选框上先做空间有效性检查，再按原CLIP×清晰度×面积选代表框；新候选的CLIP与DINO需要重新计算，作为另一项明确标注的消融。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: report[k] for k in ('coverage', 'rejection_counts', 'validation')}, ensure_ascii=False, indent=2))
    print('Official:', original_agg['all']['cs_face'], 'Gated:', agg['all']['cs_face'], flush=True)


def review(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        if r['original_status'] == 'present':
            groups[r['reason']].append(r)
    examples = []
    for reason, rs in sorted(groups.items()):
        examples.extend(sorted(rs, key=lambda r: (r['episode_id'], r['shot'], r['character']))[:3])
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    cellw, cellh = 360, 300
    canvas = Image.new('RGB', (cellw*3, 40+math.ceil(len(examples)/3)*cellh), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), 'Original canonical pixels | automatic gate decisions, no score filtering', font=font, fill='black')
    for i, row in enumerate(examples):
        x, y = (i % 3)*cellw, 40+(i//3)*cellh
        draw.text((x+5, y+3), row['reason'][:43], font=font, fill='black')
        draw.text((x+5, y+25), f"{row['episode_id'].split('__')[1]} {row['character']} {row['shot']}", font=font, fill='black')
        im = Image.open(row['original_crop']).convert('RGB')
        im.thumbnail((350, 240))
        canvas.paste(im, (x+(cellw-im.width)//2, y+53+(240-im.height)//2))
    canvas.save(OUT/'gate_review.jpg', quality=93)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'score'])
    args = parser.parse_args()
    {'prepare': prepare, 'score': score}[args.stage]()
