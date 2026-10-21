"""Independent face audit respecting the pinned official four-decimal publication."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_entitybench_qwen21_full140 import verify_adapters
from prepare_entitybench_qwen21_full140 import OUTPUT, read
from prepare_entitybench_qwen21_20ep import sha256, write_json
import evaluate_entitybench_face_only_dino_qwen140 as face


def validate_published_mean(published, per_episode_values):
    expected = round(float(np.mean(per_episode_values)), 4) if per_episode_values else None
    assert published == expected, (published, expected)
    return expected


def audit(root):
    root = Path(root).resolve()
    verify_adapters(root)
    face.ROOT = root
    face.OUT = out = root / 'evaluation/face_only_dino'
    face.ROOTS = {'A': root}
    face.helper.OFFICIAL = root / 'runtime/evaluate_benchmark_official.py'
    plan = face.initialize()
    manifest = read(out / 'face_crop_manifest.json')
    review = read(out / 'review/decisions.json')
    frozen = read(out / 'frozen_review.json')
    face.validate_review(manifest, review, sha256(out / 'face_crop_manifest.json'))
    assert manifest['plan_sha256'] == sha256(out / 'plan.json')
    assert frozen['review_sha256'] == sha256(out / 'review/decisions.json')
    assert frozen['face_crop_manifest_sha256'] == sha256(out / 'face_crop_manifest.json')
    assert frozen['reviewed_ids'] == review['reviewed_ids'] and frozen['rejected_ids'] == review['rejected_ids']
    for item in plan['inputs'] + plan['original_reports']:
        assert sha256(item['path']) == item['sha256']
    report_path = out / 'report.json'
    report = read(report_path)
    report_sha = sha256(report_path)
    assert report['status'] == 'complete' and report['vlm_calls'] == 0 and not report['insightface_used']
    assert report['plan_sha256'] == sha256(out / 'plan.json')
    assert report['frozen_review_sha256'] == sha256(out / 'frozen_review.json')
    rows = manifest['rows']
    assert len(plan['episodes']) == 140 and sum(len(e['shots']) for e in plan['episodes']) == 2491
    rejected = set(review['rejected_ids'])
    covered, appearances, missing, scores, deserialized = 0, 0, Counter(), [], []
    scorer = face.helper.official()
    for episode in plan['episodes']:
        eid = episode['episode_id']
        selected = [row for row in rows if row['episode_id'] == eid]
        assert len(selected) == sum(len(shot['anchors']) for shot in episode['shots'])
        pools, skipped = defaultdict(list), 0
        for row in selected:
            if not row['crop_path'] or row['id'] in rejected:
                missing['review_rejected_false_face' if row['id'] in rejected else row['reason']] += 1
                skipped += 1
                continue
            assert sha256(row['crop_path']) == row['crop_sha256']
            vector = np.load(out / 'embeddings' / (row['crop_sha256'] + '.npy'))
            assert np.isfinite(vector).all() and abs(np.linalg.norm(vector) - 1) < 1e-5
            pools[row['entity']].append(vector)
            covered += 1
        similarities = []
        for vectors in pools.values():
            if len(vectors) < 2:
                continue
            matrix = np.stack(vectors)
            centroid = matrix.mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            similarities.extend((matrix @ centroid).tolist())
        expected = float(np.mean(similarities)) if similarities else None
        cached = read(out / 'results/A' / (eid + '.json'))
        metric = cached['metrics']['cs_face']
        assert (metric['value'] is None) == (expected is None)
        if expected is not None:
            assert abs(metric['value'] - expected) < 1e-6
            scores.append(expected)
        assert metric['n_evaluated'] == len(similarities) and metric['n_skipped'] == skipped and metric['n_failed'] == 0
        assert cached['num_shots'] == len(episode['shots']) and cached['tier'] == episode['tier']
        appearances += len(similarities)
        deserialized.append(scorer._episode_result_from_json(eid, episode['tier'], cached))
    actual = report['routes']['A']
    assert actual['scheduled_role_shots'] == len(rows) and actual['valid_face_role_shots'] == covered
    assert actual['face_coverage'] == covered / len(rows)
    assert actual['cs_face_appearances'] == appearances and actual['n_missing_by_reason'] == dict(missing)
    split = read(root / 'split_full140.json')
    recomputed = scorer.aggregate_results(deserialized, split)
    assert recomputed == actual['aggregation'], 'Official face aggregation differs'
    published = validate_published_mean(actual['aggregation']['all']['cs_face']['mean'], scores)
    assert sha256(report_path) == report_sha
    result = dict(status='passed', episodes=140, videos=2491, scheduled_role_shots=len(rows),
        mapped_face_role_shots=covered, coverage=covered / len(rows), valid_episodes=len(scores),
        appearances=appearances, missing_by_reason=dict(missing),
        raw_episode_macro_mean=float(np.mean(scores)) if scores else None,
        official_published_mean=published, official_publication_decimals=4,
        official_deserializer_and_aggregation_verified=True, report_sha256=report_sha,
        frozen_review_sha256=sha256(out / 'frozen_review.json'), source_script_sha256=sha256(__file__))
    write_json(out / 'independent_validation.json', result)
    write_json(out / 'status.json', dict(phase='face_only_dino_complete', episodes_per_route=140,
        videos_per_route=2491, vlm_calls=0, insightface_used=False, independent_audit=str(out / 'independent_validation.json')))
    print('Face-only audit passed: per-episode raw centroids and official four-decimal aggregation unchanged', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    audit(parser.parse_args().output_root)
