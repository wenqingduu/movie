"""Full140 default region-DINO, preserving prior reviews on exact crop pixels."""
from __future__ import annotations
import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path
import random
import shutil

import qwen140_badshot_full_evaluation_20261104 as full
import qwen140_rerun_region_diagnostic as diag

ROOT, BASE = full.ROOT, full.BASE
OUT = full.OUT / 'visible_region'
OLD_REGIONS = [ROOT / 'outputs/entitybench_rerun30_latest_evaluation_20261101/visible_region',
               ROOT / 'outputs/entitybench_completed35_comparison_20261102/new5_visible_region',
               ROOT / 'outputs/entitybench_full140_completed41_snapshot_20261103/new6_visible_region',
               full.SNAPSHOT / 'new10_visible_region']
read, write, sha = diag.read, diag.write, diag.sha


def before_state():
    rows, original_exclusions = diag.historical()
    mapping = {diag.key(r): r for r in rows}
    exclusions = list(original_exclusions.values())
    for directory in OLD_REGIONS:
        assert read(directory / 'report.json').get('status', read(directory / 'report.json').get('stage')) == 'complete_with_localization_review'
        for path in (directory / 'extraction').glob('*/*.json'):
            for row in read(path)['rows']:
                mapping[diag.key(row)] = row
        review = read(directory / 'review/decisions.json')
        assert review['manifest_sha256'] == sha(directory / 'review/manifest.json')
        exclusions += [r for r in review['decisions'] if r['decision'] == 'exclude']
    assert len(mapping) == 4991
    rejected = {diag.key(r): r for r in exclusions if mapping[diag.key(r)]['original_crop_sha256'] == r['original_crop_sha256']}
    return mapping, rejected, exclusions


def prepare():
    OUT.mkdir(exist_ok=True)
    source = read(full.OUT / 'replacement_manifest.json')
    episodes = []
    changed_ids = {k.split('/')[0] for k in source['targets']}
    for ep in read(BASE / 'manifest.json')['episodes']:
        eid = ep['episode_id']
        results = full.OUT / ('affected' if eid in changed_ids else 'full140') / 'results'
        result = results / (eid + '.json')
        episodes.append({'episode_id': eid, 'episode': ep, 'output_root': str(full.OUT / 'episode_media'),
                         'result': str(result), 'result_sha256': sha(result),
                         'artifacts': str(results / '.work/eval_artifacts' / eid)})
    diag.immutable(OUT / 'input_manifest.json', {'episodes': episodes, 'selected_episode_count': 140,
                   'selected_shot_count': 2491, 'scope': 'Complete mixed full140 after selected-shot replacements; preserve all canonical regions and exact evaluation protocol.'})
    shutil.copyfile(OLD_REGIONS[0] / 'required_span_validation.json', OUT / 'required_span_validation.json')
    diag.OUT = OUT
    return episodes, source


def review_sheets(scored, episodes):
    from PIL import Image, ImageDraw, ImageOps
    candidates = sorted((r for r in scored if r['edition'] == 'rerun'), key=lambda r: (r['similarity'], diag.key(r)))
    if len(candidates) >= 80:
        chosen = candidates[:60] + random.Random(20261025).sample(candidates[60:], 20)
    else:
        chosen = list(candidates)
    random.Random(20261101).shuffle(chosen)
    rows = [{**{k: v for k, v in row.items() if k != 'similarity'}, 'review_id': f'N{i:03d}'} for i, row in enumerate(chosen, 1)]
    diag.immutable(OUT / 'review/manifest.json', {'selection': 'Changed target-video crops with new pixels only: lowest60 + deterministic random20; review all if fewer80. Shared old crops never newly corrected on just one side. Scores hidden.',
                    'scored_crops_sha256': sha(OUT / 'scored_crops.json'), 'rows': rows})
    epmap = {e['episode_id']: e['episode'] for e in episodes}
    for start in range(0, len(rows), 6):
        canvas = Image.new('RGB', (1600, 1290), '#ededed')
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(rows[start:start + 6]):
            x, y = i % 2 * 800, i // 2 * 430
            ep = epmap[row['episode_id']]
            draw.text((x + 6, y + 4), f"{row['review_id']} {ep['run']} {ep['episode_id'].split('__')[2]} {row['shot']} target:{row['character']}", font=diag.FONT, fill='black')
            context = Image.open(OUT / 'contexts' / row['episode_id'] / (row['shot'].replace(':', '_') + f"_f{row['sample_idx']}.jpg")).convert('RGB')
            mark = ImageDraw.Draw(context)
            mark.rectangle(row['region'], outline='red', width=3)
            for candidate in row['candidates']:
                mark.rectangle(candidate['bbox'], outline='yellow', width=2)
            canvas.paste(ImageOps.contain(context, (590, 315)), (x + 5, y + 35))
            crop = Image.open(row['original_crop']).convert('RGB')
            canvas.paste(ImageOps.contain(crop, (180, 285)), (x + 607, y + 36))
            prompt = next(s['prompt'] for s in ep['shots'] if s['shot_key'] == row['shot'])
            for line in range(3):
                draw.text((x + 6, y + 350 + 20 * line), prompt[line * 100:(line + 1) * 100], font=diag.FONT, fill='black')
        canvas.save(OUT / 'review' / f'page_{start // 6 + 1:02d}.jpg', quality=95)
    write(OUT / 'review/role_descriptions.json', {e['episode_id']: e['episode']['entity_descriptions'] for e in episodes})
    return len(rows)


def score(final=False):
    episodes, source = prepare()
    before_map, before_rejected, old_exclusions = before_state()
    newmap = {}
    for item in episodes:
        files = list((OUT / 'extraction' / item['episode_id']).glob('*.json'))
        assert len(files) == len(item['episode']['shots'])
        for path in files:
            for row in read(path)['rows']:
                if row['original_crop']:
                    assert sha(row['original_crop']) == row['original_crop_sha256']
                newmap[diag.key(row)] = row
    assert set(newmap) == set(before_map) and len(newmap) == 4991
    rejected = {diag.key(r): r for r in old_exclusions if newmap[diag.key(r)]['original_crop_sha256'] == r['original_crop_sha256']}
    for key, row in newmap.items():
        row['edition'] = ('rerun' if key[0] + '/' + key[2] in source['targets'] and
                          row['original_crop_sha256'] != before_map[key]['original_crop_sha256'] else 'unchanged')
    review = None
    if final:
        manifest = read(OUT / 'review/manifest.json')
        review = read(OUT / 'review/decisions.json')
        assert review['manifest_sha256'] == sha(OUT / 'review/manifest.json')
        expected = {r['review_id']: r for r in manifest['rows']}
        assert {r['review_id'] for r in review['decisions']} == set(expected)
        assert len(review['decisions']) == len(expected)
        for row in review['decisions']:
            ref = expected[row['review_id']]
            assert diag.key(ref) == diag.key(row) and ref['original_crop_sha256'] == row['original_crop_sha256']
            assert row['decision'] in ['keep', 'exclude'] and row['reason'] and row['evidence']
            if row['decision'] == 'exclude':
                rejected[diag.key(row)] = row
    module = diag.gate.load_official_pure()
    selected = {e['episode_id']: e for e in read(full.OUT / 'baseline_selection.json')['episodes']}
    baseline_metrics = {e['episode_id']: e for e in read(full.SNAPSHOT / 'reviewed_dino_full140.json')['per_episode']}
    old_results, new_results, scored, details, shot_dino = [], [], [], [], {}
    for item in episodes:
        ep, eid = item['episode'], item['episode_id']
        old, old_entities = diag.compute(ep, read(selected[eid]['selected_result']), before_map, before_rejected, module)
        expected = baseline_metrics[eid]['current']
        assert old.metrics['cs_face'].n_evaluated == expected['n_evaluated']
        actual = old.metrics['cs_face'].value
        assert (actual is None and expected['value'] is None) or abs(actual - expected['value']) < 1e-8
        new, entities = diag.compute(ep, read(item['result']), newmap, rejected, module)
        old_results.append(old)
        new_results.append(new)
        details.append({'episode_id': eid, 'before': old.metrics['cs_face'].to_json(), 'after': new.metrics['cs_face'].to_json()})
        for label, groups in [('before', old_entities), ('after', entities)]:
            for name, group in groups.items():
                for app in group['per_shot_sim_to_centroid']:
                    shot_dino.setdefault((eid, app['shot']), {}).setdefault(label, []).append(app['sim_to_centroid'])
                    if label == 'after':
                        scored.append({**newmap[eid, name, app['shot']], 'similarity': app['sim_to_centroid']})
    split = read(BASE / 'split_full140.json')
    report = {'metric': 'cs_face_visible_single_face_reviewed' if final else 'cs_face_visible_single_face_auto',
              'status': 'complete_with_localization_review' if final else 'automatic_pending_localization_review',
              'scope': 'Full140 episode-macro; same canonical person-region pixels, RetinaFace gate and centroid definition; not official raw cs_face.',
              'episodes': 140, 'scheduled_role_shots': len(newmap), 'before': module.aggregate_results(old_results, split),
              'after': module.aggregate_results(new_results, split), 'per_episode': details,
              'coverage': {'before_scored': sum(r.metrics['cs_face'].n_evaluated for r in old_results),
                           'after_scored': sum(r.metrics['cs_face'].n_evaluated for r in new_results),
                           'after_present': sum(r['original_status'] == 'present' for r in newmap.values()),
                           'after_auto_eligible': sum(r['kept'] for r in newmap.values()),
                           'after_retained': sum(r['kept'] and k not in rejected for k, r in newmap.items()),
                           'gate_reasons': dict(Counter(r['reason'] for r in newmap.values()))},
              'manual_review_items': len(review['decisions']) if review else 0, 'exclusions': list(rejected.values()), 'vlm_calls': 0}
    write(OUT / ('report.json' if final else 'automatic_report.json'), report)
    if not final:
        write(OUT / 'scored_crops.json', scored)
        count = review_sheets(scored, episodes)
        write(OUT / 'status.json', {'stage': 'localization_review_required', 'items': count, 'default_reviewed_metric_final': False})
    else:
        comparison = read(full.OUT / 'target_shot_comparison.json')
        for row in comparison['rows']:
            for label in ['before', 'after']:
                values = shot_dino.get((row['episode_id'], row['shot']), {}).get(label, [])
                row[label]['cs_face_visible_single_face_reviewed'] = math.fsum(values) / len(values) if values else None
        comparison['reviewed_dino_status'] = 'complete_with_localization_review'
        write(full.OUT / 'target_shot_comparison_with_reviewed_dino.json', comparison)
        columns = ['episode_id', 'shot'] + [f'{metric}_{label}' for metric in [*full.agg.LABELS, 'cs_face_visible_single_face_reviewed'] for label in ['before', 'after', 'delta']]
        with (full.OUT / 'target_shot_comparison_with_reviewed_dino.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in comparison['rows']:
                flat = {k: row[k] for k in ['episode_id', 'shot']}
                for name in [*full.agg.LABELS, 'cs_face_visible_single_face_reviewed']:
                    a, b = row['before'].get(name), row['after'].get(name)
                    flat.update({name + '_before': a, name + '_after': b,
                                 name + '_delta': b - a if a is not None and b is not None else None})
                writer.writerow(flat)
        write(full.OUT / 'reviewed_full140_summary.json', {'original': read(full.SNAPSHOT / 'reviewed_dino_full140.json')['baseline']['all']['cs_face'],
              'before_v2': report['before']['all']['cs_face'], 'after_v2': report['after']['all']['cs_face']})
        combined = read(full.OUT / 'full140_official_comparison.json')
        summary = read(full.OUT / 'reviewed_full140_summary.json')
        combined['metrics'].insert(0, {'metric': 'cs_face_visible_single_face_reviewed', 'label': '人物一致性DINO（可见单脸＋定位纠错）',
                                      **{k: v['mean'] for k, v in summary.items()}})
        combined['default_reviewed_dino'] = 'complete_with_localization_review'
        combined['reviewed_dino_coverage'] = report['coverage']
        write(full.OUT / 'full140_all_metrics_comparison.json', combined)
        text = '# 全140集完整可用指标对比\n\n官方strict/skip_vlm；raw cs_face/cs_object无fidelity门控，默认DINO为可见单脸＋定位纠错条件诊断。VLM依赖项仍为空。\n\n| 指标 | 原始benchmark | v2替换前 | v2替换后 |\n|---|---:|---:|---:|\n'
        for row in combined['metrics']:
            text += '| ' + row['label'] + ' | ' + ' | '.join('—' if row[k] is None else f'{row[k]:.4f}' for k in ['original', 'before_v2', 'after_v2']) + ' |\n'
        (full.OUT / 'complete_report.md').write_text(text)
        full.status('complete_all_available_metrics_with_reviewed_dino', episodes=140, reviewed_items=report['manual_review_items'])
        batch_status = read(full.BATCH / 'status.json')
        batch_status.update(state='complete_all_available_metrics_with_reviewed_dino', followup_status=str(full.OUT / 'status.json'))
        write(full.BATCH / 'status.json', batch_status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['auto', 'final'])
    args = parser.parse_args()
    if args.action == 'auto':
        prepare()
        diag.extract('cpu')
    score(args.action == 'final')
