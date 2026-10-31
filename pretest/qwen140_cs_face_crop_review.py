#!/usr/bin/env python3
"""Render saved official character crops for review; never alter score inputs."""
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def build(item: dict) -> Path:
    directory = Path(item['output_root'])
    dest = directory / 'review/cs_face_crops'
    dest.mkdir(parents=True, exist_ok=True)
    paths = {'old': Path(item['baseline_result']),
             'new': directory / 'evaluation/weak4_v1/results' / (item['episode_id'] + '.json')}
    results = {k: json.loads(p.read_text()) for k, p in paths.items()}
    roots = {k: p.parent / '.work/eval_artifacts' / item['episode_id'] for k, p in paths.items()}
    roles = sorted({name for shot in item['episode']['shots']
                    for name in shot['entity_schedule'].get('characters', [])})
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    inventory, sections = [], []
    for role in roles:
        shots = [s['shot_key'] for s in item['episode']['shots']
                 if role in s['entity_schedule'].get('characters', [])]
        cw, ch = 166, 248
        sheet = Image.new('RGB', (cw * len(shots), ch * 2 + 36), '#151515')
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 7), f'{item["episode"]["run"]} / {role} | top original, bottom rerun', font=font, fill='white')
        for version, oy in [('old', 36), ('new', 36 + ch)]:
            entity = results[version]['per_entity_scores'].get('face_' + role, {})
            scores = {v['shot']: v for v in entity.get('per_shot_sim_to_centroid', [])}
            for j, shot in enumerate(shots):
                source = roots[version] / 'intra_shot' / shot.replace(':', '_') / '_eval.json'
                entries = json.loads(source.read_text())['presence']['character']['items']
                info = next(v for v in entries if v['name'] == role)
                path = Path(info['canonical_path']) if info.get('canonical_path') else None
                score = scores.get(shot, {}).get('sim_to_centroid')
                inventory.append({'version': version, 'role': role, 'shot': shot,
                                  'status': info['status'], 'clip': info['clip_sim'],
                                  'sample_frame_index': info['frame_idx'], 'score': score,
                                  'canonical_path': str(path) if path else None})
                x = j * cw
                draw.text((x + 3, oy), f'{version} {shot} {info["status"]}', font=font, fill='white')
                if path:
                    image = Image.open(path).convert('RGB')
                    thumb = ImageOps.contain(image, (cw - 8, ch - 65))
                    sheet.paste(thumb, (x + (cw - thumb.width) // 2, oy + 24 + (ch - 65 - thumb.height) // 2))
                    draw.text((x + 3, oy + ch - 34), f'{image.width}x{image.height} f{info["frame_idx"]}', font=font, fill='#bbbbbb')
                draw.text((x + 3, oy + ch - 17), f'sim {score:.3f}' if score is not None else 'excluded from cs', font=font, fill='#ffd966')
        filename = f'role_{len(sections):02d}.jpg'
        sheet.save(dest / filename, quality=90)
        sections.append(f'<h2>{html.escape(role)}</h2><a href="{filename}"><img loading="lazy" src="{filename}" style="max-width:100%"></a>')
    (dest / 'inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    (dest / 'index.html').write_text('<meta charset="utf-8"><title>cs_face actual crop review</title><body style="background:#151515;color:#eee;max-width:1600px;margin:25px auto;font-family:sans-serif"><h1>官方cs_face真实裁剪对照</h1><p>上排原版，下排重跑。自动整理官方实际裁剪，未修改指标；人物身份、错人和脸部可见性尚需人工核对。excluded表示未进入cs_face。</p>' + ''.join(sections) + '</body>')
    return dest / 'index.html'
