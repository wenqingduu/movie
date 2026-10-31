#!/usr/bin/env python3
"""Copy completed replay images into a plain image folder with SHA provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(source, destination, shots=None):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Use an output folder outside the original replay directory')
    chosen = shots or sorted(p.name for p in source.glob('shot_*') if p.is_dir() and (p/'complete.json').exists())
    if not chosen:
        raise ValueError('No completed replay shots selected')
    records, counts = [], {}
    def copy(original, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(original)
        if target.exists():
            if sha(target) != digest:
                raise ValueError('Destination contains a different file: ' + str(target))
        else:
            shutil.copy2(original, target)
        if sha(target) != digest:
            raise ValueError('Copied bytes differ: ' + str(target))
        records.append({'file': str(target.relative_to(destination)), 'source': str(original), 'sha256': digest})
    for original in sorted((source/'inputs').glob('*')):
        if original.is_file() and original.suffix.lower() in ('.png','.jpg','.jpeg'):
            copy(original, destination/original.name)
    for shot in chosen:
        if Path(shot).name != shot or not (source/shot/'complete.json').is_file():
            raise ValueError('Expected a completed shot directory: ' + shot)
        start = len(records)
        for original in sorted((source/shot).rglob('*')):
            if original.is_file() and original.suffix.lower() in ('.png','.jpg','.jpeg'):
                copy(original, destination/shot/original.relative_to(source/shot))
        for name in ('harmonized_3d_face.png','final_injection_mask.png'):
            original = source/'inputs'/shot/name
            if original.exists(): copy(original,destination/shot/'guidance'/name)
        for name in ('qwen_prompt.txt','official_prompt.txt'):
            original = source/'inputs'/shot/name
            if original.exists(): copy(original,destination/shot/'prompts'/name)
        counts[shot] = len([r for r in records[start:] if Path(r['file']).suffix.lower() in ('.png','.jpg','.jpeg')])
    manifest={'source':str(source),'shots':chosen,'image_counts':counts,
        'total_images':sum(r['file'].lower().endswith(('.png','.jpg','.jpeg')) for r in records),
        'files':records,'pixel_changes':False,'tensor_files_remain_in_source':True}
    (destination/'image_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (destination/'README.md').write_text('''# First-frame figure images

PNG/JPG files are copied byte for byte; HTML is only an optional preview elsewhere.
Each shot has treatment.png (injected final), control.png (paired Control),
treatment/ and control/ (z_00..z_N actual state decodes and pred_x0 clean predictions),
injection/ (Euler-before versus actual-after), and optional guidance/ and prompts/.

z_00 is initial noise; z_n is the state after n solver updates. pred_x0 is a clean
prediction and is not the actual noisy state. For zero-based step k, injection
changes the Euler output before the next state z_(k+1). Keep matching replay
finals and intermediate states together. Raw tensors and run logs remain at the
source recorded in image_manifest.json. These are replay observations, not files
saved during the original run. Figure selection is not a benchmark aggregate.
''')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--shots',nargs='+')
    args=parser.parse_args()
    manifest=pack(args.source,args.out,args.shots)
    print(json.dumps({k:manifest[k] for k in ('shots','image_counts','total_images')},ensure_ascii=False))
