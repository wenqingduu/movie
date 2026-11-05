"""Original VBench dynamic score for an isolated before/after shot pair."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--before', required=True)
    args = parser.parse_args()
    import torch
    from easydict import EasyDict
    from vbench.dynamic_degree import DynamicDegree
    torch.set_num_threads(4)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    record = json.loads((args.folder / 'result.json').read_text())
    assert sha(record['video']) == record['video_sha256']
    weight = Path('/root/.cache/vbench/raft_model/models/raft-things.pth')
    model = DynamicDegree(EasyDict(model=str(weight), small=False, mixed_precision=False, alternate_corr=False), torch.device('cuda'))
    rows = []
    for label, video in [('current_integrated', args.before), ('prompt_v2', record['video'])]:
        rows.append({'edition': label, 'video': video, 'video_sha256': sha(video),
                     'dynamic_degree': int(model.infer(video)), 'params': dict(model.params)})
    source = Path(inspect.getfile(DynamicDegree))
    report = {'scope': 'Single-shot before/after, B includes transition; not episode/benchmark or causal ablation.',
              'source': str(source), 'source_sha256': sha(source), 'weight_sha256': sha(weight), 'rows': rows}
    (args.folder / 'dynamic_v2.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
