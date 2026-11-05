"""Reuse exact-video per-shot VBench values around the pinned official scorer.

Only unchanged video SHA + compatible scorer/config/model/library values qualify.
New media uses the original six-dimension implementation. No grounding cache,
threshold, sampling, cross-shot center, or aggregation changes.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path('/root/autodl-tmp/movie')
OUT=ROOT/'outputs/entitybench_v2_completed_snapshot_20261105'
CACHE=OUT/'acceleration/vbench_cache_manifest.json'
EVALUATOR=ROOT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140/runtime/evaluate_benchmark_partial.py'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cache=read(CACHE)
    assert sha(EVALUATOR)==cache['evaluator_sha256']
    for f in cache['bindings']:assert sha(f['path'])==f['sha256'],f['path']
    spec=importlib.util.spec_from_file_location('_verified_vbench_cache_official',EVALUATOR)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    assert mod.VBENCH_DIMENSIONS==cache['dimensions']
    original=mod.VBenchEvaluator.evaluate_episode_shots
    def evaluate(self,shot_videos,episode_id,work_dir):
        reused={};changed={};receipts=[]
        for sk,video in shot_videos.items():
            row=cache['shots'][episode_id+'/'+sk]
            assert sha(video)==row['video_sha256'],video
            if row['reuse']:
                reused[sk]=dict(row['scores'])
                receipts.append({'shot':sk,'video_sha256':row['video_sha256'],'sources':row['score_sources']})
            else:changed[sk]=video
        fresh,failed=original(self,changed,episode_id,work_dir) if changed else ({},[])
        assert not failed
        results={sk:(reused[sk] if sk in reused else fresh[sk]) for sk in shot_videos}
        # Preserve the exact format consumed by target diagnostics; summary first
        # element is the episode's official mean of per-shot values.
        for dimension in mod.VBENCH_DIMENSIONS:
            filename=Path(work_dir)/(episode_id+'_'+dimension+'_eval_results.json')
            if filename.exists():
                audit=filename.with_name(filename.stem+'_computed_targets_only.json')
                if not audit.exists():audit.write_bytes(filename.read_bytes())
            values=[{'video_path':str(shot_videos[sk]),'video_results':results[sk][dimension]} for sk in shot_videos]
            filename.write_text(json.dumps({dimension:[sum(v['video_results'] for v in values)/len(values),values]},indent=2)+'\n')
        proof={'episode_id':episode_id,'cache_manifest_sha256':sha(CACHE),'adapter_sha256':sha(__file__),
               'unchanged_shots_reused':len(reused),'changed_shots_computed':len(changed),'reused':receipts,
               'scope':'Only VBench reuse. All grounding and cross-shot metrics recomputed by pinned scorer.'}
        (Path(work_dir)/(episode_id+'_vbench_reuse_receipt.json')).write_text(json.dumps(proof,indent=2)+'\n')
        return results,failed
    mod.VBenchEvaluator.evaluate_episode_shots=evaluate
    mod.main()
if __name__=='__main__':main()
