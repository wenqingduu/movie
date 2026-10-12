"""Pause one formal worker, compare placement and two-worker throughput, then always resume it."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from prepare_entitybench_qwen21_20ep import OUTPUT, PROJECT, sha256, write_json
from run_entitybench_a14b_4step_20ep import environment

PYTHON='/root/autodl-tmp/qwen-image21-venv/bin/python'


def main(args):
    root=args.output_root;root.mkdir(parents=True,exist_ok=True)
    cmdline=Path(f'/proc/{args.formal_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'run_qwen21_entitybench_20ep_images.py' in cmdline and '--stage references' in cmdline
    plan=json.loads((OUTPUT/'preparation_plan.json').read_text());cases=[]
    # Freeze two existing seeds before looking at any benchmark outcomes.
    for role in [plan['roles'][0],next(r for r in plan['roles'] if r['needs_face_asset'])]:
        candidate=Path(role['reference']).parent/'candidates'/Path(role['reference']).stem/f"seed_{role['seed']}.png"
        metadata=json.loads(candidate.with_suffix('.json').read_text())
        assert metadata['sha256']==sha256(candidate)
        cases.append(dict(name=f"{role['character']}_{role['seed']}.png",prompt=role['prompt'],seed=role['seed'],
            expected_sha256=metadata['sha256'],baseline_seconds=metadata['elapsed_seconds'],source=str(candidate)))
    write_json(root/'cases.json',cases)
    original=json.loads((OUTPUT/'status.json').read_text())
    write_json(root/'formal_pause.json',dict(pid=args.formal_pid,original_status=original,time=time.time()))
    env=environment();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    successes={};children=[]
    def launch(mode,out,cases_file):
        out.mkdir(parents=True,exist_ok=True)
        stream=(out/'worker.log').open('w')
        p=subprocess.Popen([PYTHON,'-u','pretest/probe_qwen21_reference_worker.py','--cases',str(cases_file),
            '--out',str(out),'--mode',mode],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        children.append((p,stream));return p
    try:
        os.kill(args.formal_pid,signal.SIGSTOP)
        state=dict(original,phase='acceleration_probe_temporarily_paused_references',probe_root=str(root),formal_pid=args.formal_pid)
        write_json(OUTPUT/'status.json',state)
        for mode in ['model','cuda']:
            print('Testing serial '+mode,flush=True)
            out=root/('serial_'+mode);p=launch(mode,out,root/'cases.json');code=p.wait()
            if code==0:
                result=json.loads((out/'results.json').read_text());successes[mode]=result
                print(json.dumps(result),flush=True)
            else:print(f'{mode} failed ({code}); log {out}/worker.log',flush=True)
        # Admission uses measured peak reserved plus headroom, with the paused worker still allocated.
        if 'model' in successes and all(r['exact_png_match'] for r in successes['model']['rows']):
            peak=max(r['cuda_peak_reserved_mib'] for r in successes['model']['rows'])
            total,used=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(','))
            if 2*peak+used+4096<total:
                print('Testing parallel 2 x model offload',flush=True);processes=[];started=time.monotonic()
                for index,case in enumerate(cases):
                    path=root/f'cases_worker{index}.json';write_json(path,[case])
                    processes.append(launch('model',root/f'parallel_worker{index}',path))
                codes=[p.wait() for p in processes]
                results=[json.loads((root/f'parallel_worker{i}/results.json').read_text()) for i in range(2)] if codes==[0,0] else []
                successes['parallel_model']=dict(wall_seconds=time.monotonic()-started,returncodes=codes,workers=results)
                print(json.dumps(successes['parallel_model']),flush=True)
            else:print(f'Parallel admission rejected: reserved peak={peak:.0f}MiB x2; headroom insufficient',flush=True)
        write_json(root/'report.json',dict(baseline=cases,results=successes,
            baseline_inference_seconds=sum(c['baseline_seconds'] for c in cases),formal_outputs_modified=False))
    finally:
        for p,stream in children:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            stream.close()
        os.kill(args.formal_pid,signal.SIGCONT)
        write_json(OUTPUT/'status.json',original)
        write_json(root/'resume.json',dict(formal_pid=args.formal_pid,resumed_at=time.time()))
        print('Formal references resumed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--formal-pid',type=int,required=True);p.add_argument('--output-root',type=Path,
        default=PROJECT/'outputs/entitybench_qwen21_reference_acceleration_probe')
    main(p.parse_args())
