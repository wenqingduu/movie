"""User-directed, single-pass repair of run594/i75_j98 shots3:4 and3:5.

Preserve the original episode and its scores. Rebuild the solo-Leo3:4 image
with the original paired v7 path, regenerate3:4, reuse3:5's exact injected
target, then generate a reveal transition and a new action clip. No planner,
semantic retry loop, automatic fallback, or unrelated queue is started.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/root/autodl-tmp/movie')
EID='00048e6c-5e69-3381-9079-c43db78ed8f0__run594__i75_j98__T300'
SOURCE=ROOT/'outputs/entitybench_qwen140_multiface_remaining126_20261030/experiments'/EID
OLD=SOURCE/'episodes'/EID
OUT=ROOT/'outputs/run594_i75_j98_shot34_35_prompt_fix_20261101'
os.environ['QWEN140_INTEGRATED_OUTPUT_ROOT']=str(OUT)
import qwen140_integrated_episodes as core
from qwen140_single_pass_shot import decode
from qwen140_wan_api import generate_video

FRAME_PROMPT=(
    'photorealistic live-action film. 图1仅参考车内环境和日光，不沿用人物构图；图2为唯一人物Leo。'
    'Leo: Young man with short dark hair. 白天从车外透过副驾车窗，裁紧为Leo头胸部单人中近景。'
    'Leo穿深色连帽衫、灰T恤，围蓝格围巾，系安全带；面露忧虑，眼睛睁开、目光略垂，尚未低头。'
    '画面只出现一个Leo；驾驶位、后座完全在画外，玻璃无人物倒影。'
    '单幅连续电影画面，无其他人物、分屏、拼图或文字。'
)
MAIN34_PROMPT=(
    'Photorealistic live-action film. Leo is the only visible person: the young short dark-haired man '
    'in the front passenger seat, wearing a dark hoodie over a grey T-shirt and a blue tartan scarf. '
    'In the fixed close framing through the passenger-side window, his worried expression becomes distressed. '
    'He slowly lowers his head and gaze to something unseen in his lap and holds that pose. '
    'Keep the camera outside the parked car; the driver seat and rear seats stay outside the framing.'
)
TRANSITION_PROMPT=(
    'Photorealistic live-action film. Begin on Leo alone in a tight view through the front passenger-side '
    'window of the same parked car. Leo slowly raises his head from his lap. The camera gently widens '
    'and shifts left outside the car, revealing Marcus already seated in the driver seat, wearing a '
    'light-colored collared shirt. Marcus remains seated at the wheel throughout. Preserve both identities '
    'and the car interior. End on the supplied last-frame composition.'
)
MAIN35_PROMPT=(
    'Photorealistic live-action film. Leo is the young short dark-haired man in the right foreground, '
    'seated in the front passenger seat, wearing a dark hoodie and blue tartan scarf. Marcus is the '
    'bearded Black man in the left background, wearing a light-colored collared shirt, seated in the '
    'driver seat with his hands at the steering wheel. From outside the passenger-side window, Leo '
    'turns toward Marcus, speaks urgently, and gestures toward Marcus with one open hand held between '
    'his own chest and the center of the cabin. Marcus listens. Keep their seats fixed, the camera '
    'steady and the car parked. Only Marcus operates the steering wheel.'
)


def read(p):return json.loads(Path(p).read_text())
def status(phase,**extra):
    p=OUT/'status.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,**extra},ensure_ascii=False,indent=2)+'\n')
def freeze(p,data):core.write_once(Path(p),data)


def initialize():
    OUT.mkdir(exist_ok=True)
    assert shutil.disk_usage(OUT).free>8*1024**3
    task=read(OLD/'shot_3_4/frame/target.task.json')
    for item in task['face_injection']['frozen_files']:
        assert core.sha(Path(item['path']))==item['sha256'],item['path']
    for image in task['images']:assert core.sha(Path(image['path']))==image['sha256']
    target=OLD/'shot_3_5/frame/target.png';inj35=read(OLD/'shot_3_5/frame/injection/result.json')
    assert core.sha(target)==inj35['treatment_sha256']
    reference=Path(task['face_injection']['characters'][0]['reference_image'])
    assert read(Path(task['face_injection']['characters'][0]['facelift_result']))['source_reference_sha256']==core.sha(reference)
    paths=[Path(__file__),ROOT/'pretest/qwen140_integrated_episodes.py',ROOT/'pretest/qwen140_v7_frame.py',
        ROOT/'pretest/qwen140_wan_api.py',ROOT/'pretest/qwen140_single_pass_shot.py',
        OLD/'shot_3_4/frame/target.task.json',OLD/'shot_3_4/frame/target.png',
        OLD/'shot_3_4/main/video.mp4',OLD/'shot_3_4/main/last.png',
        OLD/'shot_3_5/frame/target.png',OLD/'shot_3_5/frame/injection/result.json',
        OLD/'shot_3_5/combined/video.mp4',OLD/'shot_3_5/transition/last.png',
        SOURCE/'style_specs.json']
    freeze(OUT/'input_manifest.json',{'episode_id':EID,'source_episode_root':str(SOURCE),
        'sources':[{'path':str(p),'sha256':core.sha(p)} for p in paths],
        'scope':'Only3:4 and3:5, new output version; original full episode/benchmark/metrics unchanged',
        'plan_origin':'User-directed targeted prompt change, no new Agent call',
        'kept':'Original references, Gaussians, v7 gates/settings/seeds/model; exact existing3:5 injected target',
        'changes':'Solo3:4 framing and action prompt; reveal-only3:5 transition; main prompt binds actual two-person start and hand direction',
        'retry_policy':'One generation per stage; no semantic retries or rerouting. Exceptions stop and preserve outputs.',
        'api':'One Wan3 first+last-frame3s task, reuse persisted task on resume; no VLM evaluation or repair agent'})
    for item in read(OUT/'input_manifest.json')['sources']:
        assert core.sha(Path(item['path']))==item['sha256'],item['path']
    style=OUT/'style_specs.json'
    if not style.exists():shutil.copy2(SOURCE/'style_specs.json',style)
    assert core.sha(style)==core.sha(SOURCE/'style_specs.json')
    task=copy.deepcopy(task);task['prompt']=FRAME_PROMPT
    freeze(OUT/'shot_3_4/frame/target.task.json',task)
    freeze(OUT/'prompts.json',{'frame_3_4':FRAME_PROMPT,'main_3_4':MAIN34_PROMPT,
        'transition_3_5':TRANSITION_PROMPT,'main_3_5':MAIN35_PROMPT})
    reuse=OUT/'shot_3_5/frame/target.png';reuse.parent.mkdir(parents=True,exist_ok=True)
    if not reuse.exists():shutil.copy2(target,reuse)
    assert core.sha(reuse)==inj35['treatment_sha256']
    freeze(reuse.with_suffix('.json'),{'status':'reused_existing_injected_target',
        'source':str(target),'sha256':core.sha(reuse),'injection_result':str(OLD/'shot_3_5/frame/injection/result.json'),
        'roles':[{'name':t['name'],'active_injection_steps':t['active_injection_steps']} for t in inj35['targets']]})
    print('Frozen inputs verified; original3:5 target reused:',reuse,flush=True)


def frame34():
    initialize();status('generating_3_4_keyframe')
    dest=OUT/'shot_3_4/frame/target.png'
    if dest.exists():
        rec=read(dest.with_suffix('.json'));assert core.sha(dest)==rec['sha256']
    else:core.run_worker(core.QWEN_PYTHON,'worker-frame','--task',str(dest.with_suffix('.task.json')),'--output',str(dest))
    status('3_4_keyframe_ready',path=str(dest),sha256=core.sha(dest))


def video_task(key,start,prompt):
    old=read(OLD/f"shot_{key.replace(':','_')}"/'main/task.json')
    task={**old,'input_path':str(start.resolve()),'input_sha256':core.sha(start),'prompt':prompt}
    folder=OUT/f"shot_{key.replace(':','_')}"/'main'
    freeze(folder/'task.json',task)
    result=folder/'generation.json'
    if result.exists():
        rec=read(result);assert rec['condition_sha256']==core.digest(task)
        assert core.sha(folder/'video.mp4')==rec['video_sha256']
        assert core.sha(folder/'last.png')==rec['last_sha256']
    else:core.run_worker(core.WAN_PYTHON,'worker-video','--task',str(folder/'task.json'),'--output',str(folder))
    return folder/'video.mp4'


def main34():
    initialize();status('generating_3_4_video')
    start=OUT/'shot_3_4/frame/target.png'
    assert core.sha(start)==read(start.with_suffix('.json'))['sha256']
    video=video_task('3:4',start,MAIN34_PROMPT)
    status('3_4_video_ready',video=str(video))


def transition35():
    initialize();status('generating_3_5_transition')
    first=OUT/'shot_3_4/main/last.png';last=OUT/'shot_3_5/frame/target.png'
    rec=read(OUT/'shot_3_4/main/generation.json');assert core.sha(first)==rec['last_sha256']
    video=generate_video(first,last,TRANSITION_PROMPT,3,OUT/'shot_3_5/transition')
    decode(video,video.parent)
    freeze(OUT/'shot_3_5/transition/parent_chain.json',{'parent_shot':'3:4',
        'parent_video_sha256':rec['video_sha256'],'parent_tail_sha256':core.sha(first),
        'target_sha256':core.sha(last),'actual_transition_tail_sha256':core.sha(video.parent/'last.png')})
    status('3_5_transition_ready',video=str(video))


def main35():
    initialize();status('generating_3_5_action')
    start=OUT/'shot_3_5/transition/last.png'
    assert core.sha(start)==read(OUT/'shot_3_5/transition/decoded.json')['last_sha256']
    video=video_task('3:5',start,MAIN35_PROMPT)
    status('3_5_action_ready',video=str(video))


def assemble():
    initialize();folder=OUT/'shot_3_5/combined';folder.mkdir(parents=True,exist_ok=True);video=folder/'video.mp4'
    if not video.exists():
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(OUT/'shot_3_5/transition/video.mp4'),
            '-i',str(OUT/'shot_3_5/main/video.mp4'),'-filter_complex',
            '[0:v]fps=16,scale=832:464,setsar=1,setpts=PTS-STARTPTS[a];'
            '[1:v]trim=start_frame=1,scale=832:464,setsar=1,setpts=PTS-STARTPTS[b];'
            '[a][b]concat=n=2:v=1:a=0[v]','-map','[v]','-an','-c:v','libx264','-crf','17',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    if not (folder/'decoded.json').exists():decode(video,folder)
    for label,first,second in [('rerun',OUT/'shot_3_4/main/video.mp4',video),
                               ('original',OLD/'shot_3_4/main/video.mp4',OLD/'shot_3_5/combined/video.mp4')]:
        dest=OUT/'review'/f'{label}_shots_3_4_3_5.mp4';dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(first),'-i',str(second),
                '-filter_complex','[0:v]fps=16,scale=832:464,setsar=1,setpts=PTS-STARTPTS[a];'
                '[1:v]fps=16,scale=832:464,setsar=1,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[v]',
                '-map','[v]','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
    status('generation_complete_pending_visual_review',new34=str(OUT/'shot_3_4/main/video.mp4'),new35=str(video))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['init','frame34','main34','transition35','main35','assemble','rest']);args=parser.parse_args()
    try:
        if args.action=='rest':
            for function in [main34,transition35,main35,assemble]:function()
        else:globals()[{'init':'initialize'}.get(args.action,args.action)]()
    except Exception as error:
        status('stopped',stage=args.action,error=repr(error))
        raise
