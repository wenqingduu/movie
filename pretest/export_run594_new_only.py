"""Package all existing new-version images/videos; no baseline comparison/media.

Includes every video frame using the same imageio decoder as generation records.
No generation or metric inference. Only actually saved denoising images exist.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil
import zipfile

import imageio.v2 as iio
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path('/root/autodl-tmp/movie')
RUN=ROOT/'outputs/run594_i75_j98_shot34_35_prompt_fix_20261101'
OUT=ROOT/'outputs/run594_i75_j98_new_only_control_vs_3d_20261101'
EID='00048e6c-5e69-3381-9079-c43db78ed8f0__run594__i75_j98__T300'
REUSED=(ROOT/'outputs/entitybench_qwen140_multiface_remaining126_20261030/experiments'/EID/'episodes'/EID/'shot_3_5/frame/injection')
copied=[];derived=[]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def copy(source,dest):
    source=Path(source);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(source)==sha(dest),f'Changed export input: {dest}'
    else:shutil.copy2(source,dest)
    assert sha(source)==sha(dest)
    copied.append({'source':str(source),'file':str(dest.relative_to(OUT)),'sha256':sha(dest)})


def tree(source,dest):
    for p in sorted(Path(source).rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.json','.mp4'}:
            copy(p,Path(dest)/p.relative_to(source))


def record_derived(path,**metadata):
    derived.append({'file':str(path.relative_to(OUT)),'sha256':sha(path),**metadata})


def comparison(key,result):
    source=result;row=json.loads(source.read_text());folder=OUT/key/'frame/injection'
    assert sha(folder/'control.png')==row['control_sha256']
    assert sha(folder/'treatment.png')==row['treatment_sha256']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    full=Image.new('RGB',(2048,630),'#eeeeee');draw=ImageDraw.Draw(full)
    for index,(filename,label) in enumerate([('control.png','Control: no3D injection'),('treatment.png','3D injected')]):
        im=Image.open(folder/filename).convert('RGB');assert im.size==(1024,576)
        full.paste(im,(1024*index,50));draw.text((1024*index+14,12),f'{key} / {label}',font=font,fill='black')
    dest=OUT/'comparisons'/f'{key}_full_control_vs_injected.jpg';dest.parent.mkdir(exist_ok=True)
    full.save(dest,quality=95);record_derived(dest,kind='Full-frame C/T comparison; no old version')
    for target in row['targets']:
        name=target['name'];role=OUT/key/'roles'/name;role.mkdir(parents=True,exist_ok=True)
        copy(target['reference_image'],role/'identity_reference.png')
        for field,filename in [('harmonized_reference','aligned_3d.png'),('mask','injection_mask.png')]:copy(target[field],role/filename)
        x1,y1,x2,y2=target['bbox'];dx=.25*(x2-x1);dy=.25*(y2-y1)
        roi=[max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(1024,math.ceil(x2+dx)),min(576,math.ceil(y2+dy))]
        for branch in ['control','treatment']:
            dest=role/f'{branch}_face.png';Image.open(folder/f'{branch}.png').crop(roi).save(dest)
            record_derived(dest,kind='Display crop only; not a metric input',shared_roi=roi)
        (role/'display_roi.json').write_text(json.dumps({'same_control_treatment_roi':roi,
            'source':'Saved prepared target bbox expanded25%; equal C/T coordinates and magnification',
            'new_metric_inference':False,'active_injection_steps':target['active_injection_steps'],
            'step12_identity_cosine_is_schedule_input_not_final_score':target['step12_identity_cosine']},indent=2)+'\n')
        sheet=Image.new('RGB',(1200,850),'#eeeeee');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f'{key} / {name} / Control versus3D ({target["active_injection_steps"]} injection steps)',font=font,fill='black')
        entries=[(role/'identity_reference.png','Identity reference'),(role/'control_face.png','Control face'),
            (role/'treatment_face.png','Injected face'),(role/'aligned_3d.png','Aligned 3D guide'),
            (folder/'control.png','Control full frame'),(folder/'treatment.png','Injected full frame')]
        for index,(path,label) in enumerate(entries):
            x=(index%3)*400;y=(index//3)*360+45;draw.text((x+10,y),label,font=font,fill='black')
            im=ImageOps.contain(Image.open(path).convert('RGB'),(380,315));sheet.paste(im,(x+(400-im.width)//2,y+30+(315-im.height)//2))
        draw.text((12,790),'Same-prompt/seed keyframe pair. No Control videos were generated.',font=font,fill='black')
        dest=OUT/'comparisons'/f'{key}_{name}_control_vs_injected.jpg';sheet.save(dest,quality=95)
        record_derived(dest,kind='Per-role C/T illustration; no final metric claim',shared_roi=roi)


def extract_every_frame(video,dest,first=None,last=None):
    dest.mkdir(parents=True,exist_ok=True);reader=iio.get_reader(str(video),'ffmpeg',ffmpeg_params=['-threads','1'])
    count=0;rows=[];first_pixels=None;last_pixels=None
    try:
        metadata=reader.get_meta_data();fps=float(metadata['fps'])
        for count,frame in enumerate(reader,1):
            index=count-1;path=dest/f'frame_{index:04d}.png'
            if first_pixels is None:first_pixels=frame.copy()
            last_pixels=frame.copy()
            Image.fromarray(frame).save(path,compress_level=3)
            rows.append({'index':index,'time_seconds':index/fps,'file':path.name,'sha256':sha(path)})
    finally:reader.close()
    assert count>0
    for pixels,path in [(first_pixels,first),(last_pixels,last)]:
        if path is not None:assert np.array_equal(pixels,np.asarray(Image.open(path).convert('RGB'))),f'Decoder endpoint mismatch: {path}'
    (dest/'frames_manifest.json').write_text(json.dumps({'source_video':str(video),'video_sha256':sha(video),
        'frames':count,'fps':fps,'size':list(Image.fromarray(first_pixels).size),
        'decoder':'Same imageio-ffmpeg RGB decoder as saved generation endpoints; native size/no resize/no skipping',
        'endpoint_pixels_verified_against_saved_png':first is not None,'items':rows},indent=2)+'\n')
    print(f'Decoded all{count}frames: {video.relative_to(OUT)}',flush=True)
    return {'video':str(video.relative_to(OUT)),'frames_directory':str(dest.relative_to(OUT)),'count':count,'fps':fps}


def main():
    OUT.mkdir(exist_ok=True);assert shutil.disk_usage(OUT).free>2*1024**3
    # Only this version's media. The reused3:5 pair remains this version's target.
    tree(RUN/'shot_3_4',OUT/'shot_3_4');tree(RUN/'shot_3_5',OUT/'shot_3_5')
    target_record=json.loads((RUN/'shot_3_5/frame/target.json').read_text())
    assert sha(REUSED/'treatment.png')==target_record['sha256']==sha(RUN/'shot_3_5/frame/target.png')
    tree(REUSED,OUT/'shot_3_5/frame/injection')
    for filename in ['prompts.json','acceptance_record.json']:copy(RUN/filename,OUT/filename)
    copy(RUN/'review/rerun_shots_3_4_3_5.mp4',OUT/'two_shots_injected.mp4')
    for key,source in [('shot_3_4',RUN/'shot_3_4/frame/injection/result.json'),('shot_3_5',REUSED/'result.json')]:comparison(key,source)
    videos=[(OUT/'shot_3_4/main/video.mp4',OUT/'shot_3_4/main/all_frames'),
            (OUT/'shot_3_5/transition/video.mp4',OUT/'shot_3_5/transition/all_frames'),
            (OUT/'shot_3_5/main/video.mp4',OUT/'shot_3_5/main/all_frames'),
            (OUT/'shot_3_5/combined/video.mp4',OUT/'shot_3_5/combined/all_frames'),
            (OUT/'two_shots_injected.mp4',OUT/'two_shots_all_frames')]
    frame_sets=[]
    for video,dest in videos:
        has_endpoints=(video.parent/'first.png').exists()
        frame_sets.append(extract_every_frame(video,dest,video.parent/'first.png' if has_endpoints else None,
            video.parent/'last.png' if has_endpoints else None))
    assert [f['count'] for f in frame_sets]==[81,90,81,128,209]
    text='''# run594/i75_j98：只含新版结果，Control与3D关键帧对比

本目录只包含本次新版3:4/3:5，不包含旧版双Leo视频、旧版对照视频或旧版3:4图片。3:5目标图按用户方案复用原有正确目标；其同次Control配对也完整包含。

## 快速查看Control／注入

- [3:4 / Leo](comparisons/shot_3_4_Leo_control_vs_injected.jpg)：新单Leo关键帧，6步3D注入。
- [3:5 / Leo](comparisons/shot_3_5_Leo_control_vs_injected.jpg)：复用目标帧，Leo6步。
- [3:5 / Marcus](comparisons/shot_3_5_Marcus_control_vs_injected.jpg)：同一目标帧中Marcus1步，并非只注入Leo。
- 全幅Control/注入对比也在comparisons/；各角色相同ROI裁剪、人物原参考、对齐3D与mask在各shot的roles/。

Control与Treatment共享同次prompt、seed、人物参考、模型与采样设置，Control只关闭3D残差，并非移除人物参考。这里只导出现有配对，未新增模型评测，不把step12调度余弦当最终身份分数。

## 全部新版视频

- [3:4注入路径视频，5.0625秒](shot_3_4/main/video.mp4)
- [3:5新过渡，3秒](shot_3_5/transition/video.mp4)
- [3:5新主动作，5.0625秒](shot_3_5/main/video.mp4)
- [3:5完整，8秒](shot_3_5/combined/video.mp4)
- [3:4＋3:5连播，13.0625秒](two_shots_injected.mp4)

**视频仅有注入路径**：用户选择“先导出现有结果”，本次未补跑无注入视频。没有把旧版视频标成Control。3:5从新3:4尾帧生成过渡，再从过渡实际尾帧生成主段；静态Control／Treatment是过渡目标帧，非主段实际解码首帧。

## 全部图片和视频帧

- 各shot的frame/：全部已保存关键帧、Control、Treatment、任务与生成记录。
- frame/injection/prepared/：已保存step12 pred_x0、原始3D渲染、对齐图、颜色协调图、各种mask、illumination_gain与元数据；3:5包括Leo和Marcus两套。
- 各视频同级all_frames/：视频每一帧原尺寸PNG，无抽帧，共81＋90＋81＋128=380帧。另two_shots_all_frames/保留连播209帧。总589帧含拼接视频的重复内容；不是589个独立生成时刻。
- 各all_frames/frames_manifest.json记录帧号、时间、视频SHA与逐图SHA。生成记录原first/last图全部保留，重新解码首尾像素已与其逐像素核对。

**完整20步去噪轨迹当时未记录**。本包包含全部现存中间图片与全部视频解码帧，未伪造其余去噪步，也未启动轨迹重放。

## 保留事项与来源

用户已接受3:5后半段触碰Marcus身体的动作；过渡中段短暂叠影也按原生成结果保留。原PNG/MP4逐字节复制并核验SHA；只有展示裁剪/拼图及逐视频解码PNG为派生文件，不做修脸、调色或锐化。

manifest.json区分原样复制与展示派生文件，标明3:5目标复用及没有Control视频。使用movie-firstframe-figure-assets skill（/root/.codex/skills/movie-firstframe-figure-assets/SKILL.md）。本次导出无需GPU/API，不修改生成结果或官方分数。
'''
    (OUT/'README.md').write_text(text)
    manifest={'episode_id':EID,'kind':'New version only; keyframe Control versus3D; injected-path videos',
        'old_baseline_media_included':False,'reused35_target_sha256':target_record['sha256'],
        'control_video_available':False,'new_generation_or_paid_calls':0,
        'video_frames':frame_sets,'total_decoded_pngs':sum(f['count'] for f in frame_sets),
        'copied_files':copied,'derived_display_files':derived,
        'denoising_scope':'Only saved step12 pred_x0/preparation; no full20-step trajectory recorded',
        'export_script_sha256':sha(__file__)}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'folder':str(OUT),'copied_original_files':len(copied),'decoded_video_frames':manifest['total_decoded_pngs'],
        'zip':str(archive),'zip_megabytes':round(archive.stat().st_size/1024**2,2)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
