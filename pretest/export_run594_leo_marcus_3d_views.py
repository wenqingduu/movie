"""Export four genuine existing Gaussian render views of Leo and Marcus."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/root/autodl-tmp/movie')
OUT=ROOT/'outputs/run594_leo_marcus_3d_four_views_20261101'
PREP=ROOT/'outputs/run594_new_injected_denoising_replay_20261101/inputs/shot_3_5/replay_prepared.json'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def main():
    OUT.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    views=[('01_upper_left','左上',20,-40),('02_upper_right','右上',20,40),
           ('03_lower_left','左下',-20,-40),('04_lower_right','右下',-20,40)]
    roles=[];all_rows=[]
    for role in sorted(read(PREP)['targets'],key=lambda t:t['name']!='Leo'):
        name=role['name'];result=read(role['facelift_result']);model=Path(result['model_path'])
        model_sha=sha(model);calibration=model.with_suffix('.pose_calibration.json')
        assert read(calibration)['gaussian_model_sha256']==model_sha
        assert sha(result['source_reference'])==result['source_reference_sha256']
        samples_path=Path(role['facelift_result']).parent/'pose_calibration/samples.json'
        samples=read(samples_path);dest=OUT/name;dest.mkdir(exist_ok=True)
        contact=Image.new('RGB',(1024,1024),'white')
        preview=Image.new('RGB',(1024,1088),'white');draw=ImageDraw.Draw(preview)
        selected=[]
        for i,(tag,label,pitch,yaw) in enumerate(views):
            def distance(s):
                p=s['detected_pose']
                return ((p['pitch']-pitch)/10)**2+((p['yaw']-yaw)/10)**2+(p['roll']/12)**2
            sample=min([s for s in samples if s['face_detected']],key=distance)
            source=Path(sample['image']);target=dest/(tag+'.png')
            if target.exists():assert sha(source)==sha(target)
            else:shutil.copy2(source,target)
            assert sha(source)==sha(target)
            im=Image.open(target).convert('RGB');assert im.size==(512,512)
            camera=source.with_suffix('.camera.json')
            shutil.copy2(camera,dest/(tag+'.camera.json'))
            x=(i%2)*512;y=(i//2)*512;contact.paste(im,(x,y))
            y=(i//2)*544;preview.paste(im,(x,y+32))
            draw.text((x+10,y+3),f'{name} / '+tag[3:].replace('_',' '),font=font,fill='black')
            row={'role':name,'direction_in_image':label,'file':str(target.relative_to(OUT)),
                 'source':str(source),'sha256':sha(target),'camera_pose_parameters':sample['camera_pose'],
                 'detected_pose_estimate':sample['detected_pose'],'detection_confidence':sample['detection_confidence'],
                 'camera_metadata':read(camera),'camera_source_sha256':sha(camera),
                 'selection_target_approximation':{'pitch':pitch,'yaw':yaw,'roll':0},
                 'selection_rule':'Closest existing measured pose, squared errors scaled by10/10/12 degrees; not a new render or generated portrait.'}
            selected.append(row);all_rows.append(row)
        assert len({r['source'] for r in selected})==4
        contact.save(OUT/(name+'_four_views_clean.png'))
        preview.save(OUT/(name+'_four_views_preview.jpg'),quality=95)
        roles.append({'name':name,'gaussian_model':str(model),'gaussian_model_sha256':model_sha,
            'identity_reference':result['source_reference'],'identity_reference_sha256':result['source_reference_sha256'],
            'facelift_result':role['facelift_result'],'facelift_result_sha256':sha(role['facelift_result']),
            'pose_calibration_sha256':sha(calibration),'samples_json_sha256':sha(samples_path),'views':selected})
    manifest={'episode_id':'00048e6c-5e69-3381-9079-c43db78ed8f0__run594__i75_j98__T300',
        'roles':roles,'independent_original_render_images':8,'image_size':[512,512],
        'render_source':'Existing FaceLift Gaussian calibration renders from the same3D assets used in injection',
        'new_inference_or_renders':False,'source_pixels_unchanged':True,
        'previews':'Clean2x2 sheets paste native original images; labeled JPGs are previews only.',
        'export_script_sha256':sha(__file__)}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    lines=['# Leo、Marcus：各四张真实3D Gaussian渲染','',
        '来自刚才run594/i75_j98所用的两个人物3D资产。直接复制既有相机标定渲染，全部原图512×512、白底、无文字；无新增人物生成、重建、API或修图。每个人的四张图来自同一个gaussians.ply，模型与原人物参考SHA已核验。',
        '', '四方向按画面中的头部朝向命名：左上、右上、左下、右下。选择已有渲染中接近俯仰±20°、左右偏转±40°且侧倾较小的样本；实际相机参数和人脸检测估计值均保存在manifest.json，不把目标近似角度当作精确实测角度。',
        '', '| 人物 | 四张独立PNG | 无文字2×2排版 | 带标签预览 |','|---|---|---|---|']
    for name in ['Leo','Marcus']:
        links=' · '.join(f'[{label}]({name}/{tag}.png)' for tag,label,_,_ in views)
        lines.append(f'| {name} | {links} | [PNG]({name}_four_views_clean.png) | [JPG]({name}_four_views_preview.jpg) |')
    lines+=['', '绘图优先使用各人物文件夹的四张独立原图；camera.json与manifest.json提供来源及角度记录。拼图只是原图排列，未裁剪、重着色、锐化或变换原图。',
            '', '流程遵循movie-firstframe-figure-assets skill（/root/.codex/skills/movie-firstframe-figure-assets/SKILL.md）。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'folder':str(OUT),'zip':str(archive),'zip_mib':round(archive.stat().st_size/1024**2,2),
                      'original_renders':len(all_rows)},ensure_ascii=False))


if __name__=='__main__':main()
