"""Export existing four-corner Gaussian views for Erik, Jordan, Alex and Casey."""
from pathlib import Path
import json
import shutil
import zipfile
from PIL import Image, ImageDraw, ImageFont
import export_run594_ethan_3d_views as common

NAMES = ('Erik', 'Jordan', 'Alex', 'Casey')
OUT = common.ROOT / 'outputs/run594_other_characters_3d_four_views_20261102'


def export_role(name):
    asset = common.RUN / 'assets' / common.EPISODE / 'faces_3d' / name.lower()
    result_path = asset / 'facelift_result.json'
    result = common.read(result_path)
    assert result['facelift_status'] == 'success'
    model = Path(result['model_path'])
    assert model.is_relative_to(asset)
    reference = common.RUN / 'references/run594' / (name.lower() + '.png')
    assert Path(result['source_reference']) == reference
    assert common.sha(reference) == result['source_reference_sha256']
    calibration = model.with_suffix('.pose_calibration.json')
    model_sha = common.sha(model)
    assert common.read(calibration)['gaussian_model_sha256'] == model_sha
    samples_path = asset / 'pose_calibration/samples.json'
    samples = [s for s in common.read(samples_path) if s['face_detected']]
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    clean = Image.new('RGB', (1024, 1024), 'white')
    preview = Image.new('RGB', (1024, 1088), 'white')
    draw = ImageDraw.Draw(preview)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    rows = []
    for i, (tag, label, pitch, yaw) in enumerate(common.VIEWS):
        def distance(sample):
            p = sample['detected_pose']
            return ((p['pitch'] - pitch) / 10)**2 + ((p['yaw'] - yaw) / 10)**2 + (p['roll'] / 12)**2
        sample = min(samples, key=distance)
        source = Path(sample['image'])
        assert source.is_relative_to(asset / 'pose_calibration/samples')
        target = dest / (tag + '.png')
        camera = source.with_suffix('.camera.json')
        shutil.copy2(source, target)
        camera_target = dest / (tag + '.camera.json')
        shutil.copy2(camera, camera_target)
        assert common.sha(source) == common.sha(target)
        assert common.sha(camera) == common.sha(camera_target)
        with Image.open(target) as image:
            assert image.size == (512, 512)
            im = image.convert('RGB')
        x, y = (i % 2) * 512, (i // 2) * 512
        clean.paste(im, (x, y))
        y = (i // 2) * 544
        preview.paste(im, (x, y + 32))
        draw.text((x + 10, y + 3), name + ' / ' + tag[3:].replace('_', ' '), font=font, fill='black')
        rows.append({'role': name, 'direction_in_image': label,
                     'file': str(target.relative_to(OUT)), 'source': str(source), 'sha256': common.sha(target),
                     'camera_pose_parameters': sample['camera_pose'],
                     'detected_pose_estimate': sample['detected_pose'],
                     'detection_confidence': sample['detection_confidence'],
                     'camera_metadata': common.read(camera), 'camera_source_sha256': common.sha(camera),
                     'selection_target_approximation': {'pitch': pitch, 'yaw': yaw, 'roll': 0},
                     'selection_rule': 'Closest existing measured pose; squared errors scaled by 10/10/12 degrees, matching the Leo/Marcus/Ethan exports.'})
    assert len({r['source'] for r in rows}) == 4
    clean.save(OUT / (name + '_four_views_clean.png'))
    preview.save(OUT / (name + '_four_views_preview.jpg'), quality=95)
    return {'name': name, 'gaussian_model': str(model), 'gaussian_model_sha256': model_sha,
            'identity_reference': str(reference), 'identity_reference_sha256': common.sha(reference),
            'facelift_result': str(result_path), 'facelift_result_sha256': common.sha(result_path),
            'pose_calibration_sha256': common.sha(calibration),
            'samples_json_sha256': common.sha(samples_path), 'views': rows}


def main():
    roles = [export_role(name) for name in NAMES]
    manifest = {'episode_id': common.EPISODE, 'roles': roles,
                'independent_original_render_images': 16, 'image_size': [512, 512],
                'render_source': 'Existing episode-local FaceLift Gaussian calibration renders',
                'new_inference_or_renders': False, 'source_pixels_unchanged': True,
                'previews': 'Clean 2x2 sheets paste native original images; labeled JPGs are previews only.',
                'export_script_sha256': common.sha(__file__),
                'common_module_sha256': common.sha(common.__file__)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Erik、Jordan、Alex、Casey：各四个朝向的真实3D Gaussian渲染', '',
             '来自与Leo、Marcus、Ethan相同的run594/i75_j98 episode，每人使用自己的Gaussian和身份参考。',
             '每人四张512×512白底无文字PNG，共16张。直接复制既有相机标定渲染，模型、身份参考、原图和相机记录SHA均已核验。', '',
             '方向按画面中的头部朝向命名，目标为俯仰约±20°、左右偏转约±40°、低侧倾；实际相机参数与检测姿态见manifest.json，不将目标角度当作实测值。', '',
             '| 人物 | 独立原图 | 无文字拼图 | 带标签预览 |', '|---|---|---|---|']
    for name in NAMES:
        links = ' · '.join(f'[{label}]({name}/{tag}.png)' for tag, label, _, _ in common.VIEWS)
        lines.append(f'| {name} | {links} | [PNG]({name}_four_views_clean.png) | [JPG]({name}_four_views_preview.jpg) |')
    lines += ['', '独立PNG与源文件字节一致，拼图只排列原图，未裁剪、重着色、锐化或修图。',
              '流程：[movie-firstframe-figure-assets skill](/root/.codex/skills/movie-firstframe-figure-assets/SKILL.md)。',
              '复现入口：pretest/export_run594_other_characters_3d_views.py（复用Ethan导出器的路径、方向和SHA函数）。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(OUT.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({'folder': str(OUT), 'zip': str(archive), 'original_renders': 16,
                      'roles': [{'name': r['name'], 'poses': [v['detected_pose_estimate'] for v in r['views']]} for r in roles]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
