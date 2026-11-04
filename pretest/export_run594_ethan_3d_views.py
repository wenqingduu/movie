"""Export Ethan's existing episode-local Gaussian renders without changing pixels."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/root/autodl-tmp/movie')
EPISODE = '00048e6c-5e69-3381-9079-c43db78ed8f0__run594__i75_j98__T300'
RUN = ROOT / 'outputs/entitybench_qwen140_multiface_remaining126_20261030/experiments' / EPISODE
ASSET = RUN / 'assets' / EPISODE / 'faces_3d/ethan'
OUT = ROOT / 'outputs/run594_ethan_3d_four_views_20261102'
VIEWS = [('01_upper_left', '左上', 20, -40), ('02_upper_right', '右上', 20, 40),
         ('03_lower_left', '左下', -20, -40), ('04_lower_right', '右下', -20, 40)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    result_path = ASSET / 'facelift_result.json'
    result = read(result_path)
    assert result['facelift_status'] == 'success'
    model = Path(result['model_path'])
    assert model.is_relative_to(ASSET)
    reference = RUN / 'references/run594/ethan.png'
    assert Path(result['source_reference']) == reference
    assert sha(reference) == result['source_reference_sha256']
    calibration = model.with_suffix('.pose_calibration.json')
    model_sha = sha(model)
    assert read(calibration)['gaussian_model_sha256'] == model_sha
    samples_path = ASSET / 'pose_calibration/samples.json'
    samples = [s for s in read(samples_path) if s['face_detected']]
    dest = OUT / 'Ethan'
    dest.mkdir(parents=True, exist_ok=True)
    clean = Image.new('RGB', (1024, 1024), 'white')
    preview = Image.new('RGB', (1024, 1088), 'white')
    draw = ImageDraw.Draw(preview)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    rows = []
    for i, (tag, label, pitch, yaw) in enumerate(VIEWS):
        def distance(sample):
            p = sample['detected_pose']
            return ((p['pitch'] - pitch) / 10)**2 + ((p['yaw'] - yaw) / 10)**2 + (p['roll'] / 12)**2
        sample = min(samples, key=distance)
        source = Path(sample['image'])
        assert source.is_relative_to(ASSET / 'pose_calibration/samples')
        target = dest / (tag + '.png')
        camera = source.with_suffix('.camera.json')
        shutil.copy2(source, target)
        shutil.copy2(camera, dest / (tag + '.camera.json'))
        assert sha(source) == sha(target)
        with Image.open(target) as image:
            assert image.size == (512, 512)
            im = image.convert('RGB')
        x, y = (i % 2) * 512, (i // 2) * 512
        clean.paste(im, (x, y))
        y = (i // 2) * 544
        preview.paste(im, (x, y + 32))
        draw.text((x + 10, y + 3), 'Ethan / ' + tag[3:].replace('_', ' '), font=font, fill='black')
        rows.append({'role': 'Ethan', 'direction_in_image': label,
                     'file': str(target.relative_to(OUT)), 'source': str(source), 'sha256': sha(target),
                     'camera_pose_parameters': sample['camera_pose'],
                     'detected_pose_estimate': sample['detected_pose'],
                     'detection_confidence': sample['detection_confidence'],
                     'camera_metadata': read(camera), 'camera_source_sha256': sha(camera),
                     'selection_target_approximation': {'pitch': pitch, 'yaw': yaw, 'roll': 0},
                     'selection_rule': 'Closest existing measured pose; squared errors scaled by 10/10/12 degrees, matching the Leo/Marcus export.'})
    assert len({r['source'] for r in rows}) == 4
    clean.save(OUT / 'Ethan_four_views_clean.png')
    preview.save(OUT / 'Ethan_four_views_preview.jpg', quality=95)
    manifest = {'episode_id': EPISODE, 'roles': [{
        'name': 'Ethan', 'gaussian_model': str(model), 'gaussian_model_sha256': model_sha,
        'identity_reference': str(reference), 'identity_reference_sha256': sha(reference),
        'facelift_result': str(result_path), 'facelift_result_sha256': sha(result_path),
        'pose_calibration_sha256': sha(calibration), 'samples_json_sha256': sha(samples_path), 'views': rows}],
        'independent_original_render_images': 4, 'image_size': [512, 512],
        'render_source': 'Existing FaceLift Gaussian calibration renders from Ethan in run594/i75_j98',
        'new_inference_or_renders': False, 'source_pixels_unchanged': True,
        'previews': 'Clean 2x2 sheet pastes native original images; labeled JPG is a preview only.',
        'export_script_sha256': sha(__file__)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Ethan：四个朝向的真实3D Gaussian渲染', '',
             '来自与Leo、Marcus相同的run594/i75_j98 episode，使用Ethan自己的Gaussian及身份参考。',
             '四张512×512白底无文字PNG直接复制既有相机标定渲染；模型、身份参考和图片SHA已核验。', '',
             '四方向按画面中的头部朝向命名。与Leo/Marcus相同，选择接近俯仰±20°、左右偏转±40°且侧倾较小的现有实测样本；目标为近似角度，实际相机参数与检测姿态见manifest.json。', '',
             ' · '.join(f'[{label}](Ethan/{tag}.png)' for tag, label, _, _ in VIEWS), '',
             '[无文字2×2拼图](Ethan_four_views_clean.png) · [带标签预览](Ethan_four_views_preview.jpg)', '',
             '独立PNG保持源文件字节一致，拼图仅排列原图；camera.json与manifest.json保留来源。',
             '流程：[movie-firstframe-figure-assets skill](/root/.codex/skills/movie-firstframe-figure-assets/SKILL.md)。',
             '复现入口：pretest/export_run594_ethan_3d_views.py。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(OUT.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({'folder': str(OUT), 'zip': str(archive), 'original_renders': len(rows),
                      'selected_poses': [r['detected_pose_estimate'] for r in rows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
