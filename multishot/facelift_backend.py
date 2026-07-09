import json
import os
import subprocess
import sys
from pathlib import Path
from shutil import copyfile

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACELIFT_ROOT = PROJECT_ROOT / "third_party" / "FaceLift"


def _copy_reference_fallback(reference_image_path: str, output_dir: Path, reason: str):
    """FaceLift 不可用时写失败状态；默认中断，禁止静默跳过。"""

    source_path = Path(reference_image_path)
    multi_view_images = []
    for view_name in ["front", "left", "right"]:
        view_path = output_dir / f"{view_name}.png"
        if source_path.exists() and source_path.stat().st_size > 0:
            copyfile(source_path, view_path)
        else:
            view_path.touch()
        view_path.with_suffix(".prompt.txt").write_text(
            f"Fallback reference image for {view_name}. FaceLift failed: {reason}",
            encoding="utf-8",
        )
        multi_view_images.append(str(view_path))

    status_path = output_dir / "facelift_status.json"
    status_path.write_text(
        json.dumps({
            "status": "failed",
            "reason": reason,
            "fallback": "copied_reference_image_for_pipeline_continuity",
            "fallback_allowed": os.getenv("MULTISHOT_ALLOW_FACELIFT_FALLBACK", "0") == "1",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if os.getenv("MULTISHOT_ALLOW_FACELIFT_FALLBACK", "0") != "1":
        raise RuntimeError(f"FaceLift failed and fallback is disabled: {reason}")
    return {
        "facelift_status": "failed",
        "facelift_error": reason,
        "model_path": str(status_path),
        "multi_view_images": multi_view_images,
        "facelift_output_dir": None,
    }


def _split_multiview(multiview_path: Path, output_dir: Path):
    """从 FaceLift multiview 横向拼图里切出 front/left/right 参考视角。"""

    image = Image.open(multiview_path).convert("RGB")
    view_count = max(1, image.width // image.height)
    view_size = image.height
    views = []
    for index in range(view_count):
        view = image.crop((index * view_size, 0, (index + 1) * view_size, view_size))
        views.append(view)

    # FaceLift 的 6-view 顺序接近 front/front-right/right/back/left/front-left。
    # 如果输出是 7 张，第一张通常是输入/条件图，因此跳过它选后面的视角。
    offset = 1 if view_count >= 7 else 0
    indices = {
        "front": min(offset + 0, view_count - 1),
        "right": min(offset + 2, view_count - 1),
        "left": min(offset + 4, view_count - 1),
    }

    paths = []
    for view_name, index in indices.items():
        view_path = output_dir / f"{view_name}.png"
        views[index].save(view_path)
        view_path.with_suffix(".prompt.txt").write_text(
            f"FaceLift rendered {view_name} view from {multiview_path}",
            encoding="utf-8",
        )
        paths.append(str(view_path))
    return paths


def build_facelift_asset(reference_image_path: str, output_dir: str):
    """调用本地 FaceLift 官方推理，生成 3D Gaussian head 和多视角参考图。"""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if os.getenv("MULTISHOT_DISABLE_FACELIFT", "0") == "1":
        return _copy_reference_fallback(reference_image_path, output_path, "MULTISHOT_DISABLE_FACELIFT=1")

    if not FACELIFT_ROOT.exists():
        return _copy_reference_fallback(reference_image_path, output_path, f"FaceLift repo not found: {FACELIFT_ROOT}")

    input_dir = output_path / "facelift_input"
    raw_output_dir = output_path / "facelift_raw"
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    input_image_path = input_dir / "input.png"
    copyfile(str(Path(reference_image_path).resolve()), input_image_path)

    step_2d = int(os.getenv("MULTISHOT_FACELIFT_STEP_2D", "50"))
    timeout = int(os.getenv("MULTISHOT_FACELIFT_TIMEOUT", "1800"))
    auto_crop = os.getenv("MULTISHOT_FACELIFT_AUTO_CROP", "1") != "0"
    runner = (
        "import sys; "
        f"sys.path.insert(0, {str(FACELIFT_ROOT)!r}); "
        "from inference import main; "
        f"main(input_dir={str(input_dir)!r}, output_dir={str(raw_output_dir)!r}, "
        f"auto_crop={auto_crop!r}, seed=4, guidance_scale_2D=3.0, step_2D={step_2d})"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(FACELIFT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        completed = subprocess.run(
            [sys.executable, "-c", runner],
            cwd=str(FACELIFT_ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        return _copy_reference_fallback(reference_image_path, output_path, f"FaceLift timeout after {timeout}s")
    except subprocess.CalledProcessError as exc:
        error = (exc.stderr or exc.stdout or str(exc))[-4000:]
        return _copy_reference_fallback(reference_image_path, output_path, error)

    result_dir = raw_output_dir / input_image_path.stem
    multiview_path = result_dir / "multiview.png"
    gaussians_path = result_dir / "gaussians.ply"
    output_render_path = result_dir / "output.png"
    turntable_path = result_dir / "turntable.mp4"

    if not multiview_path.exists():
        return _copy_reference_fallback(reference_image_path, output_path, "FaceLift completed but multiview.png was not found")

    multi_view_images = _split_multiview(multiview_path, output_path)
    status = {
        "status": "success",
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "raw_output_dir": str(result_dir),
        "step_2d": step_2d,
        "auto_crop": auto_crop,
        "multiview_path": str(multiview_path),
        "gaussians_path": str(gaussians_path) if gaussians_path.exists() else None,
        "output_render_path": str(output_render_path) if output_render_path.exists() else None,
        "turntable_path": str(turntable_path) if turntable_path.exists() else None,
    }
    status_path = output_path / "facelift_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "facelift_status": "success",
        "model_path": str(gaussians_path) if gaussians_path.exists() else str(status_path),
        "multi_view_images": multi_view_images,
        "facelift_output_dir": str(result_dir),
        "multiview_path": str(multiview_path),
        "output_render_path": str(output_render_path) if output_render_path.exists() else None,
        "turntable_path": str(turntable_path) if turntable_path.exists() else None,
        "facelift_status_path": str(status_path),
    }
