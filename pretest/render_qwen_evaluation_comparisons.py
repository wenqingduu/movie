#!/usr/bin/env python3
"""Render complete per-shot Qwen evaluation contact sheets.

The canonical sheet keeps the same visual evidence order as the earlier
PuLID/IP-Adapter experiments: identity portrait, continuous FaceLift render,
harmonized/aligned 3D reference, Control, and Treatment.  Multi-character
shots repeat the first three panels once per assigned character.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_SIZE = (420, 420)
LABEL_HEIGHT = 82
COLUMNS = 3


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{float(value):+.4f}" if signed else f"{float(value):.4f}"


def _fmt_triplet(value: dict[str, Any] | None) -> str:
    value = value or {}
    control = value.get("control")
    treatment = value.get("treatment")
    if control is None or treatment is None:
        return "--"
    return (
        f"{_fmt(control)}->{_fmt(treatment)} "
        f"({_fmt(value.get('delta'), signed=True)})"
    )


def _placeholder(label: str) -> Image.Image:
    image = Image.new("RGB", PANEL_SIZE, (232, 232, 232))
    draw = ImageDraw.Draw(image)
    text = "not available"
    box = draw.textbbox((0, 0), text)
    draw.text(
        ((image.width - (box[2] - box[0])) // 2, (image.height - (box[3] - box[1])) // 2),
        text,
        fill=(90, 90, 90),
    )
    return image


def _panel_image(path: Path | None, label: str) -> Image.Image:
    if path is None or not path.is_file():
        return _placeholder(label)
    return Image.open(path).convert("RGB")


def _wrapped_lines(draw: ImageDraw.ImageDraw, text: str, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width = draw.textbbox((0, 0), candidate)[2]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:4]


def _make_sheet(items: list[tuple[str, Path | None]], output: Path) -> None:
    rows = (len(items) + COLUMNS - 1) // COLUMNS
    sheet = Image.new(
        "RGB",
        (COLUMNS * PANEL_SIZE[0], rows * (PANEL_SIZE[1] + LABEL_HEIGHT)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        image = _panel_image(path, label)
        image.thumbnail(PANEL_SIZE, Image.Resampling.LANCZOS)
        cell_x = (index % COLUMNS) * PANEL_SIZE[0]
        cell_y = (index // COLUMNS) * (PANEL_SIZE[1] + LABEL_HEIGHT)
        x = cell_x + (PANEL_SIZE[0] - image.width) // 2
        y = cell_y + (PANEL_SIZE[1] - image.height) // 2
        sheet.paste(image, (x, y))
        for line_index, line in enumerate(_wrapped_lines(draw, label, PANEL_SIZE[0] - 16)):
            draw.text(
                (cell_x + 8, cell_y + PANEL_SIZE[1] + 7 + line_index * 16),
                line,
                fill="black",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def _character_metrics(shot_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    insight_path = shot_dir / "identity_metrics.json"
    if insight_path.is_file():
        for item in _load(insight_path).get("characters", []):
            result.setdefault(item["name"], {})["image_if"] = {
                "control": item.get("control_cosine"),
                "treatment": item.get("treatment_cosine"),
                "delta": item.get("delta"),
            }
    video_path = shot_dir / "video_identity_metrics.json"
    if video_path.is_file():
        video = _load(video_path)
        for name, item in video.get("treatment_minus_control", {}).items():
            result.setdefault(name, {})["video_if"] = {
                "control": video["conditions"]["control"]["characters"][name].get("mean"),
                "treatment": video["conditions"]["treatment"]["characters"][name].get("mean"),
                "delta": item.get("mean"),
            }
    dino_path = shot_dir / "dino_identity_metrics.json"
    if dino_path.is_file():
        dino = _load(dino_path)
        for name, item in dino.get("treatment_minus_control", {}).items():
            control = dino["conditions"]["control"]["characters"][name]
            treatment = dino["conditions"]["treatment"]["characters"][name]
            result.setdefault(name, {})["image_dino"] = {
                "control": control.get("first_frame"),
                "treatment": treatment.get("first_frame"),
                "delta": item.get("first_frame"),
            }
            result.setdefault(name, {})["video_dino"] = {
                "control": control.get("video_frame_mean"),
                "treatment": treatment.get("video_frame_mean"),
                "delta": item.get("video_frame_mean"),
            }
    return result


def _combined_mask(prepared: dict[str, Any] | None, shot_dir: Path) -> Path | None:
    masks = []
    for target in (prepared or {}).get("targets", []):
        path = Path(target["mask"])
        if path.is_file():
            masks.append(Image.open(path).convert("L"))
    if not masks:
        return None
    combined = masks[0]
    for mask in masks[1:]:
        combined = ImageChops.lighter(combined, mask)
    output = shot_dir / "prepared/combined_injection_mask.png"
    combined.save(output)
    return output


def _render_shot(record: dict[str, Any]) -> Path:
    shot_dir = Path(record["shot_dir"])
    prepared_path = shot_dir / "prepared/prepared.json"
    prepared = _load(prepared_path) if prepared_path.is_file() else None
    metrics = _character_metrics(shot_dir)
    prepared_targets = {
        item["name"]: item for item in (prepared or {}).get("targets", [])
    }
    result_path = shot_dir / "result.json"
    result = _load(result_path) if result_path.is_file() else {}
    applied_characters = set(
        result.get("applied_characters")
        or [item["name"] for item in (prepared or {}).get("targets", [])]
    )
    items: list[tuple[str, Path | None]] = []
    for character in record["characters"]:
        name = character["name"]
        target = prepared_targets.get(name)
        values = metrics.get(name, {})
        role_state = "injected" if name in applied_characters else "skipped"
        suffix = (
            f"role={role_state}; IF image={_fmt_triplet(values.get('image_if'))}; "
            f"IF video={_fmt_triplet(values.get('video_if'))}; "
            f"DINO image={_fmt_triplet(values.get('image_dino'))}; "
            f"DINO video={_fmt_triplet(values.get('video_dino'))}"
        )
        char_dir = shot_dir / "prepared/characters" / name.lower().replace("-", "_").replace(" ", "_")
        if target and target.get("harmonized_reference"):
            harmonized = Path(target["harmonized_reference"])
            char_dir = harmonized.parent
        else:
            harmonized = None
        items.extend(
            [
                (f"{name}: identity reference | {suffix}", Path(character["reference_image"])),
                (f"{name}: continuous FaceLift 3D render", char_dir / "rendered_3d_face.png" if target else None),
                (f"{name}: harmonized aligned 3D reference", harmonized),
            ]
        )

    injected = result.get("injection_applied") or result.get("kind") == (
        "qwen_image21_shared_prefix_pred_x0_v7_pilot"
    )
    state = "v7 injected" if injected else "Control reused"
    reason = result.get("reason")
    if reason:
        state += f" ({reason})"
    mask = _combined_mask(prepared, shot_dir)
    items.extend(
        [
            ("combined v7 final injection mask", mask),
            ("Qwen-Image-2.1 multi-reference Control", shot_dir / "control.png"),
            (f"Qwen-Image-2.1 + v7 3D residual | {state}", shot_dir / "treatment.png"),
        ]
    )
    canonical = shot_dir / "comparison.jpg"
    archived = shot_dir / "pair/comparison_control_treatment.jpg"
    if canonical.is_file() and not archived.is_file():
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(canonical, archived)
    _make_sheet(items, canonical)
    shutil.copy2(canonical, shot_dir / "evaluation_comparison.jpg")
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/entitybench_qwen_image21_v7_full3",
    )
    args = parser.parse_args()
    manifest = _load(args.output_root / "manifest.json")
    for index, record in enumerate(manifest["shots"], 1):
        path = _render_shot(record)
        print(f"[{index}/{len(manifest['shots'])}] {record['run']} {record['shot_key']} -> {path}")


if __name__ == "__main__":
    main()
