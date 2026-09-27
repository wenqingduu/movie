#!/usr/bin/env python3
"""Summarize the current three-episode Control-vs-v7 evaluation.

This script is intentionally evaluation-only: it never runs a generator and
never changes an experiment result.  It reads the six current route/episode
roots and emits both machine-readable JSON and a detailed Markdown report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multishot.prompt_injection_safety import prompt_requests_closed_eyes  # noqa: E402

CURRENT_CASES = (
    {
        "route": "PuLID-FLUX",
        "episode": "run719",
        "root": "outputs/entitybench_wan22_colornested_v7_s04_12/episode_00053051",
        "first_frame_report": "pulid_first_frame_report.json",
        "result_subdir": "pulid_flux",
        "result_name": "metrics.json",
    },
    {
        "route": "IP-Adapter",
        "episode": "run719",
        "root": "outputs/entitybench_wan22_ip_adapter_v7_s04_12/episode_00053051",
        "first_frame_report": "ip_adapter_first_frame_report.json",
        "result_subdir": "ip_adapter",
        "result_name": "result.json",
    },
    {
        "route": "PuLID-FLUX",
        "episode": "run893",
        "root": "outputs/entitybench_wan22_pulid_v7_pilot3/episode_run893",
        "first_frame_report": "pulid_first_frame_report.json",
        "result_subdir": "pulid_flux",
        "result_name": "metrics.json",
    },
    {
        "route": "IP-Adapter",
        "episode": "run893",
        "root": "outputs/entitybench_wan22_ip_adapter_v7_pilot3/episode_run893",
        "first_frame_report": "ip_adapter_first_frame_report.json",
        "result_subdir": "ip_adapter",
        "result_name": "result.json",
    },
    {
        "route": "PuLID-FLUX",
        "episode": "run1517",
        "root": "outputs/entitybench_wan22_pulid_v7_pilot3/episode_run1517",
        "first_frame_report": "pulid_first_frame_report.json",
        "result_subdir": "pulid_flux",
        "result_name": "metrics.json",
    },
    {
        "route": "IP-Adapter",
        "episode": "run1517",
        "root": "outputs/entitybench_wan22_ip_adapter_v7_pilot3/episode_run1517",
        "first_frame_report": "ip_adapter_first_frame_report.json",
        "result_subdir": "ip_adapter",
        "result_name": "result.json",
    },
)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _delta(control: float | None, treatment: float | None) -> float | None:
    if control is None or treatment is None:
        return None
    return _round(treatment - control)


def _direct_identity(route: str, result: dict[str, Any]) -> dict[str, float | None]:
    if route == "PuLID-FLUX":
        control = result.get("reference_control_insightface_cosine")
        treatment = result.get("reference_treatment_insightface_cosine")
    else:
        metrics = result.get("metrics", {})
        control = metrics.get("ip_adapter_baseline", {}).get(
            "reference_portrait_cosine"
        )
        treatment = metrics.get("ip_adapter_plus_pulid_style_residual", {}).get(
            "reference_portrait_cosine"
        )
    return {
        "control": _round(control),
        "treatment": _round(treatment),
        "delta": _delta(control, treatment),
    }


def _mean_direct(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row["direct_identity"]
        for row in rows
        if row["plugin_applied"]
        and row["direct_identity"]["control"] is not None
        and row["direct_identity"]["treatment"] is not None
    ]
    if not valid:
        return {"valid_pair_count": 0}
    return {
        "valid_pair_count": len(valid),
        "control_mean": _round(statistics.mean(item["control"] for item in valid)),
        "treatment_mean": _round(
            statistics.mean(item["treatment"] for item in valid)
        ),
        "mean_paired_delta": _round(
            statistics.mean(item["delta"] for item in valid)
        ),
        "positive_delta_count": sum(item["delta"] > 0 for item in valid),
    }


def _mean_scalar(values: list[float | None]) -> dict[str, Any]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return {"valid_pair_count": 0, "mean_paired_delta": None, "positive_delta_count": 0}
    return {
        "valid_pair_count": len(valid),
        "mean_paired_delta": _round(statistics.mean(valid)),
        "positive_delta_count": sum(value > 0 for value in valid),
    }


def _route_aggregates(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for route in sorted({case["route"] for case in cases}):
        rows = [
            shot
            for case in cases
            if case["route"] == route
            for shot in case["shots"]
            if len(shot["characters"]) == 1
        ]
        applied = [shot for shot in rows if shot["plugin_applied"]]
        def all_pair_delta(shot: dict[str, Any], value: float | None) -> float | None:
            # A safety-gated pair is byte-identical by construction, so its
            # paired delta is exactly zero even when no face was detectable.
            return value if shot["plugin_applied"] else 0.0

        result[route] = {
            "single_character_pair_count": len(rows),
            "injected_pair_count": len(applied),
            "control_reuse_pair_count": len(rows) - len(applied),
            "applied_pairs": {
                "first_frame_insightface": _mean_scalar(
                    [shot["direct_identity"]["delta"] for shot in applied]
                ),
                "video_insightface_mean": _mean_scalar(
                    [shot["video_identity_delta"].get("mean") for shot in applied]
                ),
                "first_frame_reference_dino": _mean_scalar(
                    [shot["reference_dino"]["delta"].get("first_frame") for shot in applied]
                ),
                "video_reference_dino_mean": _mean_scalar(
                    [shot["reference_dino"]["delta"].get("video_frame_mean") for shot in applied]
                ),
            },
            "all_single_pairs_including_control_reuse": {
                "first_frame_insightface": _mean_scalar(
                    [
                        all_pair_delta(shot, shot["direct_identity"]["delta"])
                        for shot in rows
                    ]
                ),
                "video_insightface_mean": _mean_scalar(
                    [
                        all_pair_delta(shot, shot["video_identity_delta"].get("mean"))
                        for shot in rows
                    ]
                ),
                "first_frame_reference_dino": _mean_scalar(
                    [
                        all_pair_delta(
                            shot, shot["reference_dino"]["delta"].get("first_frame")
                        )
                        for shot in rows
                    ]
                ),
                "video_reference_dino_mean": _mean_scalar(
                    [
                        all_pair_delta(
                            shot,
                            shot["reference_dino"]["delta"].get("video_frame_mean"),
                        )
                        for shot in rows
                    ]
                ),
            },
        }
    return result


def _lookup_cases(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for case in summary.get("cases", []):
        result[(case["route"], case["episode_id"])] = case
    return result


def _pair_lookup(case: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not case:
        return {}
    return {item["shot_key"]: item for item in case.get("pairs", [])}


def _video_pair_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["shot_key"]: item for item in report.get("paired_comparisons", [])
    }


def _episode_shot_count(episode_path: Path) -> int:
    episode = _load(episode_path)
    return sum(len(scene.get("video_prompts", [])) for scene in episode.get("scenes", []))


def _summarize_case(
    spec: dict[str, str],
    dino_cases: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    root = PROJECT_ROOT / spec["root"]
    first_report = _load(root / spec["first_frame_report"])
    video_report = _load(root / "video_identity_report_single_character.json")
    episode_id = first_report["episode_id"]
    episode_path = Path(first_report["episode_path"])
    video_pairs = _video_pair_lookup(video_report)
    dino_case = dino_cases.get((spec["route"], episode_id))
    dino_pairs = _pair_lookup(dino_case)
    rows = []
    for shot in first_report["shots"]:
        shot_key = shot["shot_key"]
        characters = shot.get("characters", []) or []
        is_single = len(characters) == 1
        recorded_skip = bool(shot.get("plugin_skipped", False))
        eye_gate = prompt_requests_closed_eyes(shot.get("prompt", ""))
        plugin_applied = is_single and not recorded_skip and not eye_gate
        result_path = (
            root
            / "first_frames"
            / f"shot_{shot_key.replace(':', '_')}"
            / spec["result_subdir"]
            / spec["result_name"]
        )
        direct = (
            _direct_identity(spec["route"], _load(result_path))
            if result_path.is_file()
            else {"control": None, "treatment": None, "delta": None}
        )
        video_pair = video_pairs.get(shot_key, {})
        dino_pair = dino_pairs.get(shot_key, {})
        skip_reason = shot.get("skip_reason") or shot.get("reason")
        skip_reason = "prompt_eye_closure_conflict" if eye_gate else skip_reason
        rows.append(
            {
                "shot_key": shot_key,
                "characters": characters,
                "status": shot.get("status"),
                "plugin_applied": plugin_applied,
                "skip_reason": skip_reason,
                "direct_identity": direct,
                "video_identity_delta": video_pair.get("treatment_minus_control", {}),
                "reference_dino": {
                    "control": dino_pair.get("control", {}),
                    "treatment": dino_pair.get("treatment", {}),
                    "delta": dino_pair.get("treatment_minus_control", {}),
                },
                "result_path": str(result_path),
            }
        )
    single_count = sum(len(item.get("characters", []) or []) == 1 for item in first_report["shots"])
    return {
        "route": spec["route"],
        "episode": spec["episode"],
        "episode_id": episode_id,
        "root": str(root),
        "total_shot_count": _episode_shot_count(episode_path),
        "single_character_shot_count": single_count,
        "injected_shots": sum(row["plugin_applied"] for row in rows),
        "control_reuse_shots": sum(
            len(row["characters"]) == 1 and not row["plugin_applied"] for row in rows
        ),
        "direct_first_frame_identity_applied": _mean_direct(rows),
        "video_identity_applied": video_report["aggregate_applied_pairs"],
        "reference_dino_applied": (
            dino_case.get("aggregate_applied_pairs") if dino_case else None
        ),
        "shots": rows,
    }


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# EntityBench 当前三 episode 评测结果",
        "",
        "> 本文件由 `pretest/summarize_current_episode_results.py` 自动生成。不要手工修改数值。",
        "",
        "## 范围",
        "",
        f"共 {summary['num_episodes']} 个完整有序 episode、{summary['num_shots']} 个镜头；当前两条路线只对单角色镜头执行 v7。",
        "",
        "| 路线 | Episode | 总镜头 | 单角色 | 实际注入 | 安全复用 Control |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| {case['route']} | {case['episode']} | {case['total_shot_count']} | "
            f"{case['single_character_shot_count']} | {case['injected_shots']} | "
            f"{case['control_reuse_shots']} |"
        )

    lines += ["", "## 跨 episode 总结果（实际注入镜头）", ""]
    lines += [
        "| 路线 | 注入/单角色 | 首帧 InsightFace Δ | 视频 InsightFace mean Δ | 首帧 DINOv2 Δ | 视频 DINOv2 mean Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route, aggregate in summary["route_aggregates"].items():
        metrics = aggregate["applied_pairs"]
        lines.append(
            f"| {route} | {aggregate['injected_pair_count']}/{aggregate['single_character_pair_count']} | "
            f"{_fmt(metrics['first_frame_insightface']['mean_paired_delta'], True)} | "
            f"{_fmt(metrics['video_insightface_mean']['mean_paired_delta'], True)} | "
            f"{_fmt(metrics['first_frame_reference_dino']['mean_paired_delta'], True)} | "
            f"{_fmt(metrics['video_reference_dino_mean']['mean_paired_delta'], True)} |"
        )

    lines += ["", "安全复用镜头按 Δ=0 纳入完整单角色口径后：", ""]
    lines += [
        "| 路线 | 单角色对数 | 首帧 InsightFace Δ | 视频 InsightFace mean Δ | 首帧 DINOv2 Δ | 视频 DINOv2 mean Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route, aggregate in summary["route_aggregates"].items():
        metrics = aggregate["all_single_pairs_including_control_reuse"]
        lines.append(
            f"| {route} | {aggregate['single_character_pair_count']} | "
            f"{_fmt(metrics['first_frame_insightface']['mean_paired_delta'], True)} | "
            f"{_fmt(metrics['video_insightface_mean']['mean_paired_delta'], True)} | "
            f"{_fmt(metrics['first_frame_reference_dino']['mean_paired_delta'], True)} | "
            f"{_fmt(metrics['video_reference_dino_mean']['mean_paired_delta'], True)} |"
        )

    lines += ["", "## Episode 汇总", ""]
    lines += [
        "| 路线 | Episode | 首帧 InsightFace Δ | 视频 InsightFace mean Δ | 参考图 DINOv2 首帧 Δ | 参考图 DINOv2 视频 mean Δ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        direct = case["direct_first_frame_identity_applied"]
        video = case["video_identity_applied"]["metrics"]["mean"]
        dino = (case["reference_dino_applied"] or {}).get("metrics", {})
        lines.append(
            f"| {case['route']} | {case['episode']} | "
            f"{_fmt(direct.get('mean_paired_delta'), True)} | "
            f"{_fmt(video.get('mean_paired_delta'), True)} | "
            f"{_fmt(dino.get('first_frame', {}).get('mean_paired_delta'), True)} | "
            f"{_fmt(dino.get('video_frame_mean', {}).get('mean_paired_delta'), True)} |"
        )

    qwen = summary.get("qwen_multiface_pilot")
    if qwen:
        lines += [
            "",
            "## Qwen-Image-2.1 多角色 pilot",
            "",
            "run719 / shot 1:1 使用 Viktor、Julian 两张身份参考图生成双人 Control，随后在两张独立 v7 mask 内同步注入。下游仍为 Wan2.2-TI2V-5B；Qwen 是首帧图像模型，不是视频模型。",
            "",
            "| 角色 | 首帧 Control | 首帧 Treatment | 首帧 Δ | 视频 Control mean | 视频 Treatment mean | 视频 Δ | 检测覆盖率 C/T |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        video_conditions = qwen["video"]["conditions"]
        video_deltas = qwen["video"]["treatment_minus_control"]
        for character in qwen["first_frame"]["characters"]:
            name = character["name"]
            control_video = video_conditions["control"]["characters"][name]
            treatment_video = video_conditions["treatment"]["characters"][name]
            lines.append(
                f"| {name} | {character['control_cosine']:.4f} | "
                f"{character['treatment_cosine']:.4f} | {_fmt(character['delta'], True)} | "
                f"{control_video['mean']:.4f} | {treatment_video['mean']:.4f} | "
                f"{_fmt(video_deltas[name]['mean'], True)} | "
                f"{control_video['detection_coverage']:.2f}/{treatment_video['detection_coverage']:.2f} |"
            )
        lines += [
            "",
            f"首帧两角色宏平均 Δ={_fmt(qwen['first_frame']['macro']['mean_paired_delta'], True)}；视频 mean 宏平均 Δ={_fmt(qwen['video']['macro']['mean_paired_delta'], True)}。该单镜头 pilot 验证了多人链路可运行，但视频增益尚未做到逐角色一致。",
            "",
            f"首帧对比：`{qwen['comparison_image']}`  ",
            f"视频并排对比：`{qwen['comparison_video']}`",
        ]

    for case in summary["cases"]:
        lines += [
            "",
            f"## {case['route']} / {case['episode']} 逐镜头",
            "",
            "| Shot | 角色 | 插件处理 | 首帧 C→T (Δ) | 视频身份 mean Δ | DINO 视频 mean Δ | 跳过原因 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for shot in case["shots"]:
            direct = shot["direct_identity"]
            direct_text = (
                "—"
                if direct["control"] is None or direct["treatment"] is None
                else f"{direct['control']:.4f}→{direct['treatment']:.4f} ({_fmt(direct['delta'], True)})"
            )
            lines.append(
                f"| {shot['shot_key']} | {', '.join(shot['characters']) or '—'} | "
                f"{'注入' if shot['plugin_applied'] else '复用 Control'} | "
                f"{direct_text} | "
                f"{_fmt(shot['video_identity_delta'].get('mean'), True)} | "
                f"{_fmt(shot['reference_dino']['delta'].get('video_frame_mean'), True)} | "
                f"{shot['skip_reason'] or '—'} |"
            )

    lines += [
        "",
        "## 指标解释",
        "",
        "- InsightFace：以角色原始参考脸为锚点，主要回答身份识别是否更接近。",
        "- 参考图 DINOv2：比较扩展人脸裁剪的整体视觉特征；与 InsightFace 不同向时，不能只按身份 cosine 宣称视觉改善。",
        "- 未通过预先固定 gate 的镜头保留在完整 episode，Treatment 复用 Control，成对增益为 0。",
        "- `prompt_eye_closure_conflict` 表示提示词明确要求闭眼，而 3D 身份参考为中性睁眼，因此安全跳过局部 residual。",
        "- VLM/LLM API 指标尚未运行，不得把缺失值当作 0。",
        "",
        "## 权威机器可读输出",
        "",
        f"`{summary['output_json']}`",
        "",
    ]
    return "\n".join(lines)


def build_summary() -> dict[str, Any]:
    dino_path = PROJECT_ROOT / "outputs/entitybench_reference_dino_pilot3/reference_dino_summary.json"
    dino_summary = _load(dino_path)
    dino_cases = _lookup_cases(dino_summary)
    cases = [_summarize_case(spec, dino_cases) for spec in CURRENT_CASES]
    episode_ids = list(dict.fromkeys(case["episode_id"] for case in cases))
    shot_counts = {
        case["episode_id"]: case["total_shot_count"] for case in cases
    }
    summary = {
        "kind": "current_entitybench_three_episode_detailed_summary",
        "num_episodes": len(episode_ids),
        "num_shots": sum(shot_counts.values()),
        "episode_ids": episode_ids,
        "cases": cases,
        "reference_dino_summary": str(dino_path),
        "entitybench_non_vlm_status": "pending_recompute_after_current_safety_gate",
    }
    summary["route_aggregates"] = _route_aggregates(cases)
    qwen_root = PROJECT_ROOT / "outputs/qwen_image21_multiface_v7/run719_shot_1_1"
    qwen_first = qwen_root / "paired_v7_s04/identity_metrics.json"
    qwen_video = qwen_root / "video_identity_metrics.json"
    if qwen_first.is_file() and qwen_video.is_file():
        summary["qwen_multiface_pilot"] = {
            "episode": "run719",
            "shot_key": "1:1",
            "first_frame": _load(qwen_first),
            "video": _load(qwen_video),
            "comparison_image": str(
                qwen_root / "paired_v7_s04/comparison.jpg"
            ),
            "comparison_video": str(
                qwen_root / "videos/comparison_side_by_side.mp4"
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "outputs/entitybench_current_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "EPISODE_TEST_RESULTS.md",
    )
    args = parser.parse_args()
    summary = build_summary()
    summary["output_json"] = str(args.output_json.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json.resolve()),
        "output_md": str(args.output_md.resolve()),
        "num_episodes": summary["num_episodes"],
        "num_shots": summary["num_shots"],
        "case_count": len(summary["cases"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
