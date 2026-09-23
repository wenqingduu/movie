#!/usr/bin/env python3
"""Build a paired Control-vs-v7 summary from EntityBench evaluator outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(directory: Path) -> dict[str, Any]:
    reports = sorted(directory.glob("report_*.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected one report_*.json in {directory}, found {len(reports)}")
    return json.loads(reports[0].read_text(encoding="utf-8"))


def _load_episodes(directory: Path) -> list[dict[str, Any]]:
    candidates = [
        path
        for path in sorted(directory.glob("*.json"))
        if not path.name.startswith("report_") and path.name != "run_manifest.json"
    ]
    if not candidates:
        raise ValueError(f"Expected episode JSON files in {directory}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in candidates]


def _metric_value(item: dict[str, Any]) -> float | None:
    value = item.get("mean", item.get("value"))
    return None if value is None else float(value)


def _read_condition(pillar1_dir: Path, no_vlm_dir: Path) -> dict[str, Any]:
    p1_report = _load_report(pillar1_dir)
    nv_report = _load_report(no_vlm_dir)
    episodes = _load_episodes(no_vlm_dir)

    vbench = {
        name: _metric_value(item)
        for name, item in p1_report["all"]["vbench"].items()
    }
    non_vlm_names = (
        "intra_character_presence",
        "intra_object_presence",
        "intra_location_presence",
        "cs_face",
        "cs_object",
        "cs_transition_boundary",
    )
    non_vlm = {
        name: _metric_value(nv_report["all"][name])
        for name in non_vlm_names
    }
    entities = {}
    for episode in episodes:
        episode_id = episode["episode_id"]
        for name, scores in episode.get("per_entity_scores", {}).items():
            entity_key = f"{episode_id}::{name}"
            entities[entity_key] = {
                key: value
                for key, value in scores.items()
                if key
                in {
                    "mean_sim_to_centroid",
                    "pairwise_median_sim",
                    "n_appearances",
                    "representative_shot",
                    "worst_shot",
                }
            }
    return {
        "episode_ids": [episode["episode_id"] for episode in episodes],
        "num_episodes": len(episodes),
        "num_shots": sum(episode["num_shots"] for episode in episodes),
        "vbench": vbench,
        "non_vlm": non_vlm,
        "per_entity": entities,
        "sources": {
            "pillar1": str(pillar1_dir),
            "non_vlm": str(no_vlm_dir),
        },
    }


def _deltas(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("vbench", "non_vlm"):
        result[group] = {}
        for name, control_value in control[group].items():
            treatment_value = treatment[group].get(name)
            result[group][name] = (
                None
                if control_value is None or treatment_value is None
                else treatment_value - control_value
            )
    common = sorted(set(control["per_entity"]) & set(treatment["per_entity"]))
    result["per_entity"] = {}
    for name in common:
        c = control["per_entity"][name]
        t = treatment["per_entity"][name]
        result["per_entity"][name] = {
            metric: t[metric] - c[metric]
            for metric in ("mean_sim_to_centroid", "pairwise_median_sim")
            if metric in c and metric in t
        }
    return result


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# EntityBench：Control vs v7",
        "",
        f"> 范围：{summary['num_episodes']} 个完整有序 episode、{summary['num_shots']} 个镜头。已跳过所有需要 VLM/大模型 API 的指标；因此这是部分官方评测，不是完整 EntityBench 排名。",
        "",
    ]
    for route_name, route in summary["routes"].items():
        lines += [f"## {route_name}", "", "### 跨镜头与实体指标", ""]
        lines += ["| 指标 | Control | v7 | Δ(v7-Control) |", "|---|---:|---:|---:|"]
        for metric in (
            "cs_face",
            "cs_object",
            "cs_transition_boundary",
            "intra_character_presence",
            "intra_object_presence",
            "intra_location_presence",
        ):
            c = route["control"]["non_vlm"][metric]
            t = route["v7"]["non_vlm"][metric]
            d = route["delta"]["non_vlm"][metric]
            lines.append(f"| `{metric}` | {_fmt(c)} | {_fmt(t)} | {_fmt(d)} |")

        lines += ["", "### 视频质量（VBench）", ""]
        lines += ["| 指标 | Control | v7 | Δ(v7-Control) |", "|---|---:|---:|---:|"]
        for metric in route["control"]["vbench"]:
            c = route["control"]["vbench"][metric]
            t = route["v7"]["vbench"][metric]
            d = route["delta"]["vbench"][metric]
            lines.append(f"| `{metric}` | {_fmt(c)} | {_fmt(t)} | {_fmt(d)} |")

        lines += ["", "### 逐实体 DINOv2（检测集合可能因条件而变化）", ""]
        lines += [
            "| 实体 | Control 均值 | v7 均值 | Δ | Control/v7 出现数 |",
            "|---|---:|---:|---:|---:|",
        ]
        for entity, delta in route["delta"]["per_entity"].items():
            c = route["control"]["per_entity"][entity]
            t = route["v7"]["per_entity"][entity]
            lines.append(
                f"| `{entity}` | {_fmt(c.get('mean_sim_to_centroid'))} | "
                f"{_fmt(t.get('mean_sim_to_centroid'))} | "
                f"{_fmt(delta.get('mean_sim_to_centroid'))} | "
                f"{c.get('n_appearances', '—')}/{t.get('n_appearances', '—')} |"
            )
        lines.append("")

    lines += [
        "## 未运行项",
        "",
        "VLM 人脸/物体/场景细节相似度、动作遵循、LLM judge 准确率均未运行；报告中的这些字段应为 null，不能当作 0。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        action="append",
        nargs=5,
        metavar=("NAME", "P1_CONTROL", "P1_V7", "NONVLM_CONTROL", "NONVLM_V7"),
        required=True,
        help="Repeat for each first-frame route.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    routes: dict[str, Any] = {}
    route_episode_ids: list[list[str]] = []
    for name, p1_control, p1_v7, nv_control, nv_v7 in args.route:
        control = _read_condition(Path(p1_control), Path(nv_control))
        treatment = _read_condition(Path(p1_v7), Path(nv_v7))
        if control["episode_ids"] != treatment["episode_ids"]:
            raise ValueError(
                f"{name}: Control and v7 episode sets differ: "
                f"{control['episode_ids']} vs {treatment['episode_ids']}"
            )
        route_episode_ids.append(control["episode_ids"])
        routes[name] = {
            "control": control,
            "v7": treatment,
            "delta": _deltas(control, treatment),
        }
    if any(ids != route_episode_ids[0] for ids in route_episode_ids[1:]):
        raise ValueError(f"Routes do not reference the same episodes: {route_episode_ids}")

    episode_ids = route_episode_ids[0]
    num_shots = next(iter(routes.values()))["control"]["num_shots"]

    summary = {
        "scope": "partial_entitybench_without_vlm",
        "skipped": [
            "VLM entity fidelity and attributes",
            "VLM action alignment",
            "LLM-judge similarity and accuracy",
        ],
        "episode_ids": episode_ids,
        "num_episodes": len(episode_ids),
        "num_shots": num_shots,
        "routes": routes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
