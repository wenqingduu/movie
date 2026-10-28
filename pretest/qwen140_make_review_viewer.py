#!/usr/bin/env python3
"""Make a local, hash-checked before/after viewer for completed integrated episodes."""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

from qwen140_integrated_episodes import BASE, OUT, episode_for, sha, source


RUNS = ("run700", "run714")
REVIEW = OUT / "review"
LABELS = {
    "verified_baseline_hardcut": "原片复用",
    "direct_continue": "A：实际尾帧直接接续",
    "new_shot": "C：新首帧与视频",
    "new_shot_clean_fallback": "C：首帧纠错后生成",
    "transition_rejected_fallback_clean_new_shot": "B 过渡失败，回退 C",
}


def video_info(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def link(target: Path, path: Path) -> None:
    assert target.is_file(), target
    if path.is_symlink():
        assert path.resolve() == target.resolve(), path
    else:
        assert not path.exists(), path
        path.symlink_to(target.resolve())


def make_one(run: str, manifest: dict, jobs: dict) -> dict:
    episode = episode_for(run, manifest)
    eid = episode["episode_id"]
    result = json.loads((OUT / "episodes" / eid / "episode_result.json").read_text())
    assert len(episode["shots"]) == len(result["shots"])
    folder = REVIEW / run
    folder.mkdir(parents=True, exist_ok=True)
    original = BASE / "episodes" / f"{eid}.mp4"
    link(original, folder / "before.mp4")
    rows = []
    concat = []
    for shot, record in zip(episode["shots"], result["shots"]):
        sk = shot["shot_key"]
        assert sk == record["shot_key"]
        prior = jobs[eid, sk]
        before = Path(prior["output_video"])
        after = Path(record["video"])
        assert sha(before) == record["baseline_video_sha256"]
        assert sha(after) == record["video_sha256"]
        tag = sk.replace(":", "_")
        link(before, folder / f"shot_{tag}_before.mp4")
        link(after, folder / f"shot_{tag}_after.mp4")
        concat.append(f"file '{after.resolve()}'")
        rows.append({"shot": sk, "changed": sha(before) != sha(after),
                     "source_kind": record["source_kind"], "route": (record.get("route") or {}).get("route"),
                     "description": shot["action_description"], "before": str(before), "after": str(after),
                     "before_sha256": record["baseline_video_sha256"], "after_sha256": record["video_sha256"]})
    (folder / "after.concat.txt").write_text("\n".join(concat) + "\n")
    combined = folder / "after.mp4"
    if not combined.exists():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                        "-safe", "0", "-i", str(folder / "after.concat.txt"), "-c", "copy",
                        "-movflags", "+faststart", str(combined)], check=True)
    before_info = video_info(original)
    after_info = video_info(combined)
    expected_frames = sum(int(video_info(Path(r["after"]))["nb_frames"]) for r in rows)
    assert int(after_info["nb_frames"]) == expected_frames, (run, after_info, expected_frames)
    assert int(before_info["nb_frames"]) == sum(int(video_info(Path(r["before"]))["nb_frames"]) for r in rows)
    assert before_info["width"] == after_info["width"] and before_info["height"] == after_info["height"]
    payload = {"run": run, "episode_id": eid, "num_shots": len(rows), "changed_shots": sum(r["changed"] for r in rows),
               "before": str(original), "after": str(combined), "before_info": before_info,
               "after_info": after_info, "before_sha256": sha(original), "after_sha256": sha(combined), "shots": rows}
    (folder / "review_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def render(items: list[dict]) -> str:
    tabs = []
    for item in items:
        run = item["run"]
        shot_rows = []
        for row in item["shots"]:
            sk = row["shot"]
            tag = sk.replace(":", "_")
            kind = LABELS.get(row["source_kind"], row["source_kind"])
            state = "已重跑" if row["changed"] else "未改动"
            shot_rows.append(f'''<section class="shot {'changed' if row['changed'] else 'same'}">
              <h3>{html.escape(sk)} · {state} · {html.escape(kind)}</h3>
              <p>{html.escape(row['description'])}</p>
              <div class="pair"><div><b>调整前</b><video controls preload="metadata" src="{run}/shot_{tag}_before.mp4"></video></div>
              <div><b>调整后</b><video controls preload="metadata" src="{run}/shot_{tag}_after.mp4"></video></div></div>
            </section>''')
        tabs.append(f'''<article id="{run}"><h2>{run} · {item['changed_shots']}/{item['num_shots']} 个镜头重跑</h2>
          <p>整集视频均为 16 fps；点击播放可在同一位置比较。下面按原镜头顺序列出逐镜对照。</p>
          <div class="pair full"><div><h3>调整前 · 整集</h3><video controls preload="metadata" src="{run}/before.mp4"></video>
          <a href="{run}/before.mp4">直接打开原版 MP4</a></div>
          <div><h3>调整后 · 整集</h3><video controls preload="metadata" src="{run}/after.mp4"></video>
          <a href="{run}/after.mp4">直接打开新版 MP4</a></div></div>
          {''.join(shot_rows)}</article>''')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Qwen140 两集重跑前后对照</title>
      <style>body{{font:16px/1.5 system-ui,sans-serif;background:#10131a;color:#e8edf6;margin:0 auto;max-width:1500px;padding:28px}}
      a{{color:#87bfff}}h1,h2{{margin:0 0 12px}}nav a{{margin-right:22px}}article{{margin-top:40px}}
      .pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.pair>div{{min-width:0}}
      video{{width:100%;display:block;background:#000;max-height:460px;margin:8px 0}}section.shot{{border-top:1px solid #3a4152;padding:16px 0}}
      section.same{{opacity:.62}}section.changed{{border-left:4px solid #e1ac55;padding-left:12px}}
      p{{color:#bcc5d4}}@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}</style></head><body>
      <h1>Qwen140 集成重跑：原版／新版视频</h1><p>橙色标记是实际重新生成的视频；灰色镜头复用原版。原始 cut 标记不因路由改变。</p>
      <nav><a href="#run700">run700</a><a href="#run714">run714</a></nav>{''.join(tabs)}</body></html>'''


def main() -> None:
    manifest, _, _, jobs = source()
    REVIEW.mkdir(parents=True, exist_ok=True)
    items = [make_one(run, manifest, jobs) for run in RUNS]
    (REVIEW / "index.html").write_text(render(items))
    print(REVIEW / "index.html")
    for item in items:
        print(item["run"], item["changed_shots"], "/", item["num_shots"], item["after_info"]["nb_frames"], "frames")


if __name__ == "__main__":
    main()
