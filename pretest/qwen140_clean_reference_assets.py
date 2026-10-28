#!/usr/bin/env python3
"""Create auditable single-person crops from contaminated episode reference sheets."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from qwen140_integrated_episodes import OUT, episode_for, sha, source, write_once

# Pixel boxes on the frozen 1024x1024 source assets, selected by visual review.
CROPS = {
    ("run700", "Arthur"): (225, 0, 800, 575),
    ("run700", "Nora"): (350, 0, 790, 340),
    ("run700", "Walter"): (400, 0, 955, 510),
    ("run714", "Nadia"): (0, 0, 510, 560),
    ("run714", "Owen"): (0, 0, 510, 560),
    ("run484", "Nora"): (0, 0, 510, 560),
}


def main(run_filter: str | None = None) -> None:
    manifest, prep, old_prep, _ = source()
    for (run, name), box in CROPS.items():
        if run_filter is not None and run != run_filter:
            continue
        ep = episode_for(run, manifest)
        rows = [r for r in prep["roles"] + old_prep["roles"]
                if r["episode_id"] == ep["episode_id"] and r["character"] == name]
        assert rows, (run, name)
        original = Path(rows[0]["reference"])
        assert original.is_file(), original
        suffix = "_v2" if run == "run700" else ""
        dest = OUT / "references" / "clean" / run / f"{name.lower()}{suffix}.png"
        record = dest.with_suffix(".json")
        condition = {"run": run, "episode_id": ep["episode_id"], "character": name,
                     "source": str(original.resolve()), "source_sha256": sha(original),
                     "crop_box": list(box), "method": "verbatim pixel crop; no generative retouch"}
        if record.exists():
            import json
            old = json.loads(record.read_text())
            assert old["inputs"] == condition and sha(dest) == old["sha256"]
            print("verified", dest)
            continue
        assert not dest.exists(), dest
        with Image.open(original) as image:
            assert image.size == (1024, 1024), (original, image.size)
            cropped = image.convert("RGB").crop(box)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(dest)
        write_once(record, {"inputs": condition, "sha256": sha(dest)})
        print("created", dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=sorted({run for run, _ in CROPS}))
    main(parser.parse_args().run)
