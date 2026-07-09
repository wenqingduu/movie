import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


GITHUB_API_ROOT = "https://api.github.com/repos/Catherine-R-He/EntityBench/contents"
DEFAULT_OUTPUT_DIR = Path("benchmarks") / "entitybench"
DEFAULT_SUBDIRS = ("data/scripts", "data/splits", "examples")


def _request_json(url: str):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "multishot-entitybench-downloader",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url: str, output_path: Path, retries: int = 3):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "multishot-entitybench-downloader",
        },
    )

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            output_path.write_bytes(data)
            return len(data)
        except (urllib.error.HTTPError, urllib.error.URLError):
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)

    return 0


def _download_file_from_api(api_url: str, output_path: Path):
    data = _request_json(api_url)
    if data.get("encoding") != "base64" or "content" not in data:
        raise RuntimeError(f"GitHub API response did not include base64 content: {api_url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(data["content"])
    output_path.write_bytes(payload)
    return len(payload)


def _list_directory(repo_subdir: str):
    url = f"{GITHUB_API_ROOT}/{repo_subdir}"
    entries = _request_json(url)
    if not isinstance(entries, list):
        raise RuntimeError(f"Expected GitHub directory listing for {repo_subdir}")
    return entries


def download_entitybench(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    subdirs: tuple[str, ...] = DEFAULT_SUBDIRS,
    limit: int | None = None,
    overwrite: bool = False,
):
    """Download EntityBench metadata for multi-shot evaluation.

    This downloads the public benchmark metadata from GitHub:
    - data/scripts: episode JSON files with shot prompts and entity schedules
    - data/splits: easy/medium/hard split JSON files
    - examples: launch examples from the benchmark repository

    It does not download generated videos, model checkpoints, or evaluator
    dependencies.
    """

    output_dir = Path(output_dir)
    total_files = 0
    downloaded_files = 0
    skipped_files = 0

    for subdir in subdirs:
        entries = [
            entry for entry in _list_directory(subdir)
            if entry.get("type") == "file" and entry.get("download_url")
        ]
        if limit is not None:
            entries = entries[:limit]

        for entry in entries:
            relative_path = Path(entry["path"])
            output_path = output_dir / relative_path
            total_files += 1

            if output_path.exists() and not overwrite:
                skipped_files += 1
                print(f"skip existing {output_path}")
                continue

            try:
                size = _download_file(entry["download_url"], output_path)
                source = "raw"
            except (urllib.error.HTTPError, urllib.error.URLError):
                size = _download_file_from_api(entry["url"], output_path)
                source = "api"
            downloaded_files += 1
            print(f"downloaded {output_path} ({size} bytes, {source})")

    summary = {
        "output_dir": str(output_dir),
        "subdirs": list(subdirs),
        "total_files": total_files,
        "downloaded_files": downloaded_files,
        "skipped_files": skipped_files,
    }
    summary_path = output_dir / "download_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"summary -> {summary_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download EntityBench benchmark scripts and splits."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Local output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--subdir",
        action="append",
        dest="subdirs",
        help=(
            "GitHub repository subdir to download. Can be repeated. "
            "Default: data/scripts, data/splits, examples"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N files from each selected subdir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist locally.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    subdirs = tuple(args.subdirs) if args.subdirs else DEFAULT_SUBDIRS
    download_entitybench(
        output_dir=Path(args.output_dir),
        subdirs=subdirs,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
