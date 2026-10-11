"""Audit cache validity and official metric preservation without loading weights."""
import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "pretest"))
from run_lightx2v_a14b_4step_continuation import fingerprint, reusable, sha256
from run_entitybench_a14b_4step_20ep import protect_evaluation_resume, environment, DIMENSIONS, episode_video_path
from prepare_entitybench_a14b_4step import OFFICIAL_SHA, PARTIAL_SHA


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ["video.mp4", "first.png", "last.png"]:
            (self.root / name).write_bytes(name.encode())
        self.expected = dict(input_sha256="unchanged_tail", parent_video_sha256="parent-v1", seed=42,
            runtime=dict(sampling_steps=4, lora_sha256="lora-v1", enable_cfg=False))
        self.record = dict(status="generated", dependency_fingerprint=fingerprint(self.expected),
            video_sha256=sha256(self.root / "video.mp4"), first_frame_sha256=sha256(self.root / "first.png"),
            last_frame_sha256=sha256(self.root / "last.png"))

    def test_valid_generation_resumes(self):
        self.assertTrue(reusable(self.record, self.expected, self.root))

    def test_changed_parent_invalidates_identical_tail(self):
        self.assertFalse(reusable(self.record, dict(self.expected, parent_video_sha256="parent-v2"), self.root))

    def test_changed_distillation_or_seed_invalidates(self):
        for key, value in [("sampling_steps", 40), ("lora_sha256", "lora-v2"), ("enable_cfg", True)]:
            changed = copy.deepcopy(self.expected)
            changed["runtime"][key] = value
            self.assertFalse(reusable(self.record, changed, self.root))
        self.assertFalse(reusable(self.record, dict(self.expected, seed=43), self.root))

    def test_corrupted_media_and_failed_jobs_never_resume(self):
        (self.root / "last.png").write_bytes(b"corrupted")
        self.assertFalse(reusable(self.record, self.expected, self.root))
        self.assertFalse(reusable(dict(self.record, status="failed"), self.expected, self.root))

    def test_official_resume_rejects_changed_video(self):
        out = self.root / "evaluation"
        protect_evaluation_resume(out, {"video": "v1", "script": "s1"})
        protect_evaluation_resume(out, {"script": "s1", "video": "v1"})
        with self.assertRaises(AssertionError):
            protect_evaluation_resume(out, {"video": "v2", "script": "s1"})

    def test_untracked_score_cache_is_rejected(self):
        (self.root / "run_manifest.json").write_text("{}")
        with self.assertRaises(AssertionError):
            protect_evaluation_resume(self.root, {"video": "v1"})

    def test_frozen_official_scores_are_not_replaced_by_project_metrics(self):
        directory = PROJECT / "benchmarks/entitybench/eval"
        official = directory / "evaluate_benchmark_official.py"
        partial = directory / "evaluate_benchmark.py"
        self.assertEqual(sha256(official), OFFICIAL_SHA)
        self.assertEqual(sha256(partial), PARTIAL_SHA)
        trees = [ast.parse(path.read_text()) for path in [official, partial]]
        functions = [{n.name: ast.dump(n, include_attributes=False) for n in ast.walk(t)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for t in trees]
        names = [name for name in functions[0] if "dino" in name or "vbench" in name or "crop" in name]
        self.assertTrue(names)
        for name in names:
            self.assertEqual(functions[0][name], functions[1][name], name)
        for tree in trees:
            assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "VBENCH_DIMENSIONS" for t in n.targets))
            self.assertEqual(ast.literal_eval(assignment.value), DIMENSIONS)

    def test_evaluation_environment_has_no_vlm_keys(self):
        env = environment()
        self.assertFalse(any("OPENAI" in k or "GEMINI" in k or "AZURE" in k for k in env))

    def test_distinct_windows_of_same_run_have_distinct_episode_movies(self):
        first = dict(run="run893", episode_id="run893__i5_j15")
        second = dict(run="run893", episode_id="run893__i95_j118")
        self.assertNotEqual(episode_video_path(self.root, first), episode_video_path(self.root, second))


if __name__ == "__main__":
    unittest.main()
