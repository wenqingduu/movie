"""Protect expensive rollout resume against parent/config/media changes."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pretest"))
from run_wan22_i2v_a14b_continuation import fingerprint, reusable, sha256
from prepare_entitybench_a14b_continuation import ordered_shots


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for name in ["video.mp4", "first.png", "last.png"]:
            (self.directory / name).write_bytes(name.encode())
        self.expected = {"input_sha256": "input", "parent_video_sha256": "parent-v1",
                         "runtime": {"sampling_steps": 40}, "seed": 42}
        self.record = {"status": "generated", "dependency_fingerprint": fingerprint(self.expected),
                       "video_sha256": sha256(self.directory / "video.mp4"),
                       "first_frame_sha256": sha256(self.directory / "first.png"),
                       "last_frame_sha256": sha256(self.directory / "last.png")}

    def test_exact_dependencies_resume(self):
        self.assertTrue(reusable(self.record, self.expected, self.directory))

    def test_changed_parent_invalidates_even_when_tail_pixels_match(self):
        changed = dict(self.expected, parent_video_sha256="parent-v2")
        self.assertFalse(reusable(self.record, changed, self.directory))

    def test_changed_config_invalidates(self):
        changed = dict(self.expected, runtime={"sampling_steps": 6})
        self.assertFalse(reusable(self.record, changed, self.directory))

    def test_corrupted_video_invalidates(self):
        (self.directory / "video.mp4").write_bytes(b"changed")
        self.assertFalse(reusable(self.record, self.expected, self.directory))

    def test_missing_tail_invalidates(self):
        (self.directory / "last.png").unlink()
        self.assertFalse(reusable(self.record, self.expected, self.directory))

    def test_failed_job_never_resumes(self):
        self.record["status"] = "failed"
        self.assertFalse(reusable(self.record, self.expected, self.directory))

    def test_order_keeps_global_index_and_cross_scene_cut(self):
        episode = {"scenes": [
            {"scene_num": 5, "video_prompts": ["A", "B"], "cut": [True, False]},
            {"scene_num": 9, "video_prompts": ["C"], "cut": [False]}], "entity_schedule": {}}
        shots = ordered_shots(episode)
        self.assertEqual([s["shot_key"] for s in shots], ["5:1", "5:2", "9:1"])
        self.assertEqual([s["shot_index"] for s in shots], [0, 1, 2])
        self.assertFalse(shots[2]["cut"])

    def test_continuation_cannot_start_episode(self):
        with self.assertRaises(AssertionError):
            ordered_shots({"scenes": [{"scene_num": 1, "video_prompts": ["A"], "cut": [False]}],
                           "entity_schedule": {}})


if __name__ == "__main__":
    unittest.main()
