import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from pretest.evaluate_cross_shot_face_centroid import aggregate, centroid_scores, paired_groups, representative_video_crop


def row(shot, control, treatment, episode="episode_a", character="same_name", applied=True):
    return {"route": "route", "episode_id": episode, "run": episode,
            "difficulty": "easy", "character": character, "shot_key": shot,
            "plugin_applied": applied, "first_frame": {
                "control": {"embedding": control, "crop_path": "control.png",
                            "missing_reason": "missing_bbox" if control is None else None},
                "treatment": {"embedding": treatment, "crop_path": "treatment.png",
                              "missing_reason": "missing_bbox" if treatment is None else None}}}


class CrossShotCentroidTest(unittest.TestCase):
    def test_known_vectors_and_order_invariance(self):
        mu, scores, median = centroid_scores([[3, 0], [0, 2]])
        np.testing.assert_allclose(mu, [2**-0.5, 2**-0.5])
        np.testing.assert_allclose(scores, [2**-0.5, 2**-0.5])
        self.assertEqual(median, 0)
        features = [[1, 0], [0, 1], [1, 1]]
        a = centroid_scores(features)
        b = centroid_scores(features[::-1])
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1][::-1])

    def test_missing_side_is_removed_before_both_centroids(self):
        group = paired_groups([row("1:1", [1, 0], [1, 0]),
                               row("1:2", [0, 1], [0, 1]),
                               row("1:3", [1, 1], None)], "first_frame")[0]
        self.assertEqual(group["matched_shot_count"], 2)
        self.assertEqual(group["delta"], 0)
        self.assertEqual(group["excluded_shots"][0]["reasons"], {"treatment": "missing_bbox"})
        summary = aggregate([group])
        self.assertEqual(summary["paired_coverage"], 2 / 3)

    def test_episode_local_groups_and_singletons(self):
        groups = paired_groups([row("1:1", [1, 0], [1, 0]),
                                row("1:1", [0, 1], [0, 1], episode="episode_b")], "first_frame")
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(g["skip_reason"] == "single_scheduled_appearance" for g in groups))
        self.assertIsNone(aggregate(groups)["control_mean"])

    def test_fallback_score_can_change_when_centroid_changes(self):
        inputs = [row("1:1", [1, 0], [1, 0], applied=False),
                  row("1:2", [0, 1], [1, 0]), row("1:3", [0, 1], [1, 0])]
        full = paired_groups(inputs, "first_frame")[0]
        self.assertGreater(full["shots"][0]["delta"], 0)
        injected = paired_groups(inputs, "first_frame", injected_only=True)[0]
        self.assertEqual(injected["matched_shot_count"], 2)
        self.assertAlmostEqual(injected["delta"], 0)

    def test_appearance_weighted_aggregation(self):
        groups = paired_groups([row("1:1", [1, 0], [1, 0]),
                                row("1:2", [0, 1], [0, 1]),
                                *[row(f"1:{i}", [1, 0], [1, 0], character="other") for i in (1, 2, 3)]], "first_frame")
        value = aggregate(groups)
        self.assertAlmostEqual(value["control_mean"], (2 * 2**-0.5 + 3) / 5)
        self.assertEqual(value["scored_paired_appearance_count"], 5)

    def test_duplicates_and_degenerate_vectors_are_not_scored(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_groups([row("1:1", [1, 0], [1, 0])] * 2, "first_frame")
        group = paired_groups([row("1:1", [1, 0], [1, 0]),
                               row("1:2", [-1, 0], [1, 0])], "first_frame")[0]
        self.assertTrue(group["skip_reason"].startswith("degenerate_features"))
        self.assertIsNone(group["delta"])

    def test_video_selection_uses_quality_not_identity_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 24, (64, 64))
            self.assertTrue(writer.isOpened())
            for i in range(6):
                frame = np.full((64, 64, 3), 128, dtype=np.uint8)
                if i == 5:
                    pattern = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
                    frame = np.repeat(pattern[:, :, None], 3, axis=2)
                writer.write(frame)
            writer.release()
            frames = [{"frame_index": i, "selected_face_bbox": [8, 8, 48, 48],
                       "identity_cosine": 1 if i == 0 else 0} for i in range(6)]
            crop, audit = representative_video_crop(path, frames, 0.15, 3)
            self.assertIsNotNone(crop)
            self.assertEqual(audit["selected"]["frame_index"], 5)
            self.assertEqual(audit["sampled_frame_indices"], [0, 2, 5])
            for i in [0, 2, 5]:
                frames[i]["selected_face_bbox"] = None
            crop, audit = representative_video_crop(path, frames, 0.15, 3)
            self.assertIsNone(crop)
            self.assertEqual(audit["missing_reason"], "no_face_in_sampled_frames")
            with self.assertRaisesRegex(ValueError, "more frames"):
                representative_video_crop(path, frames[:-1], 0.15, 3)


if __name__ == "__main__":
    unittest.main()
