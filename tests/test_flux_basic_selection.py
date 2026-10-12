"""Text-only non-face routing must be explicit and keep original shot seeds."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretest'))
from prepare_entitybench_flux_controls import _select_basic_shots


class BasicSelectionTests(unittest.TestCase):
    def setUp(self):
        self.episode = dict(scenes=[dict(scene_num=1, video_prompts=['Human', 'Dog', 'Group'])],
            entity_schedule={'1:1': {'characters': ['Person']}, '1:2': {'characters': ['Blaze']},
                             '1:3': {'characters': ['A', 'B']}})

    def test_default_still_excludes_single_characters(self):
        self.assertEqual([s['shot_key'] for s in _select_basic_shots(self.episode, None)], ['1:3'])

    def test_explicit_non_face_selection_keeps_global_index(self):
        shots = _select_basic_shots(self.episode, {'1:2', '1:3'}, True)
        self.assertEqual([s['shot_key'] for s in shots], ['1:2', '1:3'])
        self.assertEqual([s['shot_index'] for s in shots], [1, 2])

    def test_non_face_mode_requires_explicit_known_shots(self):
        with self.assertRaises(ValueError):
            _select_basic_shots(self.episode, None, True)
        with self.assertRaises(ValueError):
            _select_basic_shots(self.episode, {'99:1'}, True)


if __name__ == '__main__':
    unittest.main()
