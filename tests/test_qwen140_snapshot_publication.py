"""Failure boundaries for the independent publication pointer, without inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'pretest/qwen140_publish_full140_snapshot.py'
spec = importlib.util.spec_from_file_location('publisher', SOURCE)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / 'old'
        self.old.mkdir()
        self.snapshot = self.root / 'versions' / 'new'
        self.snapshot.mkdir(parents=True)
        self.pointer = self.root / 'published'
        self.pointer.symlink_to(self.old, target_is_directory=True)
        self.input = self.root / 'source.json'
        self.input.write_text('original input')
        publisher.write(self.snapshot / 'snapshot_inputs.json', {
            'sha256': {str(self.input): publisher.sha(self.input)}})
        publisher.write(self.snapshot / 'status.json', {
            'status': 'complete_reaggregation_only', 'replaced_episodes': 29, 'replaced_shots': 333})
        validation = {key: True for key in (
            'baseline_aggregate_reproduced_exactly_for_all_tiers',
            'no_scoring_configuration_or_model_differences',
            'all_original_and_selected_video_hashes_verified', 'canonical_links_verified',
            'original_and_rerun_parent_chains_verified', 'no_neural_inference')}
        validation.update(episode_count=140, shot_count=2491, replaced_episode_count=29,
                          replaced_shot_count=333, vlm_calls=0)
        publisher.write(self.snapshot / 'validation.json', validation)
        publisher.write(self.snapshot / 'selection_manifest.json', {
            'num_episodes': 140, 'num_shots': 2491, 'rerun_episodes': 29, 'rerun_shots': 333,
            'episodes': [{'episode_id': str(i)} for i in range(140)],
            'videos': {str(i): {} for i in range(2491)}, 'vlm_calls': 0})

    def test_valid_snapshot_replaces_only_pointer(self):
        publisher.publish(self.snapshot, self.pointer)
        self.assertEqual(self.pointer.resolve(), self.snapshot)
        self.assertTrue(self.old.is_dir())
        self.assertIn('versions/new/index.html', self.pointer.with_suffix('.html').read_text())

    def test_changed_input_preserves_previous_publication(self):
        self.input.write_text('changed during build')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            publisher.publish(self.snapshot, self.pointer)
        self.assertEqual(self.pointer.resolve(), self.old)

    def test_incomplete_snapshot_preserves_previous_publication(self):
        publisher.write(self.snapshot / 'status.json', {'status': 'validation_failed'})
        with self.assertRaises(AssertionError):
            publisher.publish(self.snapshot, self.pointer)
        self.assertEqual(self.pointer.resolve(), self.old)

    def test_real_directory_never_replaced(self):
        self.pointer.unlink()
        self.pointer.mkdir()
        with self.assertRaisesRegex(ValueError, 'real file/directory'):
            publisher.publish(self.snapshot, self.pointer)
        self.assertTrue(self.pointer.is_dir())
        self.assertFalse(self.pointer.is_symlink())


if __name__ == '__main__':
    unittest.main()
