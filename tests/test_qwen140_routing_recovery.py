import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretest'))
from qwen140_routing_recovery import gated_metadata, allocated_asset_bytes, resume_budget


class RecoverySafety(unittest.TestCase):
    def test_semantic_gate_preserves_reference_selection_and_fingerprint(self):
        original = dict(face_eligible=True, face_skip_reason=None, selected_sha256='reference',
                        selected_seed=123, candidates=[{'sha256': 'candidate'}], dependency_fingerprint='frozen')
        role = dict(episode_id='episode', character='Blaze', description='A Dalmatian pup')
        result = gated_metadata(original, role, 'Dalmatian pup', 'script')
        self.assertFalse(result['face_eligible'])
        self.assertTrue(original['face_eligible'])
        for key in ['selected_sha256', 'selected_seed', 'candidates', 'dependency_fingerprint']:
            self.assertEqual(result[key], original[key])
        with self.assertRaises(AssertionError):
            gated_metadata(original, dict(role, description='A person'), 'Dalmatian pup', 'script')

    def test_resume_credit_retains_reserve_and_other_resource_failures(self):
        approval = dict(startup_min_free_gib=50, product_budget_gib=40, reserve_gib=10)
        base = dict(startup_min_free_gib=50, free_gib=46, reasons=['startup_free_space_below_50GiB'])
        result = resume_budget(base, approval, 14 * 1024**3)
        self.assertTrue(result['ready'])
        self.assertEqual(result['resume_accounting']['required_free_gib'], 36)
        self.assertFalse(resume_budget(dict(base, free_gib=35), approval, 14*1024**3)['ready'])
        self.assertFalse(resume_budget(dict(base, reasons=base['reasons']+['gpu_busy']), approval, 14*1024**3)['ready'])
        self.assertFalse(resume_budget(dict(base, free_gib=9), approval, 100*1024**3)['ready'])
        with self.assertRaises(AssertionError):
            resume_budget(dict(base, startup_min_free_gib=40), approval, 14*1024**3)

    def test_budget_credit_excludes_links_and_other_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            assets = root / 'assets'
            assets.mkdir()
            artifact = assets / 'reference.png'
            artifact.write_bytes(b'x'*8192)
            expected = artifact.stat().st_blocks * 512
            os.link(artifact, assets / 'hardlink')
            (assets / 'symlink').symlink_to(artifact)
            elsewhere = root / 'elsewhere'
            elsewhere.mkdir()
            (elsewhere / 'huge').write_bytes(b'x'*16384)
            (assets / 'other_run').symlink_to(elsewhere, target_is_directory=True)
            self.assertEqual(allocated_asset_bytes(root), expected)


if __name__ == '__main__':
    unittest.main()
