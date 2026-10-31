from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretest'))
from qwen140_multiface_overnight_extra import can_start, estimate_seconds


class AdmissionTest(unittest.TestCase):
    def test_estimate_accounts_for_assets_and_existing_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            item = {'output_root': directory, 'episode': {'shots': [{'entity_schedule': {'characters': ['A']}}] * 10}}
            self.assertEqual(estimate_seconds(item), 10 * 300 + 150 + 480)
            p = Path(directory) / 'episodes/e/shot_1_1';p.mkdir(parents=True);(p / 'result.json').write_text('{}')
            (Path(directory) / 'face_injection_policy.json').write_text('{}')
            self.assertEqual(estimate_seconds(item), 9 * 300 + 480)
    def test_episode_must_fit_before_noon(self):
        with tempfile.TemporaryDirectory() as directory:
            item = {'output_root': directory, 'episode': {'shots': [{'entity_schedule': {'characters': []}}] * 10}}
            now = datetime(2026, 10, 30, 3, 30, tzinfo=timezone.utc)
            self.assertFalse(can_start(item, now + timedelta(minutes=30), now))
            self.assertTrue(can_start(item, now + timedelta(hours=2), now))


if __name__ == '__main__':
    unittest.main()
