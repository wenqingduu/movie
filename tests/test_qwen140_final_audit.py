import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretest'))
from audit_entitybench_qwen140_final import validate_missing_vlm, validate_episode_absent


class EmptyOfficialMetricSchema(unittest.TestCase):
    def setUp(self):
        self.names = [f'intra_face_metric_{i}' for i in range(39)]
        self.split = dict(easy=['a', 'b'], medium=['c'], hard=['d'])
        self.report = {}
        for tier, count in [('all', 4), ('easy', 2), ('medium', 1), ('hard', 1)]:
            self.report[tier] = dict(num_episodes=count, **{
                key: dict(value=None, n_episodes_evaluated=0, n_episodes_total=count) for key in self.names})

    def test_official_null_value_schema_and_all_tier_coverage(self):
        before = copy.deepcopy(self.report)
        validate_missing_vlm(self.report, self.split, self.names)
        self.assertEqual(self.report, before)

    def test_non_null_vlm_value_is_rejected(self):
        self.report['hard'][self.names[0]]['value'] = 0.0
        with self.assertRaises(AssertionError):
            validate_missing_vlm(self.report, self.split, self.names)

    def test_missing_field_or_incorrect_coverage_cannot_pass(self):
        for replacement in [{}, dict(mean=None, n_episodes_evaluated=0, n_episodes_total=4),
                            dict(value=None, n_episodes_evaluated=1, n_episodes_total=4),
                            dict(value=None, n_episodes_evaluated=0, n_episodes_total=3)]:
            with self.subTest(replacement=replacement):
                report = copy.deepcopy(self.report)
                report['all'][self.names[0]] = replacement
                with self.assertRaises(AssertionError):
                    validate_missing_vlm(report, self.split, self.names)

    def test_episode_omission_matches_official_aggregation_but_non_null_is_rejected(self):
        validate_episode_absent({}, self.names)
        validate_episode_absent({self.names[0]: dict(value=None, n_evaluated=0)}, self.names)
        for bad in [dict(value=0, n_evaluated=0), dict(value=None, n_evaluated=1)]:
            with self.assertRaises(AssertionError):
                validate_episode_absent({self.names[0]: bad}, self.names)


if __name__ == '__main__':
    unittest.main()
