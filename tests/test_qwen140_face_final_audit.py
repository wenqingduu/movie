import ast
import unittest
from pathlib import Path
import numpy as np

tree = ast.parse((Path(__file__).resolve().parents[1] / 'pretest/audit_entitybench_qwen140_face_final.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_published_mean')
namespace = dict(np=np)
exec(compile(ast.Module(body=[function], type_ignores=[]), 'published_mean_validation', 'exec'), namespace)
validate = namespace['validate_published_mean']


class PublishedPrecision(unittest.TestCase):
    def test_official_four_decimal_summary_accepts_raw_rounding_difference(self):
        self.assertEqual(validate(0.8123, [0.812341, 0.812341]), 0.8123)

    def test_wrong_published_score_is_rejected(self):
        with self.assertRaises(AssertionError):
            validate(0.8124, [0.812341, 0.812341])

    def test_empty_and_zero_are_distinct(self):
        self.assertIsNone(validate(None, []))
        self.assertEqual(validate(0.0, [0.0]), 0.0)
        with self.assertRaises(AssertionError):
            validate(0.0, [])


if __name__ == '__main__':
    unittest.main()
