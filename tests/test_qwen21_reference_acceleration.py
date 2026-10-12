"""A placement amendment must not redirect injected frames or change frozen core algorithms."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretest'))
from run_entitybench_qwen21_20ep_accelerated import reference_command,validate_probe,WORKER
from prepare_entitybench_qwen21_20ep import sha256


class ReferenceAccelerationTests(unittest.TestCase):
    def test_only_reference_stage_is_redirected_without_changing_arguments(self):
        cmd=['python','-u','pretest/run_qwen21_entitybench_20ep_images.py','--output-root','/some/root','--stage','references']
        result=reference_command(cmd)
        self.assertEqual(result[2],str(WORKER));self.assertEqual(result[:2]+result[3:],cmd[:2]+cmd[3:])
        frames=copy.deepcopy(cmd);frames[-1]='frames'
        self.assertEqual(reference_command(frames),frames)
        video=['python','pretest/run_lightx2v_a14b_4step_continuation.py']
        self.assertEqual(reference_command(video),video)

    def test_exact_output_guard_rejects_drift_and_changed_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'sample.png';source.write_bytes(b'fixture');h=sha256(source)
            row=dict(seed=1,exact_png_match=True,image_sha256=h,expected_sha256=h)
            report=dict(results=dict(cuda=dict(rows=[row,dict(row,seed=2)],inference_seconds=20)),
                baseline_inference_seconds=100,baseline=[dict(source=str(source),expected_sha256=h)])
            validate_probe(report)
            changed=copy.deepcopy(report);changed['results']['cuda']['rows'][1]['image_sha256']='drift'
            with self.assertRaises(AssertionError):validate_probe(changed)
            source.write_bytes(b'changed')
            with self.assertRaises(AssertionError):validate_probe(report)


if __name__=='__main__':unittest.main()
